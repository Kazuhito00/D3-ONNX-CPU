# -*- coding: utf-8 -*-
"""The torch modules that are exported, and that the oracle gates run directly (fp32).

DecisionGraph  prompt tokens (+ optional picture features) -> the 255 code logits of ONE question.
VisionGraph    patchified picture + precomputed positions -> one embedding row per visual token.

Both wrap the stock transformers Qwen3_5Model of the vllm-sr/d3-* checkpoint. What the vendor runtime does around
the backbone (d3_runtime.py) is reproduced here:

* ``noncausal_full_attention``: the softmax-attention layers see the whole prompt (padding masked out), the Gated
  DeltaNet layers stay causal. The vendor adds this with a forward pre-hook that hands the text model a mask dict;
  here the same dict is built inside the graph from ``attention_mask``.
* ``pooling: last``: the hidden state of the last real prompt token (``read_position``) goes through the FP32 readout
  ``[255, hidden]``; the graph returns the 255 logits.

Requests are RIGHT-padded to a fixed length L (the Gated DeltaNet chunk loop is unrolled at trace time, see README).
That equals the vendor's left padding: the padded keys are masked in the softmax layers, and the recurrent layers are
causal, so the real tokens never see the padding. ``read_position`` is the index of the last real token.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionGraph(nn.Module):
    def __init__(self, backbone, readout: torch.Tensor):
        super().__init__()
        self.language_model = backbone.language_model
        self.register_buffer("readout", readout.float().contiguous(), persistent=False)

    def forward(self, input_ids, attention_mask, position_ids, read_position, image_features, image_slots):
        lm = self.language_model
        embeds = lm.embed_tokens(input_ids)
        embeds = embeds.index_copy(1, image_slots, image_features.unsqueeze(0).to(embeds.dtype))
        mask_f = attention_mask.to(embeds.dtype)
        # additive mask for the softmax layers: 0 for real keys, a very large negative number for padding
        full = ((1.0 - mask_f) * torch.finfo(embeds.dtype).min)[:, None, None, :]
        hidden = lm(
            inputs_embeds=embeds,
            attention_mask={"full_attention": full, "linear_attention": attention_mask},
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        last = hidden[0].index_select(0, read_position)          # [1, hidden]
        return (last.float() @ self.readout.T)[0]                # [255]


def _single_image_attention_forward(self, hidden_states, cu_seqlens=None, position_embeddings=None,
                                    max_seqlen=None, **kwargs):
    """Qwen3_5VisionAttention.forward for ONE picture: plain attention over all patches (no per-image split)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb_vision

    seq_length = hidden_states.shape[0]
    q, k, v = (self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0))
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False, scale=self.scaling)
    return self.proj(out.squeeze(0).transpose(0, 1).reshape(seq_length, -1))


def patch_vision_attention():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod

    mod.Qwen3_5VisionAttention.forward = _single_image_attention_forward


class VisionGraph(nn.Module):
    """visual.forward with the grid-dependent helpers moved out: the host passes the position tables."""

    def __init__(self, backbone):
        super().__init__()
        patch_vision_attention()
        self.visual = backbone.visual
        w = self.visual.patch_embed.proj.weight                # [E, C, T, P, P]
        self.register_buffer("lin_weight", w.reshape(w.shape[0], -1).clone(), persistent=False)
        self.register_buffer("lin_bias", self.visual.patch_embed.proj.bias.clone(), persistent=False)

    def forward(self, pixel_values, interp_indices, interp_weights, position_ids):
        v = self.visual
        hidden = F.linear(pixel_values, self.lin_weight, self.lin_bias)
        pos = (v.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
        hidden = hidden + pos.to(hidden.dtype)
        position_embeddings = v.rotary_pos_emb(hidden, position_ids)
        for blk in v.blocks:
            hidden = blk(hidden, cu_seqlens=None, max_seqlen=None, position_embeddings=position_embeddings)
        return v.merger(hidden)


def load_backbone(hf_dir, dtype=torch.float32, attn="eager"):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    model = Qwen3_5Model.from_pretrained(str(hf_dir), dtype=dtype, attn_implementation=attn).eval()
    return model


def load_readout(hf_dir):
    from safetensors.torch import load_file

    return load_file(str(hf_dir) + "/readout.safetensors")["weight"]
