# -*- coding: utf-8 -*-
"""Export a vllm-sr/d3-* checkpoint to ONNX: one decoder graph per fixed length L, plus one vision encoder.

    inputs   input_ids [1, L] int64      right-padded prompt of ONE question
             attention_mask [1, L] int64 1 for real tokens, 0 for padding
             position_ids [3, 1, L] int64 M-RoPE positions (text: 0..L-1 on all three axes)
             read_position [1] int64     index of the last real token
             image_features [N, hidden]  vision encoder output (N may be 0)
             image_slots [N] int64       where in the prompt those rows go (the <|image_pad|> positions)
    output   code_logits [255] float32   FP32 readout of the last real token

The length L is static: the Gated DeltaNet layers run a Python loop over 64-token chunks, which the exporter unrolls.
Run in the export environment (torch + transformers 5.17.0 + onnx + onnxscript):

    uv run --group export python export/export_onnx.py --model edge --hf <checkpoint dir> --seq-len 384
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
from graph_modules import DecisionGraph, VisionGraph, load_backbone, load_readout  # noqa: E402
from models import MODELS, model_paths  # noqa: E402

CHUNK = 64
DEFAULT_SEQ_LEN = 384


def write_meta(out_dir: pathlib.Path, hf: pathlib.Path, pad_id: int, model_cfg: dict, name: str) -> None:
    onnx_dir = out_dir / "onnx"
    found = sorted(int(m.group(1)) for p in onnx_dir.glob("model_L*.onnx") if (m := re.fullmatch(r"model_L(\d+)\.onnx", p.name)))
    vision = onnx_dir / "vision_encoder.onnx"
    meta = {"seq_len": DEFAULT_SEQ_LEN if DEFAULT_SEQ_LEN in found else found[0], "pad_token_id": pad_id,
            "model_name": "d3-" + name, "chunk_size": CHUNK, "buckets": found, "vision_graph": vision.name if vision.exists() else None,
            "hidden_size": model_cfg["text_config"]["hidden_size"], "image_token_id": model_cfg["image_token_id"],
            "spatial_merge_size": model_cfg["vision_config"]["spatial_merge_size"],
            "num_grid_per_side": int(model_cfg["vision_config"]["num_position_embeddings"] ** 0.5),
            "vision_out_hidden": model_cfg["vision_config"]["out_hidden_size"]}
    (out_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--hf", required=True, help="directory with the vllm-sr/d3-* files")
    ap.add_argument("--out", help="default: models/<model>")
    ap.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--skip-vision", action="store_true")
    args = ap.parse_args()
    if args.seq_len % CHUNK:
        raise SystemExit("--seq-len must be a multiple of %d" % CHUNK)
    hf = pathlib.Path(args.hf)
    out_dir = pathlib.Path(args.out) if args.out else model_paths(args.model)[0]
    (out_dir / "onnx").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((hf / "config.json").read_text(encoding="utf-8"))
    dec = json.loads((hf / "decision_config.json").read_text(encoding="utf-8"))
    pad_id = 248044  # <|endoftext|>, the tokenizer's pad token

    print("loading %s (fp32, eager attention) ..." % hf, flush=True)
    backbone = load_backbone(hf)
    hidden = backbone.config.text_config.hidden_size

    if not args.skip_text:
        graph = DecisionGraph(backbone, load_readout(hf)).eval()
        L = args.seq_len
        ids = torch.full((1, L), pad_id, dtype=torch.long)
        ids[0, :100] = torch.tensor(dec["token_ids"][:100])
        mask = torch.zeros(1, L, dtype=torch.long)
        mask[0, :100] = 1
        pos = torch.arange(L).view(1, 1, L).expand(3, 1, L).contiguous()
        read = torch.tensor([99], dtype=torch.long)
        feats = torch.zeros(4, hidden)
        slots = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        with torch.no_grad():
            ref = graph(ids, mask, pos, read, feats, slots)
        print("torch logits", tuple(ref.shape), flush=True)
        target = out_dir / "onnx" / ("model_L%d.onnx" % L)
        names = ["input_ids", "attention_mask", "position_ids", "read_position", "image_features", "image_slots"]
        started = time.time()
        torch.onnx.export(graph, (ids, mask, pos, read, feats, slots), str(target), input_names=names,
                          output_names=["code_logits"],
                          dynamic_shapes={"input_ids": {}, "attention_mask": {}, "position_ids": {}, "read_position": {},
                                          "image_features": {0: "visual"}, "image_slots": {0: "visual"}},
                          opset_version=18, dynamo=True, external_data=True, optimize=False)
        print("exported decoder L=%d in %.1f s -> %s" % (L, time.time() - started, target), flush=True)

    if not args.skip_vision:
        vision = VisionGraph(backbone).eval()
        n = 256
        px = torch.randn(n, 3 * cfg["vision_config"]["temporal_patch_size"] * cfg["vision_config"]["patch_size"] ** 2)
        idx = torch.randint(0, 2304, (n, 4))
        wts = torch.rand(n, 4)
        vpos = torch.randint(0, 16, (n, 2))
        with torch.no_grad():
            probe = vision(px, idx, wts, vpos)
        print("vision features", tuple(probe.shape), flush=True)
        target = out_dir / "onnx" / "vision_encoder.onnx"
        started = time.time()
        torch.onnx.export(vision, (px, idx, wts, vpos), str(target),
                          input_names=["pixel_values", "interp_indices", "interp_weights", "position_ids"],
                          output_names=["image_features"],
                          dynamic_shapes={"pixel_values": {0: "patches"}, "interp_indices": {0: "patches"},
                                          "interp_weights": {0: "patches"}, "position_ids": {0: "patches"}},
                          opset_version=18, dynamo=True, external_data=True, optimize=False)
        print("exported vision in %.1f s -> %s" % (time.time() - started, target), flush=True)

    write_meta(out_dir, hf, pad_id, cfg, args.model)


if __name__ == "__main__":
    main()
