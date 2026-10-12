# -*- coding: utf-8 -*-
"""d3 (Decision 3.0) on onnxruntime + tokenizers + numpy.

One question is one forward pass: the prompt lists every option under a single-token answer code, the decoder graph
returns the 255 code logits of the last prompt token, and the answer is the softmax over the first N codes (N = number
of options). Nothing is generated. A request with several questions runs them one after another (the graph is fixed
at L tokens and right-padded, see README).

    engine = D3Engine("models/onnx")
    engine.system_one(state="...", questions={"route": {"type": "choice", "instructions": "...", "criteria": {...}}},
                      images=["photo.png"])
    engine.ask(state, images, kind, instr, options)       # the single-question form used by the benchmark adapter

Runtime dependencies: onnxruntime, tokenizers, numpy; pillow only when a request carries pictures.
"""
from __future__ import annotations

import gc
import json
import math
import pathlib
import re
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from . import chat
from .format import (IMAGE_PAD, MAX_OPTIONS, VIDEO_PAD, Question, normalize_question, product_answer, render)

__all__ = ["D3Engine", "DEFAULT_SEQ_LEN", "gpu_providers", "resolve_providers", "find_buckets", "load_tokenizer"]

DEFAULT_SEQ_LEN = 384
_BUCKET = re.compile(r"model_L(\d+)\.onnx")


def find_buckets(onnx_dir: pathlib.Path) -> Dict[int, pathlib.Path]:
    found = {}
    for path in pathlib.Path(onnx_dir).glob("model_L*.onnx"):
        match = _BUCKET.fullmatch(path.name)
        if match:
            found[int(match.group(1))] = path
    return dict(sorted(found.items()))


GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
DEVICES = ("cpu", "cuda", "auto")


def gpu_providers() -> List[str]:
    """GPU execution providers available in this onnxruntime build, CPU last."""
    available = ort.get_available_providers()
    return [p for p in GPU_PROVIDERS if p in available] + ["CPUExecutionProvider"]


def resolve_providers(device: str = "cpu", available: Optional[Sequence[str]] = None, require_gpu: bool = False) -> List[str]:
    """Execution providers for a ``device`` choice: ``cpu`` (default) -> CPU only; ``cuda`` -> CUDA then CPU;
    ``auto`` -> the first GPU provider this build has (CUDA, else DirectML), then CPU.
    ``require_gpu``: raise RuntimeError instead of quietly planning for the CPU when no GPU provider is available.
    ``available`` defaults to ``onnxruntime.get_available_providers()`` (a parameter so that this logic can be tested)."""
    if device not in DEVICES:
        raise ValueError("device must be one of %s, got %r" % (", ".join(DEVICES), device))
    have = list(ort.get_available_providers() if available is None else available)
    if device == "cpu":
        if require_gpu:
            raise RuntimeError("require_gpu is set but device is 'cpu'")
        return ["CPUExecutionProvider"]
    if device == "cuda":
        wanted = [p for p in ("CUDAExecutionProvider",) if p in have]
    else:
        wanted = [p for p in GPU_PROVIDERS if p in have][:1]
    if not wanted:
        if require_gpu or device == "cuda":
            if require_gpu:
                raise RuntimeError("no GPU execution provider in this onnxruntime build (have: %s); install onnxruntime-gpu "
                                   "(see README, 'Running on a GPU')" % ", ".join(have))
            print("warning: device='cuda' but this onnxruntime build has no CUDAExecutionProvider (have: %s); running on the CPU"
                  % ", ".join(have), file=sys.stderr)
        return ["CPUExecutionProvider"]
    return wanted + ["CPUExecutionProvider"]


def _session(path: pathlib.Path, threads: Optional[int], providers: Optional[Sequence[str]] = None,
             require_gpu: bool = False) -> ort.InferenceSession:
    if not path.exists():
        raise FileNotFoundError("%s not found (run download_model.py, or export/prepare_model.py)" % path)
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    wanted = list(providers) if providers else ["CPUExecutionProvider"]
    gpu = wanted[0] != "CPUExecutionProvider"
    try:
        session = ort.InferenceSession(str(path), sess_options=options, providers=wanted)
    except Exception as exc:        # a GPU that is too small or missing DLLs
        if not gpu:
            raise
        if require_gpu:
            raise RuntimeError("%s could not load %s: %s" % (wanted[0], path.name, str(exc).splitlines()[0][:200])) from exc
        print("warning: %s could not load %s (%s); using the CPU" % (wanted[0], path.name, str(exc).splitlines()[0][:160]),
              file=sys.stderr)
        return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if gpu and session.get_providers()[0] != wanted[0]:    # onnxruntime dropped the GPU provider without raising
        msg = "%s was requested for %s but the session runs on %s" % (wanted[0], path.name, session.get_providers()[0])
        if require_gpu:
            raise RuntimeError(msg)
        print("warning: %s (missing CUDA / cuDNN DLLs on PATH?)" % msg, file=sys.stderr)
    return session


#: transformers 5.17's Qwen2Tokenizer (what the vendor runtime tokenizes with) replaces the pre-tokenizer regex of tokenizer.json by
#: this one (letters without combining marks); tokenizer.json itself says ``[\p{L}\p{M}]+``. The two differ on text with combining
#: marks (Devanagari, emoji variation selectors, ...). The vendor's behaviour is the reference here.
VENDOR_PRETOKENIZE_REPLACEMENTS = (("[\\p{L}\\p{M}]+", "\\p{L}+"), ("\\p{L}\\p{M}\\p{N}", "\\p{L}\\p{N}"))


def load_tokenizer(path) -> Tokenizer:
    """tokenizer.json with the pre-tokenizer regex the vendor runtime actually uses (see VENDOR_PRETOKENIZE_REPLACEMENTS)."""
    spec = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for item in spec["pre_tokenizer"]["pretokenizers"]:
        if item["type"] == "Split":
            regex = item["pattern"]["Regex"]
            for old, new in VENDOR_PRETOKENIZE_REPLACEMENTS:
                regex = regex.replace(old, new)
            item["pattern"]["Regex"] = regex
    return Tokenizer.from_str(json.dumps(spec, ensure_ascii=False))


class D3Engine:
    """Load once, then answer System One requests (choice / noul / score questions)."""

    def __init__(self, model_dir: Union[str, pathlib.Path], *, seq_len: Optional[int] = None,
                 intra_op_threads: Optional[int] = None, providers: Optional[Sequence[str]] = None,
                 max_image_pixels: Optional[int] = None, model_name: Optional[str] = None,
                 device: str = "cpu", require_gpu: bool = False):
        """``seq_len``: the bucket to start with (default: the smallest one on disk that is >= 384, else the largest).
        A request that does not fit makes the engine swap to the smallest larger bucket on disk (the old graph is
        released first, so two decoders are never in memory); the engine then stays there.
        ``device``: ``cpu`` (default), ``cuda`` (CUDA, then CPU) or ``auto``; an explicit ``providers`` list wins. ``require_gpu``:
        raise if the GPU cannot be used instead of falling back to the CPU. What really runs is ``engine.providers_used``.
        ``max_image_pixels``: cap a picture below the vendor's 1,638,400 pixels (1600 tokens); changes the answers.
        """
        self.model_dir = pathlib.Path(model_dir)
        meta = json.loads((self.model_dir / "graph_meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.decision = json.loads((self.model_dir / "decision_config.json").read_text(encoding="utf-8"))
        self.model_name = model_name or meta.get("model_name") or self.model_dir.name
        self.pad_token_id = int(meta["pad_token_id"])
        self.hidden_size = int(meta["hidden_size"])
        self.image_token_id = int(meta["image_token_id"])
        self.merge_size = int(meta["spatial_merge_size"])
        self.codes: List[str] = list(self.decision["codes"])
        self.temperature = float(self.decision.get("temperature", 1.0))
        self._threads = intra_op_threads
        self._providers = list(providers) if providers else resolve_providers(device, require_gpu=require_gpu)
        self._require_gpu = require_gpu
        self.session = None
        self.seq_len = 0
        onnx_dir = self.model_dir / "onnx"
        self.buckets = list(find_buckets(onnx_dir))
        if not self.buckets:
            raise FileNotFoundError("no model_L*.onnx in %s" % onnx_dir)
        want = seq_len if seq_len else DEFAULT_SEQ_LEN
        larger = [n for n in self.buckets if n >= want]
        self._load_graph(min(larger) if larger else max(self.buckets))
        self.vision = None
        vision = onnx_dir / (meta.get("vision_graph") or "vision_encoder.onnx")
        if meta.get("vision_graph") and vision.exists():
            self.vision = _session(vision, intra_op_threads, self._providers, self._require_gpu)
        self.tokenizer = load_tokenizer(self.model_dir / "tokenizer.json")
        self._image_config = None
        self._max_image_pixels = max_image_pixels
        self.last_timing: Dict[str, float] = {}

    # ----------------------------------------------------------------- graphs

    def _load_graph(self, length: int) -> None:
        self.session = None                 # release the old decoder first
        gc.collect()
        self.session = _session(self.model_dir / "onnx" / ("model_L%d.onnx" % length), self._threads, self._providers, self._require_gpu)
        self.seq_len = int(self.session.get_inputs()[0].shape[1])

    def _fit(self, tokens: int) -> None:
        if tokens <= self.seq_len:
            return
        bigger = [n for n in self.buckets if n >= tokens]
        if not bigger:
            raise ValueError("the prompt has %d tokens but the largest graph holds %d; truncation is forbidden, "
                             "export a larger one (export/prepare_model.py --seq-len %d)"
                             % (tokens, max(self.buckets), -(-tokens // 64) * 64))
        self._load_graph(min(bigger))

    @property
    def max_tokens(self) -> int:
        return max(self.buckets)

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    @property
    def providers_used(self) -> Dict[str, str]:
        """The provider each session really runs on, e.g. {"decoder": "CUDAExecutionProvider", "vision": ...}."""
        used = {"decoder": self.session.get_providers()[0]}
        if self.vision is not None:
            used["vision"] = self.vision.get_providers()[0]
        return used

    @property
    def image_config(self):
        if self._image_config is None:
            from .image import ImageConfig

            overrides = {"max_pixels": int(self._max_image_pixels)} if self._max_image_pixels else {}
            self._image_config = ImageConfig.from_model_dir(self.model_dir, **overrides)
        return self._image_config

    # ------------------------------------------------------------- encoding

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def _features(self, prepared) -> np.ndarray:
        """Vision encoder output of every picture (one call per picture), concatenated: [tokens, hidden]."""
        rows = []
        for image in prepared:
            rows.append(self.vision.run(["image_features"], {
                "pixel_values": image.pixel_values, "interp_indices": image.interp_indices,
                "interp_weights": image.interp_weights, "position_ids": image.position_ids})[0])
        return np.concatenate(rows, axis=0)

    def logits(self, ids: Sequence[int], features: Optional[np.ndarray] = None, grids: Sequence[Sequence[int]] = ()) -> np.ndarray:
        """The 255 code logits for one prompt (float32). ``features``: picture rows to scatter into the image slots."""
        real = list(ids)
        self._fit(len(real))
        pad = self.seq_len - len(real)
        x = np.asarray([real + [self.pad_token_id] * pad], dtype=np.int64)
        mask = np.asarray([[1] * len(real) + [0] * pad], dtype=np.int64)
        if features is not None and len(grids):
            slots = np.flatnonzero(np.asarray(real) == self.image_token_id).astype(np.int64)
            if slots.size != features.shape[0]:
                raise RuntimeError("vision encoder produced %d rows for %d picture tokens" % (features.shape[0], slots.size))
            positions = chat.position_ids(real, grids, self.image_token_id, self.merge_size)
            tail = np.arange(pad, dtype=np.int64) + (int(positions.max()) + 1)
            positions = np.concatenate([positions, np.broadcast_to(tail, (3, 1, pad))], axis=2)
            feats = features.astype(np.float32)
        else:
            slots = np.zeros((0,), dtype=np.int64)
            feats = np.zeros((0, self.hidden_size), dtype=np.float32)
            positions = np.arange(self.seq_len, dtype=np.int64)[None, None, :].repeat(3, axis=0)
        feeds = {"input_ids": x, "attention_mask": mask, "position_ids": np.ascontiguousarray(positions),
                 "read_position": np.asarray([len(real) - 1], dtype=np.int64), "image_features": feats, "image_slots": slots}
        return self.session.run(["code_logits"], feeds)[0]

    def probabilities(self, logits: np.ndarray, count: int) -> List[float]:
        """Softmax over the first ``count`` codes (the others are masked), temperature from decision_config.json."""
        row = logits[:count].astype(np.float64) / self.temperature
        row = np.exp(row - row.max())
        return (row / row.sum()).tolist()

    # -------------------------------------------------------------- request

    def system_one(self, *, state: Any, questions: Mapping[str, Any], images: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
        """Typed Choice / Noul / Score answers about one state: ``{"model", "answers", "usage"}``.

        A question that cannot be answered gets ``{"type", "error", "message"}`` (``invalid_question`` or
        ``max_length_exceeded``); the other questions are still answered.
        """
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a nonempty mapping of question IDs to questions")
        if any(not isinstance(k, str) or not k for k in questions):
            raise ValueError("question IDs must be nonempty strings")
        json.dumps(state, ensure_ascii=False, allow_nan=False)
        started = time.perf_counter()
        prepared = []
        counts: List[int] = []
        if images:
            if self.vision is None:
                raise NotImplementedError("this build has no vision encoder; download it or export without --skip-vision")
            from .image import prepare_image

            prepared = [prepare_image(src, self.image_config) for src in images]
            counts = [p.tokens for p in prepared]
        t_prep = time.perf_counter()

        answers: Dict[str, Any] = {}
        runnable: List[tuple] = []
        for key, raw in questions.items():
            try:
                q = normalize_question(raw)
                text = render(state, q.rendered, self.codes, counts)
                if images:
                    probe = render(state, q.rendered, self.codes, [1] * len(images))
                    if probe.count(IMAGE_PAD) != len(images) or VIDEO_PAD in probe:
                        raise ValueError("the state or question contains a literal picture placeholder token")
                elif IMAGE_PAD in text or VIDEO_PAD in text:
                    raise ValueError("the state or question contains a literal picture placeholder token")
            except ValueError as exc:
                kind = raw.get("type") if isinstance(raw, Mapping) else None
                answers[key] = {"type": kind, "error": "invalid_question", "message": str(exc)}
                continue
            ids = self._encode(text)
            if len(ids) > self.max_tokens:
                answers[key] = {"type": q.kind, "error": "max_length_exceeded",
                                "message": "the question prompt has %d tokens, over the largest exported graph (%d); "
                                           "nothing was truncated" % (len(ids), self.max_tokens)}
                continue
            runnable.append((key, q, ids))
        features = self._features(prepared) if prepared and runnable else None
        grids = [p.grid for p in prepared]
        t_vis = time.perf_counter()
        total = 0
        if runnable:
            self._fit(max(len(ids) for _, _, ids in runnable))      # one bucket for the whole request
        for key, q, ids in runnable:
            lg = self.logits(ids, features, grids)
            answers[key] = product_answer(q, self.probabilities(lg, len(q.keys)))
            total += len(ids)
        done = time.perf_counter()
        self.last_timing = {"prepare_s": t_prep - started, "vision_s": t_vis - t_prep, "decode_s": done - t_vis,
                            "total_s": done - started}
        return {"model": self.model_name, "answers": {k: answers[k] for k in questions},
                "usage": {"input_tokens": total, "output_tokens": 0}}

    def ask(self, state: str, images: Sequence[Any], kind: str, instr: str, options: Any = None):
        """One question. noul -> bool (p(yes) > 0.5); choice -> index into ``options`` ([(key, text), ...]);
        score -> index of the most probable level (``options`` = the level labels)."""
        if kind == "noul":
            q = {"type": "noul", "instructions": instr}
        elif kind == "choice":
            q = {"type": "choice", "instructions": instr, "criteria": {k: t for k, t in options}}
        elif kind == "score":
            q = {"type": "score", "instructions": instr, "criteria": list(options)}
        else:
            raise ValueError("unknown question kind %r" % (kind,))
        out = self.system_one(state=state, questions={"q": q}, images=list(images) or None)["answers"]["q"]
        if "error" in out:
            raise ValueError("%s: %s" % (out["error"], out["message"]))
        if kind == "noul":
            return out["noul"] > 0.5
        probs = list(out["probabilities"].values())
        return int(max(range(len(probs)), key=probs.__getitem__))
