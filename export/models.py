# -*- coding: utf-8 -*-
"""The d3 checkpoints this repository converts, with the revision each port was made and verified against.
d3-mini (5B) is not converted: its fp32 export needs about 25 GB of RAM (see README)."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

MODELS = {
    "edge": dict(repo="vllm-sr/d3-edge", revision="18d61df78684675463af9ae732872478845c5a79", base="Qwen/Qwen3.5-0.8B",
                 weights=["model-00001-of-00001.safetensors"], params="0.59B"),
    "lite": dict(repo="vllm-sr/d3-lite", revision="a8d25a91ef75552803f7009d25b7bd435991abd9", base="Qwen/Qwen3.5-0.8B",
                 weights=["model-00001-of-00001.safetensors"], params="0.85B"),
    "nano": dict(repo="vllm-sr/d3-nano", revision="3aeb1c22a5632aac72353c36b3d0111c40e7dbf4", base="Qwen/Qwen3.5-2B",
                 weights=["model.safetensors"], params="2.21B"),
}
DEFAULT_BUCKETS = (128, 256, 384, 512, 1024, 2048)


def model_paths(model: str):
    """(onnx model dir, artifacts dir) of a model inside this repository."""
    return ROOT / "models" / model, ROOT / "artifacts" / model
