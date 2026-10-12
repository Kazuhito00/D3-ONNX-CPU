# -*- coding: utf-8 -*-
"""Where the model files of each d3 size are."""
from __future__ import annotations

import os
import pathlib
from typing import Optional, Union

#: model key -> (display name, checkpoint, environment variable that overrides the folder)
MODELS = {
    "edge": ("d3-edge 0.6B", "vllm-sr/d3-edge", "D3_EDGE_MODEL_DIR"),
    "lite": ("d3-lite 0.9B", "vllm-sr/d3-lite", "D3_LITE_MODEL_DIR"),
    "nano": ("d3-nano 2B", "vllm-sr/d3-nano", "D3_NANO_MODEL_DIR"),
}
DEFAULT_MODEL = "edge"


def default_model_dir(model: str = DEFAULT_MODEL, root: Union[str, os.PathLike] = ".", override: Optional[str] = None) -> pathlib.Path:
    """``override`` (e.g. --model-dir), else the model's environment variable, else ``<root>/models/<model>``."""
    if model not in MODELS:
        raise ValueError("model must be one of %s, not %r" % (", ".join(MODELS), model))
    if override:
        return pathlib.Path(override)
    if os.environ.get(MODELS[model][2]):
        return pathlib.Path(os.environ[MODELS[model][2]])
    return pathlib.Path(root) / "models" / model
