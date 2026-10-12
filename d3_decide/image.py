# -*- coding: utf-8 -*-
"""Picture preprocessing for the d3 vision tower (Qwen3.5-VL), with Pillow and numpy only.

Produces what ``vision_encoder.onnx`` needs for ONE picture:

    pixel_values    float32 [patches, 3 * 2 * 16 * 16]
    grid            (t, h, w) = (1, grid_h, grid_w) in 16 px patches
    interp_indices  int64   [patches, 4]   position-embedding taps (bilinear, align_corners=True), merge-block order
    interp_weights  float32 [patches, 4]
    position_ids    int64   [patches, 2]   (row, col) of every patch for the vision rotary embedding

The resize rule is the vendor runtime's: the checkpoint's own processor with ``size = {shortest_edge: 65536,
longest_edge: 1638400}`` pixels (``d3_runtime.IMAGE_MIN_PIXELS / IMAGE_MAX_PIXELS``), the aspect ratio kept, both sides a
multiple of 32. Every 32 x 32 pixels become one input token. Ported from transformers' ``smart_resize`` and the
Qwen2-VL patchify, and from ``transformers.vision_utils`` for the two position tables.
"""
from __future__ import annotations

import base64
import binascii
import io
import math
import pathlib
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from PIL import Image

ImageSource = Union[str, pathlib.Path, bytes, "Image.Image"]

IMAGE_MIN_PIXELS = 65_536
IMAGE_MAX_PIXELS = 1_638_400


@dataclass(frozen=True)
class ImageConfig:
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    num_grid_per_side: int = 48
    image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    min_pixels: int = IMAGE_MIN_PIXELS
    #: Lowering this caps the prompt (tokens = pixels / 1024) but changes the answers: the vendor reads up to 1.6 MP.
    max_pixels: int = IMAGE_MAX_PIXELS
    max_source_pixels: int = 40_000_000       # decode bomb guard

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    @classmethod
    def from_model_dir(cls, model_dir, **overrides) -> "ImageConfig":
        import json

        cfg = json.loads((pathlib.Path(model_dir) / "preprocessor_config.json").read_text(encoding="utf-8"))
        meta_path = pathlib.Path(model_dir) / "graph_meta.json"
        grid = int(json.loads(meta_path.read_text(encoding="utf-8")).get("num_grid_per_side", 48)) if meta_path.exists() else 48
        values = {"patch_size": cfg.get("patch_size", 16), "temporal_patch_size": cfg.get("temporal_patch_size", 2),
                  "merge_size": cfg.get("merge_size", 2), "num_grid_per_side": grid,
                  "image_mean": tuple(cfg.get("image_mean", (0.5, 0.5, 0.5))),
                  "image_std": tuple(cfg.get("image_std", (0.5, 0.5, 0.5)))}
        values.update(overrides)
        return cls(**values)


@dataclass
class PreparedImage:
    pixel_values: np.ndarray
    grid: Tuple[int, int, int]
    interp_indices: np.ndarray
    interp_weights: np.ndarray
    position_ids: np.ndarray

    @property
    def tokens(self) -> int:
        return self.grid[0] * self.grid[1] * self.grid[2] // 4


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    """transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200, got %s" % (max(height, width) / min(height, width)))
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def load_image(source: ImageSource, config: ImageConfig) -> "Image.Image":
    """RGB PIL image from a PIL image, a path, bytes or a base64 data URL (decoded and converted, nothing else)."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (str, pathlib.Path)):
        text = str(source)
        if text.startswith("data:image/"):
            header, sep, encoded = text.partition(",")
            if not sep or ";base64" not in header:
                raise ValueError("only base64 data URLs are supported")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except binascii.Error as exc:
                raise ValueError("image data URL is not valid base64") from exc
        elif text.startswith(("http://", "https://")):
            raise ValueError("URLs are not fetched; pass a path, bytes, a data URL or a PIL image")
        else:
            raw = pathlib.Path(text).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        raise TypeError("unsupported image source: %r" % type(source).__name__)
    try:
        with Image.open(io.BytesIO(raw)) as picture:
            if picture.width * picture.height > config.max_source_pixels:
                raise ValueError("decoded image exceeds the pixel limit")
            return picture.convert("RGB")
    except (OSError, SyntaxError) as exc:
        raise ValueError("could not decode the image") from exc


def _patchify(pixels: np.ndarray, config: ImageConfig):
    """Qwen2VLImageProcessor patchify for one still picture; ``pixels`` is [C, H, W], normalised."""
    channel, height, width = pixels.shape
    patch, merge, temporal = config.patch_size, config.merge_size, config.temporal_patch_size
    grid_h, grid_w = height // patch, width // patch
    patches = pixels.reshape(channel, grid_h // merge, merge, patch, grid_w // merge, merge, patch)
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)      # [gh/m, gw/m, m, m, C, P, P]
    patches = np.repeat(patches[..., None, :, :], temporal, axis=-3)
    flat = patches.reshape(grid_h * grid_w, channel * temporal * patch * patch)
    return np.ascontiguousarray(flat, dtype=np.float32), grid_h, grid_w


def vision_position_ids(grid_h: int, grid_w: int, merge: int) -> np.ndarray:
    """transformers.vision_utils.get_vision_position_ids for one still picture: (row, col), merge-block order."""
    rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    shape = (grid_h // merge, merge, grid_w // merge, merge)
    rows = rows.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    cols = cols.reshape(shape).transpose(0, 2, 1, 3).reshape(-1)
    return np.stack([rows, cols], axis=-1).astype(np.int64)


def _axis_taps(index: np.ndarray, size: int, side: int):
    """Two bilinear taps and weights per target index (align_corners=True, border padding)."""
    index = index.astype(np.float32)
    src = index * np.float32(side - 1) / np.float32(max(size - 1, 1))
    floor = np.floor(src)
    offsets = np.arange(2)
    raw = floor.astype(np.int64)[:, None] + offsets
    taps = np.clip(raw, 0, side - 1)
    weights = np.maximum(1 - np.abs(src[:, None] - floor[:, None] - offsets), 0).astype(np.float32)
    return taps, weights


def interpolation_tables(grid_h: int, grid_w: int, side: int, merge: int):
    """transformers.vision_utils.get_vision_interpolation_indices_and_weights (bilinear, align_corners=True)."""
    pos = vision_position_ids(grid_h, grid_w, merge)
    h_taps, h_w = _axis_taps(pos[:, 0], grid_h, side)
    w_taps, w_w = _axis_taps(pos[:, 1], grid_w, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    return indices.astype(np.int64), weights.astype(np.float32)


def prepare_image(source: ImageSource, config: ImageConfig = None, resample=Image.Resampling.BICUBIC) -> PreparedImage:
    config = config or ImageConfig()
    picture = load_image(source, config)
    factor = config.factor
    height, width = smart_resize(picture.height, picture.width, factor, config.min_pixels, config.max_pixels)
    if (width, height) != picture.size:
        picture = picture.resize((width, height), resample)
    rgb = np.asarray(picture, dtype=np.uint8)
    mean = np.asarray(config.image_mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(config.image_std, dtype=np.float32).reshape(3, 1, 1)
    pixels = (rgb.astype(np.float32).transpose(2, 0, 1) * np.float32(1 / 255.0) - mean) / std
    flat, grid_h, grid_w = _patchify(pixels, config)
    indices, weights = interpolation_tables(grid_h, grid_w, config.num_grid_per_side, config.merge_size)
    return PreparedImage(flat, (1, grid_h, grid_w), indices, weights, vision_position_ids(grid_h, grid_w, config.merge_size))
