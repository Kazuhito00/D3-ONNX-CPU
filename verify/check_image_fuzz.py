# -*- coding: utf-8 -*-
"""Oracle gate (needs transformers): our picture preprocessing against the vendor processor on random pictures of many sizes.

    uv run --group export python verify/check_image_fuzz.py <checkpoint folder (preprocessor_config.json)> [model folder]
"""
import pathlib
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from transformers import AutoProcessor  # noqa: E402

from d3_decide.image import ImageConfig, prepare_image  # noqa: E402

SIZES = [(301, 517), (224, 224), (1920, 1080), (1280, 720), (100, 40), (3000, 2000), (64, 64), (777, 333), (640, 480), (500, 20)]
TOL = 0.02


def main():
    hf = pathlib.Path(sys.argv[1])
    proc = AutoProcessor.from_pretrained(str(hf))
    proc.image_processor.size = {"shortest_edge": 65536, "longest_edge": 1638400}   # what d3_runtime.load_processor sets
    cfg = ImageConfig.from_model_dir(hf)
    rng = np.random.default_rng(1)
    bad = 0
    worst = 0.0
    for w, h in SIZES:
        arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
        small = Image.fromarray(arr).resize((max(8, w // 7), max(8, h // 7)), Image.BICUBIC).resize((w, h), Image.BICUBIC)
        enc = proc.image_processor(images=[small], return_tensors="pt")
        ref, grid = enc["pixel_values"].numpy(), enc["image_grid_thw"].tolist()
        got = prepare_image(small, cfg)
        ok = got.pixel_values.shape == ref.shape and [list(got.grid)] == grid
        d = float(np.abs(got.pixel_values - ref).max()) if ok else -1.0
        worst = max(worst, d)
        bad += not (ok and d < TOL)
        print("%s %-11s grid %s max|dpixel| %.4f" % ("ok" if ok and d < TOL else "XX", (w, h), grid, d))
    print("worst %.4f (tol %.2f); failures: %d" % (worst, TOL, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
