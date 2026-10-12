# -*- coding: utf-8 -*-
"""Gate: our numpy + Pillow picture preprocessing against the vendor processor's pixel_values (artifacts/golden*).

Needs Pillow on top of the runtime dependencies; the reference values were saved by export/make_golden.py, so no
torch or transformers is needed here.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _model import ART, MODEL, MODEL_DIR  # noqa: E402
from d3_decide.image import ImageConfig, prepare_image  # noqa: E402

TOL = 0.02   # pixel_values are in [-1, 1]; 0.02 is about 2.5 grey levels (torchvision and Pillow resample differently)


def main(model_dir=MODEL_DIR):
    golden = json.loads((ART / "golden.json").read_text(encoding="utf-8"))
    pixels = np.load(ART / "golden_pixels.npz")
    config = ImageConfig.from_model_dir(model_dir)
    bad = n = 0
    worst = 0.0
    for case in golden["cases"]:
        if not case["images"]:
            continue
        key = next(iter(case["q"]))
        grids = case["q"][key]["image_grid_thw"]
        ref = pixels[case["name"]]
        prepared = [prepare_image(str(ROOT / p), config) for p in case["images"]]
        got = np.concatenate([p.pixel_values for p in prepared], axis=0)
        ok_grid = [list(p.grid) for p in prepared] == grids
        ok_shape = got.shape == ref.shape
        delta = float(np.abs(got - ref).max()) if ok_shape else -1.0
        mean = float(np.abs(got - ref).mean()) if ok_shape else -1.0
        worst = max(worst, delta)
        good = ok_grid and ok_shape and 0 <= delta < TOL
        n += 1
        bad += not good
        print("%s %-18s grid=%s shape=%s max|dpixel|=%.4f mean=%.5f" % ("ok" if good else "XX", case["name"], ok_grid, ok_shape, delta, mean))
    print("\nworst |dpixel| %.4f (tol %.2f); failures %d / %d" % (worst, TOL, bad, n))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
