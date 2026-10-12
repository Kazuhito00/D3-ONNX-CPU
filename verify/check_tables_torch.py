# -*- coding: utf-8 -*-
"""Oracle gate (needs torch + transformers 5.17.0, i.e. the export environment): the numpy re-implementations of
 * the vision position tables (rotary positions, bilinear position-embedding taps and weights) and
 * the 3-D M-RoPE text/picture positions
against transformers' own functions, over many grid sizes and prompt layouts.

    uv run --group export python verify/check_tables_torch.py
"""
import pathlib
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from transformers.vision_utils import get_vision_interpolation_indices_and_weights, get_vision_position_ids  # noqa: E402

from d3_decide import chat  # noqa: E402
from d3_decide.image import interpolation_tables, vision_position_ids  # noqa: E402


def main():
    rng = random.Random(7)
    bad = n = 0
    worst_w = 0.0
    sizes = [(16, 16), (2, 2), (4, 6), (32, 18), (60, 106), (64, 96), (12, 26), (48, 48), (20, 48), (100, 100)]
    sizes += [(2 * rng.randint(1, 60), 2 * rng.randint(1, 60)) for _ in range(30)]
    for h, w in sizes:
        grid = torch.tensor([[1, h, w]])
        pos_ref = get_vision_position_ids(grid, 2).numpy()
        idx_ref, wt_ref = get_vision_interpolation_indices_and_weights(grid, num_grid_per_side=48, mode="bilinear",
                                                                       align_corners=True, spatial_merge_size=2)
        idx, wt = interpolation_tables(h, w, 48, 2)
        ok = (pos_ref == vision_position_ids(h, w, 2)).all() and (idx_ref.numpy() == idx).all()
        dw = float(np.abs(wt_ref.numpy() - wt).max())
        worst_w = max(worst_w, dw)
        ok = ok and dw < 1e-6
        n += 1
        bad += not ok
        if not ok:
            print("XX vision tables", (h, w), dw)
    print("vision tables: %d grids, %d mismatches, worst weight diff %.2e" % (n, bad, worst_w))

    # M-RoPE: compare with Qwen3_5Model.get_rope_index on random layouts of text runs and pictures
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    cfg = Qwen3_5Config()
    image_token = cfg.image_token_id
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    torch.nn.Module.__init__(model)
    model.config = cfg
    bad2 = n2 = 0
    for _ in range(60):
        parts, grids = [], []
        ids = [1] * rng.randint(3, 40)
        for _ in range(rng.randint(0, 3)):
            h, w = 2 * rng.randint(1, 12), 2 * rng.randint(1, 12)
            grids.append([1, h, w])
            ids += [248053] + [image_token] * (h * w // 4) + [248054] + [2] * rng.randint(1, 30)
        got = chat.position_ids(ids, grids, image_token, 2)
        types = torch.tensor([[1 if t == image_token else 0 for t in ids]])
        try:
            ref, _ = model.get_rope_index(torch.tensor([ids]), types, torch.tensor(grids) if grids else None, attention_mask=torch.ones(1, len(ids), dtype=torch.long))
        except TypeError:
            ref, _ = model.get_rope_index(input_ids=torch.tensor([ids]), mm_token_type_ids=types,
                                          image_grid_thw=torch.tensor(grids) if grids else None, attention_mask=torch.ones(1, len(ids), dtype=torch.long))
        n2 += 1
        ok = got.shape == tuple(ref.shape) and (got == ref.numpy()).all()
        bad2 += not ok
        if not ok:
            print("XX rope layout", len(ids), grids)
    print("m-rope positions: %d layouts, %d mismatches" % (n2, bad2))
    print("failures: %d" % (bad + bad2))
    return 1 if (bad or bad2) else 0


if __name__ == "__main__":
    raise SystemExit(main())
