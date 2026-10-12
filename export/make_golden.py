# -*- coding: utf-8 -*-
"""Ground truth from the vendor's own runtime (d3_runtime.py, PyTorch bf16 on CPU).

Writes artifacts/golden.json (requests, rendered prompts, token ids, image grids, probabilities of every question)
and artifacts/golden_pixels.npz (the processor's pixel_values for the image requests).

    uv run --group export python export/make_golden.py --model edge --hf <dir holding the vllm-sr/d3-* files>

Run in the export environment (torch + transformers 5.17.0). The vendor code is imported from the checkpoint directory
(never copied into this repository).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
import cases as case_set  # noqa: E402
from models import MODELS, model_paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", help="default: artifacts/<model>")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="use every N-th text request (all picture requests are kept): fewer cases for the big models")
    ap.add_argument("--no-fp32", action="store_true", help="skip the fp32 pass (needs ~4 bytes per parameter of RAM on top; for the 4.5B model)")
    args = ap.parse_args()
    hf = pathlib.Path(args.hf)
    sys.path.insert(0, str(hf))
    from d3_runtime import D3  # vendor code
    from PIL import Image

    model = D3.from_pretrained(str(hf), device="cpu", verify="none")
    cases = case_set.text_cases()[:: args.stride] + case_set.image_cases()
    if args.limit:
        cases = cases[: args.limit]
    out, pixels, kept = [], {}, []
    for case in cases:
        imgs = [Image.open(ROOT / p).convert("RGB") for p in case.get("images", [])] or None
        t = time.perf_counter()
        prepared = model.prepare(case["state"], case["questions"], imgs, None)
        probs, tokens = model.run(prepared)
        resp = model.respond(prepared, probs, tokens)
        rec = {"name": case["name"], "state": case["state"], "questions": case["questions"],
               "images": case.get("images", []), "seconds": time.perf_counter() - t,
               "input_tokens": tokens, "answers": resp["answers"], "q": {}}
        for key in prepared.runnable:
            q = prepared.questions[key]
            entry = {"keys": q.keys, "probs": probs[key]}
            if imgs is None:
                entry["text"] = model.text(case["state"], q.rendered)
                entry["input_ids"] = prepared.sequences[key]
            else:
                text = prepared.texts[key]
                enc = model.processor(text=[text], images=imgs, return_tensors="pt")
                entry["text"] = text
                entry["input_ids"] = enc["input_ids"][0].tolist()
                entry["image_grid_thw"] = enc["image_grid_thw"].tolist()
                pixels["%s/%s" % (case["name"], key)] = enc["pixel_values"].numpy()
            rec["q"][key] = entry
        out.append(rec)
        kept.append(prepared)
        print(case["name"], tokens, "%.2fs" % rec["seconds"], flush=True)
    # Second pass: the same vendor code path with the backbone cast to fp32 (the oracle for the fp32 ONNX graph;
    # the bf16 numbers above are the vendor's own).
    for rec, prepared in ([] if args.no_fp32 else zip(out, kept)):
        if not getattr(model, "_fp32", False):
            model.backbone.float()
            model._fp32 = True
        probs32, _ = model.run(prepared)
        for key, p in probs32.items():
            rec["q"][key]["probs_fp32"] = p
        print("fp32", rec["name"], flush=True)
    outdir = pathlib.Path(args.out) if args.out else model_paths(args.model)[1]
    outdir.mkdir(exist_ok=True, parents=True)
    meta = {"model": model.model_name, "pad_token_id": model.pad_id, "codes": model.codes, "token_ids": model.token_ids}
    (outdir / "golden.json").write_text(json.dumps({"meta": meta, "cases": out}, ensure_ascii=False), encoding="utf-8")
    if pixels:
        # one entry per image request is enough (the questions of a request share the picture)
        seen, keep = set(), {}
        for k, v in pixels.items():
            if k.split("/")[0] not in seen:
                seen.add(k.split("/")[0])
                keep[k.split("/")[0]] = v
        np.savez_compressed(outdir / "golden_pixels.npz", **keep)
    print("wrote", outdir / "golden.json")


if __name__ == "__main__":
    main()
