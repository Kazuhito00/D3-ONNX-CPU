# -*- coding: utf-8 -*-
"""Gate: the ONNX runtime against artifacts/golden.json.

Two references, both made by the vendor's own code (export/make_golden.py):
  fp32  the vendor runtime with its backbone cast to fp32. The ONNX graphs are fp32 too, so this is the tight check
        (threshold --tol-fp32, default 1e-3 on any probability; the measured value is far smaller).
  bf16  the vendor runtime as shipped (bf16 backbone). The difference to fp32 is bf16 rounding, so this is reported
        (max |dp| and how often the answer is the same) but only gated loosely (--tol-bf16, default 0.1).

    uv run verify/verify.py [--model-dir models/onnx] [--seq-len 384] [--only img_]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _model import ART, MODEL, MODEL_DIR  # noqa: E402
from d3_decide import D3Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--only", default="", help="only cases whose name starts with this")
    ap.add_argument("--tol-fp32", type=float, default=1e-3)
    ap.add_argument("--tol-bf16", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu", help="run the runtime on this device (GPU numbers differ slightly from the CPU: use a looser --tol-fp32)")
    ap.add_argument("--require-gpu", action="store_true")
    args = ap.parse_args()

    golden = json.loads((ART / "golden.json").read_text(encoding="utf-8"))
    engine = D3Engine(args.model_dir, seq_len=args.seq_len, intra_op_threads=args.threads, device=args.device, require_gpu=args.require_gpu)
    print("graph L=%d (buckets %s), provider %s" % (engine.seq_len, engine.buckets, engine.provider))
    worst32 = worst16 = 0.0
    same32 = same16 = n = n32 = 0
    failures = 0
    for case in golden["cases"]:
        if not case["name"].startswith(args.only):
            continue
        images = [str(ROOT / p) for p in case["images"]] or None
        t = time.perf_counter()
        got = engine.system_one(state=case["state"], questions=case["questions"], images=images)
        dt = time.perf_counter() - t
        line = []
        for key, entry in case["q"].items():
            ans = got["answers"][key]
            if "error" in ans:
                failures += 1
                line.append("%s:ERROR %s" % (key, ans["message"]))
                continue
            if "probabilities" in ans:
                p = np.asarray(list(ans["probabilities"].values()))
            else:
                p = np.asarray([1 - ans["noul"], ans["noul"]])
            ref32 = entry.get("probs_fp32")
            d32 = float(np.abs(p - np.asarray(ref32)).max()) if ref32 else 0.0
            d16 = float(np.abs(p - np.asarray(entry["probs"])).max())
            s32 = int(p.argmax()) == int(np.argmax(ref32)) if ref32 else True
            n32 += 1 if ref32 else 0
            s16 = int(p.argmax()) == int(np.argmax(entry["probs"]))
            worst32, worst16 = max(worst32, d32), max(worst16, d16)
            same32 += s32
            same16 += s16
            n += 1
            bad = d32 > args.tol_fp32 or d16 > args.tol_bf16 or not s32
            failures += bad
            line.append("%s %s dp32=%.1e dp16=%.1e" % ("XX" if bad else "ok", key, d32, d16))
        print("%-18s %5.2fs  %s" % (case["name"], dt, "; ".join(line)))
    print("\nquestions: %d" % n)
    print("vs fp32 reference: answer agreement %d/%d, max|dp| %.2e (tol %.0e)" % (same32 - (n - n32), n32, worst32, args.tol_fp32))
    print("vs bf16 vendor   : answer agreement %d/%d, max|dp| %.2e (tol %.0e)" % (same16, n, worst16, args.tol_bf16))
    print("failures: %d" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
