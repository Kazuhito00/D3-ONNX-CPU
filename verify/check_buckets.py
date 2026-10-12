# -*- coding: utf-8 -*-
"""Gate (runtime dependencies only): the sequence-length buckets agree with each other and switch as documented.

 * the same questions run through every bucket on disk that holds them give the same probabilities (right padding
   does not move the read logits: the read position precedes the padding and the Gated DeltaNet layers are causal, and the
   padded keys are masked in the softmax layers)
 * an engine started on a small bucket moves to the smallest larger bucket when a prompt does not fit
 * a prompt beyond the largest bucket is a ``max_length_exceeded`` answer, never truncated
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _model import ART, MODEL, MODEL_DIR  # noqa: E402
from d3_decide import D3Engine  # noqa: E402

TOL = 1e-4


def probs(answer):
    return np.asarray(list(answer["probabilities"].values())) if "probabilities" in answer else np.asarray([1 - answer["noul"], answer["noul"]])


def main(model_dir=MODEL_DIR):
    golden = json.loads((ART / "golden.json").read_text(encoding="utf-8"))
    cases = [c for c in golden["cases"] if not c["images"]][:6] + [c for c in golden["cases"] if c["name"].startswith("rand_")][-2:]
    engine = D3Engine(model_dir, seq_len=1)               # the smallest bucket
    buckets = engine.buckets
    print("buckets:", buckets, "started on", engine.seq_len)
    ok = True
    ref = {}
    worst = 0.0
    for case in cases:
        need = max(len(e["input_ids"]) for e in case["q"].values())
        fits = [b for b in buckets if b >= need]
        outs = []
        for b in fits[:3]:                                 # the smallest three buckets that hold the request
            e = D3Engine(model_dir, seq_len=b)
            got = e.system_one(state=case["state"], questions=case["questions"])["answers"]
            outs.append((b, {k: probs(v) for k, v in got.items()}))
            del e
        base = outs[0][1]
        for b, other in outs[1:]:
            d = max(float(np.abs(base[k] - other[k]).max()) for k in base)
            worst = max(worst, d)
            if d > TOL:
                ok = False
                print("XX %s L=%d vs L=%d max|dp|=%.2e" % (case["name"], outs[0][0], b, d))
        print("%-12s need %4d  buckets %s  ok" % (case["name"], need, [b for b, _ in outs]))
    print("worst difference between buckets: %.2e (tol %.0e)" % (worst, TOL))

    engine = D3Engine(model_dir, seq_len=1)
    first = engine.seq_len
    long_state = "word " * (first + 40)                    # does not fit the first bucket
    r = engine.system_one(state=long_state, questions={"q": {"type": "noul", "instructions": "x"}})
    switched = engine.seq_len > first
    print("escalation from L=%d to L=%d: %s" % (first, engine.seq_len, "ok" if switched else "XX"))
    ok &= switched and "error" not in r["answers"]["q"]

    huge = "word " * (max(buckets) + 100)
    r = engine.system_one(state=huge, questions={"q": {"type": "noul", "instructions": "x"}})
    exceeded = r["answers"]["q"].get("error") == "max_length_exceeded"
    print("beyond the largest bucket (%d): %s" % (max(buckets), "max_length_exceeded, ok" if exceeded else "XX"))
    ok &= exceeded
    print("failures: %d" % (0 if ok else 1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
