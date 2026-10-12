# -*- coding: utf-8 -*-
"""Text decisions with d3 (Decision 3.0: d3-edge / d3-lite / d3-nano), on onnxruntime + tokenizers + numpy.

    uv run demo_inference_text.py [--model edge|lite|nano]
    uv run demo_inference_text.py --state "..." --choice "team=billing,technical" --noul "angry=Is the customer angry?"

Each question is one forward pass over its own prompt (the options are listed under single-token codes A, B, C, ...);
the answer is the probability of every option. Nothing is generated.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from d3_decide import D3Engine
from d3_decide.paths import default_model_dir  # noqa: E402

EXAMPLE = {
    "state": "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    "questions": {
        "route": {"type": "choice", "instructions": "Which team should handle this request?",
                  "criteria": {"returns": "Refunds, replacements and damaged deliveries",
                               "billing": "Payments, invoices and charges",
                               "technical": "Product setup and faults"}},
        "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
        "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["Routine", "Soon", "Today"]},
    },
}


def parse_questions(args) -> dict:
    questions: dict = {}
    for spec in args.choice or []:
        name, _, rest = spec.partition("=")
        values = [v.strip() for v in rest.split(",") if v.strip()]
        questions[name.strip()] = {"type": "choice", "instructions": args.instructions or name.strip(), "criteria": {v: None for v in values}}
    for spec in args.score or []:
        name, _, rest = spec.partition("=")
        questions[name.strip()] = {"type": "score", "instructions": args.instructions or name.strip(),
                                   "criteria": [v.strip() for v in rest.split(",") if v.strip()]}
    for spec in args.noul or []:
        name, _, rest = spec.partition("=")
        questions[name.strip()] = {"type": "noul", "instructions": rest.strip() or name.strip()}
    return questions


def show(result: dict) -> None:
    for key, a in result["answers"].items():
        if "error" in a:
            print("%-10s ERROR %s: %s" % (key, a["error"], a["message"]))
        elif a["type"] == "noul":
            print("%-10s %-8s p(yes)=%.3f" % (key, "yes" if a["noul"] > 0.5 else "no", a["noul"]))
        else:
            spread = "  ".join("%s=%.3f" % kv for kv in sorted(a["probabilities"].items(), key=lambda kv: -kv[1]))
            head = a["choice"] if a["type"] == "choice" else "score %.2f" % a["score"]
            print("%-10s %-22s confidence %.3f\n           %s" % (key, head, a["confidence"], spread))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", help="text or JSON; omit to run the built-in example")
    parser.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    parser.add_argument("--score", action="append", metavar="NAME=low,...,high")
    parser.add_argument("--noul", action="append", metavar="NAME=question text")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--model", default="edge", choices=["edge", "lite", "nano"], help="d3-edge 0.6B, or d3-nano 2B")
    parser.add_argument("--model-dir", help="model folder (overrides --model and D3_<MODEL>_MODEL_DIR; default: models/<model>)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=0, help="bucket to start with (default: the smallest one that fits)")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu", help="cpu (default), cuda (CUDAExecutionProvider, then CPU) or auto")
    parser.add_argument("--require-gpu", action="store_true", help="fail instead of falling back to the CPU when the GPU cannot be used")
    parser.add_argument("--gpu", action="store_true", help="same as --device cuda")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.state is None:
        request = EXAMPLE
    else:
        try:
            state = json.loads(args.state)
        except json.JSONDecodeError:
            state = args.state
        questions = parse_questions(args)
        if not questions:
            parser.error("give at least one --choice, --score or --noul")
        request = {"state": state, "questions": questions}

    started = time.perf_counter()
    engine = D3Engine(default_model_dir(args.model, ROOT, args.model_dir), seq_len=args.seq_len or None, intra_op_threads=args.threads,
                      device="cuda" if args.gpu else args.device, require_gpu=args.require_gpu)
    load = time.perf_counter() - started
    started = time.perf_counter()
    result = engine.system_one(state=request["state"], questions=request["questions"])
    elapsed = time.perf_counter() - started
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print("state:\n  %s\n" % json.dumps(request["state"], ensure_ascii=False)[:160])
    show(result)
    print("\n%d prompt tokens over %d question(s), one forward pass each" % (result["usage"]["input_tokens"], len(request["questions"])))
    print("load %.1f s, inference %.2f s (graph fixed at %d tokens, %s)" % (load, elapsed, engine.seq_len, engine.provider))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
