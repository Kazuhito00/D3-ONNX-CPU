# -*- coding: utf-8 -*-
"""Picture + text decisions with d3 (Decision 3.0: d3-edge / d3-lite / d3-nano), on onnxruntime + tokenizers + numpy + Pillow.

    uv run demo_inference_image.py
    uv run demo_inference_image.py --image shot.png --choice "status=paid,pending,failed" --noul "paid=Is it paid?"

Every picture is read as the vendor does: the aspect ratio is kept, the side lengths become multiples of 32 and the area stays
between 65,536 and 1,638,400 pixels (32 x 32 pixels = one input token). ``--max-pixels`` lowers the upper limit to save time
(a 640x480 picture is 300 tokens; at 224 px it is 64 tokens), which changes the answers.
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
from demo_inference_text import parse_questions, show  # noqa: E402

DEFAULT_IMAGE = ROOT / "artifacts" / "images" / "sample_chart.png"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", action="append", help="picture file (repeatable); default: artifacts/images/sample_chart.png")
    parser.add_argument("--state", default="Look at the attached picture.")
    parser.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    parser.add_argument("--score", action="append", metavar="NAME=low,...,high")
    parser.add_argument("--noul", action="append", metavar="NAME=question text")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--model", default="edge", choices=["edge", "lite", "nano"], help="d3-edge 0.6B, or d3-nano 2B")
    parser.add_argument("--model-dir", help="model folder (overrides --model and D3_<MODEL>_MODEL_DIR; default: models/<model>)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=0, help="cap on the picture area in pixels (default: the vendor's 1,638,400)")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu", help="cpu (default), cuda (CUDAExecutionProvider, then CPU) or auto")
    parser.add_argument("--require-gpu", action="store_true", help="fail instead of falling back to the CPU when the GPU cannot be used")
    parser.add_argument("--gpu", action="store_true", help="same as --device cuda")
    args = parser.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    questions = parse_questions(args)
    if not questions:
        questions = {
            "tallest": {"type": "choice", "instructions": "Which bar is the tallest?", "criteria": {"A": None, "B": None, "C": None, "D": None}},
            "is_chart": {"type": "noul", "instructions": "Is this a bar chart?"},
        }
    images = [str(p) for p in (args.image or [DEFAULT_IMAGE])]
    started = time.perf_counter()
    engine = D3Engine(default_model_dir(args.model, ROOT, args.model_dir), intra_op_threads=args.threads, device="cuda" if args.gpu else args.device, require_gpu=args.require_gpu,
                      max_image_pixels=args.max_pixels or None)
    load = time.perf_counter() - started
    started = time.perf_counter()
    result = engine.system_one(state=args.state, questions=questions, images=images)
    elapsed = time.perf_counter() - started
    print("pictures: %s" % ", ".join(images))
    show(result)
    t = engine.last_timing
    print("\n%d prompt tokens; load %.1f s, request %.2f s (vision %.2f s, decoder %.2f s; graph L=%d, %s)"
          % (result["usage"]["input_tokens"], load, elapsed, t["vision_s"], t["decode_s"], engine.seq_len, engine.provider))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
