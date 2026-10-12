# -*- coding: utf-8 -*-
"""One measurement per process (so ORT's arena cannot drift between configurations); prints one JSON line.

    uv run verify/bench.py --model edge --seq-len 384 --case readme_text --runs 7
    uv run verify/bench.py --model lite --seq-len 512 --case img_blue

The graph is fixed-length, so the cost of one question follows the bucket L, not the prompt length. ``--case`` is a request of
artifacts/golden.json (its questions run one after another). Timings are only meaningful on a quiet machine: run the GLiNER
calibration of the benchmark (runner.calibrate, <= 600 ms) next to it and quote it.
"""
import argparse
import ctypes
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _model import ART, MODEL, MODEL_DIR  # noqa: E402
from d3_decide import D3Engine  # noqa: E402


def peak_rss_mb():
    """Peak working set in MB (Windows); argtypes are required or the handle truncates."""
    try:
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=384, help="bucket to load")
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--case", default="readme_text", help="name of a golden request")
    ap.add_argument("--questions", type=int, default=0, help="0 = all questions of the case; N = the first question repeated N times")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu", help="cpu (default), cuda (CUDAExecutionProvider, then CPU) or auto")
    ap.add_argument("--require-gpu", action="store_true", help="fail instead of falling back to the CPU when the GPU cannot be used")
    ap.add_argument("--gpu", action="store_true", help="same as --device cuda")
    args = ap.parse_args()

    golden = json.loads((ART / "golden.json").read_text(encoding="utf-8"))
    case = next(c for c in golden["cases"] if c["name"] == args.case)
    questions = case["questions"]
    if args.questions:
        first = next(iter(questions.values()))
        questions = {"q%02d" % i: first for i in range(args.questions)}
    images = [str(ROOT / p) for p in case["images"]] or None

    t0 = time.perf_counter()
    engine = D3Engine(args.model_dir, seq_len=args.seq_len, intra_op_threads=args.threads,
                      device="cuda" if args.gpu else args.device, require_gpu=args.require_gpu)
    load_s = time.perf_counter() - t0
    result = engine.system_one(state=case["state"], questions=questions, images=images)     # warm
    if engine.seq_len != args.seq_len and args.seq_len in engine.buckets:
        pass
    times, parts = [], []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        engine.system_one(state=case["state"], questions=questions, images=images)
        times.append((time.perf_counter() - t0) * 1000)
        parts.append(dict(engine.last_timing))
    times.sort()
    print(json.dumps({
        "model_dir": str(args.model_dir), "case": args.case, "graph_seq_len": engine.seq_len, "threads": args.threads,
        "provider": engine.provider, "runs": args.runs, "questions": len(questions),
        "prompt_tokens_total": result["usage"]["input_tokens"], "load_s": round(load_s, 2),
        "median_ms": round(statistics.median(times), 1), "min_ms": round(times[0], 1), "max_ms": round(times[-1], 1),
        "per_question_ms": round(statistics.median(times) / len(questions), 1),
        "vision_ms_median": round(statistics.median(p["vision_s"] for p in parts) * 1000, 1),
        "peak_rss_mb": peak_rss_mb(),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
