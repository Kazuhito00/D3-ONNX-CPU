# -*- coding: utf-8 -*-
"""Build the ONNX files of one d3 model from its original checkpoint (needs the export environment: ``uv sync --group export``).

    uv run --group export python export/prepare_model.py --model edge --list
    uv run --group export python export/prepare_model.py --model edge                  # buckets 128 ... 2048 + vision
    uv run --group export python export/prepare_model.py --model lite --seq-len 384 --seq-len 512
    uv run --group export python export/prepare_model.py --model nano --hf-dir D:/cache/d3-nano   # checkpoint kept outside the repository

Steps: download vllm-sr/d3-<model> (plain HTTPS, the revision is pinned, every file is checked against the checkpoint's own
MODEL_MANIFEST.json), write the golden data with the vendor runtime (artifacts/<model>/golden.json; skip with --skip-golden), export
one decoder graph per sequence length and the vision encoder into models/<model>/, and let every bucket share ONE weights file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.download import fetch_files  # noqa: E402
from export.models import DEFAULT_BUCKETS, MODELS, model_paths  # noqa: E402

COMMON_FILES = [
    "config.json", "decision_config.json", "MODEL_MANIFEST.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    "merges.txt", "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json",
    "readout.safetensors", "model.safetensors.index.json", "LICENSE", "NOTICE",
    # the vendor runtime: only make_golden.py imports it (the oracle); nothing here copies it into the repository
    "d3_runtime.py", "d3_format.py", "d3_fast.py", "d3_kernels.py", "d3_engine.py", "modeling_d3.py", "pipeline_d3.py",
]
#: copied next to the ONNX files: what the runtime and the demos read
RUNTIME_FILES = ["tokenizer.json", "decision_config.json", "preprocessor_config.json", "config.json", "LICENSE", "NOTICE"]


def run(argv):
    print("$ %s" % " ".join(str(a) for a in argv), flush=True)
    subprocess.run([str(a) for a in argv], check=True, cwd=ROOT)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifest(hf_dir: pathlib.Path) -> None:
    manifest = json.loads((hf_dir / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["identity"]["files_sha256"].items():
        path = hf_dir / name
        if path.exists():
            if sha256(path) != expected:
                raise SystemExit("%s does not match MODEL_MANIFEST.json (sha256): delete it and download again" % name)
            print("  sha256 ok  %s" % name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--list", action="store_true", help="show the default sequence-length buckets")
    parser.add_argument("--hf-dir", help="where the original checkpoint is (default: models/hf-<model>; downloaded when missing)")
    parser.add_argument("--seq-len", type=int, action="append", help="bucket(s) to export (multiples of 64; repeatable)")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--golden-args", default="", help="extra arguments for make_golden.py, e.g. '--stride 3 --no-fp32'")
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    spec = MODELS[args.model]
    buckets = args.seq_len or list(DEFAULT_BUCKETS)

    if args.list:
        print("sequence-length buckets (cost scales with L, not with the real prompt length):", ", ".join(map(str, DEFAULT_BUCKETS)))
        print("one question is one prompt: ~100-250 tokens for a text question, + 64 tokens per 224 px picture (256x256 after resize).")
        return 0

    onnx_dir, _ = model_paths(args.model)
    hf = pathlib.Path(args.hf_dir) if args.hf_dir else ROOT / "models" / ("hf-" + args.model)
    print("checkpoint %s@%s -> %s" % (spec["repo"], spec["revision"][:8], hf), flush=True)
    fetch_files(spec["repo"], spec["weights"] + COMMON_FILES, hf, revision=spec["revision"], optional=["model.safetensors.index.json"])
    check_manifest(hf)
    if args.fetch_only:
        return 0

    onnx_dir.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_FILES:
        if (hf / name).exists():
            (onnx_dir / name).write_bytes((hf / name).read_bytes())

    if not args.skip_golden:
        run([sys.executable, "export/make_golden.py", "--model", args.model, "--hf", hf] + args.golden_args.split())

    graphs = onnx_dir / "onnx"
    pool = graphs / "weights.onnx.data"
    first = True
    for length in buckets:
        cmd = [sys.executable, "-X", "utf8", "export/export_onnx.py", "--model", args.model, "--hf", hf, "--seq-len", length]
        if args.skip_vision or not first:
            cmd.append("--skip-vision")
        run(cmd)
        graph = graphs / ("model_L%d.onnx" % length)
        run([sys.executable, "export/share_external_data.py"] + ([] if pool.exists() else ["--adopt"]) + [graph])
        first = False

    print("\nready: %s" % onnx_dir)
    print("verify with:  uv run verify/run_all.py --model %s" % args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
