# -*- coding: utf-8 -*-
"""Download the ONNX models for the demos from this repository's GitHub release.

    python download_model.py                                    # every model (edge, lite, nano), SHA-256 checked
    python download_model.py --model edge                       # one model only (edge | lite | nano)
    python download_model.py --model lite --seq-len 384        # only the 384-token graph (+ shared weights, vision encoder, tokenizer)
    python download_model.py --model edge --exclude "vision_encoder*"   # text-only use: skip the vision encoder
    python download_model.py --tag v1                           # a given release (default: latest)
    python download_model.py --base-url http://127.0.0.1:8000   # a mirror, or a local dist/release output
    python download_model.py --list                             # show the files, download nothing
    python download_model.py --repo OWNER/NAME                  # a fork's release

The default repository is ``Kazuhito00/D3-ONNX-CPU``. Give another one with ``--repo OWNER/NAME`` or through the environment:
``D3_ONNX_REPO=OWNER/NAME`` (the argument wins over the environment, the environment over the default).
**With no arguments every model of the release is fetched (several GB per model; the 2B and 5B ones are large)**: use ``--model`` to
take one.

The vendor (vllm-sr) publishes no ONNX. The release holds the graphs exported by export/export_onnx.py; to build them yourself from the
original checkpoint use ``python export/prepare_model.py --model edge`` instead (needs ``uv sync --group export``).

The release carries one ``manifest-<model>.json`` per model (written by release/make_release.py): every file's path, size and sha256
and, for files that were split to stay under GitHub's 2 GiB asset limit, the parts. Assets are named ``d3-<model>__...``. This script
fetches what the manifest lists, joins the parts and checks every sha256 (a mismatch deletes the file and stops). A file already in place
with the right sha256 is skipped and an interrupted download continues where it stopped (HTTP Range), so the same command can simply be
run again. Plain HTTPS from the standard library: no git, no git-lfs, no huggingface_hub.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_REPO = "Kazuhito00/D3-ONNX-CPU"
MODELS = ("edge", "lite", "nano")
USER_AGENT = "d3-onnx-cpu/0.1 (urllib)"
CHUNK = 1 << 20  # 1 MiB


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return "%.1f %s" % (size, unit)
        size /= 1024
    return "%.1f GB" % size


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def base_url(args) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    repo = args.repo or os.environ.get("D3_ONNX_REPO") or DEFAULT_REPO
    if "<OWNER>" in repo:
        raise SystemExit("no release repository set yet: pass --repo OWNER/NAME (or set D3_ONNX_REPO, or --base-url)")
    if args.tag == "latest":
        return "https://github.com/%s/releases/latest/download" % repo
    return "https://github.com/%s/releases/download/%s" % (repo, args.tag)


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch(url: str, dest: pathlib.Path, size: int, label: str = "", tries: int = 8) -> pathlib.Path:
    """Download ``url`` to ``dest`` (``size`` bytes), resuming the bytes already there; retries on dropped connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for attempt in range(1, tries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == size:
            return dest
        if have > size:
            dest.unlink()
            have = 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:
                    have = 0  # the server ignored the Range header: start over
                with open(dest, "ab" if have else "wb") as handle:
                    done, last = have, 0.0
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        done += len(block)
                        now = time.time()
                        if now - last > 0.5:
                            last = now
                            rate = (done - have) / max(now - started, 1e-6)
                            sys.stdout.write("\r  %-44s %5.1f%%  %s/s   "
                                             % (label or dest.name, 100 * done / size, _human(rate)))
                            sys.stdout.flush()
            if dest.stat().st_size == size:
                return dest
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            wait = min(60, 2 ** attempt)
            print("\n  %s: %s (try %d/%d, again in %d s)" % (dest.name, exc, attempt, tries, wait), flush=True)
            time.sleep(wait)
    raise SystemExit("could not download %s completely" % url)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", help="GitHub repository that holds the release (OWNER/NAME; default: $D3_ONNX_REPO, else %s)"
                        % DEFAULT_REPO)
    parser.add_argument("--model", default="all", choices=MODELS + ("all",), help="which model(s) to fetch (default: all)")
    parser.add_argument("--tag", default="latest", help="release tag (default: latest)")
    parser.add_argument("--base-url", help="fetch the assets from this URL instead of a GitHub release (mirror, local server)")
    parser.add_argument("--dest", default=str(ROOT / "models"), help="where the files go (default: models/)")
    parser.add_argument("--include", action="append", help="only paths matching this glob (repeatable)")
    parser.add_argument("--exclude", action="append", help="skip paths matching this glob (repeatable)")
    parser.add_argument("--seq-len", type=int, action="append",
                        help="fetch only the graph(s) of this sequence-length bucket (repeatable); the shared weights, vision "
                             "encoder and tokenizer files always come along. Default: every bucket in the release")
    parser.add_argument("--list", action="store_true", help="show the files of the release and exit")
    args = parser.parse_args()

    base = base_url(args)
    manifests = []
    for model in (MODELS if args.model == "all" else (args.model,)):
        try:
            manifest = json.loads(fetch_bytes("%s/manifest-%s.json" % (base, model)))
        except urllib.error.URLError as exc:
            if args.model == "all":
                print("note: no manifest-%s.json in this release (%s); skipped" % (model, exc))
                continue
            raise SystemExit("could not read %s/manifest-%s.json: %s" % (base, model, exc))
        if manifest.get("format") != 1:
            raise SystemExit("unknown manifest format: %r" % manifest.get("format"))
        manifests.append(manifest)
    if not manifests:
        raise SystemExit("nothing to fetch")

    def wanted(path: str) -> bool:
        name = path.rsplit("/", 1)[-1]
        bucket = re.fullmatch(r"model_L(\d+)\.onnx", name)      # a bucket's graph; its weights are shared (weights.onnx.data*)
        if args.seq_len and bucket and int(bucket.group(1)) not in args.seq_len:
            return False
        if args.include and not any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(name, g) for g in args.include):
            return False
        return not (args.exclude and any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(name, g) for g in args.exclude))

    files = [f for m in manifests for f in m["files"] if wanted(f["path"])]
    total = sum(f["size"] for f in files)
    print("%s: %d files, %s" % (", ".join("%s %s" % (m.get("name"), m.get("version")) for m in manifests), len(files), _human(total)))
    if args.list:
        for f in files:
            print("  %10s  %s%s" % (_human(f["size"]), f["path"], "  (%d parts)" % len(f["parts"]) if f.get("parts") else ""))
        return 0

    dest = pathlib.Path(args.dest)
    for f in files:
        out = dest / f["path"]
        if out.exists() and out.stat().st_size == f["size"] and _sha256(out) == f["sha256"]:
            print("  have  %-44s %s" % (f["path"], _human(f["size"])))
            continue
        tmp = out.with_name(out.name + ".part")
        if not f.get("parts"):
            fetch("%s/%s" % (base, f["asset"]), tmp, f["size"], label=f["path"])
        else:  # join the parts into one file, one part at a time
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as joined:
                for part in f["parts"]:
                    piece = tmp.with_name(part["asset"] + ".dl")
                    fetch("%s/%s" % (base, part["asset"]), piece, part["size"], label=part["asset"])
                    if _sha256(piece) != part["sha256"]:
                        piece.unlink()
                        raise SystemExit("%s: SHA-256 mismatch, file deleted" % part["asset"])
                    with open(piece, "rb") as src:
                        for block in iter(lambda: src.read(CHUNK), b""):
                            joined.write(block)
                    piece.unlink()
        if _sha256(tmp) != f["sha256"]:
            tmp.unlink()
            raise SystemExit("%s: SHA-256 mismatch, file deleted" % f["path"])
        os.replace(tmp, out)
        sys.stdout.write("\r  got   %-44s %s\n" % (f["path"], _human(f["size"])))
        sys.stdout.flush()

    print("\nready: %s" % dest)
    first = manifests[0].get("model", "edge")
    print("run:    uv run demo_inference_text.py --model %s" % first)
    print("verify: uv run verify/run_all.py --model %s" % first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
