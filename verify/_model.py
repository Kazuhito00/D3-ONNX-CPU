# -*- coding: utf-8 -*-
"""``--model edge|lite|nano`` for the verification scripts (removed from sys.argv so each script keeps its own options)."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ("edge", "lite", "nano")


def select(default="edge"):
    model = default
    for i, a in enumerate(sys.argv):
        if a == "--model" and i + 1 < len(sys.argv) and sys.argv[i + 1] in MODELS:
            model = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            break
        if a.startswith("--model=") and a.split("=", 1)[1] in MODELS:
            model = a.split("=", 1)[1]
            del sys.argv[i]
            break
    return model, ROOT / "models" / model, ROOT / "artifacts" / model


MODEL, MODEL_DIR, ART = select()
