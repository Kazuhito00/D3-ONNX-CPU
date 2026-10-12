# -*- coding: utf-8 -*-
"""Gate: prompt strings and token ids match the vendor runtime (artifacts/golden.json) for every question.

Needs the runtime dependencies only (tokenizers; no torch, no transformers). Text requests are rendered with
d3_decide.format.render; picture requests with the picture token counts the vendor processor produced.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _model import ART, MODEL, MODEL_DIR  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from d3_decide.format import normalize_question, render  # noqa: E402
from d3_decide.runtime import load_tokenizer  # noqa: E402


def main(model_dir=MODEL_DIR):
    model_dir = pathlib.Path(model_dir)
    golden = json.loads((ART / "golden.json").read_text(encoding="utf-8"))
    codes = json.loads((model_dir / "decision_config.json").read_text(encoding="utf-8"))["codes"]
    tok = load_tokenizer(model_dir / "tokenizer.json")
    bad = n = 0
    for case in golden["cases"]:
        for key, entry in case["q"].items():
            q = normalize_question(case["questions"][key])
            counts = []
            if case["images"]:
                grids = entry["image_grid_thw"]
                counts = [t * h * w // 4 for t, h, w in grids]
            text = render(case["state"], q.rendered, codes, counts)
            ids = tok.encode(text, add_special_tokens=False).ids
            ref_text = entry["text"]
            if case["images"]:   # the vendor string holds one <|image_pad|> per picture; the processor expands it
                parts = ref_text.split("<|image_pad|>")
                ref_text = parts[0] + "".join(("<|image_pad|>" * c) + p for c, p in zip(counts, parts[1:]))
            ok_text, ok_ids = text == ref_text, ids == entry["input_ids"]
            n += 1
            if not (ok_text and ok_ids):
                bad += 1
                print("XX %s/%s text=%s ids=%s" % (case["name"], key, ok_text, ok_ids))
    print("prompts: %d questions, %d mismatches" % (n, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
