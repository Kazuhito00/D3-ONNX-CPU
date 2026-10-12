# -*- coding: utf-8 -*-
"""Oracle gate (needs transformers: the export environment): the plain ``tokenizers`` file loader used at run time against
transformers' AutoTokenizer (what the vendor runtime uses) on random mixed-script text and on full d3 prompts.

    uv run --group export python verify/check_tokenizer_fuzz.py <hf dir with tokenizer files> [N]
"""
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from d3_decide.format import normalize_question, render  # noqa: E402
from d3_decide.runtime import load_tokenizer  # noqa: E402

POOLS = [
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789.,:;!?()[]{}<>/\\'\"-_=+*&^%$#@~` \t\n\r",
    "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでど",
    "日本語東京大阪夏目漱石吾輩猫請求不具合緊急度分類料金支払障害地理文学画像円赤青白",
    "ａｂｃ１２３　、。「」・ー〜！？",
    "é ü ß ñ ç ø å ł ő Ж Щ ф ы 한국어 中文 العربية עברית ไทย हिन्दी",
    "😀🎉🚀👍🏽❤️🇯🇵𠮷",
]
SPECIAL = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<think>", "</think>", "<|endoftext|>"]


def rand_text(rng):
    out = []
    for _ in range(rng.randint(1, 40)):
        r = rng.random()
        if r < 0.04:
            out.append(rng.choice(SPECIAL))
        else:
            pool = rng.choice(POOLS)
            out.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 12))))
    return "".join(out)


def main():
    hf = pathlib.Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    ref = AutoTokenizer.from_pretrained(str(hf))
    mine = load_tokenizer(hf / "tokenizer.json")
    codes = json.loads((hf / "decision_config.json").read_text(encoding="utf-8"))["codes"]
    rng = random.Random(11)
    bad = 0
    for i in range(n):
        text = rand_text(rng)
        if i % 4 == 0:        # a real d3 prompt around the random text
            q = normalize_question({"type": "choice", "instructions": rand_text(rng), "criteria": {rand_text(rng) or "x": rand_text(rng) for _ in range(3)}})
            text = render(rand_text(rng), q.rendered, codes)
        a = ref(text, add_special_tokens=False)["input_ids"]
        b = mine.encode(text, add_special_tokens=False).ids
        if a != b:
            bad += 1
            if bad <= 5:
                print("XX", repr(text[:80]))
    print("cases: %d   mismatches: %d" % (n, bad))
    print("failures: %d" % bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
