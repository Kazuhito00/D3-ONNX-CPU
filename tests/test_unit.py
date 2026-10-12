# -*- coding: utf-8 -*-
"""Unit tests that need no model files: ``uv run python -m unittest discover -s tests`` (runtime dependencies only)."""
import json
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d3_decide import chat  # noqa: E402
from d3_decide.format import normalize_question, product_answer, render, user_prompt  # noqa: E402
from d3_decide.paths import MODELS, default_model_dir  # noqa: E402
from d3_decide.runtime import resolve_providers  # noqa: E402
from d3_decide.image import interpolation_tables, smart_resize, vision_position_ids  # noqa: E402

CODES = ["A", "B", "C", "D", "E", "F", "G", "H"]


class QuestionTests(unittest.TestCase):
    def test_choice_prompt(self):
        q = normalize_question({"type": "choice", "instructions": "Pick", "criteria": {"x": "first", "y": None}})
        text = user_prompt("S", q.rendered, CODES)
        self.assertEqual(text, "State:\nS\n\nQuestion:\nPick\n\nOptions:\nA: x: first\nB: y\n\nReply with only the code of the best option.")

    def test_noul_default_descriptions(self):
        q = normalize_question({"type": "noul", "instructions": "Ok?"})
        self.assertEqual(q.keys, ["false", "true"])
        self.assertIn("A: No / false\nB: Yes / true", user_prompt("", q.rendered, CODES))
        self.assertIn("(empty)", user_prompt("", q.rendered, CODES))

    def test_score_levels_become_options(self):
        q = normalize_question({"type": "score", "instructions": "Rate", "criteria": ["low", "high"]})
        self.assertEqual(q.keys, ["0", "1"])
        self.assertIn("A: low\nB: high", user_prompt("s", q.rendered, CODES))

    def test_invalid_questions(self):
        for bad in ({"type": "choice"}, {"type": "choice", "criteria": {}}, {"type": "score", "criteria": ["only"]},
                    {"type": "score", "criteria": ["a", "a"]}, {"type": "noul", "criteria": {"maybe": "x"}}, {"type": "text"}, "x"):
            with self.assertRaises(ValueError):
                normalize_question(bad)

    def test_render_trims_and_opens_the_assistant_turn(self):
        q = normalize_question({"type": "noul", "instructions": "Ok?"})
        text = render("S", q.rendered, CODES, [2])
        self.assertTrue(text.startswith("<|im_start|>system\nYou are a decision engine."))
        self.assertIn("<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|vision_end|>State:\nS", text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_answers(self):
        c = normalize_question({"type": "choice", "criteria": {"x": None, "y": None}})
        a = product_answer(c, [0.25, 0.75])
        self.assertEqual(a["choice"], "y")
        self.assertAlmostEqual(sum(a["probabilities"].values()), 1.0)
        n = normalize_question({"type": "noul"})
        self.assertAlmostEqual(product_answer(n, [0.2, 0.6])["noul"], 0.75)
        s = normalize_question({"type": "score", "criteria": ["a", "b", "c"]})
        self.assertAlmostEqual(product_answer(s, [0, 0, 1])["score"], 2.0)
        with self.assertRaises(ValueError):
            product_answer(c, [0.0, 0.0])


class VisionTests(unittest.TestCase):
    def test_smart_resize(self):
        self.assertEqual(smart_resize(224, 224, 32, 65536, 1638400), (256, 256))     # small pictures are scaled up to 65,536 px
        self.assertEqual(smart_resize(480, 640, 32, 65536, 1638400), (480, 640))
        h, w = smart_resize(3000, 4000, 32, 65536, 1638400)
        self.assertLessEqual(h * w, 1638400)
        self.assertEqual((h % 32, w % 32), (0, 0))
        with self.assertRaises(ValueError):
            smart_resize(10, 3000, 32, 65536, 1638400)

    def test_tables(self):
        idx, wt = interpolation_tables(8, 6, 48, 2)
        self.assertEqual(idx.shape, (48, 4))
        np.testing.assert_allclose(wt.sum(1), 1.0, atol=1e-6)
        self.assertEqual(vision_position_ids(4, 4, 2).tolist()[:4], [[0, 0], [0, 1], [1, 0], [1, 1]])   # merge-block order


class RopeTests(unittest.TestCase):
    def test_text_only_counts_up(self):
        pos = chat.position_ids([1, 2, 3, 4], [], 99)
        self.assertEqual(pos.shape, (3, 1, 4))
        self.assertEqual(pos[0, 0].tolist(), [0, 1, 2, 3])

    def test_picture_block(self):
        # 2 text tokens, a 4x4-patch picture (4 merged tokens), 2 text tokens
        ids = [1, 2] + [99] * 4 + [3, 4]
        pos = chat.position_ids(ids, [[1, 4, 4]], 99)
        self.assertEqual(pos[1, 0].tolist(), [0, 1, 2, 2, 2, 3, 4, 5][:2] + [2, 2, 3, 3] + [4, 5])
        self.assertEqual(pos[2, 0].tolist()[2:6], [2, 3, 2, 3])


class ProviderTests(unittest.TestCase):
    CPU = ["CPUExecutionProvider", "AzureExecutionProvider"]
    CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    DML = ["DmlExecutionProvider", "CPUExecutionProvider"]

    def test_cpu_is_the_default(self):
        self.assertEqual(resolve_providers("cpu", self.CUDA), ["CPUExecutionProvider"])
        self.assertEqual(resolve_providers(available=self.CUDA), ["CPUExecutionProvider"])

    def test_cuda_then_cpu(self):
        self.assertEqual(resolve_providers("cuda", self.CUDA), ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(resolve_providers("auto", self.CUDA), ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(resolve_providers("auto", self.DML), ["DmlExecutionProvider", "CPUExecutionProvider"])

    def test_missing_gpu_falls_back_loudly_or_fails(self):
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(resolve_providers("cuda", self.CPU), ["CPUExecutionProvider"])
        self.assertIn("warning", err.getvalue())
        self.assertEqual(resolve_providers("auto", self.CPU), ["CPUExecutionProvider"])      # auto: quiet by design
        with self.assertRaises(RuntimeError):
            resolve_providers("cuda", self.CPU, require_gpu=True)
        with self.assertRaises(RuntimeError):
            resolve_providers("auto", self.CPU, require_gpu=True)
        with self.assertRaises(RuntimeError):
            resolve_providers("cpu", self.CUDA, require_gpu=True)
        with self.assertRaises(ValueError):
            resolve_providers("tpu", self.CUDA)


class PathTests(unittest.TestCase):
    def test_default_model_dir(self):
        import os
        os.environ.pop("D3_LITE_MODEL_DIR", None)
        self.assertEqual(default_model_dir("lite", "root"), pathlib.Path("root") / "models" / "lite")
        self.assertEqual(default_model_dir("lite", "root", "elsewhere"), pathlib.Path("elsewhere"))
        os.environ["D3_LITE_MODEL_DIR"] = "env_dir"
        try:
            self.assertEqual(default_model_dir("lite", "root"), pathlib.Path("env_dir"))
        finally:
            del os.environ["D3_LITE_MODEL_DIR"]
        with self.assertRaises(ValueError):
            default_model_dir("huge")
        self.assertEqual(sorted(MODELS), ["edge", "lite", "nano"])


class GoldenTests(unittest.TestCase):
    def test_golden_file_is_consistent(self):
        paths = sorted((ROOT / "artifacts").glob("*/golden.json"))
        if not paths:
            self.skipTest("no artifacts/<model>/golden.json present")
        for path in paths:
            golden = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(golden["cases"])
            for case in golden["cases"]:
                for entry in case["q"].values():
                    self.assertAlmostEqual(sum(entry["probs"]), 1.0, places=4)
                    if "probs_fp32" in entry:
                        self.assertAlmostEqual(sum(entry["probs_fp32"]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
