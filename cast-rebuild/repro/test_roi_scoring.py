"""CPU-only unit test of the recovered CAST counterfactual ROI scoring formula.

This is not an Ariel/Phi-3.5V GPU parity test. It validates the original
CounterfactualSelector.score implementation with controlled likelihood values.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from PIL import Image


FILE = Path(__file__).resolve().parents[1] / "cast" / "roi_selection.py"
SPEC = importlib.util.spec_from_file_location("_recovered_cast_roi_selection", FILE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScoringProbe(MODULE.CounterfactualSelector):
    def __init__(self):
        super().__init__(model=object(), processor=object(),
                         occlusion_modes=["blur", "mean"],
                         lambda_area=0.1, mu_consistency=0.1, T_score=32)
        self.calls = []

    def occlude(self, image, mask, mode="blur"):
        self.calls.append("occlude:" + mode)
        result = image.copy()
        result[0, 0, 0] = 1 if mode == "blur" else 2
        return result

    def draft_answer(self, image, question, max_new_tokens=None):
        return ""

    def loglikelihood(self, image, question, answer, T=None):
        marker = int(np.asarray(image)[0, 0, 0])
        self.calls.append("ll:" + str(marker))
        return {0: -4.0, 1: -7.0, 2: -5.0}[marker]


class TestRecoveredScoring(unittest.TestCase):
    def test_formula_and_call_order(self):
        probe = ScoringProbe()
        image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1  # area fraction = 4 / 16
        # Deltas = (3, 1); mean=2; -0.1*area=-0.025;
        # np.var((3,1))=1 so consistency term=-0.1.
        actual = probe.score(image, "question", mask, "draft answer")
        self.assertAlmostEqual(actual, 1.875, places=7)
        self.assertEqual(probe.calls, [
            "ll:0", "occlude:blur", "ll:1", "occlude:mean", "ll:2"
        ])

    def test_empty_answer_returns_zero(self):
        probe = ScoringProbe()
        image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        self.assertEqual(probe.score(image, "question", np.zeros((4, 4), np.uint8), ""), 0.0)
        # This test focuses on score math; draft generation normally requires
        # a VLM, so an empty draft is tested by overriding draft_answer.


if __name__ == "__main__":
    unittest.main()
