import unittest

from app.agent3.config import Agent3Config
from app.agent3.fusion import decide
from app.agent3.schemas import AideModelResult, ImlVitModelResult


def models(ai, tamper, area):
    return (
        AideModelResult(status="ok", ai_probability=ai, real_probability=1 - ai, model_version="aide"),
        ImlVitModelResult(status="ok", tamper_probability=tamper, mask_area_ratio=area, model_version="iml"),
    )


class Agent3FusionTests(unittest.TestCase):
    def setUp(self):
        self.config = Agent3Config(runtime_dir=self._tmp())

    def _tmp(self):
        from pathlib import Path
        import tempfile
        return Path(tempfile.mkdtemp())

    def test_all_fusion_branches(self):
        cases = [
            ((0.90, 0.90, 0.80), "likely_ai_generated"),
            ((0.90, 0.80, 0.20), "suspected_ai_assisted_edit"),
            ((0.90, 0.20, 0.00), "likely_ai_generated"),
            ((0.10, 0.80, 0.20), "edited_photo"),
            ((0.10, 0.20, 0.00), "no_anomaly_detected"),
            ((0.55, 0.80, 0.20), "inconclusive"),
        ]
        for values, expected in cases:
            with self.subTest(values=values):
                self.assertEqual(decide(*models(*values), self.config).source_type, expected)

    def test_single_model_degradation_is_explicit(self):
        aide = AideModelResult(status="ok", ai_probability=0.9, real_probability=0.1, model_version="aide")
        iml = ImlVitModelResult(status="timeout", model_version="iml")
        result = decide(aide, iml, self.config)
        self.assertEqual(result.source_type, "likely_ai_generated")
