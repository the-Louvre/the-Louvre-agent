import unittest

from app.agent3.schemas import AideModelResult, VisualReviewResult


class Agent3SchemaTests(unittest.TestCase):
    def test_probability_is_bounded(self):
        with self.assertRaises(ValueError):
            AideModelResult(status="ok", ai_probability=1.1, real_probability=0.0, model_version="x")

    def test_unknown_fields_are_rejected_on_result(self):
        with self.assertRaises(ValueError):
            VisualReviewResult.model_validate({"unexpected": True})
