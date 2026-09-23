import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from fastapi.testclient import TestClient

from app.agent3.config import Agent3Config
from app.agent3.detectors.fake import FakeAideDetector, FakeImlVitLocalizer
from app.agent3.repository import ReviewRepository
from app.agent3.schemas import AideModelResult, ModelError, RawMaskResult
from app.agent3.service import VisualReviewService
from app.main import app


def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(output, format="PNG")
    return output.getvalue()


class FailingAide:
    async def predict(self, image_bytes, request_id):
        return AideModelResult(status="timeout", model_version="aide-test", error=ModelError(code="model_timeout", message="timeout", retryable=True))


class FailingIml:
    async def predict(self, image_bytes, request_id):
        return RawMaskResult(status="unavailable", model_version="iml-test", error=ModelError(code="unavailable", message="down", retryable=True))


class Agent3DegradationTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_models_failed_is_failed_not_degraded(self):
        with TemporaryDirectory() as directory:
            config = Agent3Config(runtime_dir=Path(directory))
            service = VisualReviewService(ReviewRepository(config.runtime_dir), config, FailingAide(), FailingIml())
            accepted = await service.create_review("task_fail", image_bytes())
            await service.process_review(accepted.review_id)
            result = await service.get(accepted.review_id)
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.source_type, "inconclusive")


class Agent3ValidationTests(unittest.TestCase):
    def test_invalid_product_context_is_422(self):
        client = TestClient(app)
        response = client.post("/api/v1/agent3/reviews", data={"task_id": "task_001", "product_context": "not-json"}, files={"image": ("fixture.png", image_bytes(), "image/png")})
        self.assertEqual(response.status_code, 422)

    def test_unicode_task_id_is_422(self):
        client = TestClient(app)
        response = client.post("/api/v1/agent3/reviews", data={"task_id": "任务_1"}, files={"image": ("fixture.png", image_bytes(), "image/png")})
        self.assertEqual(response.status_code, 422)
