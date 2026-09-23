import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def png_bytes():
    image = Image.new("RGB", (64, 64), "white")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class Agent3ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_create_poll_and_get_artifact(self):
        response = self.client.post("/api/v1/agent3/reviews", data={"task_id": "task_001"}, files={"image": ("fixture.png", png_bytes(), "image/png")})
        self.assertEqual(response.status_code, 202)
        accepted = response.json()
        result = self.client.get(accepted["status_url"])
        self.assertEqual(result.status_code, 200)
        payload = result.json()
        self.assertIn(payload["status"], {"completed", "degraded", "failed"})
        self.assertEqual(payload["schema_version"], "1.0.0")
        artifact = self.client.get(f"/api/v1/agent3/reviews/{accepted['review_id']}/artifacts/overlay.png")
        self.assertEqual(artifact.status_code, 200)
        self.assertTrue(artifact.headers["content-type"].startswith("image/png"))

    def test_bad_image_and_bad_artifact_are_rejected(self):
        response = self.client.post("/api/v1/agent3/reviews", data={"task_id": "task_001"}, files={"image": ("fixture.png", b"not-an-image", "image/png")})
        self.assertEqual(response.status_code, 400)
        response = self.client.get("/api/v1/agent3/reviews/vr_doesnotexist/artifacts/../../main.py")
        self.assertIn(response.status_code, {404, 400})

    def test_health_contract(self):
        response = self.client.get("/api/v1/agent3/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["fusion_version"], "fusion-rules-v1")
