import base64
import io
import unittest

from PIL import Image

from app.agent3.detectors.iml_vit_client import _decode_probability_mask


class Agent3ImlVitClientTests(unittest.TestCase):
    def test_decode_gray16_probability_mask(self):
        image = Image.new("I;16", (2, 1))
        image.putdata([0, 65535])
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        width, height, mask = _decode_probability_mask(base64.b64encode(buffer.getvalue()).decode("ascii"))
        self.assertEqual((width, height), (2, 1))
        self.assertAlmostEqual(mask[0][0], 0.0, places=5)
        self.assertAlmostEqual(mask[0][1], 1.0, places=5)
