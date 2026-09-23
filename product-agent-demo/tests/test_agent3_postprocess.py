import unittest

from app.agent3.postprocess import summarize_mask


class Agent3PostprocessTests(unittest.TestCase):
    def test_connected_region_and_area(self):
        mask = [[0.0 for _ in range(10)] for _ in range(10)]
        for y in range(2, 8):
            for x in range(2, 8):
                mask[y][x] = 0.9
        tamper, area, regions, final_mask = summarize_mask(mask)
        self.assertAlmostEqual(tamper, 0.9)
        self.assertAlmostEqual(area, 0.36)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].bbox, (2, 2, 6, 6))
        self.assertEqual(sum(sum(row) for row in final_mask), 36)

    def test_small_region_is_removed(self):
        mask = [[0.0 for _ in range(20)] for _ in range(20)]
        mask[0][0] = 1.0
        _, area, regions, _ = summarize_mask(mask)
        self.assertEqual(regions, [])
        self.assertEqual(area, 0.0)
