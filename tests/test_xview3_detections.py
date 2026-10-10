import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from import_xview3_detections import make_candidates


class XView3DetectionImportTests(unittest.TestCase):
    def test_converts_detections_and_matches_ais_one_to_one(self):
        detections = [
            {
                "lat": "37.4400",
                "lon": "126.5500",
                "objectness_p": "0.9",
                "is_vessel_p": "0.8",
                "vessel_length_m": "42",
            },
            {
                "lat": "37.3600",
                "lon": "126.4500",
                "objectness_p": "0.7",
                "is_vessel_p": "0.6",
                "vessel_length_m": "25",
            },
        ]
        ais = [
            {
                "lat": "37.4400",
                "lon": "126.5500",
                "mmsi": "440123456",
                "sog": "4.0",
                "cog": "90",
                "max_gap_seconds": "0",
                "confidence": "high",
            },
        ]

        candidates = make_candidates(detections, ais, "scene-1", "2026-09-30T21:31:22Z")

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["status"], "matched")
        self.assertEqual(candidates[0]["nearest_mmsi"], "440123456")
        self.assertEqual(candidates[1]["status"], "unmatched-candidate")
        self.assertEqual(candidates[1]["candidate_id"], "X3-0002")
        self.assertEqual(candidates[1]["length_m"], 25.0)
        self.assertEqual(candidates[1]["detector"], "xView3 First Place ensemble")

    def test_rejects_invalid_probability_scores(self):
        detections = [{
            "lat": "37.4400",
            "lon": "126.5500",
            "objectness_p": "1.2",
            "is_vessel_p": "0.8",
            "vessel_length_m": "42",
        }]

        with self.assertRaisesRegex(ValueError, "expected 0..1"):
            make_candidates(detections, [], "scene-1", "2026-09-30T21:31:22Z")

    def test_missing_length_is_not_replaced_with_an_assumed_vessel_size(self):
        detections = [{
            "lat": "37.4400",
            "lon": "126.5500",
            "objectness_p": "0.9",
            "is_vessel_p": "0.8",
        }]

        candidate = make_candidates(detections, [], "scene-1", "2026-09-30T21:31:22Z")[0]

        self.assertEqual(candidate["length_m"], 0.0)
        self.assertEqual(candidate["score_size"], 0)


if __name__ == "__main__":
    unittest.main()
