import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from risk_scoring import score_candidate
from pretrained_vessel_detection import analyze_maritime_anomalies


def vessel(**overrides):
    result = {
        "candidate_id": "T-1",
        "lat": 37.44,
        "lon": 126.59,
        "status": "matched",
        "nearest_mmsi": "440123456",
        "max_gap_seconds": 0,
        "sog": 3,
        "length_m": 50,
        "declared_length": 50,
    }
    result.update(overrides)
    return result


class RiskScoringTests(unittest.TestCase):
    def test_normal_harbor_vessel_scores_each_axis(self):
        scores = score_candidate(vessel())

        self.assertEqual(scores["zone"], "harbor")
        self.assertEqual(scores["score_identity"], 0)
        self.assertEqual(scores["score_kinematics"], 0)
        self.assertEqual(scores["score_size"], 10)
        self.assertEqual(scores["score_behavior"], 0)
        self.assertEqual(scores["risk_score"], 10)
        self.assertEqual(scores["action_level"], "정상")

    def test_ais_gap_bands(self):
        self.assertEqual(score_candidate(vessel(max_gap_seconds=300))["score_identity"], 15)
        self.assertEqual(score_candidate(vessel(max_gap_seconds=301))["score_identity"], 15)
        self.assertEqual(score_candidate(vessel(max_gap_seconds=900))["score_identity"], 25)
        self.assertEqual(score_candidate(vessel(max_gap_seconds=901))["score_identity"], 25)
        self.assertEqual(score_candidate(vessel(max_gap_seconds=3600))["score_identity"], 25)
        self.assertEqual(score_candidate(vessel(max_gap_seconds=3601))["score_identity"], 40)

    def test_coastal_and_offshore_kinematics(self):
        coastal = score_candidate(vessel(lon=126.65))
        inside_buffer = score_candidate(vessel(lon=126.685))
        outside_buffer = score_candidate(vessel(lon=126.688))
        fast_offshore = score_candidate(vessel(lon=126.70, sog=5.1))
        drifting_offshore = score_candidate(vessel(lon=126.70, sog=0.4))

        self.assertEqual(coastal["zone"], "coastal")
        self.assertEqual(coastal["score_kinematics"], 5)
        self.assertEqual(inside_buffer["zone"], "coastal")
        self.assertEqual(outside_buffer["zone"], "offshore")
        self.assertEqual(fast_offshore["score_kinematics"], 15)
        self.assertEqual(drifting_offshore["score_kinematics"], 20)

    def test_unmatched_or_unknown_speed_does_not_get_drift_points(self):
        unmatched = score_candidate(
            vessel(lon=126.70, status="unmatched-candidate", nearest_mmsi="", sog=0)
        )
        unknown_speed = score_candidate(vessel(lon=126.70, sog=""))

        self.assertEqual(unmatched["score_kinematics"], 0)
        self.assertEqual(unknown_speed["score_kinematics"], 0)

    def test_dimension_bands(self):
        self.assertEqual(score_candidate(vessel(length_m=34.9))["score_size"], 5)
        self.assertEqual(score_candidate(vessel(length_m=35))["score_size"], 10)
        self.assertEqual(score_candidate(vessel(length_m=90))["score_size"], 10)
        self.assertEqual(score_candidate(vessel(length_m=90.1))["score_size"], 20)

    def test_spoof_bonus_is_capped_and_assigns_five_level_action(self):
        scores = score_candidate(
            vessel(
                lon=126.70,
                status="matched",
                nearest_mmsi="12",
                max_gap_seconds=3601,
                sog=0.4,
                length_m=120,
                declared_length=50,
            ),
            behavior_points=20,
            behavior_flags=["STS-Dark"],
        )

        self.assertEqual(scores["score_spoof_bonus"], 15)
        self.assertEqual(scores["risk_score"], 100)
        self.assertEqual(scores["action_level"], "심각")
        self.assertEqual(scores["action_color"], "#ef4444")

    def test_all_action_level_boundaries(self):
        cases = [
            (vessel(length_m=34.9), 20, 25, "정상"),
            (vessel(max_gap_seconds=300, length_m=35), 1, 26, "관심"),
            (
                vessel(status="unmatched-candidate", nearest_mmsi="", length_m=34.9),
                6, 51, "주의",
            ),
            (
                vessel(
                    lon=126.70, status="unmatched-candidate",
                    nearest_mmsi="", length_m=120,
                ),
                11, 71, "경계",
            ),
            (
                vessel(
                    lon=126.70, max_gap_seconds=3601, nearest_mmsi="12",
                    sog=3, length_m=120, declared_length=50,
                ),
                15, 90, "심각",
            ),
        ]
        for row, behavior_points, score, level in cases:
            with self.subTest(score=score):
                actual = score_candidate(row, behavior_points=behavior_points)
                self.assertEqual(actual["risk_score"], score)
                self.assertEqual(actual["action_level"], level)

    def test_cluster_and_sts_behavior_rules(self):
        clustered = [
            vessel(candidate_id="C-1", lon=126.70, lat=37.440),
            vessel(candidate_id="C-2", lon=126.70, lat=37.444),
            vessel(candidate_id="C-3", lon=126.704, lat=37.442),
        ]
        analyze_maritime_anomalies(clustered)
        self.assertTrue(all(row["score_behavior"] == 8 for row in clustered))

        sts = [
            vessel(candidate_id="A", lon=126.70, lat=37.44),
            vessel(
                candidate_id="B", lon=126.70, lat=37.441,
                status="unmatched-candidate", nearest_mmsi="",
            ),
        ]
        analyze_maritime_anomalies(sts)
        self.assertEqual([row["score_behavior"] for row in sts], [20, 20])


if __name__ == "__main__":
    unittest.main()
