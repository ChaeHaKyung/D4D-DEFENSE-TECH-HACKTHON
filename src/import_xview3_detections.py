#!/usr/bin/env python3
"""Convert the xView3 First Place notebook CSV to the dashboard candidate schema."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from pretrained_vessel_detection import analyze_maritime_anomalies, dead_reckon_ais, haversine_m


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
AIS_PATH = OUTPUTS / "ais_interpolated_at_sar.csv"
STAC_PATH = OUTPUTS / "latest_stac_item.json"
DEFAULT_DESTINATION = OUTPUTS / "sar_ship_candidates.csv"
HARBOR_BBOX = [126.56, 37.40, 126.63, 37.48]
MAX_MATCH_RADIUS_M = 1800.0

FIELDS = [
    "candidate_id", "lat", "lon", "pixel_x", "pixel_y",
    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "area_pixels", "peak_gray",
    "zone", "coast_distance_m", "status", "nearest_mmsi", "distance_m",
    "ais_confidence", "sog", "repeat_scene_count", "risk_score", "anomaly_type",
    "scene_id", "sar_time", "length_m", "objectness_p", "is_vessel_p",
    "is_fishing_p", "is_fishing", "detector", "max_gap_seconds", "declared_length",
    "score_identity", "score_kinematics", "score_size", "score_behavior",
    "score_spoof_bonus", "action_level", "action_label", "action_color", "risk_flags",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        required = {"lat", "lon", "objectness_p", "is_vessel_p", "source_vh"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required xView3 columns: {', '.join(sorted(missing))}")
        return list(reader)


def _number(row: dict[str, str], key: str, row_number: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key} in detections.csv row {row_number}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Invalid {key} in detections.csv row {row_number}: expected a finite number")
    return value


def make_candidates(
    detections: list[dict[str, str]],
    ais_records: list[dict[str, str]],
    scene_id: str,
    sar_time: str,
) -> list[dict[str, object]]:
    """Associate detections with the SAR-time AIS snapshot and shape them for app.py."""
    valid_ais = []
    for row in ais_records:
        try:
            if not math.isfinite(float(row.get("lat", ""))) or not math.isfinite(float(row.get("lon", ""))):
                continue
            valid_ais.append(dead_reckon_ais(row))
        except (TypeError, ValueError, ZeroDivisionError):
            continue

    coordinates = [
        (_number(row, "lat", index), _number(row, "lon", index))
        for index, row in enumerate(detections, start=2)
    ]
    for row_number, (lat, lon) in enumerate(coordinates, start=2):
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"Invalid geographic coordinates in detections.csv row {row_number}")
    matched: dict[int, tuple[dict, float]] = {}
    if valid_ais and coordinates:
        costs = np.full((len(valid_ais), len(coordinates)), 1e6, dtype=float)
        for ais_index, vessel in enumerate(valid_ais):
            for detection_index, (lat, lon) in enumerate(coordinates):
                distance = haversine_m(vessel["lat"], vessel["lon"], lat, lon)
                if distance <= MAX_MATCH_RADIUS_M:
                    costs[ais_index, detection_index] = distance
        ais_indices, detection_indices = linear_sum_assignment(costs)
        for ais_index, detection_index in zip(ais_indices, detection_indices):
            distance = float(costs[ais_index, detection_index])
            if distance <= MAX_MATCH_RADIUS_M:
                matched[int(detection_index)] = (valid_ais[int(ais_index)], distance)

    west, south, east, north = HARBOR_BBOX
    candidates: list[dict[str, object]] = []
    for index, detection in enumerate(detections):
        lat, lon = coordinates[index]
        row_number = index + 2
        objectness = _number(detection, "objectness_p", row_number)
        vessel_score = _number(detection, "is_vessel_p", row_number)
        fishing_score = (
            _number(detection, "is_fishing_p", row_number)
            if detection.get("is_fishing_p")
            else ""
        )
        for score_name, score in (
            ("objectness_p", objectness),
            ("is_vessel_p", vessel_score),
            ("is_fishing_p", fishing_score),
        ):
            if score != "" and not 0 <= score <= 1:
                raise ValueError(f"Invalid {score_name} in detections.csv row {row_number}: expected 0..1")
        zone = "harbor" if west <= lon <= east and south <= lat <= north else "offshore"
        if index in matched:
            vessel, distance = matched[index]
            status = "matched"
            risk = 15
            mmsi = vessel["mmsi"]
            ais_confidence = vessel["confidence"]
            sog = vessel["sog"]
            distance_text = f"{distance:.1f}"
        else:
            status = "unmatched-candidate"
            risk = 85 if zone == "offshore" else 45
            mmsi = ""
            ais_confidence = ""
            sog = 0.0
            distance_text = ""

        try:
            length_m = float(detection.get("vessel_length_m") or 0.0)
        except ValueError as exc:
            raise ValueError(f"Invalid vessel_length_m in detections.csv row {index + 2}") from exc
        if not math.isfinite(length_m) or length_m < 0:
            raise ValueError(f"Invalid vessel_length_m in detections.csv row {index + 2}")

        candidates.append({
            "candidate_id": f"X3-{index + 1:04d}",
            "lat": f"{lat:.7f}",
            "lon": f"{lon:.7f}",
            "pixel_x": "",
            "pixel_y": "",
            "bbox_x0": "",
            "bbox_y0": "",
            "bbox_x1": "",
            "bbox_y1": "",
            "area_pixels": "",
            "peak_gray": "",
            "zone": zone,
            "coast_distance_m": "",
            "status": status,
            "nearest_mmsi": mmsi,
            "distance_m": distance_text,
            "ais_confidence": ais_confidence,
            "max_gap_seconds": vessel.get("max_gap_seconds", 0.0) if index in matched else 0.0,
            "declared_length": vessel.get("declared_length", 0.0) if index in matched else 0.0,
            "sog": sog if index in matched else "",
            "repeat_scene_count": 0,
            "risk_score": risk,
            "anomaly_type": "정상",
            "scene_id": scene_id,
            "sar_time": sar_time,
            "length_m": length_m,
            "objectness_p": objectness,
            "is_vessel_p": vessel_score,
            "is_fishing_p": fishing_score,
            "is_fishing": detection.get("is_fishing", ""),
            "detector": "xView3 First Place ensemble",
        })

    analyze_maritime_anomalies(candidates)
    return candidates


def import_detections(
    detections_path: Path,
    scene_id: str,
    destination: Path = DEFAULT_DESTINATION,
) -> int:
    item = json.loads(STAC_PATH.read_text(encoding="utf-8"))
    expected_scene = item["id"]
    if scene_id != expected_scene:
        raise ValueError(
            f"Scene mismatch: detections are for {scene_id}, but latest_stac_item.json is {expected_scene}. "
            "Run inference on the currently selected scene or select matching scene metadata first."
        )

    detections = read_rows(detections_path)
    source_paths = {Path(row["source_vh"]).name for row in detections if row.get("source_vh")}
    if source_paths and any(scene_id not in source_name for source_name in source_paths):
        raise ValueError(
            "The detections.csv source_vh filename does not contain the requested scene ID. "
            "Run inference on the selected GeoTIFF without renaming its scene-based filename."
        )
    with AIS_PATH.open(newline="", encoding="utf-8") as fp:
        ais_records = list(csv.DictReader(fp))
    properties = item.get("properties", {})
    sar_time = properties.get("start_datetime") or properties.get("datetime")
    if not sar_time:
        raise ValueError("latest_stac_item.json has no SAR acquisition timestamp")
    candidates = make_candidates(detections, ais_records, scene_id, sar_time)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(candidates)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return len(candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", required=True, type=Path, help="detections.csv produced by the Colab notebook")
    parser.add_argument("--scene-id", required=True, help="STAC scene ID used for inference; checked against latest_stac_item.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION, help="Dashboard candidate CSV output path")
    args = parser.parse_args()
    count = import_detections(args.detections, args.scene_id, args.output)
    print(f"Imported {count} xView3 detections into {args.output}")
    print("AIS unmatched detections are candidates only, not confirmed dark vessels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
