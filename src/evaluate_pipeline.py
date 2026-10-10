#!/usr/bin/env python3
"""AIS 시간보간과 SAR-AIS 매칭의 과장 없는 내부 검증 지표를 만든다.

정답 선박 바운딩박스가 아직 없으므로 실제 detector precision/recall은 계산할 수 없다.
대신 AIS 중간점을 숨기는 홀드아웃 보간오차와, 고신뢰 AIS 중 CFAR 후보가 매칭한
비율을 대리지표로 제공한다. JSON의 warning 문구를 반드시 결과와 함께 제시해야 한다.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TRACKS = OUTPUTS / "incheon_ais_at_sar_time.csv"
SNAPSHOT = OUTPUTS / "ais_interpolated_at_sar.csv"
CANDIDATES = OUTPUTS / "sar_ship_candidates.csv"
REPORT = OUTPUTS / "evaluation.json"
BBOX = [126.40, 37.32, 126.62, 37.48]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위·경도 사이의 대권거리를 미터로 계산한다."""
    # 지구 평균 반지름. 짧은 부산항 거리에서는 충분한 정확도를 제공한다.
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def validate() -> dict:
    """보간 홀드아웃과 AIS 지지 탐지율을 계산하고 JSON 보고서를 저장한다."""
    from datetime import datetime

    def dt(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    # MMSI별 시계열을 만들기 위해 행을 선박별 리스트로 모은다.
    tracks: dict[str, list[dict]] = defaultdict(list)
    with TRACKS.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row.get("time") and row.get("lat") and row.get("lon"):
                row["dt"] = dt(row["time"])
                tracks[row["mmsi"]].append(row)

    # Leave-one-message-out: 가운데 실제 AIS 점을 숨기고 양옆 두 점으로 예측한다.
    errors = []
    for rows in tracks.values():
        rows.sort(key=lambda row: row["dt"])
        # 세 점씩 미끄러지며 left-middle-right 묶음을 만든다.
        for left, middle, right in zip(rows, rows[1:], rows[2:]):
            span = (right["dt"] - left["dt"]).total_seconds()
            # 시간 역전 또는 10분 넘는 긴 공백은 짧은 보간 검증에서 제외한다.
            if span <= 0 or span > 600:
                continue
            fraction = (middle["dt"] - left["dt"]).total_seconds() / span
            pred_lat = float(left["lat"]) + fraction * (float(right["lat"]) - float(left["lat"]))
            pred_lon = float(left["lon"]) + fraction * (float(right["lon"]) - float(left["lon"]))
            # 예측 좌표와 숨긴 실제 좌표 사이 오차를 미터로 누적한다.
            errors.append(haversine_m(pred_lat, pred_lon, float(middle["lat"]), float(middle["lon"])))

    with SNAPSHOT.open(encoding="utf-8") as fp:
        snapshot = list(csv.DictReader(fp))
    with CANDIDATES.open(encoding="utf-8") as fp:
        candidates = list(csv.DictReader(fp))

    west, south, east, north = BBOX
    # 분석영역 안에 있고 high/medium인 AIS만 탐지 대리지표의 분모로 사용한다.
    eligible = [
        row for row in snapshot
        if west <= float(row["lon"]) <= east and south <= float(row["lat"]) <= north
    ]
    high_medium = [row for row in eligible if row["confidence"] in {"high", "medium"}]
    # 한 선박 주변에 후보가 여러 개 있어도 MMSI는 한 번만 세도록 set을 사용한다.
    matched_mmsi = {row["nearest_mmsi"] for row in candidates if row["status"] == "matched" and row["nearest_mmsi"]}
    errors_array = np.asarray(errors, dtype=float)
    # 실제 정답 평가와 혼동하지 않도록 각 지표의 의미와 경고를 같이 기록한다.
    report = {
        "interpolation_holdout": {
            "samples": len(errors),
            "median_error_m": round(float(np.median(errors_array)), 2) if len(errors) else None,
            "p95_error_m": round(float(np.percentile(errors_array, 95)), 2) if len(errors) else None,
            "note": "AIS 중간점을 숨기고 전후 점 선형 보간으로 예측한 내부 검증",
        },
        "time_alignment": {
            "snapshot_vessels": len(snapshot),
            "high": sum(row["confidence"] == "high" for row in snapshot),
            "medium": sum(row["confidence"] == "medium" for row in snapshot),
            "low": sum(row["confidence"] == "low" for row in snapshot),
        },
        "detector_ais_support": {
            "sar_candidates": len(candidates),
            "ais_matched_candidates": sum(row["status"] == "matched" for row in candidates),
            "unique_matched_mmsi": len(matched_mmsi),
            "eligible_high_medium_ais": len(high_medium),
            "ais_supported_recall_proxy": round(len(matched_mmsi) / len(high_medium), 4) if high_medium else None,
            "warning": "완전한 정답 라벨이 아니므로 실제 precision/recall이 아닌 AIS 기반 점검 지표",
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"평가 보고서: {REPORT}")
    return report


if __name__ == "__main__":
    validate()
