#!/usr/bin/env python3
"""각 MMSI의 위치를 정확한 SAR 촬영시각에 맞춘 단일 AIS 스냅숏으로 만든다.

SAR는 한 시각의 영상이지만 AIS 메시지는 불규칙한 시각에 들어온다. 따라서 선박별
전후 메시지 사이를 선형 보간한다. ±5분 이내 자료는 high, 한쪽 점만 있으면 medium,
±30분까지 사용한 자료는 low로 표시해 불확실성을 숨기지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# 모든 단계가 같은 파일을 읽도록 경로를 중앙에서 정의한다.
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
TRACKS_CSV = OUTPUTS / "incheon_ais_at_sar_time.csv"
if not TRACKS_CSV.exists():
    TRACKS_CSV = OUTPUTS / "incheon_ais_at_sar_time.csv"
STAC_JSON = OUTPUTS / "latest_stac_item.json"
SNAPSHOT_CSV = OUTPUTS / "ais_interpolated_at_sar.csv"
# 출력 CSV 스키마. left/right_time은 보간 근거를 추적하기 위해 함께 보존한다.
FIELDS = [
    "sar_time", "mmsi", "lat", "lon", "sog", "cog", "method", "confidence",
    "max_gap_seconds", "left_time", "right_time", "source",
]


def parse_time(value: str) -> datetime:
    """문자열 끝의 Z(UTC)를 Python이 이해하는 +00:00으로 바꿔 파싱한다."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def interpolate_number(left: dict, right: dict, key: str, fraction: float) -> str:
    """SOG·COG 같은 숫자 필드를 동일한 시간비율로 선형 보간한다."""
    try:
        # CSV 값은 문자열이므로 계산 전에 실수로 바꾼다.
        a, b = float(left[key]), float(right[key])
        return f"{a + fraction * (b - a):.4f}"
    except (KeyError, TypeError, ValueError):
        # 값이 비어 있으면 계산을 포기하고 존재하는 원본값을 보존한다.
        return left.get(key) or right.get(key) or ""


def align(strict_minutes: int = 5, fallback_minutes: int = 30) -> list[dict]:
    """원시 항적을 MMSI별로 정렬하고 SAR 시각의 위치를 계산해 저장한다."""
    # 선택된 SAR 장면의 촬영 시작시각을 유일한 기준시각으로 사용한다.
    item = json.loads(STAC_JSON.read_text(encoding="utf-8"))
    sar_raw = item["properties"].get("start_datetime", item["properties"]["datetime"])
    sar_time = parse_time(sar_raw)

    # defaultdict(list)는 처음 보는 MMSI도 빈 리스트를 자동으로 만든다.
    tracks: dict[str, list[dict]] = defaultdict(list)
    with TRACKS_CSV.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            if row.get("mmsi") and row.get("time") and row.get("lat") and row.get("lon"):
                # 반복 비교를 위해 파싱한 datetime을 임시 필드에 저장한다.
                row["dt"] = parse_time(row["time"])
                tracks[row["mmsi"]].append(row)

    # 계산 단위를 분이 아닌 초로 통일한다.
    strict_seconds = strict_minutes * 60
    fallback_seconds = fallback_minutes * 60
    snapshot: list[dict] = []
    for mmsi, rows in tracks.items():
        # API 응답 순서를 믿지 않고 반드시 시간순으로 다시 정렬한다.
        rows.sort(key=lambda row: row["dt"])
        # SAR보다 이른 점 중 가장 가까운 것과 늦은 점 중 가장 가까운 것을 찾는다.
        left = next((row for row in reversed(rows) if row["dt"] <= sar_time), None)
        right = next((row for row in rows if row["dt"] >= sar_time), None)

        if left and right and right["dt"] != left["dt"]:
            # 정상적인 양방향 보간: SAR 시각이 두 AIS 메시지 사이에 있다.
            left_gap = (sar_time - left["dt"]).total_seconds()
            right_gap = (right["dt"] - sar_time).total_seconds()
            max_gap = max(left_gap, right_gap)
            if max_gap > fallback_seconds:
                continue
            # 0이면 왼쪽 점, 1이면 오른쪽 점이며 그 사이 비율로 좌표를 이동한다.
            fraction = left_gap / (left_gap + right_gap)
            lat = float(left["lat"]) + fraction * (float(right["lat"]) - float(left["lat"]))
            lon = float(left["lon"]) + fraction * (float(right["lon"]) - float(left["lon"]))
            method = "linear"
            # 양쪽 근거가 모두 ±5분 안이면 high, 아니면 low다.
            confidence = "high" if max_gap <= strict_seconds else "low"
            sog = interpolate_number(left, right, "sog", fraction)
            cog = interpolate_number(left, right, "cog", fraction)
        else:
            # 한쪽에만 메시지가 있거나 같은 시각이면 가장 가까운 실제 점을 사용한다.
            nearest = min(rows, key=lambda row: abs((row["dt"] - sar_time).total_seconds()))
            max_gap = abs((nearest["dt"] - sar_time).total_seconds())
            # ±30분도 넘는 오래된 위치는 매칭 근거로 사용하지 않는다.
            if max_gap > fallback_seconds:
                continue
            left = right = nearest
            lat, lon = float(nearest["lat"]), float(nearest["lon"])
            method = "nearest"
            confidence = "medium" if max_gap <= strict_seconds else "low"
            sog, cog = nearest.get("sog", ""), nearest.get("cog", "")

        # 웹·탐지·평가가 똑같은 좌표를 읽도록 최종 행을 만든다.
        snapshot.append(
            {
                "sar_time": sar_raw,
                "mmsi": mmsi,
                "lat": f"{lat:.7f}",
                "lon": f"{lon:.7f}",
                "sog": sog,
                "cog": cog,
                "method": method,
                "confidence": confidence,
                "max_gap_seconds": f"{max_gap:.1f}",
                "left_time": left["time"],
                "right_time": right["time"],
                "source": "openwaters-track",
            }
        )

    with SNAPSHOT_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(snapshot)
    # 실행 직후 데이터 품질을 확인할 수 있도록 신뢰도별 수를 출력한다.
    high = sum(row["confidence"] == "high" for row in snapshot)
    medium = sum(row["confidence"] == "medium" for row in snapshot)
    low = sum(row["confidence"] == "low" for row in snapshot)
    print(
        f"SAR 시각 AIS 스냅숏 {len(snapshot)}척 저장: "
        f"high(±{strict_minutes}분)={high}, medium={medium}, low(±{fallback_minutes}분)={low}"
    )
    print(f"보간 결과: {SNAPSHOT_CSV}")
    return snapshot


def main() -> int:
    """엄격·대체 시간창을 터미널에서 조절할 수 있게 한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-minutes", type=int, default=5)
    parser.add_argument("--fallback-minutes", type=int, default=30)
    args = parser.parse_args()
    align(args.strict_minutes, args.fallback_minutes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
