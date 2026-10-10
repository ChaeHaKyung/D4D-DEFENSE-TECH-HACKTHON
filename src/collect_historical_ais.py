#!/usr/bin/env python3
"""현재 확인된 MMSI들의 Sentinel-1 촬영시각 전후 과거 AIS 항적을 조회한다.

중요한 한계: ``busan_ais_live.csv``에 현재 등장한 MMSI만 역조회한다. 따라서 SAR
촬영 당시에는 있었지만 현재 목록에 없는 선박과 AIS를 끈 선박은 포함되지 않는다.
결과는 완전한 역사 AIS 스냅숏이 아니라 부분 표본이다.
"""

from __future__ import annotations

import argparse
import csv
import json
import ssl
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path


# 프로젝트의 공통 입력·출력 위치를 절대경로로 만든다.
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
# 이 파일의 좌표 순서는 일반 GeoJSON 순서인 [서, 남, 동, 북]이다.
BBOX = [128.95, 34.95, 129.25, 35.20]
SSL_CONTEXT = ssl.create_default_context(
    cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None
)
FIELDS = ["time", "mmsi", "lat", "lon", "sog", "cog", "heading", "nav_status", "source"]


def iso(value: datetime) -> str:
    """datetime을 API가 요구하는 UTC ISO 8601 문자열로 바꾼다."""
    return value.isoformat().replace("+00:00", "Z")


def coordinates(feature: dict) -> list[list[float]]:
    """GeoJSON Point 또는 LineString에서 [경도, 위도] 좌표 배열을 꺼낸다."""
    geometry = feature.get("geometry")
    if not geometry:
        return []
    if geometry["type"] == "Point":
        return [geometry["coordinates"]]
    if geometry["type"] == "LineString":
        return geometry["coordinates"]
    return []


def collect(max_vessels: int | None = None) -> Path:
    """SAR 시각 ±30분의 항적을 MMSI별로 조회해 하나의 CSV로 합친다."""
    # STAC 메타데이터가 영상과 AIS를 연결하는 기준 시각의 원천이다.
    item = json.loads((OUTPUTS / "latest_stac_item.json").read_text(encoding="utf-8"))
    sar_raw = item["properties"].get("start_datetime", item["properties"]["datetime"])
    sar_time = datetime.fromisoformat(sar_raw.replace("Z", "+00:00"))
    # 넓게 ±30분을 수집하고, 뒤 단계에서 ±5분 고신뢰와 저신뢰를 나눈다.
    start, end = sar_time - timedelta(minutes=30), sar_time + timedelta(minutes=30)

    # 현재 스트림에서 확인한 MMSI를 중복 없이 정렬한다.
    with (OUTPUTS / "busan_ais_live.csv").open(encoding="utf-8") as fp:
        mmsis = sorted({row["mmsi"] for row in csv.DictReader(fp) if row.get("mmsi")})
    if max_vessels:
        mmsis = mmsis[:max_vessels]

    rows: list[dict] = []
    west, south, east, north = BBOX
    # Open Waters track API는 MMSI 한 척씩 조회해야 하므로 반복한다.
    for index, mmsi in enumerate(mmsis, 1):
        query = urllib.parse.urlencode(
            {"from": iso(start), "to": iso(end), "interval": "0", "limit": "1000", "format": "geojson"}
        )
        url = f"https://ais.openwaters.io/v1/vessels/{mmsi}/track?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "dark-vessel-hackathon/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as response:
                feature = json.load(response)
        except Exception as exc:
            # 한 척 조회 실패가 전체 작업을 중단시키지 않도록 건너뛴다.
            print(f"[{index}/{len(mmsis)}] {mmsi}: 조회 실패 ({exc})")
            time.sleep(0.2)
            continue
        props = feature.get("properties", {})
        coords = coordinates(feature)
        times = props.get("times", [])
        # GeoJSON 좌표와 properties의 시간·속력 배열은 같은 인덱스를 사용한다.
        for i, coord in enumerate(coords):
            lon, lat = coord[:2]
            # 부산 분석영역 밖의 항적점은 저장하지 않는다.
            if not (west <= lon <= east and south <= lat <= north):
                continue
            rows.append(
                {
                    "time": times[i] if i < len(times) else "",
                    "mmsi": mmsi,
                    "lat": lat,
                    "lon": lon,
                    "sog": (props.get("sog") or [None] * len(coords))[i],
                    "cog": (props.get("cog") or [None] * len(coords))[i],
                    "heading": (props.get("heading") or [None] * len(coords))[i],
                    "nav_status": (props.get("nav_status") or [None] * len(coords))[i],
                    "source": "openwaters-track",
                }
            )
        print(f"[{index}/{len(mmsis)}] {mmsi}: {len(coords)}개 항적점")
        # 공개 API에 너무 빠르게 연속 요청하지 않도록 짧게 쉰다.
        time.sleep(0.15)

    output = OUTPUTS / "busan_ais_at_sar_time.csv"
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"SAR 시각 ±30분 부산항 AIS {len(rows)}건 저장: {output}")
    return output


def main() -> int:
    """디버깅 시 조회 선박 수를 제한할 수 있는 CLI 진입점."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-vessels", type=int)
    args = parser.parse_args()
    collect(args.limit_vessels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
