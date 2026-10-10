#!/usr/bin/env python3
"""확인된 MMSI들의 Sentinel-1 촬영시각 전후 과거 AIS 항적을 Open Waters에서 조회한다."""

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

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
# GeoJSON 좌표 순서: [서, 남, 동, 북]
BBOX = [126.40, 37.32, 126.62, 37.48]
SSL_CONTEXT = ssl.create_default_context(
    cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None
)
FIELDS = ["time", "mmsi", "lat", "lon", "sog", "cog", "heading", "nav_status", "source"]


def iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def collect(max_vessels: int | None = None) -> Path:
    item = json.loads((OUTPUTS / "latest_stac_item.json").read_text(encoding="utf-8"))
    sar_raw = item["properties"].get("start_datetime", item["properties"]["datetime"])
    sar_time = datetime.fromisoformat(sar_raw.replace("Z", "+00:00"))
    start, end = sar_time - timedelta(minutes=30), sar_time + timedelta(minutes=30)

    live_csv = OUTPUTS / "incheon_ais_live.csv"
    if not live_csv.exists():
        raise FileNotFoundError(f"{live_csv} 파일이 없습니다. 먼저 collect_ais.py를 실행하세요.")

    with live_csv.open(encoding="utf-8") as fp:
        mmsis = sorted({row["mmsi"] for row in csv.DictReader(fp) if row.get("mmsi")})
    if max_vessels:
        mmsis = mmsis[:max_vessels]

    rows = []
    west, south, east, north = BBOX

    for index, mmsi in enumerate(mmsis, 1):
        query = urllib.parse.urlencode({
            "from": iso(start),
            "to": iso(end),
            "interval": "0",
            "limit": "1000",
            "format": "geojson",
        })
        url = f"https://ais.openwaters.io/v1/vessels/{mmsi}/track?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "dark-vessel-hackathon/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30, context=SSL_CONTEXT) as response:
                feature = json.load(response)
        except Exception as exc:
            print(f"[{index}/{len(mmsis)}] {mmsi}: 조회 실패 ({exc})")
            time.sleep(0.2)
            continue

        props = feature.get("properties", {})
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates", []) if geometry.get("type") == "LineString" else []
        times = props.get("times", [])

        for i, coord in enumerate(coords):
            lon, lat = coord[:2]
            if not (west <= lon <= east and south <= lat <= north):
                continue
            rows.append({
                "time": times[i] if i < len(times) else "",
                "mmsi": mmsi,
                "lat": lat,
                "lon": lon,
                "sog": (props.get("sog") or [None] * len(coords))[i],
                "cog": (props.get("cog") or [None] * len(coords))[i],
                "heading": (props.get("heading") or [None] * len(coords))[i],
                "nav_status": (props.get("nav_status") or [None] * len(coords))[i],
                "source": "openwaters-track",
            })
        print(f"[{index}/{len(mmsis)}] {mmsi}: {len(coords)}개 항적점")
        time.sleep(0.15)

    output = OUTPUTS / "incheon_ais_at_sar_time.csv"
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"SAR 시각 ±30분 인천항 AIS {len(rows)}건 저장: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-vessels", type=int)
    args = parser.parse_args()
    collect(args.limit_vessels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())