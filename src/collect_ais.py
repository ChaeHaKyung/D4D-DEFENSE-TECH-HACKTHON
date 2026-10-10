#!/usr/bin/env python3
"""인천항 실시간 AIS 위치 이벤트를 Open Waters SSE 스트림에서 수집한다."""

from __future__ import annotations

import argparse
import csv
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path

# 인천항 외항 BBOX: [남위, 서경, 북위, 동경] (Open Waters 표준 규격)
SOUTH = 37.32
WEST = 126.40
NORTH = 37.48
EAST = 126.62

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
SSL_CONTEXT = ssl.create_default_context(
    cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None
)

FIELDS = [
    "time", "mmsi", "lat", "lon", "sog", "cog", "heading",
    "nav_status", "source", "synthesized",
]


def collect(seconds: int = 30) -> Path:
    """지정한 초 동안 Open Waters SSE 스트림을 읽어 CSV로 저장"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "incheon_ais_live.csv"

    # API 표준 bbox 규격: min_lat,min_lon,max_lat,max_lon
    bbox_str = f"{SOUTH},{WEST},{NORTH},{EAST}"
    params = {
        "bbox": bbox_str,
        "snapshot": "1"
    }
    url = f"https://ais.openwaters.io/v1/stream?{urllib.parse.urlencode(params)}"
    
    print(f"-> AIS 스트림 연결 시도: {url}")
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "text/event-stream",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        },
    )

    deadline = time.monotonic() + seconds
    seen = set()
    count = 0

    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()

        try:
            with urllib.request.urlopen(req, timeout=max(seconds + 10, 30), context=SSL_CONTEXT) as response:
                print(f"-> 스트림 연결 성공. {seconds}초간 수신 대기...")
                while time.monotonic() < deadline:
                    line = response.readline().decode("utf-8", errors="replace").strip()
                    if not line.startswith("data: "):
                        continue

                    try:
                        event = json.loads(line[6:])
                    except Exception:
                        continue

                    if event.get("type") != "event" or event.get("msg_type") != "PositionReport":
                        continue
                    if event.get("lat") is None or event.get("lon") is None:
                        continue

                    message = event.get("message", {})
                    key = (event.get("mmsi"), event.get("time"), event.get("lat"), event.get("lon"))
                    if key in seen:
                        continue
                    seen.add(key)

                    writer.writerow({
                        "time": event.get("time"),
                        "mmsi": event.get("mmsi"),
                        "lat": event.get("lat"),
                        "lon": event.get("lon"),
                        "sog": message.get("Sog"),
                        "cog": message.get("Cog"),
                        "heading": message.get("TrueHeading"),
                        "nav_status": message.get("NavigationalStatus"),
                        "source": "openwaters-live",
                        "synthesized": event.get("synthesized", False),
                    })
                    fp.flush()
                    count += 1
        except Exception as exc:
            print(f"[알림] 스트림 수신 종료 또는 오류 ({exc})")

    print(f"인천항 AIS 위치 {count}건 저장 완료: {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()
    collect(args.seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())