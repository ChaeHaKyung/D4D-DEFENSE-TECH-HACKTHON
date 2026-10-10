#!/usr/bin/env python3
"""부산항의 현재 AIS 위치 이벤트를 Open Waters SSE 스트림에서 수집한다.

SSE(Server-Sent Events)는 서버가 연결을 유지하면서 새 이벤트를 한 줄씩 보내는
방식이다. 이 파일은 PositionReport 메시지만 골라 CSV로 저장한다. 저장한 MMSI
목록은 이후 ``collect_historical_ais.py``가 SAR 촬영시각의 과거 항적을 조회할 때
사용한다. 따라서 이 CSV는 그 자체로 완전한 과거 AIS 데이터가 아니다.
"""

from __future__ import annotations

import argparse
import csv
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path


# Open Waters API가 요구하는 순서는 [최소위도, 최소경도, 최대위도, 최대경도]다.
BBOX = [34.95, 128.95, 35.20, 129.25]
# __file__은 현재 파일 경로이고 parents[1]은 프로젝트 루트다.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
# macOS와 일반 Linux 환경 모두에서 HTTPS 인증서를 찾도록 SSL 컨텍스트를 만든다.
SSL_CONTEXT = ssl.create_default_context(
    cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None
)
# CSV 열 순서를 한 곳에서 고정하면 DictWriter가 항상 같은 스키마를 만든다.
FIELDS = [
    "time",
    "mmsi",
    "lat",
    "lon",
    "sog",
    "cog",
    "heading",
    "nav_status",
    "source",
    "synthesized",
]


def collect(seconds: int) -> Path:
    """지정한 초 동안 AIS 스트림을 읽어 중복을 제거한 CSV를 반환한다."""
    # 출력 폴더가 없으면 생성한다. exist_ok=True라서 이미 있어도 오류가 없다.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "busan_ais_live.csv"
    # snapshot=1은 연결 직후 최근 상태를 먼저 받고 이후 새 이벤트를 이어 받는다.
    query = urllib.parse.urlencode(
        {"bbox": ",".join(map(str, BBOX)), "snapshot": "1"}
    )
    # Accept 헤더로 일반 JSON 요청이 아니라 SSE 스트림 요청임을 알린다.
    req = urllib.request.Request(
        f"https://ais.openwaters.io/v1/stream?{query}",
        headers={"Accept": "text/event-stream", "User-Agent": "dark-vessel-hackathon/0.1"},
    )
    # monotonic 시계는 시스템 시간이 바뀌어도 증가하므로 제한시간 계산에 안전하다.
    deadline = time.monotonic() + seconds
    count = 0
    # 동일 MMSI·시각·좌표 이벤트가 중복 저장되지 않도록 키를 기억한다.
    seen = set()
    with output_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDS)
        writer.writeheader()
        with urllib.request.urlopen(req, timeout=max(seconds + 10, 30), context=SSL_CONTEXT) as response:
            while time.monotonic() < deadline:
                # SSE 한 줄을 UTF-8로 읽고 줄바꿈을 제거한다.
                line = response.readline().decode("utf-8", errors="replace").strip()
                # SSE 데이터 본문은 "data: " 접두사로 시작한다.
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                # 선박 위치가 아닌 연결 상태·메타데이터 이벤트는 제외한다.
                if event.get("type") != "event" or event.get("msg_type") != "PositionReport":
                    continue
                # 위도나 경도가 없는 메시지는 지도에 표시할 수 없어 제외한다.
                if event.get("lat") is None or event.get("lon") is None:
                    continue
                message = event.get("message", {})
                key = (event.get("mmsi"), event.get("time"), event.get("lat"), event.get("lon"))
                if key in seen:
                    continue
                seen.add(key)
                # 원본 이벤트와 내부 AIS 메시지 필드를 평평한 한 행으로 변환한다.
                writer.writerow(
                    {
                        "time": event.get("time"),
                        "mmsi": event.get("mmsi"),
                        "lat": event.get("lat"),
                        "lon": event.get("lon"),
                        "sog": message.get("Sog"),
                        "cog": message.get("Cog"),
                        "heading": message.get("TrueHeading"),
                        "nav_status": message.get("NavigationalStatus"),
                        "source": event.get("source"),
                        "synthesized": event.get("synthesized", False),
                    }
                )
                # 프로그램이 중간에 종료돼도 이미 받은 데이터는 남도록 즉시 디스크에 쓴다.
                fp.flush()
                count += 1
    print(f"AIS 위치 {count}건 저장: {output_path}")
    return output_path


def main() -> int:
    """터미널 인자를 해석해 collect()를 실행하는 CLI 진입점."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()
    collect(args.seconds)
    return 0


if __name__ == "__main__":
    # 다른 파일에서 import할 때는 실행하지 않고 직접 실행했을 때만 main()을 호출한다.
    raise SystemExit(main())
