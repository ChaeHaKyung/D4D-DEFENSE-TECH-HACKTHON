#!/usr/bin/env python3
"""CDSE에서 Sentinel-1 장면을 검색하고 부산 관심영역 영상을 생성한다.

STAC API는 영상 자체가 아니라 장면 ID·촬영시각·궤도 같은 카탈로그 정보를 찾는다.
Process API는 선택된 촬영시각과 영역의 실제 VV 래스터를 GeoTIFF 또는 PNG로 만든다.
``search``와 ``catalog``는 인증이 필요 없지만 ``download``와 ``preview``는 CDSE
OAuth client id/secret이 환경변수에 있어야 한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Copernicus Data Space STAC 검색 엔드포인트.
STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
# client credentials를 짧은 수명의 access token으로 교환하는 OAuth 주소.
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
# 선택한 장면을 실제 래스터로 처리하는 Sentinel Hub Process API 주소.
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
# 모든 파일에서 공통으로 쓰는 부산 분석영역 [서, 남, 동, 북].
BBOX = [126.40, 37.32, 126.62, 37.48]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
SSL_CONTEXT = ssl.create_default_context(
    cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None
)


def request_json(url: str, *, payload: dict | None = None, headers: dict | None = None) -> dict:
    """GET 또는 JSON POST 요청을 보내고 응답 본문을 Python dict로 반환한다."""
    # payload가 없으면 GET, 있으면 UTF-8 JSON 본문을 가진 POST 요청이 된다.
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=90, context=SSL_CONTEXT) as response:
        return json.load(response)


def search_latest(days: int = 45) -> dict:
    """최근 기간의 호환 장면을 검색하고 가장 최신 한 장을 반환한다."""
    candidates = search_scenes(days=days, limit=10)
    if not candidates:
        raise RuntimeError("최근 검색 범위에서 IW/VV Sentinel-1 GRD 장면을 찾지 못했습니다.")
    return candidates[0]


def search_scenes(days: int = 90, limit: int = 100) -> list[dict]:
    """부산 BBOX와 교차하는 Sentinel-1 IW/VV 장면들을 최신순으로 반환한다."""
    # STAC 검색 시간범위는 현재 UTC에서 days만큼 과거까지다.
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    # urllib.urlencode가 쿼리 문자열로 바꿀 수 있도록 모든 값을 문자열로 구성한다.
    query = {
        "collections": "sentinel-1-grd",
        "bbox": ",".join(map(str, BBOX)),
        "datetime": f"{start.isoformat().replace('+00:00', 'Z')}/{end.isoformat().replace('+00:00', 'Z')}",
        "limit": str(limit),
        "sortby": "-datetime",
    }
    result = request_json(f"{STAC_SEARCH}?{urllib.parse.urlencode(query)}")
    # 같은 컬렉션에도 모드와 편파가 다르므로 우리 분석과 호환되는 장면만 남긴다.
    candidates = [
        item
        for item in result.get("features", [])
        if item.get("properties", {}).get("sar:instrument_mode") == "IW"
        and "VV" in item.get("properties", {}).get("sar:polarizations", [])
    ]
    return candidates


def save_catalog(items: list[dict]) -> Path:
    """여러 날짜 장면의 핵심 메타데이터를 팀이 보기 쉬운 CSV로 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "sar_scene_catalog.csv"
    fields = ["scene_id", "start_datetime", "end_datetime", "platform", "mode", "polarizations", "orbit_state"]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for item in items:
            props = item.get("properties", {})
            # 편파 배열 ['VV', 'VH']는 CSV 한 칸에 VV+VH로 기록한다.
            writer.writerow(
                {
                    "scene_id": item.get("id"),
                    "start_datetime": props.get("start_datetime", props.get("datetime")),
                    "end_datetime": props.get("end_datetime", props.get("datetime")),
                    "platform": props.get("platform"),
                    "mode": props.get("sar:instrument_mode"),
                    "polarizations": "+".join(props.get("sar:polarizations", [])),
                    "orbit_state": props.get("sat:orbit_state"),
                }
            )
    print(f"최근 SAR 장면 {len(items)}개 카탈로그: {path}")
    return path


def save_search_result(item: dict) -> None:
    """선택된 최신 장면의 전체 STAC JSON과 제공 썸네일을 저장한다."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT_DIR / "latest_stac_item.json"
    metadata_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")

    # 일부 STAC 항목에만 thumbnail asset이 있으므로 존재할 때만 받는다.
    thumbnail = item.get("assets", {}).get("thumbnail", {}).get("href")
    if thumbnail:
        req = urllib.request.Request(thumbnail)
        with urllib.request.urlopen(req, timeout=90, context=SSL_CONTEXT) as response:
            (OUTPUT_DIR / "latest_thumbnail.png").write_bytes(response.read())

    # 사람이 즉시 선택 결과를 확인할 수 있도록 핵심 필드를 터미널에 출력한다.
    props = item["properties"]
    print(f"ID: {item['id']}")
    print(f"촬영 시작(UTC): {props.get('start_datetime', props.get('datetime'))}")
    print(f"플랫폼: {props.get('platform')}")
    print(f"모드/편파: {props.get('sar:instrument_mode')} / {props.get('sar:polarizations')}")
    print(f"궤도: {props.get('sat:orbit_state')}")
    print(f"메타데이터: {metadata_path}")


def get_access_token() -> str:
    """환경변수의 CDSE client credentials로 access token을 발급받는다."""
    # 비밀키는 소스코드나 명령행 인자로 받지 않고 환경변수에서만 읽는다.
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "CDSE_CLIENT_ID와 CDSE_CLIENT_SECRET을 환경변수로 설정해야 합니다. "
            "비밀키를 채팅이나 소스코드에 붙여 넣지 마세요."
        )
    # OAuth2 client_credentials grant는 form-urlencoded 형식을 사용한다.
    form = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=60, context=SSL_CONTEXT) as response:
        return json.load(response)["access_token"]


def expanded_time_range(item: dict, minutes: int = 5) -> tuple[str, str]:
    """Process API가 장면을 확실히 찾도록 STAC 촬영구간 앞뒤에 여유를 준다."""
    props = item["properties"]
    raw_start = props.get("start_datetime", props["datetime"])
    raw_end = props.get("end_datetime", props["datetime"])
    start = datetime.fromisoformat(raw_start.replace("Z", "+00:00")) - timedelta(minutes=minutes)
    end = datetime.fromisoformat(raw_end.replace("Z", "+00:00")) + timedelta(minutes=minutes)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def download_geotiff(item: dict) -> Path:
    """선택 장면을 분석용 2밴드 FLOAT32 GeoTIFF(VV dB, dataMask)로 저장한다."""
    token = get_access_token()
    start, end = expanded_time_range(item)
    # Evalscript는 서버가 각 픽셀에서 어떤 입력 밴드를 어떻게 출력할지 정의한다.
    # VV는 선형 power로 들어오므로 10*log10(VV)로 dB 단위로 바꾼다.
    evalscript = """//VERSION=3
function setup() {
  return {input: [\"VV\", \"dataMask\"], output: {bands: 2, sampleType: \"FLOAT32\"}};
}
function evaluatePixel(s) {
  let db = s.VV > 0 ? 10 * Math.log(s.VV) / Math.LN10 : -40;
  return [db, s.dataMask];
}
"""
    # bounds는 공간, timeRange는 시간, dataFilter는 SAR 제품 조건을 제한한다.
    payload = {
        "input": {
            "bounds": {
                "bbox": BBOX,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [
                {
                    "type": "S1GRD",
                    "dataFilter": {
                        "timeRange": {"from": start, "to": end},
                        "mosaickingOrder": "mostRecent",
                        "acquisitionMode": "IW",
                        "polarization": "DV",
                        "resolution": "HIGH",
                    },
                    "processing": {
                        # 지표 좌표로 정사보정하고 타원체 기준 gamma0를 계산한다.
                        "orthorectify": True,
                        "backCoeff": "GAMMA0_ELLIPSOID",
                        # 3×3 Lee 필터로 SAR 고유의 speckle 잡음을 일부 완화한다.
                        "speckleFilter": {"type": "LEE", "windowSizeX": 3, "windowSizeY": 3},
                    },
                }
            ],
        },
        # 현재 1024²는 데모 크기다. 최종 학습용은 더 작은 AOI/원해상도로 바꿔야 한다.
        "output": {
            "width": 1024,
            "height": 1024,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }
    # 발급된 bearer token을 Authorization 헤더에 넣어 Process API를 호출한다.
    req = urllib.request.Request(
        PROCESS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    output_path = OUTPUT_DIR / "incheon_vv_db.tiff"
    with urllib.request.urlopen(req, timeout=180, context=SSL_CONTEXT) as response:
        output_path.write_bytes(response.read())
    print(f"GeoTIFF: {output_path}")
    return output_path


def download_visual_preview(item: dict) -> Path:
    """웹 지도와 일반 이미지 앱에서 볼 수 있는 8-bit 회색조 PNG를 만든다."""
    token = get_access_token()
    start, end = expanded_time_range(item)
    evalscript = """//VERSION=3
function setup() {
  return {input: [\"VV\", \"dataMask\"], output: {bands: 4}};
}
function evaluatePixel(s) {
  let db = s.VV > 0 ? 10 * Math.log(s.VV) / Math.LN10 : -40;
  // -30~0 dB를 화면의 0~1 밝기로 자르고 나머지는 포화시킨다.
  let gray = Math.max(0, Math.min(1, (db + 30) / 30));
  return [gray, gray, gray, s.dataMask];
}
"""
    payload = {
        "input": {
            "bounds": {
                "bbox": BBOX,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [
                {
                    "type": "S1GRD",
                    "dataFilter": {
                        "timeRange": {"from": start, "to": end},
                        "mosaickingOrder": "mostRecent",
                        "acquisitionMode": "IW",
                        "polarization": "DV",
                        "resolution": "HIGH",
                    },
                    "processing": {
                        "orthorectify": True,
                        "backCoeff": "GAMMA0_ELLIPSOID",
                        "speckleFilter": {"type": "LEE", "windowSizeX": 3, "windowSizeY": 3},
                    },
                }
            ],
        },
        # GeoTIFF와 같은 BBOX/크기를 사용해야 지도에서 정확히 겹친다.
        "output": {
            "width": 1024,
            "height": 1024,
            "responses": [{"identifier": "default", "format": {"type": "image/png"}}],
        },
        "evalscript": evalscript,
    }
    req = urllib.request.Request(
        PROCESS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    output_path = OUTPUT_DIR / "incheon_vv_visual.png"
    with urllib.request.urlopen(req, timeout=180, context=SSL_CONTEXT) as response:
        output_path.write_bytes(response.read())
    print(f"시각화 PNG: {output_path}")
    return output_path


def main() -> int:
    """search/catalog/download/preview 중 사용자가 고른 명령을 실행한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["search", "catalog", "download", "preview"])
    parser.add_argument("--days", type=int, default=45)
    args = parser.parse_args()

    try:
        if args.command == "catalog":
            # catalog는 여러 장의 목록만 저장하고 영상 처리는 하지 않는다.
            save_catalog(search_scenes(days=args.days, limit=100))
            return 0
        item = search_latest(args.days)
        save_search_result(item)
        if args.command == "download":
            download_geotiff(item)
        elif args.command == "preview":
            download_visual_preview(item)
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
