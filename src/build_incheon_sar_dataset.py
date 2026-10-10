#!/usr/bin/env python3
"""Build a reproducible Sentinel-1 VV/VH time-series dataset for Incheon Port.

The catalogue step is public and does not require credentials.  The download
step exchanges CDSE client credentials from environment variables for a short
lived access token, then creates one analysis GeoTIFF per acquisition.

Two catalogues are produced:

* training_pool.csv: every compatible IW/DV scene, maximising training variety.
* temporal_core.csv: the densest orbit-direction/relative-orbit group, suitable
  for comparisons through time because the viewing geometry stays consistent.

The script never stores the client id, secret, or access token on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path


STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

# Focused outer-port/approach area: enough sea for ships, with some harbour edge
# for testing coastal false positives. Order is west, south, east, north.
AOI_NAME = "incheon_outer_port"
BBOX = [126.40, 37.32, 126.62, 37.48]

# At latitude 37.4 this fixed canvas is approximately 10 m per output pixel.
# The GeoTIFF remains georeferenced in CRS84. Do not resize it before training.
WIDTH = 1950
HEIGHT = 1780

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "outputs" / "datasets" / "incheon"
CATALOG_DIR = DATASET_DIR / "catalog"
METADATA_DIR = DATASET_DIR / "metadata"
RAW_DIR = DATASET_DIR / "sar" / "geotiff_raw"
LEE_DIR = DATASET_DIR / "sar" / "geotiff_lee3"
LOG_PATH = DATASET_DIR / "download_log.jsonl"

SSL_CONTEXT = ssl.create_default_context(
    cafile="/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None
)

CSV_FIELDS = [
    "scene_id",
    "start_datetime",
    "end_datetime",
    "platform",
    "orbit_state",
    "relative_orbit",
    "absolute_orbit",
    "mode",
    "polarizations",
    "product_type",
    "split",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def request_json(
    url: str,
    *,
    payload: dict | None = None,
    headers: dict | None = None,
    form: dict | None = None,
    timeout: int = 90,
) -> dict:
    if payload is not None and form is not None:
        raise ValueError("payload and form are mutually exclusive")
    body = None
    request_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    elif form is not None:
        body = urllib.parse.urlencode(form).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    request_headers.setdefault("User-Agent", "incheon-dark-vessel-hackathon/1.0")
    request = urllib.request.Request(url, data=body, headers=request_headers)
    with urllib.request.urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
        return json.load(response)


def search_scenes(days: int) -> list[dict]:
    """Return every Incheon IW GRD acquisition containing both VV and VH."""
    end = utc_now()
    start = end - timedelta(days=days)
    payload = {
        "collections": ["sentinel-1-grd"],
        "bbox": BBOX,
        "datetime": f"{iso(start)}/{iso(end)}",
        "limit": 100,
        "sortby": [{"field": "datetime", "direction": "desc"}],
    }
    result = request_json(STAC_SEARCH, payload=payload)
    compatible: dict[str, dict] = {}
    for item in result.get("features", []):
        props = item.get("properties", {})
        polarizations = set(props.get("sar:polarizations", []))
        if (
            props.get("sar:instrument_mode") == "IW"
            and {"VV", "VH"}.issubset(polarizations)
            and props.get("product:type") == "IW_GRDH_1S"
        ):
            compatible[item["id"]] = item
    return sorted(
        compatible.values(),
        key=lambda item: scene_time(item),
    )


def scene_time(item: dict) -> datetime:
    props = item["properties"]
    raw = props.get("start_datetime", props.get("datetime"))
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def temporal_core(items: list[dict]) -> tuple[list[dict], tuple[str, int | None]]:
    """Choose the orbit group with the most scenes, preserving geometry."""
    groups = Counter(
        (
            item["properties"].get("sat:orbit_state", "unknown"),
            item["properties"].get("sat:relative_orbit"),
        )
        for item in items
    )
    if not groups:
        return [], ("unknown", None)
    orbit_key, _ = groups.most_common(1)[0]
    selected = [
        item
        for item in items
        if (
            item["properties"].get("sat:orbit_state", "unknown"),
            item["properties"].get("sat:relative_orbit"),
        )
        == orbit_key
    ]
    return selected, orbit_key


def split_by_time(items: list[dict]) -> dict[str, str]:
    """Chronological 70/15/15 split prevents near-future leakage into training."""
    total = len(items)
    train_end = math.ceil(total * 0.70)
    val_end = math.ceil(total * 0.85)
    result = {}
    for index, item in enumerate(items):
        result[item["id"]] = "train" if index < train_end else "val" if index < val_end else "test"
    return result


def row_for(item: dict, splits: dict[str, str]) -> dict:
    props = item["properties"]
    return {
        "scene_id": item["id"],
        "start_datetime": props.get("start_datetime", props.get("datetime")),
        "end_datetime": props.get("end_datetime", props.get("datetime")),
        "platform": props.get("platform"),
        "orbit_state": props.get("sat:orbit_state"),
        "relative_orbit": props.get("sat:relative_orbit"),
        "absolute_orbit": props.get("sat:absolute_orbit"),
        "mode": props.get("sar:instrument_mode"),
        "polarizations": "+".join(props.get("sar:polarizations", [])),
        "product_type": props.get("product:type"),
        "split": splits[item["id"]],
    }


def write_catalog(path: Path, items: list[dict]) -> None:
    splits = split_by_time(items)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(row_for(item, splits) for item in items)


def write_dataset_config(days: int, items: list[dict], core: list[dict], orbit_key: tuple) -> None:
    intervals = [
        (scene_time(b) - scene_time(a)).total_seconds() / 86400
        for a, b in zip(items, items[1:])
    ]
    config = {
        "generated_at": iso(utc_now()),
        "aoi_name": AOI_NAME,
        "bbox_crs84": BBOX,
        "output_crs": "OGC:CRS84",
        "width": WIDTH,
        "height": HEIGHT,
        "nominal_pixel_spacing_m": 10,
        "bands": ["VV_dB", "VH_dB", "dataMask"],
        "dtype": "FLOAT32",
        "collection": "sentinel-1-grd",
        "product_type": "IW_GRDH_1S",
        "acquisition_mode": "IW",
        "polarization": "DV (VV+VH)",
        "backscatter": "GAMMA0_ELLIPSOID",
        "orthorectified": True,
        "canonical_speckle_filter": "NONE",
        "search_days": days,
        "training_pool_scenes": len(items),
        "temporal_core_scenes": len(core),
        "temporal_core_orbit_state": orbit_key[0],
        "temporal_core_relative_orbit": orbit_key[1],
        "training_pool_median_interval_days": round(sorted(intervals)[len(intervals) // 2], 2)
        if intervals
        else None,
    }
    (DATASET_DIR / "dataset_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_catalog(days: int) -> list[dict]:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    items = search_scenes(days)
    if not items:
        raise RuntimeError("인천 AOI에서 호환되는 IW VV/VH GRD 장면을 찾지 못했습니다.")
    core, orbit_key = temporal_core(items)
    write_catalog(CATALOG_DIR / "training_pool.csv", items)
    write_catalog(CATALOG_DIR / "temporal_core.csv", core)
    for item in items:
        (METADATA_DIR / f"{item['id']}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    write_dataset_config(days, items, core, orbit_key)
    print(f"학습 풀: {len(items)}장 -> {CATALOG_DIR / 'training_pool.csv'}")
    print(
        f"시계열 코어: {len(core)}장 "
        f"({orbit_key[0]}, relative orbit {orbit_key[1]}) -> "
        f"{CATALOG_DIR / 'temporal_core.csv'}"
    )
    return items


def get_access_token() -> str:
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError(
            "현재 터미널에 CDSE_CLIENT_ID와 CDSE_CLIENT_SECRET을 export한 뒤 다시 실행하세요. "
            "값을 코드나 채팅에 저장하지 마세요."
        )
    response = request_json(
        TOKEN_URL,
        form={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=60,
    )
    return response["access_token"]


def expanded_scene_range(item: dict, seconds: int = 2) -> tuple[str, str]:
    props = item["properties"]
    start_raw = props.get("start_datetime", props.get("datetime"))
    end_raw = props.get("end_datetime", props.get("datetime"))
    start = datetime.fromisoformat(start_raw.replace("Z", "+00:00")) - timedelta(seconds=seconds)
    end = datetime.fromisoformat(end_raw.replace("Z", "+00:00")) + timedelta(seconds=seconds)
    return iso(start), iso(end)


def evalscript() -> str:
    return """//VERSION=3
function setup() {
  return {
    input: ["VV", "VH", "dataMask"],
    output: {bands: 3, sampleType: "FLOAT32"}
  };
}
function toDb(x) {
  return x > 0 ? 10 * Math.log(x) / Math.LN10 : -40;
}
function evaluatePixel(s) {
  return [toDb(s.VV), toDb(s.VH), s.dataMask];
}
"""


def process_payload(item: dict, use_lee: bool) -> dict:
    start, end = expanded_scene_range(item)
    props = item["properties"]
    processing: dict[str, object] = {
        "orthorectify": True,
        "backCoeff": "GAMMA0_ELLIPSOID",
    }
    if use_lee:
        processing["speckleFilter"] = {
            "type": "LEE",
            "windowSizeX": 3,
            "windowSizeY": 3,
        }
    return {
        "input": {
            "bounds": {
                "bbox": BBOX,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [
                {
                    "type": "sentinel-1-grd",
                    "dataFilter": {
                        "timeRange": {"from": start, "to": end},
                        "mosaickingOrder": "mostRecent",
                        "acquisitionMode": "IW",
                        "polarization": "DV",
                        "resolution": "HIGH",
                        "orbitDirection": str(props.get("sat:orbit_state", "")).upper(),
                    },
                    "processing": processing,
                }
            ],
        },
        "output": {
            "width": WIDTH,
            "height": HEIGHT,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript(),
    }


def append_log(**event: object) -> None:
    event = {"logged_at": iso(utc_now()), **event}
    with LOG_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(event, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(item: dict, token: str, use_lee: bool) -> Path:
    output_dir = LEE_DIR if use_lee else RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "lee3_gamma0_db" if use_lee else "raw_gamma0_db"
    output_path = output_dir / f"{item['id']}__{AOI_NAME}__{suffix}.tif"
    if output_path.exists() and output_path.stat().st_size > 10_000:
        print(f"건너뜀(기존 파일): {output_path.name}")
        return output_path

    request = urllib.request.Request(
        PROCESS_URL,
        data=json.dumps(process_payload(item, use_lee)).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "incheon-dark-vessel-hackathon/1.0",
        },
    )
    temporary_path = output_path.with_suffix(".part")
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(
                request, timeout=300, context=SSL_CONTEXT
            ) as response:
                temporary_path.write_bytes(response.read())
            temporary_path.replace(output_path)
            append_log(
                status="downloaded",
                scene_id=item["id"],
                file=str(output_path.relative_to(ROOT)),
                bytes=output_path.stat().st_size,
                sha256=sha256(output_path),
                lee3=use_lee,
            )
            return output_path
        except (urllib.error.URLError, TimeoutError) as exc:
            if temporary_path.exists():
                temporary_path.unlink()
            if attempt == 3:
                append_log(status="failed", scene_id=item["id"], error=str(exc), lee3=use_lee)
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def read_selected_items(dataset_set: str, max_scenes: int) -> list[dict]:
    catalog_name = "training_pool.csv" if dataset_set == "training" else "temporal_core.csv"
    catalog_path = CATALOG_DIR / catalog_name
    if not catalog_path.exists():
        raise FileNotFoundError("먼저 catalog 명령을 실행하세요.")
    with catalog_path.open(encoding="utf-8") as fp:
        ids = [row["scene_id"] for row in csv.DictReader(fp)]
    # Most recent first is convenient during a hackathon; max_scenes=0 means all.
    ids.reverse()
    if max_scenes > 0:
        ids = ids[:max_scenes]
    return [json.loads((METADATA_DIR / f"{scene_id}.json").read_text(encoding="utf-8")) for scene_id in ids]


def download_dataset(dataset_set: str, max_scenes: int, use_lee: bool) -> None:
    selected = read_selected_items(dataset_set, max_scenes)
    token = get_access_token()
    print(f"{dataset_set} 세트 {len(selected)}장 다운로드 시작")
    for index, item in enumerate(selected, 1):
        started = time.monotonic()
        path = download_one(item, token, use_lee)
        elapsed = time.monotonic() - started
        print(f"[{index}/{len(selected)}] {path.name} ({elapsed:.1f}초)")
    # Every download run ends by rebuilding map PNGs and the CSV manifest.
    # Import here avoids requiring tifffile for the public catalogue-only step.
    from prepare_dataset_assets import build_area

    print("GeoTIFF 다운로드 완료. PNG와 dataset_manifest.csv를 갱신합니다.")
    build_area("incheon")


def show_status() -> None:
    raw = list(RAW_DIR.glob("*.tif")) if RAW_DIR.exists() else []
    lee = list(LEE_DIR.glob("*.tif")) if LEE_DIR.exists() else []
    print(f"원본 GeoTIFF: {len(raw)}장")
    print(f"Lee 3x3 GeoTIFF: {len(lee)}장")
    print(f"데이터셋 위치: {DATASET_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog_parser = subparsers.add_parser("catalog", help="장면 목록 생성(인증 불필요)")
    catalog_parser.add_argument("--days", type=int, default=365)

    download_parser = subparsers.add_parser("download", help="목록의 GeoTIFF 다운로드")
    download_parser.add_argument("--set", choices=["training", "temporal"], default="training")
    download_parser.add_argument(
        "--max-scenes", type=int, default=12, help="0이면 선택 세트 전체 다운로드"
    )
    download_parser.add_argument("--lee", action="store_true", help="별도 Lee 3x3 파생본 생성")

    subparsers.add_parser("status", help="현재 다운로드 장수 확인")
    args = parser.parse_args()
    try:
        if args.command == "catalog":
            build_catalog(args.days)
        elif args.command == "download":
            download_dataset(args.set, args.max_scenes, args.lee)
        else:
            show_status()
    except Exception as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
