#!/usr/bin/env python3
"""Create human-readable PNGs and a single manifest CSV for SAR datasets.

GeoTIFF remains the analysis source. PNG files are derived quicklooks for the
web map and presentations; they must not replace the original floating-point
VV/VH values. The manifest keeps the bounds needed to place each PNG on a map.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "outputs" / "datasets"

AREA_CONFIG = {
    "busan": {
        "bbox": [128.95, 34.95, 129.25, 35.20],
        "expected_bands": ["VV_dB", "dataMask"],
    },
    "incheon": {
        "bbox": [126.40, 37.32, 126.62, 37.48],
        "expected_bands": ["VV_dB", "VH_dB", "dataMask"],
    },
}

MANIFEST_FIELDS = [
    "area",
    "scene_id",
    "acquisition_time_utc",
    "platform",
    "orbit_state",
    "relative_orbit",
    "width",
    "height",
    "bands",
    "west",
    "south",
    "east",
    "north",
    "valid_fraction",
    "vv_min_db",
    "vv_max_db",
    "vv_mean_db",
    "vh_min_db",
    "vh_max_db",
    "vh_mean_db",
    "status",
    "geotiff_path",
    "vv_png_path",
    "rgb_png_path",
    "notes",
]


def relative(path: Path | None) -> str:
    return "" if path is None else str(path.relative_to(ROOT))


def normalize_db(array: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map a fixed dB interval to uint8 so dates remain visually comparable."""
    clipped = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.round(clipped * 255).astype(np.uint8)


def metadata_for(area: str, scene_id: str) -> dict:
    if area == "incheon":
        path = DATASETS / area / "metadata" / f"{scene_id}.json"
    else:
        path = DATASETS / area / "metadata" / "latest_stac_item.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def scene_id_from_name(path: Path, area: str) -> str:
    if "__" in path.stem:
        return path.stem.split("__", 1)[0]
    metadata = metadata_for(area, "")
    return metadata.get("id", path.stem)


def band_stats(array: np.ndarray, mask: np.ndarray) -> tuple[str, str, str]:
    values = array[mask]
    if values.size == 0:
        return "", "", ""
    return tuple(f"{value:.4f}" for value in (values.min(), values.max(), values.mean()))


def process_tiff(area: str, tif_path: Path) -> dict:
    config = AREA_CONFIG[area]
    array = tifffile.imread(tif_path)
    if array.ndim == 2:
        array = array[..., np.newaxis]
    if array.ndim != 3 or array.shape[-1] not in (2, 3, 4):
        raise ValueError(f"지원하지 않는 TIFF 배열 형태: {array.shape}")

    height, width, band_count = array.shape
    vv = array[..., 0].astype(np.float32)
    vh = array[..., 1].astype(np.float32) if band_count >= 3 else None
    mask_band = array[..., 2] if band_count >= 3 else array[..., 1]
    valid = np.isfinite(vv) & (mask_band > 0.5)
    if vh is not None:
        valid &= np.isfinite(vh)
    valid_fraction = float(valid.mean())

    png_dir = DATASETS / area / "sar" / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    vv_png = png_dir / f"{tif_path.stem}__vv.png"
    rgb_png = png_dir / f"{tif_path.stem}__vv_vh_rgb.png" if vh is not None else None

    vv_u8 = normalize_db(vv, -30.0, 5.0)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    Image.fromarray(np.dstack([vv_u8, vv_u8, vv_u8, alpha]), "RGBA").save(vv_png)

    if vh is not None:
        vh_u8 = normalize_db(vh, -35.0, -5.0)
        ratio_u8 = normalize_db(vv - vh, 0.0, 20.0)
        Image.fromarray(np.dstack([vv_u8, vh_u8, ratio_u8, alpha]), "RGBA").save(rgb_png)

    scene_id = scene_id_from_name(tif_path, area)
    item = metadata_for(area, scene_id)
    props = item.get("properties", {})
    west, south, east, north = config["bbox"]
    vv_min, vv_max, vv_mean = band_stats(vv, valid)
    vh_min, vh_max, vh_mean = band_stats(vh, valid) if vh is not None else ("", "", "")

    status = "valid" if valid_fraction > 0.01 else "invalid_no_data"
    notes = ""
    if status != "valid":
        notes = "유효 데이터가 거의 없음. 학습과 탐지에서 제외할 것."
    elif vh is None:
        notes = "VV 단일 분석본. VV+VH 인천 규격과 직접 비교하지 말 것."

    return {
        "area": area,
        "scene_id": scene_id,
        "acquisition_time_utc": props.get("start_datetime", props.get("datetime", "")),
        "platform": props.get("platform", ""),
        "orbit_state": props.get("sat:orbit_state", ""),
        "relative_orbit": props.get("sat:relative_orbit", ""),
        "width": width,
        "height": height,
        "bands": "+".join(config["expected_bands"]),
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "valid_fraction": f"{valid_fraction:.6f}",
        "vv_min_db": vv_min,
        "vv_max_db": vv_max,
        "vv_mean_db": vv_mean,
        "vh_min_db": vh_min,
        "vh_max_db": vh_max,
        "vh_mean_db": vh_mean,
        "status": status,
        "geotiff_path": relative(tif_path),
        "vv_png_path": relative(vv_png),
        "rgb_png_path": relative(rgb_png),
        "notes": notes,
    }


def build_area(area: str) -> list[dict]:
    tiff_dir = DATASETS / area / "sar" / "geotiff_raw"
    manifest_path = DATASETS / area / "dataset_manifest.csv"
    tiffs = sorted([*tiff_dir.glob("*.tif"), *tiff_dir.glob("*.tiff")])
    rows: list[dict] = []
    for index, tif_path in enumerate(tiffs, 1):
        try:
            row = process_tiff(area, tif_path)
            rows.append(row)
            print(f"[{area} {index}/{len(tiffs)}] {row['status']}: {tif_path.name}")
        except Exception as exc:
            print(f"[{area} {index}/{len(tiffs)}] 변환 실패: {tif_path.name}: {exc}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{area} manifest {len(rows)}건: {manifest_path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", choices=["busan", "incheon", "all"], default="all")
    args = parser.parse_args()
    areas = AREA_CONFIG if args.area == "all" else [args.area]
    all_rows: list[dict] = []
    for area in areas:
        all_rows.extend(build_area(area))
    if args.area == "all":
        combined_path = DATASETS / "dataset_index.csv"
        with combined_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"전체 데이터 인덱스 {len(all_rows)}건: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
