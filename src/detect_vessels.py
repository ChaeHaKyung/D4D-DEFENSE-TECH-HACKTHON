#!/usr/bin/env python3
"""SAR에서 밝은 선박 후보를 찾고 같은 촬영시각의 AIS와 공간 매칭한다.

이 파일은 학습된 AI가 아니라 설명 가능한 CA-CFAR 형태의 기준선이다. 현재 입력
PNG는 VV dB의 -30~0 dB를 밝기 0~255로 선형 변환했으므로 국소 대비는 유지된다.
Natural Earth 육지 마스크, 해안거리, 항만/외해 임계값, 과거 장면 반복 위치를 함께
사용한다. ``unmatched-candidate``는 Dark Vessel 확정이 아니라 검토 후보라는 점이
가장 중요하다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# 프로젝트 안에서 사용하는 모든 입력·출력 경로.
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SAR_PNG = OUTPUTS / "busan_vv_visual.png"
STAC_JSON = OUTPUTS / "latest_stac_item.json"
AIS_CSV = OUTPUTS / "ais_interpolated_at_sar.csv"
CANDIDATE_CSV = OUTPUTS / "sar_ship_candidates.csv"
OVERLAY_PNG = OUTPUTS / "sar_ship_candidates_overlay.png"
LAND_MASK_PNG = OUTPUTS / "land_mask.png"
COASTLINE_GEOJSON = ROOT / "data" / "ne_10m_land.geojson"
HISTORY_DIR = OUTPUTS / "history"
# SAR 이미지 전체가 차지하는 지리 범위 [서, 남, 동, 북].
BBOX = [128.95, 34.95, 129.25, 35.20]
# 항만은 인공 고반사체가 많아 외해보다 엄격한 임계값을 적용하는 분석 구역이다.
# 현재는 근사 사각형이며 최종판에서는 공식 항만경계 GeoJSON으로 교체해야 한다.
HARBOR_BBOX = [128.98, 35.00, 129.14, 35.13]


def box_sum(array: np.ndarray, radius: int) -> np.ndarray:
    """적분영상을 이용해 모든 픽셀 주변 정사각형 합계를 빠르게 계산한다.

    일반 반복문으로 각 픽셀의 주변 창을 더하면 매우 느리다. 누적합 적분영상에서는
    사각형 꼭짓점 네 개의 값만 더하고 빼서 같은 결과를 얻는다.
    """
    h, w = array.shape
    # 위·왼쪽에 0 한 줄을 붙이면 경계 사각형도 같은 공식으로 계산할 수 있다.
    integral = np.pad(array, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    # 각 중심 픽셀의 창 시작/끝 좌표를 영상 경계 안으로 자른다.
    y = np.arange(h)
    x = np.arange(w)
    y0 = np.maximum(0, y - radius)
    y1 = np.minimum(h, y + radius + 1)
    x0 = np.maximum(0, x - radius)
    x1 = np.minimum(w, x + radius + 1)
    # 오른쪽아래 - 왼쪽아래 - 오른쪽위 + 왼쪽위가 사각형 내부 합이다.
    return (
        integral[y1[:, None], x1[None, :]]
        - integral[y0[:, None], x1[None, :]]
        - integral[y1[:, None], x0[None, :]]
        + integral[y0[:, None], x0[None, :]]
    )


def box_stats(array: np.ndarray, valid: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """유효 픽셀만 사용해 국소 평균과 표준편차 영상을 만든다."""
    # counts가 0이면 나눗셈 오류가 나므로 최소 1로 제한한다.
    counts = np.maximum(box_sum(valid.astype(np.float64), radius), 1.0)
    sums = box_sum(np.where(valid, array, 0.0), radius)
    squares = box_sum(np.where(valid, array * array, 0.0), radius)
    mean = sums / counts
    # Var(X)=E[X²]-E[X]². 부동소수점 오차로 음수가 되지 않게 0으로 자른다.
    variance = np.maximum(squares / counts - mean * mean, 0.0)
    return mean, np.sqrt(variance)


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """참 픽셀 주변을 넓혀 끊어진 작은 산란점을 하나의 후보로 연결한다."""
    result = np.zeros_like(mask)
    h, w = mask.shape
    # 마스크를 상하좌우·대각선으로 이동한 결과를 OR 연산한다.
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ys = slice(max(0, dy), min(h, h + dy))
            xs = slice(max(0, dx), min(w, w + dx))
            src_y = slice(max(0, -dy), min(h, h - dy))
            src_x = slice(max(0, -dx), min(w, w - dx))
            result[ys, xs] |= mask[src_y, src_x]
    return result


def components(mask: np.ndarray, intensity: np.ndarray) -> list[dict]:
    """8방향으로 붙은 참 픽셀들을 한 물체 후보로 묶는다."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    found: list[dict] = []
    # np.argwhere는 참인 픽셀 좌표만 반환하므로 희소한 후보 마스크에서 효율적이다.
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        pixels: list[tuple[int, int]] = []
        # BFS로 현재 픽셀과 연결된 모든 이웃을 찾는다.
        while queue:
            cy, cx = queue.popleft()
            pixels.append((cy, cx))
            for ny in range(max(0, cy - 1), min(h, cy + 2)):
                for nx in range(max(0, cx - 1), min(w, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        ys = np.array([p[0] for p in pixels])
        xs = np.array([p[1] for p in pixels])
        # 밝은 산란점이 중심에 더 큰 영향을 주도록 밝기 가중 중심을 계산한다.
        weights = np.maximum(intensity[ys, xs], 1.0)
        found.append(
            {
                "area": len(pixels),
                "x0": int(xs.min()),
                "y0": int(ys.min()),
                "x1": int(xs.max()),
                "y1": int(ys.max()),
                "cx": float(np.average(xs, weights=weights)),
                "cy": float(np.average(ys, weights=weights)),
                "peak": float(intensity[ys, xs].max()),
            }
        )
    return found


def load_ais_snapshot() -> list[dict]:
    """시간정렬 단계가 만든 SAR 촬영시각 AIS 위치들을 읽는다."""
    if not AIS_CSV.exists():
        return []
    with AIS_CSV.open(encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """위·경도 두 점 사이 지표면 거리를 미터로 계산한다."""
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def pixel_to_lonlat(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    """이미지 픽셀 중심을 BBOX 안의 경도·위도로 선형 변환한다."""
    west, south, east, north = BBOX
    lon = west + (x + 0.5) / width * (east - west)
    # 이미지 y는 아래로 증가하지만 위도는 북쪽으로 증가하므로 north에서 뺀다.
    lat = north - (y + 0.5) / height * (north - south)
    return lon, lat


def lonlat_to_pixel(lon: float, lat: float, width: int, height: int) -> tuple[float, float]:
    """경도·위도를 이미지 픽셀 좌표로 바꾸는 위 함수의 역변환."""
    west, south, east, north = BBOX
    x = (lon - west) / (east - west) * width - 0.5
    y = (north - lat) / (north - south) * height - 0.5
    return x, y


def geographic_land_mask(width: int, height: int) -> np.ndarray:
    """Natural Earth 육지 폴리곤을 SAR와 같은 크기의 불리언 마스크로 래스터화한다."""
    if not COASTLINE_GEOJSON.exists():
        raise FileNotFoundError(f"해안선 파일이 없습니다: {COASTLINE_GEOJSON}")
    data = json.loads(COASTLINE_GEOJSON.read_text(encoding="utf-8"))
    # 모드 '1'은 픽셀당 육지(1)/바다(0)만 저장하는 1-bit 이미지다.
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    west, south, east, north = BBOX

    def ring_pixels(ring: list[list[float]]) -> list[tuple[float, float]]:
        """GeoJSON 폴리곤의 경위도 점들을 현재 영상 픽셀점으로 바꾼다."""
        return [lonlat_to_pixel(float(lon), float(lat), width, height) for lon, lat, *_ in ring]

    for feature in data.get("features", []):
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates", [])
        polygons = [coords] if geometry.get("type") == "Polygon" else coords if geometry.get("type") == "MultiPolygon" else []
        for polygon in polygons:
            if not polygon or not polygon[0]:
                continue
            outer = polygon[0]
            lons = [point[0] for point in outer]
            lats = [point[1] for point in outer]
            # 부산 BBOX와 전혀 겹치지 않는 전 세계 폴리곤은 그리지 않는다.
            if max(lons) < west or min(lons) > east or max(lats) < south or min(lats) > north:
                continue
            # 바깥 고리는 육지로 채우고, 호수 같은 내부 구멍은 다시 바다로 지운다.
            draw.polygon(ring_pixels(outer), fill=1)
            for hole in polygon[1:]:
                draw.polygon(ring_pixels(hole), fill=0)
    mask = np.asarray(canvas, dtype=bool)
    Image.fromarray((mask * 255).astype(np.uint8)).save(LAND_MASK_PNG)
    return mask


def candidate_coast_distance_m(comp: dict, boundary_yx: np.ndarray, width: int, height: int) -> float:
    """후보 중심에서 가장 가까운 육지 경계 픽셀까지 근사거리를 계산한다."""
    if not len(boundary_yx):
        return math.inf
    west, south, east, north = BBOX
    center_lat = (south + north) / 2
    # 경도 1도 길이는 위도에 따라 cos(latitude)만큼 짧아진다.
    meters_x = 111_320.0 * math.cos(math.radians(center_lat)) * (east - west) / width
    meters_y = 110_574.0 * (north - south) / height
    dy = (boundary_yx[:, 0] - comp["cy"]) * meters_y
    dx = (boundary_yx[:, 1] - comp["cx"]) * meters_x
    return float(np.sqrt(dx * dx + dy * dy).min())


def previous_detections(scene_id: str) -> list[list[dict]]:
    """현재 장면을 제외한 과거 장면별 후보 CSV들을 읽는다."""
    groups = []
    if not HISTORY_DIR.exists():
        return groups
    for path in HISTORY_DIR.glob("*.csv"):
        # 같은 장면을 다시 분석한 것은 다중 날짜 반복관측으로 세면 안 된다.
        if path.stem == scene_id:
            continue
        with path.open(encoding="utf-8") as fp:
            groups.append(list(csv.DictReader(fp)))
    return groups


def detect(k: float = 4.2, match_radius_m: float = 900.0) -> list[dict]:
    """전처리, 구역별 CFAR, AIS 매칭, 위험점수를 수행해 후보 행들을 반환한다."""
    # PNG를 RGBA로 통일하면 RGB 밝기와 dataMask 역할의 alpha를 함께 읽을 수 있다.
    image = Image.open(SAR_PNG).convert("RGBA")
    rgba = np.asarray(image)
    # 현재 PNG는 회색조지만 RGB 평균을 쓰면 형식 변화에도 안전하다.
    gray = rgba[..., :3].mean(axis=2).astype(np.float64)
    valid = rgba[..., 3] > 0

    # 지리 육지 마스크와 SAR 밝기/질감 조건을 함께 사용해 분석 가능한 바다를 정한다.
    land = geographic_land_mask(image.width, image.height)
    broad_mean, broad_std = box_stats(gray, valid, radius=25)
    # 너무 밝거나 거친 수면은 육지 잔여물·항만 구조물 가능성이 커 1차 제외한다.
    water = valid & ~land & (broad_mean < 125.0) & (broad_std < 28.0)

    west, south, east, north = BBOX
    lon_axis = west + (np.arange(image.width) + 0.5) / image.width * (east - west)
    lat_axis = north - (np.arange(image.height) + 0.5) / image.height * (north - south)
    # 각 열의 경도와 각 행의 위도를 만들어 항만 구역 불리언 배열을 구성한다.
    harbor = (
        (lon_axis[None, :] >= HARBOR_BBOX[0])
        & (lon_axis[None, :] <= HARBOR_BBOX[2])
        & (lat_axis[:, None] >= HARBOR_BBOX[1])
        & (lat_axis[:, None] <= HARBOR_BBOX[3])
    )
    meters_per_pixel = (
        111_320.0 * math.cos(math.radians((south + north) / 2)) * (east - west) / image.width
        + 110_574.0 * (north - south) / image.height
    ) / 2
    # 1km를 현재 영상 픽셀 수로 바꾸고, 그 안에 육지가 있으면 near_coast다.
    coast_radius_px = max(1, round(1_000.0 / meters_per_pixel))
    near_coast = water & (box_sum(land.astype(np.float64), coast_radius_px) > 0)

    # CFAR 배경 창은 반경 12, 보호 창은 반경 2로 두어 표적 자체가 배경을 올리는 것을 줄인다.
    outer_mean, outer_std = box_stats(gray, water, radius=12)
    guard_mean, _ = box_stats(gray, water, radius=2)
    # 기본 임계값은 평균+k·표준편차. 해안과 항만에서는 k를 높여 오탐을 억제한다.
    zone_k = np.full(gray.shape, k)
    zone_k[near_coast] += 0.7
    zone_k[harbor & water] += 0.7
    threshold = outer_mean + zone_k * np.maximum(outer_std, 4.0)
    # 세 조건을 모두 만족해야 후보: CFAR 초과, 절대 밝기, 보호창 대비 충분한 밝기.
    bright = water & (gray > threshold) & (gray > 120.0) & (gray > guard_mean + 20.0)
    bright = dilate(bright, radius=1)

    raw = components(bright, gray)
    kept = []
    # 연결요소의 면적과 가로·세로 크기로 큰 항만시설·긴 줄무늬를 제거한다.
    for comp in raw:
        bw = comp["x1"] - comp["x0"] + 1
        bh = comp["y1"] - comp["y0"] + 1
        # With this low-resolution preview, very small components are retained;
        # large port structures and streaks are rejected.
        if not (3 <= comp["area"] <= 180 and bw <= 28 and bh <= 28):
            continue
        # 영상 가장자리 후보는 잘린 장면 또는 처리 경계일 수 있어 제외한다.
        if comp["x0"] < 3 or comp["y0"] < 3 or comp["x1"] >= image.width - 3 or comp["y1"] >= image.height - 3:
            continue
        kept.append(comp)

    item = json.loads(STAC_JSON.read_text(encoding="utf-8"))
    raw_time = item["properties"].get("start_datetime", item["properties"]["datetime"])
    # 장면 ID는 역사 파일명과 다중 날짜 중복 방지 키로 사용한다.
    scene_id = item["id"].replace("/", "_")
    ais_positions = load_ais_snapshot()
    history = previous_detections(scene_id)
    # 육지이면서 바로 옆에 바다가 있는 픽셀만 해안선 경계로 간주한다.
    boundary = land & dilate(~land, radius=1)
    boundary_yx = np.argwhere(boundary)

    rows: list[dict] = []
    for index, comp in enumerate(kept, 1):
        lon, lat = pixel_to_lonlat(comp["cx"], comp["cy"], image.width, image.height)
        coast_distance = candidate_coast_distance_m(comp, boundary_yx, image.width, image.height)
        in_harbor = HARBOR_BBOX[0] <= lon <= HARBOR_BBOX[2] and HARBOR_BBOX[1] <= lat <= HARBOR_BBOX[3]
        zone = "harbor" if in_harbor else "nearshore" if coast_distance <= 1_000 else "offshore"
        # 후보마다 모든 보간 AIS와 거리를 비교해 가장 가까운 선박을 찾는다.
        nearest = None
        nearest_distance = math.inf
        for position in ais_positions:
            distance = haversine_m(lat, lon, float(position["lat"]), float(position["lon"]))
            if distance < nearest_distance:
                nearest, nearest_distance = position, distance
        # 현재 저해상도·위치편이를 고려해 기본 900m 이내를 매칭으로 둔다.
        matched = nearest is not None and nearest_distance <= match_radius_m
        repeat_count = 0
        # 같은 과거 장면 안에서는 몇 번 겹쳐도 한 번만 반복 관측으로 센다.
        for old_scene in history:
            if any(haversine_m(lat, lon, float(old["lat"]), float(old["lon"])) <= 150 for old in old_scene):
                repeat_count += 1
        # 과거 서로 다른 두 장면에서도 반복되면 현재 포함 3회이므로 고정물 후보가 된다.
        persistent = repeat_count >= 2 and not matched
        status = "matched" if matched else "persistent-static" if persistent else "unmatched-candidate"
        # 점수는 설명 가능한 규칙 기반 우선순위이며 확률값이 아니다.
        risk = 10 if matched else 20 if persistent else 75
        if zone == "nearshore":
            risk -= 20
        elif zone == "harbor":
            risk -= 30
        # 가장 가까운 AIS 자체가 저신뢰이면 미매칭 불확실성이 커져 5점을 더한다.
        if nearest and nearest.get("confidence") == "low":
            risk += 5
        risk = max(0, min(100, risk))
        rows.append(
            {
                "candidate_id": f"SAR-{index:03d}",
                "lat": f"{lat:.7f}",
                "lon": f"{lon:.7f}",
                "pixel_x": f"{comp['cx']:.2f}",
                "pixel_y": f"{comp['cy']:.2f}",
                "bbox_x0": comp["x0"],
                "bbox_y0": comp["y0"],
                "bbox_x1": comp["x1"],
                "bbox_y1": comp["y1"],
                "area_pixels": comp["area"],
                "peak_gray": f"{comp['peak']:.1f}",
                "zone": zone,
                "coast_distance_m": f"{coast_distance:.1f}",
                "status": status,
                "nearest_mmsi": nearest["mmsi"] if matched else "",
                "distance_m": f"{nearest_distance:.1f}" if nearest else "",
                "ais_confidence": nearest.get("confidence", "") if matched else "",
                "repeat_scene_count": repeat_count,
                "risk_score": risk,
                "scene_id": scene_id,
                "sar_time": raw_time,
            }
        )

    fieldnames = list(rows[0]) if rows else [
        "candidate_id", "lat", "lon", "pixel_x", "pixel_y", "bbox_x0", "bbox_y0",
        "bbox_x1", "bbox_y1", "area_pixels", "peak_gray", "zone", "coast_distance_m",
        "status", "nearest_mmsi", "distance_m", "ais_confidence", "repeat_scene_count",
        "risk_score", "scene_id", "sar_time",
    ]
    # 지도·평가·사람 검토가 사용할 표준 후보 CSV를 저장한다.
    with CANDIDATE_CSV.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 원본 PNG를 훼손하지 않고 별도 검토용 바운딩박스 이미지를 만든다.
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for row, comp in zip(rows, kept):
        # 초록은 AIS 매칭, 주황은 미매칭/검토 후보다.
        color = (34, 197, 94, 255) if row["status"] == "matched" else (249, 115, 22, 255)
        draw.rectangle((comp["x0"] - 2, comp["y0"] - 2, comp["x1"] + 2, comp["y1"] + 2), outline=color, width=2)
    overlay.save(OVERLAY_PNG)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    # 장면별 복사본은 다음 날짜 분석에서 고정 위치 구조물을 찾는 근거가 된다.
    shutil.copyfile(CANDIDATE_CSV, HISTORY_DIR / f"{scene_id}.csv")

    matched_count = sum(row["status"] == "matched" for row in rows)
    print(f"SAR 후보 {len(rows)}개: AIS 매칭 {matched_count}, 미매칭 후보 {len(rows) - matched_count}")
    print(f"후보 CSV: {CANDIDATE_CSV}")
    print(f"검토용 오버레이: {OVERLAY_PNG}")
    return rows


def main() -> int:
    """CFAR 민감도 k와 AIS 매칭반경을 실험할 수 있는 CLI 진입점."""
    parser = argparse.ArgumentParser(description="CA-CFAR-style SAR ship candidate baseline")
    parser.add_argument("--k", type=float, default=4.2, help="local standard-deviation multiplier")
    parser.add_argument("--match-radius-m", type=float, default=900.0)
    args = parser.parse_args()
    detect(args.k, args.match_radius_m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
