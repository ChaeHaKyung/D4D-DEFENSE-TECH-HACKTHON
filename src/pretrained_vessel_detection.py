#!/usr/bin/env python3
"""인천항 SAR 영상 기반 선박 탐지 및 AIS Re-ID (인천대교 다구간 궤적 및 픽셀 마스크 적용)."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, find_objects
from scipy.optimize import linear_sum_assignment

from risk_scoring import score_candidate

# ------------------------------------------------------------------------------
# 1. 경로 및 지리 좌표 정의
# ------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

# 영상 파일 탐색
SAR_PNG = OUTPUTS / "incheon_vv_visual.png"
if not SAR_PNG.exists():
    SAR_PNG = OUTPUTS / "busan_vv_visual.png"

STAC_JSON = OUTPUTS / "latest_stac_item.json"
AIS_CSV = OUTPUTS / "ais_interpolated_at_sar.csv"
CANDIDATE_CSV = OUTPUTS / "sar_ship_candidates.csv"
OVERLAY_PNG = OUTPUTS / "sar_ship_candidates_overlay.png"

# 인천 외항 BBOX: [서경, 남위, 동경, 북위]
BBOX = [126.40, 37.32, 126.62, 37.48]
HARBOR_BBOX = [126.56, 37.40, 126.63, 37.48]

# 인천대교 실제 곡선 궤적 픽셀 좌표 (영종IC 부근 -> 주탑 -> 송도 방향)
BRIDGE_PIXELS = [
    (530, 80),
    (590, 180),
    (670, 310),
    (740, 410),
    (810, 490),
    (900, 520),
    (1024, 530),
]


@dataclass
class SARDetection:
    det_id: int
    lat: float
    lon: float
    pixel_x: float
    pixel_y: float
    bbox: list
    length_m: float
    width_m: float
    heading_deg: float
    confidence: float


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def pixel_to_lonlat(x: float, y: float, width: int, height: int) -> tuple[float, float]:
    west, south, east, north = BBOX
    lon = west + (x + 0.5) / width * (east - west)
    lat = north - (y + 0.5) / height * (north - south)
    return lon, lat


def dist_to_polyline_px(px: float, py: float, points: list) -> float:
    """픽셀 좌표가 인천대교 궤적으로부터 떨어진 최단 거리(px)"""
    min_dist = float("inf")
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        l2 = dx * dx + dy * dy
        if l2 == 0:
            d = math.hypot(px - x1, py - y1)
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            d = math.hypot(px - proj_x, py - proj_y)
        if d < min_dist:
            min_dist = d
    return min_dist


# ------------------------------------------------------------------------------
# 2. SAR 영상 선박 탐지 및 정밀 오탐 제거
# ------------------------------------------------------------------------------
def detect_sar_vessels(image_path: Path) -> List[SARDetection]:
    """군집 구조물 분리 및 진짜 단독 선박을 보존하는 SAR 탐지기"""
    if not image_path.exists():
        print(f"[오류] 영상 파일이 없습니다: {image_path}")
        return []

    img = Image.open(image_path).convert("RGB")
    gray = np.array(img.convert("L"))
    w_px, h_px = img.width, img.height

    # 상위 99.6% 밝기 기준 고반사체 추출
    threshold = min(max(100, int(gray.max() * 0.7)), np.percentile(gray, 99.6))
    bright_mask = gray > threshold

    labeled, num_features = label(bright_mask)
    slices = find_objects(labeled)

    raw_candidates = []
    # 1차 파싱: 픽셀 영역 및 기본 제원 계산
    for idx, slc in enumerate(slices, 1):
        if slc is None:
            continue
        y_slice, x_slice = slc
        area = int(np.sum(labeled[slc] > 0))
        bw = x_slice.stop - x_slice.start
        bh = y_slice.stop - y_slice.start
        cx = (x_slice.start + x_slice.stop) / 2.0
        cy = (y_slice.start + y_slice.stop) / 2.0

        major = max(bw, bh)
        minor = max(min(bw, bh), 1.0)
        aspect_ratio = major / minor

        lon, lat = pixel_to_lonlat(cx, cy, w_px, h_px)

        raw_candidates.append({
            "idx": idx,
            "cx": cx,
            "cy": cy,
            "bw": bw,
            "bh": bh,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "lon": lon,
            "lat": lat,
            "slice": slc,
        })

    detections = []
    det_id = 1
    filtered_cnt = 0

    # 2차 필터링: 육지 제거 + (교각/충돌방지공 군집 vs 단독 선박 분리)
    for c in raw_candidates:
        cx, cy = c["cx"], c["cy"]
        area = c["area"]
        ar = c["aspect_ratio"]
        lat, lon = c["lat"], c["lon"]

        # --- A. 기본 육지/해안선 컷오프 ---
        # --- A. 기본 육지/해안선 컷오프 ---
        if cy < 70:  # 최상단 북항
            filtered_cnt += 1; continue
        if cy < (-1.05 * cx + 640):  # 영종도 남동측 갯벌 경계
            filtered_cnt += 1; continue
        if cx < 220 and cy > 360:  # 좌하단 무의도/용유도 갯벌
            filtered_cnt += 1; continue
        if cx > 820 or (cx > 700 and cy > 650):  # 송도 신항/남항 부두
            filtered_cnt += 1; continue
        if area < 5 or area > 350:  # 미세 노이즈나 거대 섬 제외
            filtered_cnt += 1; continue

        # [추가] 팔미도 섬 및 팔미도 해수욕장/사구 오탐 제거
        # 위경도 기준: 팔미도 중심 (37.3580, 126.5120) 반경 700m 이내 섬/모래톱 제외
        # 또는 픽셀 기준: cx 약 520, cy 약 780 주변 반경 35px 제외
        palmido_dist_m = haversine_m(lat, lon, 37.3580, 126.5120)
        if palmido_dist_m < 750.0:
            filtered_cnt += 1
            continue

        # --- B. 인천대교 및 충돌방지공(Dolphin Protector) 군집 판별 ---
        bridge_dist = dist_to_polyline_px(cx, cy, BRIDGE_PIXELS)
        
        # 반경 28픽셀(~280m) 이내의 이웃 반사체 개수 계산 (밀집도)
        neighbors = sum(
            1 for other in raw_candidates
            if other["idx"] != c["idx"] and math.hypot(other["cx"] - cx, other["cy"] - cy) < 28.0
        )

        # 1) 교량 중심부 바로 위(30px 이내)는 무조건 교각으로 간주
        if bridge_dist < 30.0:
            filtered_cnt += 1
            continue

        # 2) 교량 근처(30~85px 사이)이면서 주변에 2개 이상 몰려있는 군집 = 인공 충돌방지공!
        #    단, 이웃이 적거나 길쭉한 선체 형태(ar >= 1.6)는 실제 배로 통과
        if bridge_dist < 85.0:
            if neighbors >= 2 and ar < 1.6:
                filtered_cnt += 1
                continue

        # 유효 선박으로 채택
        slc = c["slice"]
        y_slice, x_slice = slc
        length_m = float(max(c["bw"], c["bh"]) * 10.0)
        width_m = float(min(c["bw"], c["bh"]) * 10.0)

        detections.append(SARDetection(
            det_id=det_id,
            lat=lat,
            lon=lon,
            pixel_x=cx,
            pixel_y=cy,
            bbox=[int(x_slice.start), int(y_slice.start), int(x_slice.stop), int(y_slice.stop)],
            length_m=length_m,
            width_m=width_m,
            heading_deg=0.0,
            confidence=float(gray[int(cy), int(cx)] / 255.0),
        ))
        det_id += 1

    print(f"영상 분석 완료: 총 {len(raw_candidates)}개 중 유효 해상 선박 {len(detections)}개 "
          f"(교량/충돌방지공/육지 {filtered_cnt}개 제거)")
    return detections


# ------------------------------------------------------------------------------
# 3. AIS Dead Reckoning & Hungarian Re-ID
# ------------------------------------------------------------------------------
def dead_reckon_ais(row: dict) -> dict:
    lat = float(row["lat"])
    lon = float(row["lon"])
    try:
        sog = float(row.get("sog", ""))
        if not math.isfinite(sog):
            sog = None
    except (TypeError, ValueError):
        sog = None
    cog = float(row.get("cog") or 0.0)
    gap_sec = float(row.get("max_gap_seconds") or 0.0)

    speed_mps = (sog or 0.0) * 0.514444
    dist_m = speed_mps * min(gap_sec, 600.0)
    delta_north = dist_m * math.cos(math.radians(cog))
    delta_east = dist_m * math.sin(math.radians(cog))

    r_earth = 6378137.0
    pred_lat = lat + (delta_north / r_earth) * (180.0 / math.pi)
    pred_lon = lon + (delta_east / (r_earth * math.cos(math.radians(lat)))) * (180.0 / math.pi)

    return {
        "mmsi": row.get("mmsi", ""),
        "lat": pred_lat,
        "lon": pred_lon,
        "length": float(row.get("length") or 60.0),
        "declared_length": float(row.get("length") or 0.0),
        "confidence": row.get("confidence", "medium"),
        "max_gap_seconds": gap_sec,
        "sog": sog,
    }


# ------------------------------------------------------------------------------
# 이상 징후 분석 함수 (선박 간 접선 STS, 군집, 표류)
# ------------------------------------------------------------------------------
def analyze_maritime_anomalies(results: list[dict]) -> None:
    """Score every detection with the published four-axis rubric and 100-point cap."""
    for index, row in enumerate(results):
        lat1, lon1 = float(row["lat"]), float(row["lon"])
        row.update(score_candidate(row))
        zone = row["zone"]
        nearby = []
        close = []
        for other_index, other in enumerate(results):
            if index == other_index:
                continue
            distance_m = haversine_m(lat1, lon1, float(other["lat"]), float(other["lon"]))
            if distance_m <= 600:
                nearby.append(other)
            if distance_m <= 180:
                close.append(other)

        behavior_points = 0
        behavior_flags = []
        if zone == "offshore" and len(nearby) >= 2:
            behavior_points = 8
            behavior_flags.append(f"외해 이상 군집 ({len(nearby) + 1}척/600m, +8)")
        if zone == "offshore" and close:
            dark_partner = row.get("status") != "matched" or any(
                other.get("status") != "matched" for other in close
            )
            sts_points = 20 if dark_partner else 12
            behavior_points = max(behavior_points, sts_points)
            partner_ids = ",".join(str(other.get("candidate_id", "?")) for other in close)
            behavior_flags.append(
                f"{'Dark 의심 선박 포함 ' if dark_partner else ''}근접 접선(STS, 180m, +{sts_points}; {partner_ids})"
            )

        scores = score_candidate(
            row,
            behavior_points=behavior_points,
            behavior_flags=behavior_flags,
        )
        row.update(scores)
        row["anomaly_type"] = " | ".join(behavior_flags) if behavior_flags else "정상"


# ------------------------------------------------------------------------------
# 기존 run_reid_and_matching 함수 교체
# ------------------------------------------------------------------------------
def run_reid_and_matching(detections: List[SARDetection], ais_records: List[dict]) -> list[dict]:
    pred_ais = [dead_reckon_ais(v) for v in ais_records if v.get("lat") and v.get("lon")]
    n_ais = len(pred_ais)
    n_sar = len(detections)

    matched_pairs = {}
    max_match_radius_m = 1800.0

    if n_ais > 0 and n_sar > 0:
        cost_matrix = np.full((n_ais, n_sar), 1e6)
        for i, a in enumerate(pred_ais):
            for j, s in enumerate(detections):
                d_m = haversine_m(a["lat"], a["lon"], s.lat, s.lon)
                if d_m <= max_match_radius_m:
                    cost_matrix[i, j] = d_m / max_match_radius_m

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1.0:
                d_m = haversine_m(pred_ais[r]["lat"], pred_ais[r]["lon"], detections[c].lat, detections[c].lon)
                matched_pairs[c] = (pred_ais[r], d_m)

    sar_time = "2026-09-30T21:31:22Z"
    if STAC_JSON.exists():
        try:
            item = json.loads(STAC_JSON.read_text(encoding="utf-8"))
            sar_time = item.get("properties", {}).get("start_datetime", sar_time)
        except Exception:
            pass

    results = []
    west, south, east, north = HARBOR_BBOX

    for idx, s in enumerate(detections):
        in_harbor = (west <= s.lon <= east) and (south <= s.lat <= north)
        zone = "harbor" if in_harbor else "offshore"

        if idx in matched_pairs:
            ais_vessel, dist_m = matched_pairs[idx]
            status = "matched"
            nearest_mmsi = ais_vessel["mmsi"]
            ais_conf = ais_vessel["confidence"]
            sog = ais_vessel.get("sog", 0.0)
            max_gap_seconds = ais_vessel.get("max_gap_seconds", 0.0)
        else:
            status = "unmatched-candidate"
            nearest_mmsi = ""
            dist_m = ""
            ais_conf = ""
            sog = 0.0
            max_gap_seconds = 0.0

        results.append({
            "candidate_id": f"AI-{s.det_id:03d}",
            "lat": f"{s.lat:.7f}",
            "lon": f"{s.lon:.7f}",
            "pixel_x": f"{s.pixel_x:.2f}",
            "pixel_y": f"{s.pixel_y:.2f}",
            "bbox_x0": s.bbox[0],
            "bbox_y0": s.bbox[1],
            "bbox_x1": s.bbox[2],
            "bbox_y1": s.bbox[3],
            "area_pixels": (s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]),
            "peak_gray": f"{s.confidence * 255.0:.1f}",
            "zone": zone,
            "coast_distance_m": "1500.0",
            "status": status,
            "nearest_mmsi": nearest_mmsi,
            "distance_m": f"{dist_m:.1f}" if dist_m != "" else "",
            "ais_confidence": ais_conf,
            "max_gap_seconds": max_gap_seconds,
            "declared_length": ais_vessel.get("declared_length", 0.0) if idx in matched_pairs else 0.0,
            "sog": sog,
            "repeat_scene_count": 0,
            "length_m": s.length_m,
            "risk_score": 0,
            "anomaly_type": "정상",  # 아래 함수에서 자동 산출
            "scene_id": "incheon_scene",
            "sar_time": sar_time,
        })

    # [핵심] 이상 패턴(STS 접선, 군집, 표류) 판정 실행
    analyze_maritime_anomalies(results)

    return results


# ------------------------------------------------------------------------------
# 4. 메인 실행 진입점
# ------------------------------------------------------------------------------
def run_ai_vessel_detection():
    print(f"-> 대상 SAR 영상: {SAR_PNG}")
    detections = detect_sar_vessels(SAR_PNG)

    ais_records = []
    if AIS_CSV.exists():
        with AIS_CSV.open(encoding="utf-8") as fp:
            ais_records = list(csv.DictReader(fp))

    rows = run_reid_and_matching(detections, ais_records)

    matched_cnt = sum(1 for r in rows if r["status"] == "matched")
    dark_cnt = sum(1 for r in rows if r["status"] == "unmatched-candidate")

    if rows:
        fieldnames = list(rows[0].keys())
        with CANDIDATE_CSV.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        img = Image.open(SAR_PNG).convert("RGBA")
        overlay = img.copy()
        draw = ImageDraw.Draw(overlay)
        for r in rows:
            color = (34, 197, 94, 255) if r["status"] == "matched" else (249, 115, 22, 255)
            draw.rectangle(
                (int(r["bbox_x0"]), int(r["bbox_y0"]), int(r["bbox_x1"]), int(r["bbox_y1"])),
                outline=color, width=2
            )
        overlay.save(OVERLAY_PNG)

    print(f"SAR 영상 탐지 객체 : {len(detections)}개 | 보간된 AIS 선박 : {len(ais_records)}척")
    print(f"AI 재식별 완료 : 총 {len(rows)}건 중 매칭(정상)={matched_cnt}, 미매칭(Dark Vessel 의심)={dark_cnt}")
    return rows

# MMSI 앞 3자리(MID: Maritime Identification Digits) 기반 고위험 편의치적국(FOC) 정의
FOC_COUNTRY_CODES = {
    "351": "Panama", "352": "Panama", "353": "Panama", "354": "Panama", "355": "Panama", "356": "Panama", "357": "Panama",
    "636": "Liberia", "637": "Liberia",
    "312": "Belize",
    "667": "Sierra Leone",
    "671": "Togo",
    "577": "Vanuatu",
    "370": "Panama",
    "445": "North Korea", # 대북 제재 감시 대상
}

def verify_vessel_identity(mmsi: str, sar_length_m: float, declared_length_m: float) -> list[str]:
    """선박 등록·소유 위장 의심 플래그 검증"""
    flags = []
    mmsi_str = str(mmsi).strip()

    if not mmsi_str or mmsi_str == "Unknown":
        return ["미등록/AIS미송출(Dark)"]

    # 1. 허위/비정상 MMSI 도용 검사
    if len(mmsi_str) != 9 or mmsi_str.startswith("0") or len(set(mmsi_str)) <= 2:
        flags.append("허위MMSI도용의심(Spoofed-MMSI)")

    # 2. 고위험 편의치적국(Flag of Convenience) 교차검증
    mid = mmsi_str[:3]
    if mid in FOC_COUNTRY_CODES:
        country = FOC_COUNTRY_CODES[mid]
        flags.append(f"편의치적의심({country})")
    elif mid in ["440", "441"]:
        # 한국 국적 선박
        pass

    # 3. SAR 실측 크기 vs 공시 등록 제원(길이) 교차검증
    if declared_length_m > 0 and sar_length_m > 0:
        ratio = abs(sar_length_m - declared_length_m) / declared_length_m
        if ratio >= 0.45:  # 크기 차이 45% 이상
            flags.append(f"제원위장의심(등록:{int(declared_length_m)}m vs 실측:{int(sar_length_m)}m)")

    return flags


if __name__ == "__main__":
    run_ai_vessel_detection()