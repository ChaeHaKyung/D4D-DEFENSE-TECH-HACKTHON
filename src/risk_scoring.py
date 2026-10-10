"""Transparent four-axis vessel-risk scoring and five-level action classification."""

from __future__ import annotations

import math


HARBOR_BBOX = (126.56, 37.40, 126.63, 37.48)
COASTAL_BUFFER_M = 5_000.0

ACTION_LEVELS = (
    (25, "정상", "안전 / 정상 통항", "#22c55e"),
    (50, "관심", "저위험 / 관찰", "#3b82f6"),
    (70, "주의", "중위험 / 주의", "#eab308"),
    (89, "경계", "고위험 / 경계", "#f97316"),
    (100, "심각", "초고위험 / 즉시 검토", "#ef4444"),
)


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _distance_to_harbor_bbox_m(lat: float, lon: float) -> float:
    west, south, east, north = HARBOR_BBOX
    closest_lon = min(max(lon, west), east)
    closest_lat = min(max(lat, south), north)
    if closest_lon == lon and closest_lat == lat:
        return 0.0
    mean_lat = math.radians((lat + closest_lat) / 2)
    north_m = (lat - closest_lat) * 111_320.0
    east_m = (lon - closest_lon) * 111_320.0 * math.cos(mean_lat)
    return math.hypot(north_m, east_m)


def _action_level(score: int) -> tuple[str, str, str]:
    for maximum, level, label, color in ACTION_LEVELS:
        if score <= maximum:
            return level, label, color
    raise AssertionError(f"Risk score outside supported range: {score}")


def score_candidate(
    row: dict,
    *,
    behavior_points: int = 0,
    behavior_flags: list[str] | None = None,
) -> dict[str, object]:
    """Compute the published identity, kinematics, size, and behavior rubric."""
    flags = list(behavior_flags or [])
    status = str(row.get("status", "unmatched-candidate"))
    mmsi = str(row.get("nearest_mmsi") or "").strip()
    matched = status == "matched" and bool(mmsi)

    gap_seconds = _finite_float(row.get("max_gap_seconds"), 0.0)
    if not matched:
        identity_score = 40
        flags.append("AIS 미매칭 (+40)")
    elif gap_seconds > 3_600:
        identity_score = 40
        flags.append("AIS 공백 60분 초과 (+40)")
    elif gap_seconds >= 900:
        identity_score = 25
        flags.append("AIS 공백 15~60분 (+25)")
    elif gap_seconds >= 300:
        identity_score = 15
        flags.append("AIS 공백 5~15분 (+15)")
    else:
        identity_score = 0

    lat = _finite_float(row.get("lat"))
    lon = _finite_float(row.get("lon"))
    west, south, east, north = HARBOR_BBOX
    if west <= lon <= east and south <= lat <= north:
        zone = "harbor"
        kinematics_score = 0
        flags.append("항만 정박지/내항 (동역학 +0)")
    else:
        distance_to_harbor = _distance_to_harbor_bbox_m(lat, lon)
        if distance_to_harbor <= COASTAL_BUFFER_M:
            zone = "coastal"
            kinematics_score = 5
            flags.append("연안 완충구역 5km 이내 (+5)")
        else:
            zone = "offshore"
            sog = _finite_float(row.get("sog"), -1.0)
            if matched and sog > 5.0:
                kinematics_score = 15
                flags.append("외해 고속 이동 5kn 초과 (+15)")
            elif matched and 0 <= sog < 0.5:
                kinematics_score = 20
                flags.append("외해 정지/표류 0.5kn 미만 (+20)")
            else:
                kinematics_score = 0

    length_m = _finite_float(row.get("length_m"), -1.0)
    if length_m <= 0:
        x_extent = _finite_float(row.get("bbox_x1"), -1) - _finite_float(row.get("bbox_x0"), -1)
        y_extent = _finite_float(row.get("bbox_y1"), -1) - _finite_float(row.get("bbox_y0"), -1)
        length_m = max(x_extent, y_extent) * 10 if max(x_extent, y_extent) > 0 else 0
    if 0 < length_m < 35:
        size_score = 5
        flags.append("소형 선박 35m 미만 (+5)")
    elif 35 <= length_m <= 90:
        size_score = 10
        flags.append("중형 선박 35~90m (+10)")
    elif length_m > 90:
        size_score = 20
        flags.append("대형 선박 90m 초과 (+20)")
    else:
        size_score = 0

    behavior_score = min(20, max(0, behavior_points))

    suspicious_identity = False
    if matched and (len(mmsi) != 9 or not mmsi.isdigit() or mmsi.startswith("0") or len(set(mmsi)) <= 2):
        suspicious_identity = True
        flags.append("비정상 MMSI 형식 (위장 가산 +15)")
    declared_length = _finite_float(row.get("declared_length"), 0.0)
    if matched and declared_length > 0 and length_m > 0:
        if abs(length_m - declared_length) / declared_length >= 0.45:
            suspicious_identity = True
            flags.append("공시/실측 길이 45% 이상 불일치 (위장 가산 +15)")
    spoof_bonus = 15 if suspicious_identity else 0

    raw_score = identity_score + kinematics_score + size_score + behavior_score + spoof_bonus
    score = min(100, raw_score)
    level, action_label, color = _action_level(score)
    return {
        "zone": zone,
        "risk_score": score,
        "score_identity": identity_score,
        "score_kinematics": kinematics_score,
        "score_size": size_score,
        "score_behavior": behavior_score,
        "score_spoof_bonus": spoof_bonus,
        "action_level": level,
        "action_label": action_label,
        "action_color": color,
        "risk_flags": " | ".join(dict.fromkeys(flags)) if flags else "특이사항 없음",
    }
