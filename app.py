"""Streamlit으로 SAR·AIS 결과, 검증값, 실행 로그를 보여주는 웹 대시보드.

이 파일은 직접 탐지 알고리즘을 계산하지 않는다. 왼쪽 버튼이 ``run_pipeline.py``를
별도 프로세스로 실행하고, 완료된 CSV/JSON/PNG를 읽어 지도와 지표로 표현한다.
따라서 터미널 실행과 웹 버튼 실행은 같은 처리 코드를 공유한다.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import folium
import streamlit as st
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium


# app.py가 프로젝트 루트에 있으므로 parent가 곧 프로젝트 경로다.
ROOT = Path(__file__).resolve().parent
# 아래 상수들은 화면이 읽을 각 단계의 표준 출력 파일이다.
OUTPUTS = ROOT / "outputs"
AIS_PATH = OUTPUTS / "ais_interpolated_at_sar.csv"
SAR_PATH = OUTPUTS / "busan_vv_visual.png"
CANDIDATE_PATH = OUTPUTS / "sar_ship_candidates.csv"
STAC_PATH = OUTPUTS / "latest_stac_item.json"
LOG_PATH = OUTPUTS / "pipeline_log.jsonl"
MANIFEST_PATH = OUTPUTS / "pipeline_manifest.json"
EVALUATION_PATH = OUTPUTS / "evaluation.json"
# Folium ImageOverlay의 bounds 순서는 [[남, 서], [북, 동]]이다.
SAR_BOUNDS = [[34.95, 128.95], [35.20, 129.25]]


def parse_time(value: str) -> datetime:
    """UTC ISO 문자열을 시간대 정보가 있는 datetime으로 변환한다."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@st.cache_data(ttl=30)
def load_data() -> tuple[dict, list[dict], str]:
    """STAC 메타데이터와 시간보간이 끝난 AIS 스냅숏을 30초 캐시한다."""
    # 원시 항적이나 마지막 AIS 점이 아니라 탐지기와 동일한 보간 CSV를 읽는다.
    item = json.loads(STAC_PATH.read_text(encoding="utf-8"))
    with AIS_PATH.open(encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    return item, rows, "SAR 촬영시각 보간 AIS"


@st.cache_data(ttl=30)
def load_candidates() -> list[dict]:
    """CFAR 후보 CSV가 있으면 읽고, 아직 없으면 빈 리스트를 반환한다."""
    if not CANDIDATE_PATH.exists():
        return []
    with CANDIDATE_PATH.open(encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


# Streamlit 명령 중 가장 먼저 페이지 제목·아이콘·넓은 레이아웃을 설정한다.
st.set_page_config(page_title="Dark Vessel 부산항", page_icon="🛰️", layout="wide")
st.title("🛰️ 공개 SAR·AIS 기반 부산항 감시 프로토타입")

# 핵심 입력이 없으면 아래 계산에서 애매한 오류가 나기 전에 명확히 중단한다.
if not all(path.exists() for path in (AIS_PATH, SAR_PATH, STAC_PATH)):
    st.error("outputs 폴더에 SAR PNG, STAC JSON, AIS CSV가 모두 필요합니다.")
    st.stop()

# 캐시된 파일들을 한 번 읽어 화면 전체에서 재사용한다.
item, vessels, ais_mode = load_data()
candidates = load_candidates()
sar_time = parse_time(item["properties"].get("start_datetime", item["properties"]["datetime"]))
# 시간 정렬 품질을 사용자가 즉시 볼 수 있도록 신뢰도별 척수를 센다.
high_count = sum(row["confidence"] == "high" for row in vessels)
medium_count = sum(row["confidence"] == "medium" for row in vessels)
low_count = sum(row["confidence"] == "low" for row in vessels)
# 이상치에 덜 민감한 중앙값으로 AIS와 SAR 사이 대표 시간 간격을 표시한다.
median_gap_minutes = sorted(float(row["max_gap_seconds"]) for row in vessels)[len(vessels) // 2] / 60 if vessels else 0

# 핵심 상태 네 개를 같은 행의 카드로 배치한다.
c1, c2, c3, c4 = st.columns(4)
c1.metric(ais_mode, f"{len(vessels)}척")
c2.metric("SAR 촬영 UTC", sar_time.strftime("%m-%d %H:%M"))
c3.metric("±5분 신뢰 AIS", f"{high_count + medium_count}척")
c4.metric("보간 간격 중앙값", f"{median_gap_minutes:.1f}분")

st.success(
    f"지도와 탐지는 모두 SAR 촬영시각으로 보간한 동일 AIS 스냅숏을 사용합니다. "
    f"high={high_count}, medium={medium_count}, low={low_count}. 전체 과거 AIS가 아닌 부분 표본입니다."
)

with st.sidebar:
    # 첫 버튼은 네트워크 없이 기존 파일로 알고리즘만 다시 실행한다.
    st.header("파이프라인 실행")
    if st.button("현재 데이터 재분석", width="stretch"):
        with st.spinner("시간 정렬 → CFAR → 검증 실행 중"):
            # 현재 Streamlit과 같은 Python을 사용해야 같은 가상환경 패키지를 쓴다.
            result = subprocess.run(
                [sys.executable, str(ROOT / "src" / "run_pipeline.py"), "--mode", "analyze"],
                cwd=ROOT, text=True, capture_output=True,
            )
        # 운영체제 종료코드 0은 성공, 그 외는 실패를 의미한다.
        if result.returncode == 0:
            st.success("재분석 완료")
            # 이전 CSV 캐시를 비운 뒤 화면 전체를 다시 그린다.
            st.cache_data.clear()
            st.rerun()
        else:
            st.error("재분석 실패. 아래 실행 로그를 확인하세요.")
            st.code(result.stdout + result.stderr)
    # 두 번째 버튼은 네트워크·CDSE 인증이 필요한 refresh 모드를 호출한다.
    if st.button("최신 SAR·AIS 수집 후 분석", width="stretch"):
        # 비밀키 값을 화면에 출력하지 않고 존재 여부만 확인한다.
        if not os.environ.get("CDSE_CLIENT_ID") or not os.environ.get("CDSE_CLIENT_SECRET"):
            st.error("CDSE 인증 환경변수가 없습니다. 터미널에서 설정한 뒤 웹을 다시 실행하세요.")
        else:
            with st.spinner("최신 SAR 검색과 AIS 수집 중입니다"):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "src" / "run_pipeline.py"), "--mode", "refresh"],
                    cwd=ROOT, text=True, capture_output=True,
                )
            if result.returncode == 0:
                st.success("최신 데이터 파이프라인 완료")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("수집 또는 분석 실패. 실행 로그를 확인하세요.")
                st.code(result.stdout + result.stderr)
    st.divider()
    st.header("레이어")
    # 각 체크박스는 Folium 지도에 해당 FeatureGroup을 넣을지 결정한다.
    show_sar = st.checkbox("Sentinel-1 SAR", value=True)
    show_ais = st.checkbox("AIS 위치", value=True)
    show_candidates = st.checkbox("SAR 선박 후보", value=True)
    sar_opacity = st.slider("SAR 투명도", 0.0, 1.0, 0.65, 0.05)
    st.caption("지도: OpenStreetMap")
    st.caption("SAR: Copernicus Sentinel-1")
    st.caption("AIS: Open Waters / AISHub")

# 부산항 중심으로 OpenStreetMap 기본 지도를 만든다.
m = folium.Map(location=[35.075, 129.08], zoom_start=11, tiles="OpenStreetMap")

if show_sar:
    # PNG 네 모서리를 실제 BBOX에 고정해 지도 좌표 위에 반투명하게 겹친다.
    ImageOverlay(
        image=str(SAR_PATH),
        bounds=SAR_BOUNDS,
        opacity=sar_opacity,
        name="Sentinel-1 VV SAR",
        interactive=True,
        cross_origin=False,
        zindex=2,
    ).add_to(m)

if show_ais:
    # 보간 AIS는 후보 레이어와 분리해 사용자가 켜고 끌 수 있게 한다.
    ais_group = folium.FeatureGroup(name=ais_mode, show=True)
    for row in vessels:
        # 0.5 knot 초과 이동선은 초록, 정박/저속선은 파랑으로 표시한다.
        sog = float(row["sog"] or 0)
        color = "#22c55e" if sog > 0.5 else "#38bdf8"
        # 클릭 팝업에는 보간 결과의 근거와 신뢰도를 포함한다.
        popup = folium.Popup(
            f"<b>MMSI</b>: {row['mmsi']}<br>"
            f"<b>SAR UTC</b>: {row['sar_time']}<br>"
            f"<b>속력</b>: {row['sog']} kn<br>"
            f"<b>침로</b>: {row['cog']}°<br>"
            f"<b>보간</b>: {row['method']} / {row['confidence']}<br>"
            f"<b>최대 시간 간격</b>: {float(row['max_gap_seconds']) / 60:.1f}분<br>"
            f"<b>출처</b>: {row['source']}",
            max_width=320,
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=4,
            color="#082f49",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=popup,
            tooltip=f"MMSI {row['mmsi']}",
        ).add_to(ais_group)
    ais_group.add_to(m)

if show_candidates and candidates:
    # CFAR 후보는 상태별로 초록(매칭), 회색(고정물), 주황(미매칭)이다.
    candidate_group = folium.FeatureGroup(name="SAR 선박 후보", show=True)
    for row in candidates:
        matched = row["status"] == "matched"
        persistent = row["status"] == "persistent-static"
        color = "#22c55e" if matched else "#94a3b8" if persistent else "#f97316"
        label = "AIS 매칭" if matched else "반복 고정물 후보" if persistent else "미매칭 후보"
        # 위험점수와 해안거리는 사람이 우선 검토할 후보를 고르는 설명 근거다.
        popup = folium.Popup(
            f"<b>{row['candidate_id']}</b><br>"
            f"<b>판정</b>: {label}<br>"
            f"<b>최근접 MMSI</b>: {row['nearest_mmsi'] or '-'}<br>"
            f"<b>거리</b>: {row['distance_m'] or '-'} m<br>"
            f"<b>구역</b>: {row.get('zone', '-')}<br>"
            f"<b>해안 거리</b>: {row.get('coast_distance_m', '-')} m<br>"
            f"<b>위험점수</b>: {row.get('risk_score', '-')} / 100<br>"
            f"<b>주의</b>: 1차 CFAR 후보이며 확정 선박이 아님",
            max_width=320,
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=7,
            color=color,
            weight=2,
            fill=False,
            popup=popup,
            tooltip=f"{row['candidate_id']} · {label}",
        ).add_to(candidate_group)
    candidate_group.add_to(m)

# 지도 우상단 레이어 컨트롤을 항상 펼친 상태로 표시한다.
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, use_container_width=True, height=680, returned_objects=[])

with st.expander("현재 데이터 시간과 판정 기준"):
    # 결과 숫자만 보지 않도록 현재 판정의 전제와 제한을 함께 보여준다.
    st.write(f"SAR 촬영시각: `{sar_time.isoformat()}`")
    st.write(f"AIS 보간 결과: high `{high_count}` · medium `{medium_count}` · low `{low_count}`")
    st.write(
        "Dark Vessel 판정은 동일 SAR 촬영시각의 AIS가 확보된 경우에만 수행합니다. "
        "현재 화면은 레이어와 데이터 파이프라인 검증용입니다."
    )
    if candidates:
        matched = sum(row["status"] == "matched" for row in candidates)
        st.write(
            f"1차 CFAR 후보: `{len(candidates)}개` · AIS 매칭: `{matched}개` · "
            f"미매칭 후보: `{len(candidates) - matched}개`"
        )
        st.warning(
            "미매칭은 Dark Vessel 확정이 아닙니다. 현재 과거 AIS가 부분 표본이고, "
            "해안 마스크와 고해상도 GeoTIFF 검증 전이므로 분석 후보로만 표시합니다."
        )

if EVALUATION_PATH.exists():
    # evaluate_pipeline.py가 만든 값만 표시하며 웹에서 지표를 다시 계산하지 않는다.
    evaluation = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    with st.expander("검증 결과", expanded=True):
        interpolation = evaluation["interpolation_holdout"]
        detector = evaluation["detector_ais_support"]
        v1, v2, v3 = st.columns(3)
        v1.metric("보간 오차 중앙값", f"{interpolation['median_error_m']:.2f} m")
        v2.metric("보간 오차 p95", f"{interpolation['p95_error_m']:.2f} m")
        v3.metric("AIS 지지 재현율 대리지표", f"{detector['ais_supported_recall_proxy']:.1%}")
        st.caption(detector["warning"])

if LOG_PATH.exists():
    # JSONL의 마지막 40개 이벤트만 보여줘 화면이 무한히 길어지지 않게 한다.
    with st.expander("파이프라인 실행 로그", expanded=True):
        events = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()][-40:]
        st.dataframe(events, width="stretch", hide_index=True)
