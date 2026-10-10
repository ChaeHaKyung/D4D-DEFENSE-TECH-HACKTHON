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
import pandas as pd
import streamlit as st
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium


# app.py가 프로젝트 루트에 있으므로 parent가 곧 프로젝트 경로다.
ROOT = Path(__file__).resolve().parent
# 아래 상수들은 화면이 읽을 각 단계의 표준 출력 파일이다.
OUTPUTS = ROOT / "outputs"
AIS_PATH = OUTPUTS / "ais_interpolated_at_sar.csv"
SAR_PATH = OUTPUTS / "incheon_vv_visual.png"
CANDIDATE_PATH = OUTPUTS / "sar_ship_candidates.csv"
STAC_PATH = OUTPUTS / "latest_stac_item.json"
LOG_PATH = OUTPUTS / "pipeline_log.jsonl"
MANIFEST_PATH = OUTPUTS / "pipeline_manifest.json"
EVALUATION_PATH = OUTPUTS / "evaluation.json"
INFERENCE_OUTPUT_PATH = OUTPUTS / "firstplace" / "detections.csv"
# Folium ImageOverlay의 bounds 순서는 [[남, 서], [북, 동]]이다.
SAR_BOUNDS = [[37.32, 126.40], [37.48, 126.62]]


def find_xview3_geotiff(scene_id: str) -> Path:
    """Find the dual-polarization raw GeoTIFF for the selected STAC scene."""
    roots = (
        OUTPUTS / "datasets" / "incheon" / "sar" / "geotiff_raw",
        OUTPUTS,
    )
    matches = [
        path
        for folder in roots
        if folder.is_dir()
        for path in folder.glob(f"{scene_id}__incheon_outer_port__raw_gamma0_db.tif")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No dual-polarization raw GeoTIFF found for {scene_id}. "
            "Download that scene with the Incheon dataset downloader first."
        )
    return matches[0]


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
st.set_page_config(page_title="Dark Vessel 인천항", page_icon="🛰️", layout="wide")
st.title("🛰️ 공개 SAR·AIS 기반 인천항 감시 프로토타입")

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
    if st.button("Colab GPU에서 xView3 추론", width="stretch"):
        api_url = os.environ.get("XVIEW3_API_URL", "")
        api_token = os.environ.get("XVIEW3_API_TOKEN", "")
        if not api_url or not api_token:
            st.error(
                "Colab API 설정이 없습니다. Colab API 노트북 셀을 실행하고 "
                "XVIEW3_API_URL 및 XVIEW3_API_TOKEN 환경변수를 설정한 뒤 앱을 다시 실행하세요."
            )
        elif not (AIS_PATH.exists() and STAC_PATH.exists()):
            st.error("GPU 탐지 결과를 AIS와 매칭하려면 현재 장면의 STAC JSON과 AIS CSV가 필요합니다.")
        else:
            try:
                selected_item = json.loads(STAC_PATH.read_text(encoding="utf-8"))
                selected_scene_id = selected_item["id"]
                scene_path = find_xview3_geotiff(selected_scene_id)
                sys.path.insert(0, str(ROOT / "src"))
                from colab_gpu_client import submit_inference

                with st.spinner("SAR 영상을 Colab으로 전송하고 GPU 추론을 기다리는 중입니다..."):
                    detections_csv = submit_inference(
                        api_url,
                        api_token,
                        selected_scene_id,
                        scene_path,
                    )
                    INFERENCE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    INFERENCE_OUTPUT_PATH.write_text(detections_csv, encoding="utf-8")
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(ROOT / "src" / "import_xview3_detections.py"),
                            "--detections",
                            str(INFERENCE_OUTPUT_PATH),
                            "--scene-id",
                            selected_scene_id,
                        ],
                        cwd=ROOT,
                        text=True,
                        capture_output=True,
                    )
                if result.returncode == 0:
                    st.success(result.stdout.strip() or "Colab GPU 추론 결과를 앱에 반영했습니다.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Colab 추론은 완료했지만 앱 결과 변환에 실패했습니다.")
                    st.code(result.stdout + result.stderr)
            except Exception as exc:
                st.error(f"Colab GPU 추론 실패: {exc}")
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

st.divider()
st.header("위험도 스코어링 · 감시 우선순위")
st.caption(
    "점수는 탐지 후보의 검토 순서를 위한 규칙 기반 지표입니다. "
    "불법행위나 선박 신원을 확정하지 않습니다."
)

if candidates:
    risk_df = pd.DataFrame(candidates).copy()
    if "risk_score" not in risk_df:
        risk_df["risk_score"] = pd.NA
    risk_df["risk_score"] = pd.to_numeric(risk_df["risk_score"], errors="coerce")
    risk_df["candidate_id"] = risk_df.get(
        "candidate_id", pd.Series(index=risk_df.index, dtype="object")
    ).fillna("ID 미상").astype(str)
    risk_df["status"] = risk_df.get(
        "status", pd.Series(index=risk_df.index, dtype="object")
    ).fillna("unknown").astype(str)
    risk_df["action_level"] = risk_df.get(
        "action_level", pd.Series(index=risk_df.index, dtype="object")
    ).fillna("미평가").astype(str)
    calculated_levels = risk_df["risk_score"].map(
        lambda score: (
            "정상" if score <= 25 else
            "관심" if score <= 50 else
            "주의" if score <= 70 else
            "경계" if score <= 89 else
            "심각"
        ) if pd.notna(score) else "미평가"
    )
    risk_df.loc[risk_df["action_level"].eq("미평가"), "action_level"] = calculated_levels
    risk_df["zone"] = risk_df.get(
        "zone", pd.Series(index=risk_df.index, dtype="object")
    ).fillna("미분류").astype(str)
    evaluated_df = risk_df[risk_df["risk_score"].notna()].copy()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("전체 탐지 후보", f"{len(risk_df):,}건")
    kpi2.metric("경계 이상", f"{int(evaluated_df['risk_score'].ge(71).sum()):,}건")
    kpi3.metric("심각 · 즉시 검토", f"{int(evaluated_df['risk_score'].ge(90).sum()):,}건")
    kpi4.metric(
        "AIS 미매칭 후보",
        f"{int(risk_df['status'].eq('unmatched-candidate').sum()):,}건",
    )

    if evaluated_df.empty:
        st.warning("위험 점수가 포함된 후보가 없습니다. 현재 데이터를 다시 분석해 주세요.")
    else:
        level_order = ["정상", "관심", "주의", "경계", "심각"]
        level_counts = (
            evaluated_df["action_level"]
            .value_counts()
            .reindex(level_order, fill_value=0)
            .rename("후보 수")
            .to_frame()
        )
        left_chart, right_chart = st.columns([1, 2])
        with left_chart:
            st.markdown("**관제 단계별 후보 분포**")
            st.bar_chart(level_counts, height=220)
        with right_chart:
            st.markdown("**우선순위 후보**")
            available_levels = [
                level for level in level_order if level in set(evaluated_df["action_level"])
            ]
            selected_levels = st.multiselect(
                "표시할 관제 단계",
                options=available_levels,
                default=[level for level in available_levels if level in ("주의", "경계", "심각")],
                key="risk_dashboard_levels",
            )
            available_zones = sorted(evaluated_df["zone"].unique().tolist())
            selected_zones = st.multiselect(
                "표시할 해역",
                options=available_zones,
                default=available_zones,
                key="risk_dashboard_zones",
            )
            priority_df = evaluated_df[
                evaluated_df["action_level"].isin(selected_levels)
                & evaluated_df["zone"].isin(selected_zones)
            ].sort_values("risk_score", ascending=False)
            table_columns = [
                "candidate_id", "action_level", "risk_score", "status",
                "nearest_mmsi", "zone", "risk_flags",
            ]
            visible_columns = [column for column in table_columns if column in priority_df]
            st.dataframe(
                priority_df[visible_columns],
                width="stretch",
                hide_index=True,
                height=260,
            )

        if len(risk_df) > len(evaluated_df):
            st.warning(f"위험 점수가 없어 우선순위 계산에서 제외된 후보 {len(risk_df) - len(evaluated_df)}건이 있습니다.")

        detail_options = evaluated_df.sort_values(
            "risk_score", ascending=False
        )["candidate_id"].drop_duplicates().tolist()
        selected_candidate = st.selectbox(
            "후보별 점수 근거",
            options=detail_options,
            key="risk_dashboard_candidate",
        )
        detail = evaluated_df[evaluated_df["candidate_id"] == selected_candidate].iloc[0]
        score_columns = [
            ("신원", "score_identity"),
            ("동역학·구역", "score_kinematics"),
            ("선박 제원", "score_size"),
            ("이상행동", "score_behavior"),
            ("위장 가산", "score_spoof_bonus"),
        ]
        detail_metrics = st.columns(len(score_columns) + 1)
        detail_metrics[0].metric(
            f"총 위험 점수 · {detail.get('action_level', '미평가')}",
            f"{int(detail['risk_score'])} / 100",
        )
        for metric, (label, column) in zip(detail_metrics[1:], score_columns):
            value = pd.to_numeric(pd.Series([detail.get(column, 0)]), errors="coerce").iloc[0]
            metric.metric(label, f"{int(value) if pd.notna(value) else 0}점")
        st.info(f"**점수 근거:** {detail.get('risk_flags', '근거 정보 없음')}")
else:
    st.info("아직 탐지 후보가 없습니다. SAR·AIS 분석을 실행하면 감시 우선순위가 표시됩니다.")

# 인천항 중심으로 OpenStreetMap 기본 지도를 만든다.
m = folium.Map(location=[37.40, 126.51], zoom_start=11, tiles="OpenStreetMap")

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
    ais_group = folium.FeatureGroup(name=ais_mode, show=True)
    for row in vessels:
        sog = float(row.get("sog") or 0)
        color = "#22c55e" if sog > 0.5 else "#38bdf8"
        mmsi = row.get("mmsi", "Unknown")

        # 클릭 시 상세 팝업창
        popup_html = f"""
        <div style="font-family: sans-serif; font-size: 13px; min-width: 180px;">
            <b style="color: #0369a1;">🚢 AIS 선박 정보</b><hr style="margin: 4px 0;">
            <b>MMSI</b>: <span style="color: #dc2626; font-weight: bold;">{mmsi}</span><br>
            <b>속력 (SOG)</b>: {row.get('sog', '-')} kn<br>
            <b>침로 (COG)</b>: {row.get('cog', '-')}°<br>
            <b>보간 신뢰도</b>: {row.get('confidence', '-')}<br>
            <b>보간 시각</b>: {row.get('sar_time', '-')}
        </div>
        """
        popup = folium.Popup(popup_html, max_width=300)

        # 마우스 오버 시 바로 뜨는 툴팁 (MMSI + 속력)
        tooltip_text = f"MMSI: {mmsi} ({row.get('sog', '0')} kn)"

        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=5,
            color="#082f49",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=popup,
            tooltip=tooltip_text,
        ).add_to(ais_group)
    ais_group.add_to(m)

if show_candidates and candidates:
    candidate_group = folium.FeatureGroup(name="SAR 선박 후보", show=True)
    for row in candidates:
        matched = row["status"] == "matched"
        persistent = row.get("status") == "persistent-static"
        risk = int(float(row.get("risk_score", 0)))
        anomaly = str(row.get("anomaly_type", "정상"))
        matched_mmsi = row.get("nearest_mmsi") or "AIS 미매칭 후보"
        detector = row.get("detector", "기존 후보 탐지기")
        level = row.get("action_level", "미평가")
        action_label = row.get("action_label", "위험 점수 기준 미적용")
        color = row.get("action_color", "#64748b")
        radius = 6.0 + min(risk / 30, 4)
        weight = 2.0 + min(risk / 40, 2)
        vessel_state = "AIS 매칭" if matched else "AIS 미매칭 탐지 후보"
        if persistent:
            vessel_state = "고정물 의심 후보"
        badge_title = f"[{level}] {action_label} · {vessel_state}"

        # 위험 점수에 반영된 세부 근거를 지도 팝업에 함께 표시한다.
        flag_tags_html = ""
        for flag in str(row.get("risk_flags", anomaly)).split(" | "):
            flag_color = "#dc2626" if ("STS" in flag or "불일치" in flag or "MMSI 형식" in flag) else "#475569"
            flag_tags_html += (
                f'<span style="background:{flag_color}; color:white; padding:2px 6px; '
                f'border-radius:4px; font-size:11px; margin-right:4px; display:inline-block; '
                f'margin-bottom:2px;">{flag}</span>'
            )

        score_breakdown = (
            f"신원 {row.get('score_identity', 0)} + 동역학 {row.get('score_kinematics', 0)} + "
            f"제원 {row.get('score_size', 0)} + 이상거동 {row.get('score_behavior', 0)} + "
            f"위장 가산 {row.get('score_spoof_bonus', 0)}"
        )

        model_scores = ""
        if row.get("objectness_p") or row.get("is_vessel_p"):
            objectness = f"{float(row['objectness_p']):.3f}" if row.get("objectness_p") else "-"
            vessel_score = f"{float(row['is_vessel_p']):.3f}" if row.get("is_vessel_p") else "-"
            length_m = f"{float(row['length_m']):.1f}" if row.get("length_m") else "-"
            model_scores = (
                f"<b>xView3 objectness</b>: {objectness}<br>"
                f"<b>xView3 vessel</b>: {vessel_score}<br>"
                f"<b>추정 길이</b>: {length_m} m<br>"
            )

        popup_html = f"""
        <div style="font-family: -apple-system, sans-serif; font-size: 13px; min-width: 230px; line-height: 1.5;">
            <b style="color: {color}; font-size: 14px;">{badge_title}</b>
            <hr style="margin: 6px 0; border: none; border-top: 1px solid #e2e8f0;">
            <b>선박 ID</b>: {row['candidate_id']}<br>
            <b>MMSI</b>: <b>{matched_mmsi}</b><br>
            <b>탐지기</b>: {detector}<br>
            {model_scores}
            <b>관제 레벨</b>: <span style="color: {color}; font-weight: bold;">{level} · {action_label}</span><br>
            <b>위험 점수</b>: <span style="color: {color}; font-weight: bold; font-size: 15px;">{risk}</span> / 100<br>
            <b>점수 구성</b>: {score_breakdown}<br>
            <b>위치</b>: ({float(row['lat']):.4f}, {float(row['lon']):.4f})<br>
            <div style="margin-top: 6px;">
                <b>교차검증 플래그:</b><br>
                {flag_tags_html}
            </div>
        </div>
        """
        popup = folium.Popup(popup_html, max_width=320)

        # 마우스 오버 툴팁
        tooltip_text = f"[{row['candidate_id']}] {badge_title} (MMSI: {matched_mmsi})"

        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=radius,
            color=color,
            weight=weight,
            fill=False,
            popup=popup,
            tooltip=tooltip_text,
        ).add_to(candidate_group)

    candidate_group.add_to(m)

# 지도 우상단 레이어 컨트롤을 항상 펼친 상태로 표시한다.
folium.LayerControl(collapsed=False).add_to(m)
st_folium(m, use_container_width=True, height=680, returned_objects=[])

# 지도 아래(st_folium 다음)에 추가
st.divider()
st.subheader("📋 탐지 선박 및 MMSI 상세 내역")

tab1, tab2 = st.tabs(["🎯 SAR 탐지 & 매칭 결과 (Dark Vessel 포함)", "📡 수신된 AIS 선박 목록"])

with tab1:
    # 지도 아래 테이블 표시 부분[cite: 1]
    if candidates:
        import pandas as pd
        df_cand = pd.DataFrame(candidates)
    
    # 핵심 열 우선 배치 (anomaly_type, risk_score 전진 배치)[cite: 1]
        target_cols = [
            "candidate_id", "action_level", "action_label", "risk_score",
            "score_identity", "score_kinematics", "score_size", "score_behavior",
            "score_spoof_bonus", "risk_flags", "status", "anomaly_type",
            "nearest_mmsi", "distance_m", "zone", "lat", "lon", "detector",
            "objectness_p", "is_vessel_p", "is_fishing_p", "length_m"
        ]
        show_cols = [c for c in target_cols if c in df_cand.columns]
    
    # 위험도 순으로 기본 내림차순 정렬[cite: 1]
        if "risk_score" in df_cand.columns:
            df_cand["risk_score"] = pd.to_numeric(df_cand["risk_score"], errors="coerce")
            df_cand = df_cand.sort_values(by="risk_score", ascending=False)
        
        st.dataframe(df_cand[show_cols], use_container_width=True, hide_index=True)
        st.caption(
            "관제 기준: 0–25 정상 · 26–50 관심 · 51–70 주의 · "
            "71–89 경계 · 90–100 심각. 점수는 검토 우선순위이며 불법행위 확정 판정이 아닙니다."
        )
with tab2:
    if vessels:
        import pandas as pd
        df_ais = pd.DataFrame(vessels)
        cols = ["mmsi", "sog", "cog", "confidence", "lat", "lon", "method"]
        display_cols = [c for c in cols if c in df_ais.columns]
        st.dataframe(df_ais[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("수신된 AIS 데이터가 없습니다.")

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
            f"SAR 탐지 후보: `{len(candidates)}개` · AIS 매칭: `{matched}개` · "
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

st.divider()
st.subheader("📋 선박 식별 및 위장·이상 거동 교차검증 내역")

tab1, tab2, tab3 = st.tabs([
    "🚨 위장 의심 및 고위험 선박 (Alerts)", 
    "🎯 전체 SAR 탐지 & 매칭 결과", 
    "📡 수신된 AIS 항적 목록"
])

import pandas as pd
if candidates:
    df_cand = pd.DataFrame(candidates)
    if "risk_score" in df_cand.columns:
        df_cand["risk_score"] = pd.to_numeric(df_cand["risk_score"], errors="coerce")
        df_cand = df_cand.sort_values(by="risk_score", ascending=False)

    with tab1:
        # 경계 이상 점수 또는 명시적인 위장/접선 근거를 검토 대상으로 표시한다.
        suspect_mask = df_cand["anomaly_type"].str.contains("위장|허위MMSI|STS", na=False) | (df_cand["risk_score"] >= 71)
        df_suspect = df_cand[suspect_mask]
        
        if not df_suspect.empty:
            st.warning(f"⚠️ 검토 우선 대상 {len(df_suspect)}건입니다. 이는 불법행위 확정 판정이 아닙니다.")
            alert_cols = [
                "candidate_id", "action_level", "action_label", "risk_score",
                "risk_flags", "nearest_mmsi", "zone", "lat", "lon",
            ]
            st.dataframe(
                df_suspect[[c for c in alert_cols if c in df_suspect.columns]], 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.success("현재 경계 이상 점수 또는 위장·접선 근거가 있는 후보가 없습니다.")

    with tab2:
        display_cols = [
            "candidate_id", "action_level", "risk_score", "score_identity",
            "score_kinematics", "score_size", "score_behavior", "score_spoof_bonus",
            "status", "nearest_mmsi", "distance_m", "lat", "lon",
        ]
        st.dataframe(df_cand[[c for c in display_cols if c in df_cand.columns]], use_container_width=True, hide_index=True)

with tab3:
    if vessels:
        df_ais = pd.DataFrame(vessels)
        st.dataframe(df_ais, use_container_width=True, hide_index=True)
