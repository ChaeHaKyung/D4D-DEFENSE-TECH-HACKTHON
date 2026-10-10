#!/usr/bin/env python3
"""분리된 SAR/AIS 처리 단계를 한 번에 실행하고 구조화 로그를 남긴다.

``analyze`` 모드는 이미 저장된 SAR/AIS로 보간·탐지·검증만 다시 수행한다.
``refresh`` 모드는 최신 SAR 검색, 영상 생성, 현재 AIS와 과거 항적 수집부터 수행한다.
웹의 두 실행 버튼도 이 파일을 subprocess로 호출하므로 CLI와 웹 결과가 동일하다.
"""

from __future__ import annotations
# 상단 import 부분에 추가
from pretrained_vessel_detection import run_ai_vessel_detection

import argparse
import json
import os
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# 각 파일의 핵심 함수를 가져오되 역할이 드러나도록 별칭을 붙인다.
from collect_ais import collect as collect_live_ais
from collect_historical_ais import collect as collect_historical_ais
from detect_vessels import detect
from evaluate_pipeline import validate
from sentinel_pipeline import download_geotiff, download_visual_preview, save_search_result, search_latest
from time_align_ais import align


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LOG_PATH = OUTPUTS / "pipeline_log.jsonl"
MANIFEST_PATH = OUTPUTS / "pipeline_manifest.json"
STAC_PATH = OUTPUTS / "latest_stac_item.json"


def utc_now() -> str:
    """로그와 실행 ID에 사용할 현재 UTC 시각을 ISO 문자열로 반환한다."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log(stage: str, status: str, message: str, **details: object) -> None:
    """한 로그 이벤트를 JSONL 파일과 터미널 양쪽에 기록한다."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    # **details에는 장면 ID, 후보 수, 검증값 같은 단계별 추가 정보가 들어간다.
    event = {"time": utc_now(), "stage": stage, "status": status, "message": message, **details}
    with LOG_PATH.open("a", encoding="utf-8") as fp:
        # JSONL은 한 줄에 JSON 객체 하나여서 스트리밍 로그로 읽기 쉽다.
        fp.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[{event['time']}] [{stage}] [{status}] {message}", flush=True)


@contextmanager
def stage(name: str, start_message: str):
    """with 블록의 시작·완료·실패와 실행시간을 자동 기록하는 컨텍스트 관리자."""
    log(name, "STARTED", start_message)
    started = datetime.now(timezone.utc)
    try:
        yield
    except Exception as exc:
        # 예외를 기록한 뒤 다시 raise하여 상위 파이프라인도 실패를 알게 한다.
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        log(name, "FAILED", str(exc), elapsed_seconds=round(elapsed, 2))
        raise
    else:
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        log(name, "COMPLETED", "완료", elapsed_seconds=round(elapsed, 2))


def existing_item() -> dict:
    """analyze 모드에서 이미 선택된 STAC 장면 메타데이터를 읽는다."""
    return json.loads(STAC_PATH.read_text(encoding="utf-8"))


def run(mode: str, ais_seconds: int, strict_minutes: int, fallback_minutes: int) -> dict:
    """요청한 모드로 전체 파이프라인을 실행하고 최종 manifest를 반환한다."""
    # 초 단위까지 포함한 실행 ID로 서로 다른 실행을 구분한다.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log("pipeline", "STARTED", f"파이프라인 {run_id} 시작", mode=mode)
    try:
        if mode == "refresh":
            # 네트워크와 CDSE 인증이 필요한 전체 최신화 경로다.
            with stage("sar_search", "가장 최근 Sentinel-1 IW/VV 장면 검색"):
                item = search_latest()
                save_search_result(item)
                log("sar_search", "INFO", "장면 선택", scene_id=item["id"], acquisition_time=item["properties"].get("start_datetime"))
            with stage("sar_process", "분석 GeoTIFF와 지도용 PNG 생성"):
                download_geotiff(item)
                download_visual_preview(item)
            with stage("ais_live", f"현재 AIS 스트림 {ais_seconds}초 수집"):
                collect_live_ais(ais_seconds)
            with stage("ais_history", "현재 MMSI 목록의 SAR 시각 ±30분 항적 조회"):
                collect_historical_ais()
        else:
            # 기존 파일로 알고리즘만 빠르게 반복 실험하는 경로다.
            with stage("sar_existing", "저장된 SAR 장면과 AIS 원시 항적 확인"):
                item = existing_item()
                if not (OUTPUTS / "incheon_ais_at_sar_time.csv").exists():
                    raise FileNotFoundError("outputs/incheon_ais_at_sar_time.csv가 없습니다.")
                log("sar_existing", "INFO", "기존 장면 사용", scene_id=item["id"], acquisition_time=item["properties"].get("start_datetime"))

        # 아래 세 단계는 refresh/analyze 양쪽에서 공통으로 반드시 실행한다.
        with stage("time_alignment", f"AIS를 SAR 시각으로 보간: ±{strict_minutes}분 우선, ±{fallback_minutes}분 대체"):
            snapshot = align(strict_minutes, fallback_minutes)
            confidence = {level: sum(row["confidence"] == level for row in snapshot) for level in ("high", "medium", "low")}
            log("time_alignment", "INFO", "보간 스냅숏 생성", vessels=len(snapshot), **confidence)

        # run() 함수 내부의 cfar_detection 단계를 아래와 같이 교체:
        with stage("ai_vessel_detection", "YOLO-OBB SAR 탐지 및 AIS 보간 신뢰도 기반 Hungarian Re-ID"):
            candidates = run_ai_vessel_detection()
            counts = {
                "matched": sum(row["status"] == "matched" for row in candidates),
                "unmatched": sum(row["status"] == "unmatched-candidate" for row in candidates),
                "persistent": sum(row["status"] == "persistent-static" for row in candidates),
            }
            log("ai_vessel_detection", "INFO", "탐지 결과 저장", candidates=len(candidates), **counts)

        with stage("validation", "AIS 보간 홀드아웃과 AIS 지지 탐지율 계산"):
            report = validate()
            log(
                "validation", "INFO", "검증 지표 저장",
                interpolation_median_m=report["interpolation_holdout"]["median_error_m"],
                interpolation_p95_m=report["interpolation_holdout"]["p95_error_m"],
                ais_supported_recall_proxy=report["detector_ais_support"]["ais_supported_recall_proxy"],
            )

        # 웹이 가장 최근 실행 요약을 빠르게 읽도록 작은 manifest를 별도로 저장한다.
        manifest = {
            "run_id": run_id,
            "mode": mode,
            "completed_at": utc_now(),
            "scene_id": item["id"],
            "sar_time": item["properties"].get("start_datetime", item["properties"].get("datetime")),
            "ais_snapshot_vessels": len(snapshot),
            "sar_candidates": len(candidates),
            "validation": report,
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log("pipeline", "COMPLETED", f"파이프라인 {run_id} 완료")
        return manifest
    except Exception:
        # 전체 실패 로그에는 디버깅용 짧은 traceback도 포함한다.
        log("pipeline", "FAILED", f"파이프라인 {run_id} 실패", traceback=traceback.format_exc(limit=4))
        raise


def main() -> int:
    """CLI 옵션을 파싱하고 성공 0, 실패 1 종료코드를 운영체제에 반환한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["analyze", "refresh"], default="analyze")
    parser.add_argument("--ais-seconds", type=int, default=15)
    parser.add_argument("--strict-minutes", type=int, default=5)
    parser.add_argument("--fallback-minutes", type=int, default=30)
    args = parser.parse_args()
    try:
        run(args.mode, args.ais_seconds, args.strict_minutes, args.fallback_minutes)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
