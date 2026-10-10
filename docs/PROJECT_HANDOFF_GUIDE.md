# 공개 SAR·AIS 기반 Dark Vessel Detection

## 내일 팀원을 위한 10분 인수인계 가이드

### 0. 이 프로젝트를 한 문장으로

실시간에 가까운 공개 AIS와 가장 최근의 공개 Sentinel-1 SAR 영상을 **같은 촬영시각으로
맞춘 뒤**, SAR에는 보이지만 AIS와 매칭되지 않는 선박 후보를 찾아 지도에서 감시
우선순위로 보여주는 OSINT 프로토타입이다.

중요: 현재 시스템은 미매칭 후보를 찾아주는 단계다. 미매칭 후보를 곧바로 불법선박이나
Dark Vessel로 확정하지 않는다.

---

## 1. 왜 이 순서로 만들었나

처음에는 Copernicus Browser에서 SAR 영상을 눈으로 확인했다. 그러나 사람이 매번 날짜와
지역을 고르면 자동화와 재현이 어렵기 때문에 STAC API와 Process API로 전환했다.

그다음 현재 AIS를 지도에 표시했지만, 최신 SAR가 이틀 전 영상이라 현재 AIS와 바로
겹치면 잘못된 비교가 된다는 문제가 생겼다. 그래서 SAR 촬영시각 전후의 AIS 항적을
찾고, 선박별로 시간순 정렬한 뒤 SAR 촬영시각에 맞춰 선형 보간했다.

이후 SAR의 밝은 점을 모두 선박으로 보면 육지·부두·방파제가 대량 오탐되는 문제가 생겨
다음 처리를 추가했다.

1. Natural Earth 육지 마스크
2. 해안 1km 이내 별도 임계값
3. 부산항 구역과 외해의 임계값 분리
4. 여러 날짜 같은 위치에 반복되는 후보를 고정시설로 판별하는 이력 구조
5. CFAR 결과와 AIS 위치의 공간 매칭
6. 보간오차와 탐지 대리지표 검증
7. 모든 단계를 한 번에 실행하고 웹에서 로그를 보여주는 통합 파이프라인

---

## 2. 전체 데이터 연계 구조

```text
CDSE STAC API
  └─ 최신 Sentinel-1 장면 ID와 촬영시각 검색
       ├─ latest_stac_item.json
       └─ 촬영시각이 모든 데이터의 기준시각이 됨

CDSE Process API
  └─ 해당 장면·부산 BBOX의 VV 영상 생성
       ├─ busan_vv_db.tiff      분석용 원본 형태
       └─ busan_vv_visual.png   지도·현재 CFAR용 표시 영상

Open Waters AIS
  ├─ 현재 SSE 스트림 → busan_ais_live.csv
  └─ MMSI별 과거 항적 → busan_ais_at_sar_time.csv
                              ↓
                         시간순 정렬·보간
                              ↓
                    ais_interpolated_at_sar.csv
                              ↓
SAR PNG + 육지 마스크 + 보간 AIS
  └─ 구역별 CFAR + 최근접 AIS 매칭
       ├─ sar_ship_candidates.csv
       ├─ sar_ship_candidates_overlay.png
       └─ history/<scene-id>.csv
                              ↓
검증·로그·지도
  ├─ evaluation.json
  ├─ pipeline_log.jsonl
  ├─ pipeline_manifest.json
  └─ app.py Streamlit 대시보드
```

가장 중요한 연계는 `latest_stac_item.json`의 **SAR 촬영시각**이다. 과거 AIS 조회,
시간보간, 지도 표시, SAR-AIS 매칭이 전부 이 시각을 공유해야 한다.

---

## 3. 질문에 나온 터미널 명령은 정확히 무엇을 하나

```bash
cd "/Users/ha-kyungchae/Documents/Codex/2026-10-08/referenced-chatgpt-conversation-this-is-an"

export CDSE_CLIENT_ID='발급받은 새 ID'
export CDSE_CLIENT_SECRET='발급받은 새 Secret'

python3 src/sentinel_pipeline.py preview
```

### `cd ...`

터미널의 현재 작업 폴더를 프로젝트 루트로 옮긴다. 그래야 `src/...`, `outputs/...` 같은
상대경로가 이 프로젝트를 기준으로 해석된다.

### `export CDSE_CLIENT_ID=...`, `export CDSE_CLIENT_SECRET=...`

CDSE가 우리 프로그램을 인증할 수 있도록 자격증명을 **현재 터미널 세션의 환경변수**에
넣는다. 파일에 저장하는 명령이 아니며 새 터미널을 열면 다시 설정해야 한다.

실제 Secret을 코드, README, 채팅, GitHub에 올리면 안 된다. 이미 노출된 Secret은
폐기하고 새로 발급해야 한다.

### `python3 src/sentinel_pipeline.py preview`

내부에서는 다음 순서가 실행된다.

```text
1. argparse가 preview 명령을 읽음
2. STAC API에 부산 BBOX와 최근 날짜를 전송
3. IW·VV 조건을 만족하는 가장 최신 장면 선택
4. 장면 ID와 촬영시각을 latest_stac_item.json에 저장
5. ID·Secret을 CDSE OAuth 서버에 POST
6. 짧은 수명의 access token 수신
7. Process API에 다음을 POST
   - 부산 BBOX
   - 선택 장면의 촬영시각
   - Sentinel-1 GRD
   - IW / VV+VH
   - 정사보정
   - Gamma0
   - Lee 3×3 필터
   - VV를 dB로 바꾸는 Evalscript
8. Process API 응답 PNG를 busan_vv_visual.png로 저장
```

즉 `preview`는 웹사이트 화면을 캡처한 것이 아니다. API가 실제 Sentinel-1 데이터를
서버에서 처리하고 그 결과 이미지를 프로그램에 돌려준 것이다.

### 실시간 업데이트인가

아니다. 명령을 실행한 순간 **카탈로그에서 가장 최근에 공개된 SAR 한 장**을 가져온다.
Sentinel-1은 부산을 계속 보는 CCTV가 아니며 궤도 통과 때만 촬영한다.

- AIS SSE: 연결 중에는 수초~수분 단위로 갱신되는 실시간에 가까운 데이터
- SAR: 위성 촬영·지상전송·처리 후 공개되는 최신 스냅숏
- 웹 버튼: 누를 때마다 새로 검색·수집하는 수동 갱신
- 현재 백그라운드 자동 스케줄러: 없음

따라서 발표 표현은 “실시간 AIS + 최신 공개 SAR 기반 준실시간 감시”가 정확하다.

---

## 4. Python 파일별 역할

| 파일 | 역할 | 입력 | 출력 |
|---|---|---|---|
| `src/sentinel_pipeline.py` | SAR 검색·영상 생성 | CDSE STAC/Process API | STAC JSON, TIFF, PNG, 장면 카탈로그 |
| `src/collect_ais.py` | 현재 AIS SSE 수집 | Open Waters stream | `busan_ais_live.csv` |
| `src/collect_historical_ais.py` | 현재 MMSI의 SAR 시각 전후 항적 조회 | STAC JSON, 현재 MMSI | `busan_ais_at_sar_time.csv` |
| `src/time_align_ais.py` | MMSI별 정렬·SAR 시각 보간 | 과거 AIS CSV | `ais_interpolated_at_sar.csv` |
| `src/detect_vessels.py` | 육지 제거·CFAR·AIS 매칭·위험점수 | SAR PNG, 육지 GeoJSON, 보간 AIS | 후보 CSV, 오버레이, 이력 |
| `src/evaluate_pipeline.py` | 보간 홀드아웃·탐지 대리지표 | AIS 항적, 후보 CSV | `evaluation.json` |
| `src/run_pipeline.py` | 위 파일들을 순서대로 실행·로그 기록 | `analyze` 또는 `refresh` | manifest, JSONL 로그 |
| `app.py` | 결과 지도·지표·로그 표시 | outputs 파일들 | Streamlit 웹 화면 |

`app.py`가 탐지 알고리즘을 직접 수행하는 것이 아니다. 웹 버튼이 `run_pipeline.py`를
실행하고, 웹은 완성된 결과 파일을 읽어 표현한다.

---

## 5. 사용한 라이브러리와 필요한 이유

| 라이브러리 | 사용 이유 |
|---|---|
| `urllib` | 별도 SDK 없이 STAC·Process·AIS HTTP API 호출 |
| `ssl` | macOS에서도 HTTPS 인증서를 안정적으로 사용 |
| `csv`, `json` | API 응답과 단계별 데이터 저장 |
| `numpy` | SAR 픽셀 배열, 적분영상, 평균·표준편차, CFAR 계산 |
| `Pillow` | PNG 읽기, 육지 마스크 래스터화, 후보 박스 그리기 |
| `Folium` | OpenStreetMap 위 SAR·AIS·후보 레이어 구성 |
| `Streamlit` | 웹 대시보드, 버튼, 지표, 실행 로그 표시 |
| `streamlit-folium` | Folium 지도를 Streamlit 안에 삽입 |

현재 `rasterio`, `scipy`, `opencv`, YOLO 라이브러리는 사용하지 않는다. 다음 고해상도
GeoTIFF·AI 단계에서 추가할 수 있다.

---

## 6. 주요 출력물은 무엇이며 왜 필요한가

| 출력물 | 무엇인가 | 어디서 생성 | 어디에 사용 |
|---|---|---|---|
| `latest_stac_item.json` | 선택된 SAR 장면 전체 메타데이터 | `sentinel_pipeline.py` | 모든 단계의 SAR 기준시각 |
| `sar_scene_catalog.csv` | 최근 90일 부산 SAR 장면 9개 목록 | `sentinel_pipeline.py catalog` | 다중 날짜 처리 계획 |
| `busan_vv_db.tiff` | VV dB + dataMask 분석형 TIFF | `sentinel_pipeline.py download` | 향후 정밀 분석 |
| `busan_vv_visual.png` | -30~0 dB를 화면 밝기로 변환한 PNG | `sentinel_pipeline.py preview` | 지도와 현재 CFAR |
| `busan_ais_live.csv` | 현재 스트림에서 본 MMSI·위치 | `collect_ais.py` | 역조회할 MMSI 목록 |
| `busan_ais_at_sar_time.csv` | SAR 시각 ±30분 원시 항적 | `collect_historical_ais.py` | 시간보간 입력 |
| `ais_interpolated_at_sar.csv` | SAR 시각의 선박별 단일 위치 | `time_align_ais.py` | 탐지·지도·평가 공통 AIS |
| `land_mask.png` | Natural Earth 기반 육지/바다 마스크 | `detect_vessels.py` | 육상 고반사체 제거 |
| `sar_ship_candidates.csv` | 후보 좌표·구역·매칭·위험점수 | `detect_vessels.py` | 지도·검증·사람 검토 |
| `sar_ship_candidates_overlay.png` | 후보 상자가 그려진 SAR | `detect_vessels.py` | 빠른 육안 검토 |
| `history/<scene-id>.csv` | 장면별 후보 이력 | `detect_vessels.py` | 반복 고정시설 판별 |
| `evaluation.json` | 보간·탐지 점검 결과 | `evaluate_pipeline.py` | 성능을 과장하지 않는 근거 |
| `pipeline_log.jsonl` | 단계별 시작·완료·실패 로그 | `run_pipeline.py` | 웹 로그와 디버깅 |
| `pipeline_manifest.json` | 가장 최근 성공 실행 요약 | `run_pipeline.py` | 실행 상태 빠른 확인 |

---

## 7. 현재 설정값과 의미

| 설정 | 현재 값 | 이유·주의점 |
|---|---:|---|
| 부산 SAR BBOX | `[128.95, 34.95, 129.25, 35.20]` | 부산항과 외해 일부 포함 |
| 항만 근사 BBOX | `[128.98, 35.00, 129.14, 35.13]` | 항만 오탐 임계값 강화, 공식 경계는 아님 |
| Process 출력 | `1024×1024` | 빠른 데모용, 최종 분석에는 거침 |
| SAR 밴드 | `VV` 탐지 | 현재 기준선, 향후 VH·입사각 추가 |
| Speckle 필터 | Lee `3×3` | 잡음 감소, 작은 표적 흐림 가능성 있음 |
| CFAR 기본 k | `4.2` | 주변 평균 + 4.2×표준편차 |
| CFAR 배경 반경 | `12 px` | 주변 해수 통계 계산 |
| CFAR 보호 반경 | `2 px` | 표적이 배경 통계를 올리는 것 방지 |
| 해안 구역 | `1,000m` | 부두·방파제 오탐 별도 처리 |
| AIS 엄격 시간창 | `±5분` | high/medium 신뢰도 |
| AIS 대체 시간창 | `±30분` | 자료 부족 시 low 신뢰도 |
| SAR-AIS 매칭반경 | `900m` | 저해상도·위치편이 고려한 임시값 |
| 반복 고정물 반경 | `150m` | 여러 날짜 같은 위치 후보 결합 |
| 고정물 판정 | 과거 2장+현재 1장 | 총 3개 서로 다른 장면 반복 |

설정값은 현재 데이터에 대한 임시 기준이다. 정답 라벨로 실험해 조정하기 전에는 최적값이라고
말하면 안 된다.

---

## 8. 현재까지 실제로 확인된 결과

- 사용 SAR: Sentinel-1C, 2026-10-07 21:23:59 UTC
- 최근 90일 호환 장면 카탈로그: 9개
- 실제 처리 완료 영상: 최신 1개
- SAR 시각으로 보간된 AIS: 62척
- high: 41척 / medium: 3척 / low: 18척
- CFAR 후보: 92개
- AIS 매칭 후보: 12개
- 고유 매칭 MMSI: 11척
- 미매칭 검토 후보: 80개
- 보간 홀드아웃 표본: 1,321개
- 보간오차 중앙값: 1.18m
- 보간오차 p95: 10.63m
- AIS 지지 재현율 대리지표: 0.25

`0.25`는 실제 탐지 recall이 아니다. 완전한 정답 바운딩박스가 없으므로 “고신뢰 AIS 중
현재 CFAR가 매칭한 비율”로만 설명해야 한다.

---

## 9. 팀원이 처음 실행하는 가장 안전한 순서

### 기존 파일로 재현

```bash
cd D4D-DEFENSE-TECH-HACKTHON
python3 src/run_pipeline.py --mode analyze
python3 -m streamlit run app.py
```

브라우저에서 `http://localhost:8501`을 연다.

### 최신 데이터부터 갱신

CDSE 새 자격증명을 환경변수에 설정한 후:

```bash
python3 src/run_pipeline.py --mode refresh
```

또는 웹 왼쪽의 `최신 SAR·AIS 수집 후 분석` 버튼을 누른다.

주의: `refresh`의 과거 AIS는 현재 MMSI 목록을 역조회하므로 완전한 과거 스냅숏이 아니다.

---

## 10. 심사위원에게 설명하는 30초 버전

> 공개 Sentinel-1 SAR와 AIS를 단순히 겹쳐 놓은 것이 아니라, 먼저 SAR 촬영시각으로
> AIS 항적을 보간해 시간축을 맞췄습니다. SAR에서는 지리적 육지 마스크와 항만·해안·
> 외해별 CFAR를 적용해 선박 후보를 찾고, 보간 AIS와 매칭되지 않는 후보에 우선순위를
> 부여합니다. 여러 날짜 같은 위치에 반복되는 물체는 고정시설 후보로 낮추며, 모든 단계와
> 불확실성을 로그와 검증 지표로 공개합니다. 현재는 설명 가능한 CFAR 기준선이며 다음
> 단계에서 다중 날짜 VV·VH·입사각 데이터로 YOLO를 학습할 구조입니다.

---

## 11. 현재 한계와 다음 팀 작업

현재 한계:

1. 실제 처리 영상이 한 장이라 일반화 성능을 말할 수 없음
2. 현재 CFAR가 1024 PNG를 사용해 작은 선박 정보가 부족함
3. 과거 AIS가 현재 MMSI 기반 부분 표본임
4. 항만 경계가 공식 폴리곤이 아닌 근사 BBOX임
5. SAR 이동표적의 Doppler 위치편이를 보정하지 않음
6. 정답 바운딩박스가 없어 실제 precision/recall이 없음
7. SAR는 실시간 연속영상이 아님

다음 우선순위:

1. 카탈로그 9개 장면을 날짜별로 다운로드
2. 원해상도에 가까운 VV·VH·입사각 GeoTIFF 생성
3. 각 장면의 신뢰 가능한 역사 AIS 확보
4. CFAR 후보를 `선박/고정물/파도/모호`로 사람이 검수
5. 장면 날짜 기준 train/validation/test 분리
6. YOLO 1-class ship detector 미세조정
7. CFAR와 YOLO가 모두 지지한 후보를 고신뢰로 분류

내일 팀원이 먼저 해야 할 것은 YOLO 실행이 아니라 **여러 날짜의 데이터와 정답 라벨을
만드는 것**이다. 한 장으로 YOLO를 학습하면 선박이 아니라 그 영상의 노이즈를 외운다.
