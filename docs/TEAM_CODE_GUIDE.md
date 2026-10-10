# Dark Vessel Detection 팀 코드 설명서

이 문서는 프로젝트를 처음 보는 팀원이 코드의 실행 순서, 각 파일의 책임, 입력과 출력,
핵심 알고리즘을 이해하도록 만든 안내서입니다. 실제 소스에도 같은 내용을 한국어 주석과
docstring으로 넣었습니다.

## 1. 프로그램을 한 문장으로 설명하면

최신 Sentinel-1 SAR 장면의 촬영시각을 기준으로 AIS 항적을 보간하고, SAR에서 밝은
선박 후보를 CFAR로 찾은 뒤, AIS 매칭 여부·해안거리·반복관측을 이용해 검토 우선순위를
지도에 표시하는 OSINT 프로토타입입니다.

## 2. 실행 순서

```text
app.py의 웹 버튼
  └─ src/run_pipeline.py
       ├─ src/sentinel_pipeline.py
       ├─ src/collect_ais.py
       ├─ src/collect_historical_ais.py
       ├─ src/time_align_ais.py
       ├─ src/detect_vessels.py
       └─ src/evaluate_pipeline.py
```

기존 데이터로 다시 분석할 때:

```bash
python3 src/run_pipeline.py --mode analyze
python3 -m streamlit run app.py
```

CDSE 인증 후 최신 데이터부터 다시 받을 때:

```bash
python3 src/run_pipeline.py --mode refresh
```

## 3. app.py - 웹 화면

- `ROOT`, `OUTPUTS`: 프로젝트와 결과 폴더 위치입니다.
- `AIS_PATH`: 원시 AIS가 아니라 SAR 시각으로 보간된 한 척당 한 행의 CSV입니다.
- `load_data()`: STAC JSON과 보간 AIS CSV를 읽습니다.
- `@st.cache_data(ttl=30)`: 30초 동안 같은 파일을 다시 읽지 않아 웹 반응속도를 높입니다.
- `subprocess.run(...)`: 웹 버튼이 별도의 파이프라인 프로세스를 실행합니다.
- `returncode == 0`: 정상 완료를 뜻합니다.
- `ImageOverlay`: SAR PNG 네 모서리를 실제 위·경도에 맞춰 OpenStreetMap 위에 올립니다.
- AIS 초록색: 0.5 knot 초과 이동선입니다.
- AIS 파란색: 정박 또는 저속선입니다.
- SAR 후보 초록색: AIS가 매칭된 후보입니다.
- SAR 후보 주황색: AIS 미매칭 검토 후보입니다.
- SAR 후보 회색: 여러 날짜 같은 위치에서 반복된 고정시설 후보입니다.

웹은 탐지 계산을 직접 하지 않습니다. 완료된 파일을 표현하는 역할만 담당합니다.

## 4. sentinel_pipeline.py - 위성 장면 검색과 영상 생성

### STAC API

STAC은 위성영상 카탈로그 표준입니다. 영상 픽셀 자체가 아니라 다음 정보를 검색합니다.

- 장면 ID
- 촬영 시작·종료시각
- 위성 플랫폼
- 관측 모드
- 편파
- 궤도 방향
- 영상 자산 주소

`search_scenes()`는 부산 BBOX와 겹치고 `IW`, `VV` 조건을 만족하는 장면만 남깁니다.
`search_latest()`는 그중 최신 장면 한 장을 반환합니다.
`save_catalog()`는 여러 날짜 장면을 `sar_scene_catalog.csv`로 저장합니다.

### Process API

`download_geotiff()`는 다음 처리를 서버에 요청합니다.

```text
VV power
→ 10 × log10(VV)
→ VV dB
→ GeoTIFF의 첫 번째 밴드
```

두 번째 밴드는 유효 픽셀을 나타내는 `dataMask`입니다.

- `orthorectify=True`: 레이더 영상 위치를 실제 지표 좌표에 맞춥니다.
- `GAMMA0_ELLIPSOID`: 타원체 기준 후방산란계수입니다.
- `LEE 3×3`: 작은 창으로 speckle 잡음을 완화합니다.
- `1024×1024`: 현재 데모 출력 크기이며 최종 학습용 해상도가 아닙니다.

`download_visual_preview()`는 -30~0 dB를 0~255 밝기로 바꿔 사람이 볼 수 있는 PNG를
만듭니다. 이 PNG는 시각화용이지만 현재 CFAR 기준선도 이 PNG를 사용합니다.

## 5. collect_ais.py - 현재 AIS 수집

Open Waters의 SSE 연결을 일정 시간 유지합니다.

```text
SSE 한 줄 수신
→ "data: " 확인
→ JSON 파싱
→ PositionReport만 선택
→ 위·경도 없는 메시지 제외
→ 중복 키 검사
→ CSV 한 행 저장
```

`snapshot=1`은 연결 직후 서버가 알고 있는 최근 선박 상태를 먼저 전달합니다.
이 파일의 결과는 현재 MMSI 목록이며 완전한 역사 AIS가 아닙니다.

## 6. collect_historical_ais.py - SAR 시각 주변 항적

1. `latest_stac_item.json`에서 SAR 촬영시각을 읽습니다.
2. 현재 AIS CSV에서 고유 MMSI를 만듭니다.
3. 각 MMSI의 SAR 시각 ±30분 track API를 호출합니다.
4. 부산 BBOX 안의 점만 저장합니다.

한계: 현재 목록에 없는 과거 선박은 조회하지 못합니다. 따라서 결과를 전체 과거 AIS라고
부르면 안 됩니다.

## 7. time_align_ais.py - 시간 정렬

SAR는 한 시각의 영상이고 AIS는 불규칙 시계열이므로 좌표 시각을 일치시켜야 합니다.

```text
MMSI별 행 모으기
→ 시간순 정렬
→ SAR 직전 점 left 찾기
→ SAR 직후 점 right 찾기
→ 시간 비율 fraction 계산
→ 위도·경도 선형 보간
```

보간식:

```text
fraction = (SAR 시각 - left 시각) / (right 시각 - left 시각)
position = left + fraction × (right - left)
```

신뢰도:

- `high`: 전후 점이 있고 최대 간격이 ±5분 이내
- `medium`: 한쪽 최근접 실제 점이 ±5분 이내
- `low`: ±5분은 넘지만 ±30분 이내
- 제외: ±30분도 넘음

## 8. detect_vessels.py - SAR 후보 탐지

### 적분영상

각 픽셀마다 주변 25×25 영역을 반복 합산하면 느리기 때문에 누적합으로 사각형 합을
계산합니다. `box_sum()`이 이 역할을 합니다.

### 육지와 분석 가능 해수면

Natural Earth GeoJSON의 경위도 폴리곤을 SAR 픽셀로 바꿔 육지 마스크를 만듭니다.
그 후 SAR 자체의 넓은 영역 평균과 표준편차를 이용해 지나치게 밝고 거친 영역을 제외합니다.

### CA-CFAR 기준선

기본 임계값:

```text
threshold = 주변 평균 + k × 주변 표준편차
기본 k = 4.2
```

- 해안 1km 이내: k에 0.7 추가
- 항만 구역: k에 0.7 추가
- 절대 밝기 120 초과 조건 추가
- 보호창 평균보다 20 이상 밝아야 함

항만과 해안에서 임계값을 높이는 이유는 크레인, 방파제, 건물 등 고반사 구조물의 오탐이
많기 때문입니다.

### 연결요소

인접한 밝은 픽셀을 BFS로 묶습니다. 너무 크거나 긴 물체는 항만시설·줄무늬로 보고
제외합니다. 남은 픽셀의 밝기 가중 중심을 선박 후보 좌표로 사용합니다.

### AIS 매칭

각 SAR 후보와 모든 보간 AIS의 Haversine 거리를 계산하고 가장 가까운 AIS를 찾습니다.
현재는 900m 이내를 매칭으로 사용합니다. 이 값은 저해상도와 SAR 이동표적 위치 편이를
고려한 임시 값이며 최종적으로 검증 데이터로 조정해야 합니다.

### 반복 고정시설

현재 장면 결과를 `outputs/history/<scene-id>.csv`로 저장합니다. 과거 서로 다른 두 장면에서
150m 이내 같은 위치가 발견되면 현재까지 총 세 장면 반복이므로 `persistent-static`으로
분류합니다.

### 위험점수

위험점수는 확률이 아니라 검토 우선순위입니다.

- AIS 매칭: 낮은 기본점수
- 반복 고정물: 낮은 기본점수
- AIS 미매칭: 높은 기본점수
- 항만: 30점 감점
- 해안: 20점 감점
- 가장 가까운 AIS가 low 신뢰도: 5점 가산

## 9. evaluate_pipeline.py - 검증

### 보간 홀드아웃

AIS 세 점 `left-middle-right`에서 middle을 숨기고 left/right로 예측합니다. 예측한 위치와
실제 middle 사이 거리를 계산합니다. 현재 결과는 1,321개 표본에서 중앙값 1.18m,
p95 10.63m입니다.

### AIS-supported recall proxy

고신뢰 AIS 중 CFAR 후보와 매칭된 고유 MMSI 비율입니다. 완전한 정답 바운딩박스가
아니므로 실제 recall이 아닙니다. 현재 값 0.25를 탐지 정확도 25%라고 표현하면 안 됩니다.

## 10. run_pipeline.py - 전체 실행 관리자

각 단계를 `with stage(...)`로 감쌉니다.

- 시작: `STARTED`
- 중간 결과: `INFO`
- 성공: `COMPLETED`와 실행시간
- 실패: `FAILED`와 오류·traceback

로그는 JSONL 형식으로 한 줄에 이벤트 하나를 기록합니다. 웹은 최근 40개를 읽어 표로
표시합니다. `pipeline_manifest.json`은 가장 최근 성공 실행의 작은 요약입니다.

## 11. 현재 결과를 발표할 때 지켜야 할 표현

올바른 표현:

> 실시간 AIS와 최신 공개 SAR를 결합해 AIS 미매칭 SAR 후보를 우선순위화한다.

피해야 할 표현:

> 실시간 위성으로 Dark Vessel 80척을 확정 탐지했다.

현재 SAR는 연속 실시간이 아니며, 역사 AIS는 부분 표본이고, 미매칭 후보에는 파도·부표·
남은 구조물이 포함될 수 있습니다.

## 12. 다음 개발 순서

1. 카탈로그 9개 장면을 날짜별 폴더로 내려받기
2. 각 장면을 원해상도에 가까운 VV·VH·입사각으로 만들기
3. 각 장면 시각의 완전한 AIS 또는 신뢰 가능한 역사 AIS 확보
4. 사람 검수용 선박·고정물·파도 라벨 화면 만들기
5. 장면 날짜 단위 train/validation/test 분리
6. YOLO 1-class ship detector 미세조정
7. CFAR와 YOLO의 합의·불일치 분석
8. 칼만 필터와 선형 보간의 홀드아웃 오차 비교
