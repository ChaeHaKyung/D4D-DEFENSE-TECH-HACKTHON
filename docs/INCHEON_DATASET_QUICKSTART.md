# 인천항 SAR 시계열 데이터셋 빠른 실행

## 데이터셋 구성

- `training_pool.csv`: IW·VV/VH 조건을 만족하는 모든 촬영 장면. 탐지 모델 학습용.
- `temporal_core.csv`: 같은 궤도 방향과 상대궤도의 장면만 선별. 날짜별 변화 비교용.
- GeoTIFF 밴드: Band 1 `VV dB`, Band 2 `VH dB`, Band 3 `dataMask`.
- 범위: 인천 외항과 접근 수역 `[126.40, 37.32, 126.62, 37.48]`.
- 영상 크기: `1950 × 1780`, 약 10m 픽셀 간격.

## 1. 공개 장면 목록 갱신

```bash
cd D4D-DEFENSE-TECH-HACKTHON
python3 src/build_incheon_sar_dataset.py catalog --days 730
```

## 2. CDSE 인증정보 설정

아래 값은 반드시 본인의 새 값으로 입력한다. 채팅·코드·Git에 저장하지 않는다.

```bash
export CDSE_CLIENT_ID='새로운-client-id'
export CDSE_CLIENT_SECRET='새로운-client-secret'
```

## 3. 먼저 최신 3장 시험

```bash
python3 src/build_incheon_sar_dataset.py download --set training --max-scenes 3
python3 src/build_incheon_sar_dataset.py status
```

세 파일이 정상적으로 생성되면 수집 범위를 확대한다.

현재 생성된 2년 카탈로그에는 학습 풀 78장과 동일 궤도 시계열 코어 49장이 있다.

## 4. 학습 풀 전체 다운로드

```bash
python3 src/build_incheon_sar_dataset.py download --set training --max-scenes 0
```

중간에 중단돼도 같은 명령을 다시 실행하면 이미 받은 파일은 건너뛴다.
다운로드가 끝나면 각 GeoTIFF의 VV PNG, VV·VH 합성 PNG와
`outputs/datasets/incheon/dataset_manifest.csv`가 자동으로 갱신된다.

## 5. 선택 사항: Lee 3×3 파생본

```bash
python3 src/build_incheon_sar_dataset.py download --set training --max-scenes 0 --lee
```

원본 학습 자료는 무필터 GeoTIFF다. Lee 영상은 비교 실험용이며 원본을 대체하지 않는다.

## 저장 위치

```text
outputs/datasets/incheon/
├── sar/geotiff_raw/   # VV·VH 분석 원본
├── sar/png/           # 지도와 발표용 PNG
├── catalog/           # 78장 전체 풀과 49장 시계열 코어
├── metadata/          # 각 촬영 장면의 원본 STAC 정보
├── ais/               # 이후 촬영시각별 AIS 저장 위치
└── dataset_manifest.csv
```

## AIS와 결합할 때

각 GeoTIFF 파일명에 들어 있는 장면 ID의 `start_datetime`을 기준으로 AIS를 수집한다.
AIS가 관측된 선박은 짧은 시간 공백만 보간할 수 있다. AIS를 계속 끈 선박의 실제
과거 위치를 보간값으로 만들면 안 되며, 해당 SAR 탐지는 `AIS 미매칭 후보`로 남긴다.
