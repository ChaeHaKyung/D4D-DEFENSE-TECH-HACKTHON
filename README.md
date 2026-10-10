# Dark Vessel Detection with Open SAR and AIS

공개 Sentinel-1 SAR 영상과 AIS를 결합해 해상 선박을 탐지하고, SAR에는 보이지만
AIS와 매칭되지 않는 대상을 **감시 우선 후보**로 제시하는 해커톤 프로젝트입니다.

> AIS 미매칭은 불법 선박 또는 Dark Vessel의 확정 판정이 아닙니다. AIS 수신 누락,
> 촬영시각 차이, 위치 오차, 부표·방파제 같은 고정 구조물이 원인일 수 있습니다.

## 현재 구현된 기능

- Copernicus Data Space STAC API에서 Sentinel-1 IW VV+VH 장면 검색
- CDSE Process API를 이용한 10m급 VV·VH GeoTIFF 생성
- 인천항 2년 장면 카탈로그 생성
  - 학습 풀: 78장
  - 동일 궤도 시계열 코어: 49장
- GeoTIFF를 VV PNG 및 VV·VH 합성 PNG로 변환
- 현재 AIS SSE 수집 및 과거 항적 조회
- SAR 촬영시각 기준 AIS 보간
- 육지·해안 패널티를 포함한 CFAR 선박 후보 탐지
- Streamlit/Folium 지도 대시보드
- 단계별 JSONL 로그와 간단한 검증 지표
- Colab xView3 First Place 추론 CSV를 AIS 매칭 후 대시보드 후보로 가져오기

## 폴더 구조

```text
.
├── app.py                         # Streamlit 웹 대시보드
├── requirements.txt
├── src/
│   ├── build_incheon_sar_dataset.py  # 인천 SAR 카탈로그·다중 장면 다운로드
│   ├── prepare_dataset_assets.py      # GeoTIFF → PNG 및 manifest CSV
│   ├── sentinel_pipeline.py           # 기존 부산 단일 장면 실험
│   ├── collect_ais.py                 # 현재 AIS 수집
│   ├── collect_historical_ais.py      # SAR 시각 전후 AIS 항적 조회
│   ├── time_align_ais.py              # AIS 시간 보간
│   ├── detect_vessels.py              # CFAR 후보 탐지
│   ├── evaluate_pipeline.py           # 검증 지표
│   └── run_pipeline.py                # 부산 실험 파이프라인 실행기
├── docs/                           # 팀 인수인계·연구 포지셔닝 문서
├── data/                           # 육지 마스크용 공개 GeoJSON
└── sample_data/                    # 영상 없이 공유 가능한 장면 목록·설정
```

## 설치 - 가상환경을 사용하지 않는 경우

Python 3가 설치된 터미널에서 실행합니다.

```bash
python3 -m pip install -r requirements.txt
```

설치 권한 오류가 발생하면 사용자 계정 범위로 설치합니다.

```bash
python3 -m pip install --user -r requirements.txt
```

## CDSE 인증정보

Client ID와 Secret은 코드나 GitHub에 저장하지 않고 현재 터미널의 환경변수로만
설정합니다.

```bash
export CDSE_CLIENT_ID='본인의-client-id'
export CDSE_CLIENT_SECRET='본인의-client-secret'
```

노출된 Secret은 즉시 폐기하고 새로 발급해야 합니다.

## 인천 SAR 데이터 수집

### 1. 장면 목록 만들기 - 인증 불필요

```bash
python3 src/build_incheon_sar_dataset.py catalog --days 730
```

### 2. 최신 한 장 시험

```bash
python3 src/build_incheon_sar_dataset.py download --set training --max-scenes 1
```

### 3. 전체 학습 풀 이어받기

```bash
python3 src/build_incheon_sar_dataset.py download --set training --max-scenes 0
```

이미 존재하는 GeoTIFF는 건너뜁니다. 다운로드가 끝나면 VV PNG, VV·VH 합성 PNG,
`dataset_manifest.csv`가 자동으로 갱신됩니다.

## 대시보드 실행

```bash
python3 -m streamlit run app.py
```

## xView3 First Place Colab GPU 연결

Streamlit 사이드바의 **Colab GPU에서 xView3 추론** 버튼은 실행 중인 Colab GPU API로 현재 장면 GeoTIFF를 전송하고 결과를 앱에 표시합니다. Colab API 준비 방법, 보안 토큰, 실행 명령은 [xView3 First Place 연결 가이드](docs/XVIEW3_FIRSTPLACE_INTEGRATION.md)를 참고하세요. AIS 미매칭은 미확인 탐지 후보이지 Dark Vessel 확정이 아닙니다.

## 위험 점수와 관제 우선순위

점수는 탐지 후보를 검토할 순서를 정하는 설명 가능한 규칙 점수이며, 불법행위의 확률이나 판정이 아닙니다. AIS 수신 공백·위치 오차·SAR 길이 추정 오차와 항만 구조물 오탐 가능성을 함께 검토해야 합니다.

| 축 | 기준 | 점수 |
|---|---|---:|
| 신원 | AIS 미매칭 후보 | +40 |
| 신원 | AIS 공백 5분 이상~15분 미만 / 15분 이상~60분 / 60분 초과 | +15 / +25 / +40 |
| 동역학·구역 | 항만 사각형 내부 / 바깥 5km 연안 완충구역 | +0 / +5 |
| 동역학·구역 | 5km 바깥에서 AIS SOG 5kn 초과 / 0.5kn 미만 | +15 / +20 |
| 선박 제원 | 추정 길이 35m 미만 / 35~90m / 90m 초과 | +5 / +10 / +20 |
| 이상행동 | 외해 600m 안에 3척 이상 / 180m 이내 일반 STS / 미매칭 선박 포함 STS | +8 / +12 / +20 |
| 위장 가산 | 비정상 MMSI 또는 AIS 등록 길이와 SAR 추정 길이 45% 이상 차이 | +15 |

동역학의 속도 점수는 AIS 매칭과 유효한 SOG가 모두 있을 때만 부여합니다. 미매칭의 빈 SOG를 0kn로 간주하지 않습니다. 총점은 100점으로 제한하며 관제 단계는 **정상 0–25, 관심 26–50, 주의 51–70, 경계 71–89, 심각 90–100**입니다. 연안 5km는 현재 항만 경계 대신 프로젝트의 근사 항만 사각형 가장자리에서 계산하므로 공식 항만 경계 데이터와 동일하지 않습니다.

## 데이터 파일 정책

GeoTIFF와 생성 PNG는 파일이 크고 다시 생성할 수 있으므로 Git에 올리지 않습니다.
각 팀원은 동일한 STAC 카탈로그와 설정을 이용해 CDSE에서 다시 생성합니다.

- Git에 포함: 코드, 문서, 장면 목록 CSV, 설정 JSON, 작은 예시 manifest
- Git에서 제외: OAuth Secret, `.env`, GeoTIFF, PNG, 실행 로그, 로컬 `outputs/`

## 현재 한계

- Sentinel-1은 연속 실시간 영상이 아니며 인천 동일 궤도 관측은 보통 약 12일 간격입니다.
- 학습 풀 78장은 서로 다른 궤도 기하를 포함하므로 픽셀 단위 변화 비교에는 부적합합니다.
- 공개 AIS만으로 과거의 모든 선박을 완전하게 복원할 수 없습니다.
- AIS를 장시간 끈 선박은 보간으로 정답 위치를 만들 수 없습니다.
- 현재 CFAR 결과는 후보 생성 단계이며 Dark Vessel 확정 판정이 아닙니다.
- 부산 시험 GeoTIFF는 유효 마스크가 없어 학습에서 제외해야 합니다.

## 권장 다음 단계

1. 동일 궤도 인천 GeoTIFF를 우선 수집
2. SAR 촬영시각별 AIS 원시 항적 확보
3. SAR 선박 탐지점과 AIS 보간 위치 매칭
4. `AIS 매칭 / AIS 미매칭 후보 / 고정시설 / 불확실` 라벨 작성
5. SSDD·HRSID 등 공개 SAR 선박 데이터로 탐지기 사전학습
6. 인천 수동 검수 라벨로 추가 학습 및 평가
