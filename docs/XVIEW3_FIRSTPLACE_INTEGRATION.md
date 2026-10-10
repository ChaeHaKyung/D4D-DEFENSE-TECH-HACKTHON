# xView3 First Place Colab GPU 연결

로컬 Streamlit 앱의 **Colab GPU에서 xView3 추론** 버튼을 누르면 현재 선택된 SAR GeoTIFF가 Colab GPU로 전송되고, 추론 결과가 AIS와 매칭되어 지도와 표에 표시됩니다. Colab 런타임은 앱을 쓰는 동안 계속 실행 상태여야 합니다.

## 1. Colab GPU API 시작

1. [GitHub의 인천 추론 노트북](https://github.com/ChaeHaKyung/D4D-DEFENSE-TECH-HACKTHON/blob/main/XView3_FirstPlace_Incheon_Inference.ipynb)을 열고 **Open in Colab**을 선택합니다.
2. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
3. 프로젝트의 `src/xview3_colab_gpu_api.py`를 Colab 왼쪽 파일 창에 업로드합니다.
4. **원본 노트북 셀은 실행하지 않아도 됩니다.** 아래 API 스크립트가 공식 모델을 직접 GPU에 로드합니다. 같은 GPU에서 원본 노트북의 모델을 먼저 올리면 메모리가 부족할 수 있습니다.
5. 새 코드 셀에서 다음을 실행합니다.

```python
%pip -q install rasterio tqdm
!python /content/xview3_colab_gpu_api.py
```

API 스크립트는 공식 약 1.3GB 모델을 GPU 메모리에 한 번 로드하고 Cloudflare 임시 HTTPS 터널을 엽니다. 셀 출력에 `XVIEW3_API_URL`과 `XVIEW3_API_TOKEN`이 나타나면 이 두 값을 로컬 터미널에 복사합니다. **셀을 중지하거나 Colab 런타임을 종료하면 API가 끊어집니다.** 토큰이 외부에 노출되면 Colab 셀을 중지하고 다시 시작해 새 토큰을 발급하세요.

공식 xView3 노트북은 추론 로직을 확인하고 단독 실행할 때 사용할 수 있습니다. 앱의 자동 연결 버튼은 노트북 셀의 변수를 공유하지 않고, 별도 API 프로세스에서 같은 공개 TorchScript 앙상블을 실행합니다.

## 2. 로컬 앱에서 API 접속

Colab API 셀은 계속 실행한 상태로 두고, 새 로컬 터미널을 열어 프로젝트 루트에서 아래를 실행합니다. 따옴표 안에 Colab 셀에서 출력된 실제 URL과 토큰을 넣습니다.

```bash
cd "/Users/jiwon/Downloads/D4D-DEFENSE-TECH-HACKTHON 3"
python3 -m pip install -r requirements.txt
export XVIEW3_API_URL='https://Colab에서-출력된-주소.trycloudflare.com'
export XVIEW3_API_TOKEN='Colab에서-출력된-토큰'
python3 -m streamlit run app.py
```

HTTPS 연결에 사용하는 CA 인증서는 `certifi`로 확인합니다. 조직 프록시 또는 사용자 지정 CA를 쓰는 네트워크에서는 신뢰할 수 있는 CA PEM 파일 경로를 `SSL_CERT_FILE`에 지정하세요. TLS 검증을 끄지 마세요.

화면 사이드바에서 **Colab GPU에서 xView3 추론**을 누릅니다. 앱은 `outputs/latest_stac_item.json`의 장면 ID와 일치하는 `outputs/datasets/incheon/sar/geotiff_raw/` 또는 `outputs/`의 원본 GeoTIFF를 찾아 업로드합니다. 모델 추론은 Colab에서, AIS 매칭과 지도 표시는 로컬에서 처리합니다. 추론이 끝나면 후보 레이어가 자동으로 갱신됩니다. 12개 모델 앙상블이라 장면에 따라 수 분 걸릴 수 있습니다.

필요한 장면 GeoTIFF가 없다는 안내가 나오면 해당 장면의 원본 데이터셋 GeoTIFF를 먼저 내려받으세요. 입력 파일은 scene ID가 포함된 원래 파일명을 유지해야 합니다.

## 주의

- 이 API는 xView3 모델을 재학습하지 않고, 공개된 사전학습 앙상블로 추론만 합니다.
- 공개 앙상블은 약 1.3GB를 다운로드하며 GPU 메모리를 많이 사용합니다. GPU 메모리가 부족하면 Colab에서 더 큰 GPU 런타임이 필요할 수 있습니다.
- 모델 입력은 Band 2 VH, Band 1 VV의 dB 데이터로 처리합니다. 현재 데이터셋의 GeoTIFF는 Band 1 VV, Band 2 VH, Band 3 dataMask입니다.
- 2048 타일과 1536 간격의 균등 평균 병합은 공식 대회 추론과 완전히 같지 않습니다. 인천에서의 성능은 별도 검증이 필요합니다.
- Colab Quick Tunnel 주소는 임시 주소이며 세션을 재시작하면 바뀝니다. 로컬 앱 실행 전 URL과 토큰을 새 값으로 설정하세요.
- 요청은 인증 토큰으로 보호되며, Colab 서버는 업로드된 GeoTIFF를 임시 폴더에서 처리한 뒤 삭제합니다. 민감한 입력을 공개 터널에 보낼 때는 주의하세요.
- xView3 결과는 `outputs/sar_ship_candidates.csv`로 저장됩니다. 기존 파이프라인의 `analyze` 또는 `refresh` 실행 시 기존 탐지 결과로 덮어씌워질 수 있습니다.
- GeoTIFF, 탐지 결과 CSV 및 모델 파일은 용량이 크거나 재생성 가능한 산출물이므로 Git에 커밋하지 마세요.
