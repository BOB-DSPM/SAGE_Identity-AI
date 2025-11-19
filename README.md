# SAGE Identity-AI

XLM-RoBERTa 기반 토큰 분류 모델을 사용하여 텍스트에서 이메일, 전화번호 등 개인식별정보(PII)를 실시간으로 탐지하는 FastAPI 서비스입니다.

## 주요 기능

- **PII 스팬 탐지**: 이메일, 전화번호, 이름, 주소 등 개인정보 위치와 유형을 정확하게 식별
- **다국어 지원**: XLM-RoBERTa 기반으로 한국어, 영어 등 100개 이상 언어 처리
- **배치 처리**: 여러 문장을 한 번에 분석할 수 있는 배치 엔드포인트 지원
- **CORS 설정**: 웹 애플리케이션과의 통합을 위한 CORS 지원
- **자동 문서화**: Swagger UI를 통한 대화형 API 문서 제공
- **스코어 필터링**: 신뢰도 임계값을 통한 정밀한 탐지 제어

## 빠른 시작

### 원클릭 배포
```bash
#!/bin/bash

# 기존 프로세스 종료
PID=$(lsof -ti tcp:8900 || true)
if [ -n "$PID" ]; then
  echo "포트 8900 사용 중 -> PID: $PID 종료"
  sudo kill -9 $PID
fi

# 프로젝트 클론
git clone https://github.com/BOB-DSPM/SAGE_Identity-AI
cd SAGE_Identity-AI

# Python 가상환경 설정
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 모델 다운로드 및 압축 해제
sudo apt install wget tar zstd -y
wget https://github.com/BOB-DSPM/SAGE_Identity-AI/releases/download/v0.1.0/xlmr-large-min.tar.zst
tar --zstd -xf xlmr-large-min.tar.zst

# 환경 변수 설정 및 서버 시작
export MODEL_DIR=./xlmr-large-min
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8900 > iden-ai.log 2>&1 & echo $! > iden-ai.pid

echo "✅ 서버 시작 완료!"
echo "📍 API 문서: http://localhost:8900/docs"
```

### 환경 구성
```bash
# 가상환경 설정
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU 사용 시 (CUDA 12.1)
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 서버 실행
```bash
# 로컬 실행 (포트 8900)
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8900 --reload

# 백그라운드 실행
nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8900 > iden-ai.log 2>&1 & echo $! > iden-ai.pid
```

API 문서: http://localhost:8900/docs

## 프로젝트 구조
```
SAGE_Identity-AI/
├── app/
│   ├── __init__.py           # 패키지 초기화
│   ├── main.py               # FastAPI 엔드포인트
│   └── runtime.py            # PiiModel 클래스 (모델 로드 및 추론)
├── tests/
│   ├── __init__.py
│   └── test_smoke.py         # 스모크 테스트 (FakeModel 사용)
├── xlmr-large-min/           # 모델 파일 디렉터리 (압축 해제 후)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── .env.example              # 환경 변수 템플릿
├── .gitignore                # Git 제외 파일 목록
├── dockerfile                # Docker 이미지 빌드 파일
├── README.md                 # 프로젝트 문서
├── requirements.txt          # Python 의존성 목록
├── run.sh                    # 실행 스크립트
├── xlmr-large-min.tar.zst    # 모델 압축 파일
└── xlmr-large-min.tar.zst.sha256
```

## API 엔드포인트

### 기본 정보
```bash
# 루트
GET /

# Health Check
GET /health

# 지원 라벨 조회
GET /labels
```

### PII 탐지
```bash
# 단일 텍스트 분석
POST /infer
Content-Type: application/json
{
  "text": "연락처는 alice@example.com 또는 010-1234-5678입니다.",
  "mask": false
}

# 배치 분석
POST /infer/batch
Content-Type: application/json
{
  "texts": [
    "문의: support@company.com",
    "전화번호는 02-1234-5678",
    "개인정보 없는 텍스트"
  ],
  "mask": false
}
```

## 응답 예시

### Health Check
```json
{
  "ok": true,
  "model_dir": "./xlmr-large-min",
  "device": "cpu",
  "labels": ["EMAIL", "PHONE", "PERSON", "ADDRESS"],
  "init_error": null
}
```

### 단일 분석
```json
{
  "text": "연락처는 alice@example.com 또는 010-1234-5678입니다.",
  "spans": [
    {
      "start": 6,
      "end": 24,
      "label": "EMAIL",
      "text": "alice@example.com",
      "score": 0.9876
    },
    {
      "start": 29,
      "end": 42,
      "label": "PHONE",
      "text": "010-1234-5678",
      "score": 0.9654
    }
  ],
  "masked": null
}
```

### 배치 분석
```json
{
  "results": [
    {
      "text": "문의: support@company.com",
      "spans": [
        {
          "start": 4,
          "end": 23,
          "label": "EMAIL",
          "text": "support@company.com",
          "score": 0.9912
        }
      ]
    },
    {
      "text": "전화번호는 02-1234-5678",
      "spans": [
        {
          "start": 7,
          "end": 19,
          "label": "PHONE",
          "text": "02-1234-5678",
          "score": 0.9543
        }
      ]
    },
    {
      "text": "개인정보 없는 텍스트",
      "spans": []
    }
  ]
}
```

## 환경 변수 설정

`.env` 파일 생성:
```bash
MODEL_DIR=./xlmr-large-min
DEVICE=cpu                    # cpu, cuda:0, auto
AGGREGATION_STRATEGY=simple   # simple, average, first, max
SCORE_THRESHOLD=0.5           # 최소 신뢰도 점수 (선택)
WARMUP=1                      # 초기 워밍업 여부 (0, 1)
CORS_ALLOW_ORIGINS=           # 추가 허용 오리진 (쉼표 구분)
```

| 변수명 | 설명 | 기본값 | 예시 |
|--------|------|--------|------|
| `MODEL_DIR` | 모델 파일 디렉터리 경로 | `./models/xlm-roberta-large` | `./xlmr-large-min` |
| `DEVICE` | 추론 디바이스 | 자동 감지 | `cpu`, `cuda:0` |
| `AGGREGATION_STRATEGY` | 토큰 병합 전략 | `simple` | `simple`, `average`, `first`, `max` |
| `SCORE_THRESHOLD` | 최소 신뢰도 점수 필터링 | 없음 | `0.5`, `0.7` |
| `WARMUP` | 초기 워밍업 추론 실행 | `1` | `0`, `1`, `false` |
| `CORS_ALLOW_ORIGINS` | 추가 허용 오리진 | 로컬 주소 | `https://example.com` |

## 데이터 처리 흐름
```
1. 요청 수신 (FastAPI)
   ↓
2. 입력 검증 (Pydantic)
   ↓
3. 토크나이저 처리 (AutoTokenizer)
   ↓
4. 모델 추론 (AutoModelForTokenClassification)
   ↓
5. 토큰 병합 (Pipeline aggregation_strategy)
   ↓
6. 스코어 필터링 (SCORE_THRESHOLD)
   ↓
7. 스팬 정보 생성 (start/end/label/text/score)
   ↓
8. JSON 응답 (FastAPI)
```

## Frontend 연동

### API 호출 예시 (TypeScript)
```typescript
// 단일 분석
const response = await fetch('http://localhost:8900/infer', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: '연락처: alice@example.com'
  })
});
const data = await response.json();

// 배치 분석
const batchResponse = await fetch('http://localhost:8900/infer/batch', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    texts: ['텍스트1', '텍스트2', '텍스트3']
  })
});
const batchData = await batchResponse.json();
```

### Python 클라이언트
```python
import requests

BASE_URL = "http://localhost:8900"

# 단일 분석
response = requests.post(
    f"{BASE_URL}/infer",
    json={"text": "연락처: alice@example.com"}
)
result = response.json()

# 배치 분석
response = requests.post(
    f"{BASE_URL}/infer/batch",
    json={"texts": ["텍스트1", "텍스트2"]}
)
results = response.json()
```


## 테스트 예시
```bash
# 1. Health Check
curl http://localhost:8900/health

# 2. 지원 라벨 조회
curl http://localhost:8900/labels

# 3. 단일 텍스트 분석
curl -X POST http://localhost:8900/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "메일: test@example.com"}'

# 4. 배치 분석
curl -X POST http://localhost:8900/infer/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["이메일: bob@test.io", "전화: 010-9876-5432"]}'

# 5. API 문서 확인
open http://localhost:8900/docs
```

## 테스트 실행
```bash
# pytest 설치 (requirements.txt에 포함)
pip install pytest httpx

# 전체 테스트 실행
pytest

# 상세 출력
pytest -v

# 특정 테스트 파일만 실행
pytest tests/test_smoke.py

# 커버리지 확인
pytest --cov=app tests/
```

### 테스트 구조

`tests/test_smoke.py`는 실제 모델 없이 **FakeModel**을 사용하여 API 엔드포인트를 테스트합니다:

- 정규식 기반 간단한 EMAIL/PHONE 탐지
- `/health`, `/labels`, `/infer`, `/infer/batch` 엔드포인트 검증
- 응답 구조 및 데이터 형식 확인
- 배치 처리 로직 검증

## 트러블슈팅

### 모델 로드 실패
```bash
# 모델 파일 확인
ls -lh xlmr-large-min/
# config.json, model.safetensors, tokenizer.json 등이 있어야 함

# SHA256 체크섬 검증
sha256sum -c xlmr-large-min.tar.zst.sha256

# 재다운로드
rm -rf xlmr-large-min xlmr-large-min.tar.zst
wget https://github.com/BOB-DSPM/SAGE_Identity-AI/releases/download/v0.1.0/xlmr-large-min.tar.zst
tar --zstd -xf xlmr-large-min.tar.zst
```

### 포트 충돌
```bash
# 사용 중인 프로세스 확인
lsof -i :8900

# 프로세스 종료
kill $(lsof -ti tcp:8900)

# 다른 포트 사용
uvicorn app.main:app --port 8901 --reload
```

### CUDA Out of Memory
```bash
# CPU 모드로 전환
export DEVICE=cpu

# 배치 크기 줄이기 (코드 수정 필요)
# runtime.py에서 배치 처리 시 chunk 단위로 분할
```

### CORS 오류
```bash
# .env 파일에 오리진 추가
CORS_ALLOW_ORIGINS=https://myapp.example.com,https://admin.example.com

# 또는 main.py에서 allow_origins 수정
allow_origins=["*"]  # 개발 환경에서만 사용
```

### 느린 응답 속도
```bash
# 워밍업 활성화 (기본값)
export WARMUP=1

# GPU 사용 (CUDA 설치 필요)
export DEVICE=cuda:0
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 워커 수 증가 (프로덕션)
uvicorn app.main:app --workers 4 --port 8900
```

## 프로덕션 배포

### 권장 사항
- **리버스 프록시**: Nginx 또는 Traefik을 통한 HTTPS 설정
- **프로세스 관리**: systemd, Supervisor, PM2로 자동 재시작
- **로깅**: 구조화된 로깅 및 모니터링 도구 연동
- **보안**: API 키 인증, Rate Limiting 추가
- **성능**: GPU 사용 시 배치 크기 조정 및 워커 수 증가

### Systemd 서비스 예시
```bash
# /etc/systemd/system/sage-identity-ai.service
[Unit]
Description=SAGE Identity-AI Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/SAGE_Identity-AI
Environment="MODEL_DIR=/home/ubuntu/SAGE_Identity-AI/xlmr-large-min"
Environment="DEVICE=cpu"
ExecStart=/home/ubuntu/SAGE_Identity-AI/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8900
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 등록 및 시작
sudo systemctl daemon-reload
sudo systemctl enable sage-identity-ai
sudo systemctl start sage-identity-ai
sudo systemctl status sage-identity-ai
```