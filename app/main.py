# app/main.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from .runtime import PiiModel

# ──────────────────────────────────────────────────────────────
# 환경 변수 로드 & 모델 초기화
# ──────────────────────────────────────────────────────────────
load_dotenv(override=True)

MODEL_DIR = os.getenv("MODEL_DIR", "./models/xlm-roberta-large")
DEVICE = os.getenv("DEVICE")  # 예: "cuda:0" / "cpu" / None(auto)
SCORE_THRESHOLD = os.getenv("SCORE_THRESHOLD")
try:
    SCORE_THRESHOLD_F = float(SCORE_THRESHOLD) if SCORE_THRESHOLD is not None else None
except ValueError:
    SCORE_THRESHOLD_F = None

try:
    MODEL = PiiModel(
        model_dir=MODEL_DIR,
        device=DEVICE,
        aggregation_strategy=os.getenv("AGGREGATION_STRATEGY", "simple"),
        score_threshold=SCORE_THRESHOLD_F,
        warmup=os.getenv("WARMUP", "1") not in {"0", "false", "False"},
    )
except Exception as e:
    # 앱 기동은 하되, 초기화 실패 정보를 담아둔다.
    MODEL = None
    INIT_ERROR = str(e)
else:
    INIT_ERROR = None

# ──────────────────────────────────────────────────────────────
# FastAPI 앱
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="PII Model API",
    version="1.0.0",
    description="토큰분류 기반 PII 스팬 탐지 API",
)

# ──────────────────────────────────────────────────────────────
# Pydantic 스키마
# ──────────────────────────────────────────────────────────────
class InReqSingle(BaseModel):
    text: str = Field(..., description="분석할 원문 텍스트")
    mask: Optional[bool] = Field(False, description="(옵션) 마스킹 결과 동봉 여부 (미구현 자리표시자)")

class InReqBatch(BaseModel):
    texts: List[str] = Field(..., description="분석할 문장 리스트")
    mask: Optional[bool] = Field(False, description="(옵션) 마스킹 결과 동봉 여부 (미구현 자리표시자)")

class OutSpan(BaseModel):
    start: int
    end: int
    label: str
    text: str
    score: Optional[float] = None

class OutSingle(BaseModel):
    text: str
    spans: List[OutSpan]
    masked: Optional[str] = None  # 확장 포인트

# ──────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────
def _ensure_ready():
    if MODEL is None:
        raise HTTPException(status_code=503, detail={"error": "MODEL_NOT_READY", "message": INIT_ERROR or "Model init failed"})

# ── CORS 설정 ────────────────────────────────────────────────────────────────
# 개발 기본(Next/Vite dev) + Swagger 로컬 확인
DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8103",     # 이 API의 Swagger를 브라우저로 열 때
    "http://127.0.0.1:8103",
]

# 배포/개발 환경에서 추가로 허용할 오리진을 쉼표로 주입 가능:
#   CORS_ALLOW_ORIGINS="https://admin.example.com,https://app.example.com"
extra = os.getenv("CORS_ALLOW_ORIGINS", "")
if extra.strip():
    DEFAULT_ORIGINS += [o.strip() for o in extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # credentials=True면 "*" 금지. 정확히 나열
    allow_credentials=False,          # 쿠키/세션/인증 포함 요청 지원
    allow_methods=["*"],             # 필요시 ["GET","POST","PUT","DELETE","OPTIONS"]
    allow_headers=["*"],             # Authorization, Content-Type 등
    # expose_headers=["ETag"],         # 프론트에서 읽어야 하는 응답 헤더 있으면 추가
    # max_age=86400,                   # 프리플라이트 캐시(초)
)
# ──────────────────────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"])
def root() -> Dict[str, Any]:
    return {"ok": True, "service": "PII Model API", "docs": "/docs"}

@app.get("/health", tags=["meta"])
def health() -> Dict[str, Any]:
    return {
        "ok": MODEL is not None,
        "model_dir": MODEL_DIR,
        "device": getattr(MODEL, "device", DEVICE or "auto"),
        "labels": getattr(MODEL, "labels", []),
        "init_error": INIT_ERROR,
    }

@app.get("/labels", tags=["model"])
def labels() -> Dict[str, Any]:
    _ensure_ready()
    return {"labels": MODEL.labels}

@app.post("/infer", response_model=OutSingle, tags=["inference"])
def infer(req: InReqSingle) -> OutSingle:
    _ensure_ready()
    spans = MODEL.infer(req.text)
    return OutSingle(text=req.text, spans=[OutSpan(**s) for s in spans], masked=None)

@app.post("/infer/batch", tags=["inference"])
def infer_batch(req: InReqBatch) -> Dict[str, Any]:
    _ensure_ready()
    results = []
    for t in req.texts:
        spans = MODEL.infer(t)
        results.append({"text": t, "spans": [OutSpan(**s).model_dump() for s in spans]})
    return {"results": results}
