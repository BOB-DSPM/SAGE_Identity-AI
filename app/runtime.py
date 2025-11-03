# app/runtime.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import os

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)


class PiiModel:
    """
    토큰 분류(PII 스팬 탐지) 모델을 로드해 스팬(start/end/label/text/score)을 반환하는 런타임 래퍼.
    - Hugging Face 로컬 체크포인트(디렉터리)에서 직접 로드
    - FastAPI 엔드포인트에서 재사용할 수 있도록 경량 API 제공
    """

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = None,
        aggregation_strategy: str = "simple",
        score_threshold: Optional[float] = None,
        warmup: bool = True,
    ):
        """
        Args:
            model_dir: 모델/토크나이저 파일이 있는 로컬 디렉터리 (config.json, model.safetensors, tokenizer.json 등)
            device:  "cuda:0" / "cpu" 등 지정. 기본(None)이면 자동 감지
            aggregation_strategy: pipeline에서 토큰→엔티티 병합 방식("simple","average","first","max")
            score_threshold: 이 점수 미만 엔티티는 필터링(None이면 필터링 안 함)
            warmup: 초기 1회 더미 추론으로 파이프라인 워밍업
        """
        self.model_dir = model_dir
        self.score_threshold = score_threshold

        # ── 디바이스 자동 감지(선택)
        if device is None:
            try:
                import torch

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        # ── 로드
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        except Exception as e:
            raise RuntimeError(
                f"[PiiModel] 모델/토크나이저 로드 실패: {model_dir}\n"
                f"필요 파일(예: config.json, model.safetensors, tokenizer.json 등)을 확인하세요.\n{e}"
            )

        # ── 추론 파이프라인 (엔티티 병합 포함)
        # device 인수는 정수 GPU index 또는 str("cpu"/"cuda:0") 모두 허용됨
        pipe_device = 0 if self.device.startswith("cuda") else -1
        self.pipe = pipeline(
            task="token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=aggregation_strategy,
            device=pipe_device,
        )

        # ── 라벨 목록(id2label에서 추출)
        cfg = self.model.config
        self.labels = sorted(set(cfg.id2label.values())) if hasattr(cfg, "id2label") else []

        # ── 워밍업(초기 응답 지연 줄이기)
        if warmup:
            try:
                _ = self.pipe("warmup")  # 짧은 더미 텍스트
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────
    # Public APIs
    # ──────────────────────────────────────────────────────────────────────
    def infer(self, text: str) -> List[Dict[str, Any]]:
        """
        단일 문장 추론 → 스팬 리스트 반환
        """
        ents = self.pipe(text)  # [{'entity_group','score','start','end','word'}, ...]
        spans = []
        for e in ents:
            score = float(e.get("score", 0.0))
            if self.score_threshold is not None and score < self.score_threshold:
                continue
            start, end = int(e["start"]), int(e["end"])
            label = str(e.get("entity_group") or e.get("entity") or "UNK")
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": text[start:end],
                    "score": score,
                }
            )
        return spans

    def infer_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        배치 추론 → 각 텍스트별 스팬 리스트
        """
        results: List[List[Dict[str, Any]]] = []
        # 파이프라인은 리스트 입력도 받지만, 결과 후처리를 명확히 하기 위해 한 개씩 처리
        for t in texts:
            results.append(self.infer(t))
        return results
