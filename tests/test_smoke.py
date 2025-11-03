# tests/test_smoke.py
import re
import pytest
from fastapi.testclient import TestClient

# 앱 임포트 (주의: app.main에서 전역 MODEL을 생성하므로, 테스트에서 바로 교체합니다)
from app.main import app

class FakeModel:
    """무거운 실제 모델 대신 테스트용 가짜 모델.
    간단한 정규식으로 EMAIL/PHONE 스팬을 흉내냅니다.
    """
    def __init__(self):
        self.labels = ["EMAIL", "PHONE"]

    def infer(self, text: str):
        spans = []

        # 이메일 탐지 (아주 단순한 패턴)
        for m in re.finditer(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text):
            spans.append({
                "start": m.start(),
                "end": m.end(),
                "label": "EMAIL",
                "text": text[m.start():m.end()],
                "score": 0.99,
            })

        # 한국 휴대전화 숫자 형태(간단 예시): 010-1234-5678 or 01012345678
        for m in re.finditer(r"(?:01[016789])[- ]?\d{3,4}[- ]?\d{4}", text):
            spans.append({
                "start": m.start(),
                "end": m.end(),
                "label": "PHONE",
                "text": text[m.start():m.end()],
                "score": 0.98,
            })

        return spans

@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    """모든 테스트에서 전역 MODEL을 FakeModel로 교체."""
    import app.main as main_mod
    main_mod.MODEL = FakeModel()
    yield
    # 필요 시 원복 로직 추가 가능

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    # 가짜 모델이 주입되었으므로 labels 노출 확인
    assert "labels" in data and data["labels"] == ["EMAIL", "PHONE"]

def test_labels():
    r = client.get("/labels")
    assert r.status_code == 200
    data = r.json()
    assert data.get("labels") == ["EMAIL", "PHONE"]

def test_infer_single_email_and_phone():
    payload = {"text": "메일 alice@example.com / 번호 010-1234-5678 입니다."}
    r = client.post("/infer", json=payload)
    assert r.status_code == 200
    out = r.json()
    assert out["text"] == payload["text"]
    spans = out["spans"]
    labels = {s["label"] for s in spans}
    assert "EMAIL" in labels
    assert "PHONE" in labels
    # 스팬 기본 필드 확인
    for s in spans:
        assert {"start", "end", "label", "text"}.issubset(s.keys())
        assert isinstance(s["start"], int) and isinstance(s["end"], int)

def test_infer_batch():
    payload = {"texts": ["no pii here", "contact: bob@test.io", "전화 01012345678"]}
    r = client.post("/infer/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "results" in data and len(data["results"]) == len(payload["texts"])
    # 각 결과 구조 확인
    for item in data["results"]:
        assert "text" in item and "spans" in item
    # 두 번째 문장에서는 EMAIL, 세 번째 문장에서는 PHONE이 나와야 함
    labels_1 = {s["label"] for s in data["results"][0]["spans"]}
    labels_2 = {s["label"] for s in data["results"][1]["spans"]}
    labels_3 = {s["label"] for s in data["results"][2]["spans"]}
    assert labels_1 == set()                    # no pii
    assert "EMAIL" in labels_2                  # bob@test.io
    assert "PHONE" in labels_3                  # 01012345678
