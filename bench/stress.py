"""안정성 시험: LLM 2줄·임베딩/리랭크 1줄·STT 1줄을 동시에 N초 돌려 오류·GPU 행이 없는지 본다.

사용: .venv/bin/python bench/stress.py [초=90] [LLM모델=qwen3-4b]
"""
import random
import sys
import threading
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8100/v3"
OUT = Path(__file__).resolve().parent / "out"
SECS = int(sys.argv[1]) if len(sys.argv) > 1 else 90
LLM = sys.argv[2] if len(sys.argv) > 2 else "qwen3-4b"
TEXTS = ["불량품 골라내 줘", "빨간 볼 옮겨 줘", "파란 공 다 담아", "빨간 거 하나 내 손에 줘", "결과 알려 줘"]
stats = {}
lock = threading.Lock()


def note(kind, ok, err=None):
    with lock:
        s = stats.setdefault(kind, {"ok": 0, "fail": 0, "errors": []})
        s["ok" if ok else "fail"] += 1
        if err and len(s["errors"]) < 3:
            s["errors"].append(err[:160])


def loop(kind, fn, end):
    while time.time() < end:
        try:
            fn()
            note(kind, True)
        except Exception as e:  # 실패도 세고 계속
            note(kind, False, repr(e))
            time.sleep(1)


def llm():
    r = requests.post(f"{BASE}/chat/completions", json={
        "model": LLM, "messages": [{"role": "user", "content": random.choice(TEXTS) + " 라고 했어. 로봇이 할 일을 설명해."}],
        "max_tokens": random.choice([32, 96, 192]), "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
    r.raise_for_status()


def rag():
    r = requests.post(f"{BASE}/embeddings", json={"model": "qwen3-embedding", "input": TEXTS}, timeout=120)
    r.raise_for_status()
    r = requests.post(f"{BASE}/rerank", json={"model": "qwen3-reranker", "query": "불량 공", "documents": TEXTS}, timeout=120)
    r.raise_for_status()


def stt():
    with open(OUT / f"tts_{random.randrange(6):02d}.wav", "rb") as f:
        r = requests.post(f"{BASE}/audio/transcriptions", files={"file": f},
                          data={"model": "whisper", "language": "ko"}, timeout=300)
    r.raise_for_status()


if __name__ == "__main__":
    end = time.time() + SECS
    ths = [threading.Thread(target=loop, args=(k, f, end)) for k, f in
           [("llm-a", llm), ("llm-b", llm), ("embed+rerank", rag), ("stt", stt)]]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for k, s in stats.items():
        print(f"{k:13} 성공 {s['ok']:4} 실패 {s['fail']:3} {s['errors']}")
