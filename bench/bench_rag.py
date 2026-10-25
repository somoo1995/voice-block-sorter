"""임베딩·리랭크 한국어 호출 지연 실측(OVMS /v3/embeddings, /v3/rerank).

사용: .venv/bin/python bench/bench_rag.py
결과: bench/out/rag.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import requests

BASE = "http://127.0.0.1:8100/v3"
OUT = Path(__file__).resolve().parent / "out"
EMB, RR = "qwen3-embedding", "qwen3-reranker"

QUERY = "찢어진 공은 어떻게 처리해야 해?"
DOCS = [
    "불량품 판정: 겉면이 찢어지거나 솜이 삐져나온 공, 지름이 기준보다 2cm 이상 작은 공은 불량으로 분류해 불량 그릇에 넣는다.",
    "빨간 공과 파란 공은 색깔별 그릇에 나눠 담는다. 그릇이 가득 차면 작업을 멈추고 알린다.",
    "로봇 팔이 움직이는 동안에는 작업대 안으로 손을 넣지 않는다.",
    "작업이 끝나면 옮긴 공의 개수와 불량 개수를 음성으로 보고한다.",
    "사람에게 공을 건넬 때는 손 자리에 손이 보일 때만 그리퍼를 연다.",
    "카메라 렌즈에 먼지가 묻으면 검출 정확도가 떨어지므로 마른 천으로 닦는다.",
    "전원을 끌 때는 정책 실행이 끝났는지 먼저 확인한다.",
    "부직포 공은 세탁하지 않는다. 오염된 공은 불량으로 본다.",
]


def timed(fn, n=5):
    fn()  # 예열
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


# Qwen3 임베딩·리랭커는 지시문 템플릿을 클라이언트가 붙여야 한다(OVMS 는 원문 그대로 넣음). 모델 카드 형식.
TASK = "Given a web search query, retrieve relevant passages that answer the query"
RR_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
             'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
RR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def embed(texts):
    r = requests.post(f"{BASE}/embeddings", json={"model": EMB, "input": texts}, timeout=120)
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def rerank(query, docs, template):
    if template:
        query = f"{RR_PREFIX}<Instruct>: {TASK}\n<Query>: {query}\n"
        docs = [f"<Document>: {d}{RR_SUFFIX}" for d in docs]
    r = requests.post(f"{BASE}/rerank", json={"model": RR, "query": query, "documents": docs}, timeout=120)
    r.raise_for_status()
    return r.json()["results"]


def main(out_name):
    res = {}
    t8, v8 = timed(lambda: embed(DOCS))
    for template in (False, True):
        tag = "템플릿" if template else "원문"
        q_text = f"Instruct: {TASK}\nQuery:{QUERY}" if template else QUERY
        t1, v1 = timed(lambda: embed([q_text]))
        sims = np.array(v8) @ np.array(v1[0])
        top = int(sims.argmax())
        res[f"embed_{'tpl' if template else 'raw'}"] = {
            "dim": len(v1[0]), "single_ms": t1 * 1000, "batch8_ms": t8 * 1000,
            "top_doc": top, "top_ok": top == 0, "cos": sims.round(3).tolist()}
        print(f"[임베딩·{tag}] 차원 {len(v1[0])} | 질문 1문장 {t1*1000:.0f}ms | 문서 8개 묶음 {t8*1000:.0f}ms | "
              f"1위 #{top} ({'정답' if top == 0 else '오답'}) cos {sims.round(3).tolist()}")

        tr5, _ = timed(lambda: rerank(QUERY, DOCS[:5], template))
        tr8, r8 = timed(lambda: rerank(QUERY, DOCS, template))
        best = max(r8, key=lambda x: x["relevance_score"])
        scores = sorted(((x["index"], round(x["relevance_score"], 4)) for x in r8), key=lambda t: -t[1])
        res[f"rerank_{'tpl' if template else 'raw'}"] = {
            "docs5_ms": tr5 * 1000, "docs8_ms": tr8 * 1000, "top_doc": best["index"], "top_ok": best["index"] == 0,
            "scores": scores}
        print(f"[리랭크·{tag}] 문서 5개 {tr5*1000:.0f}ms | 문서 8개 {tr8*1000:.0f}ms | 1위 #{best['index']} "
              f"({'정답' if best['index'] == 0 else '오답'}) 상위 {scores[:4]}")

    OUT.mkdir(exist_ok=True)
    (OUT / out_name).write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--emb", default=EMB)
    ap.add_argument("--rr", default=RR)
    ap.add_argument("--out", default="rag.json")
    a = ap.parse_args()
    BASE, EMB, RR = a.base, a.emb, a.rr
    main(a.out)
