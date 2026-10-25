"""LLM·임베딩·리랭크 클라이언트(OVMS OpenAI 호환 주소)."""
import functools
import time

import httpx
from langchain_core.embeddings import Embeddings

from . import config


_info: tuple[float, dict] | None = None


def model_info() -> dict:
    """{name, wanted, fallback}. 서버 목록을 60초마다 다시 본다(인프라가 모델을 바꿔 띄우는 일이 잦았다, 10-21)."""
    global _info
    if _info and time.monotonic() - _info[0] < 60:
        return _info[1]
    _info = (time.monotonic(), _model_info())
    return _info[1]


def _model_info() -> dict:
    wanted = config.LLM_MODEL
    try:
        r = httpx.get(f"{config.LLM_BASE_URL}/models", timeout=3)
        ids = [m["id"] for m in r.json().get("data", [])]
    except Exception as e:                  # 목록을 못 읽으면 지정값 그대로(호출 때 오류가 난다)
        return {"name": wanted, "wanted": wanted, "fallback": f"목록 확인 실패: {type(e).__name__}"}
    if wanted in ids:
        return {"name": wanted, "wanted": wanted, "fallback": None}
    for name in config.LLM_FALLBACKS:
        if name in ids:
            return {"name": name, "wanted": wanted, "fallback": f"{wanted} 가 목록에 없어 {name} 사용"}
    return {"name": wanted, "wanted": wanted, "fallback": f"목록에 {wanted}·대체 모델 없음: {ids}"}


def model_name() -> str:
    return model_info()["name"]


def chat(max_tokens: int = 512, timeout: float | None = None):
    return _chat(model_name(), max_tokens, timeout or config.LLM_TIMEOUT_S)


@functools.cache
def _chat(model: str, max_tokens: int, timeout: float):
    from langchain_openai import ChatOpenAI

    extra = {} if config.LLM_THINKING else {"chat_template_kwargs": {"enable_thinking": False}}
    return ChatOpenAI(
        model=model, base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
        temperature=0, max_tokens=max_tokens, timeout=timeout, max_retries=0,
        extra_body=extra,
    )


class QwenEmbeddings(Embeddings):
    """Qwen3-Embedding: 질의에만 지시문을 붙이고 문서는 그대로 넣는다."""

    def __init__(self, instruct: str):
        from langchain_openai import OpenAIEmbeddings

        self.instruct = instruct
        kw = dict(model=config.EMB_MODEL, base_url=config.EMB_BASE_URL, api_key=config.LLM_API_KEY,
                  check_embedding_ctx_length=False, max_retries=1)
        self.base = OpenAIEmbeddings(timeout=20, **kw)            # 질의 한 문장
        # 색인(문서 조각·라우터 예문): 16개씩, 넉넉한 시간. 세 문서 99조각은 조각당 0.2~0.9초(CPU)에 쉬고 난 첫 요청이
        # 18초라 한 번에 보내면 20초를 넘었다(10-22)
        self.docs = OpenAIEmbeddings(timeout=120, chunk_size=16, **kw)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.docs.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        q = f"Instruct: {self.instruct}\nQuery: {text}" if self.instruct else text
        return self.base.embed_query(q)


# Qwen3-Reranker(seq-cls 변환본)는 이 틀을 씌워야 점수가 제대로 갈린다(10-21 실측: 0.83 vs 틀 없이 0.02)
_RR_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the '
              'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
              '<|im_end|>\n<|im_start|>user\n')
_RR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def rerank(query: str, documents: list[str], instruct: str, top_n: int | None = None) -> list[tuple[int, float]]:
    """OVMS /v3/rerank(Cohere 형식). [(문서 번호, 점수)] 점수 내림차순."""
    body = {
        "model": config.RERANK_MODEL,
        "query": f"{_RR_PREFIX}<Instruct>: {instruct}\n<Query>: {query}\n",
        "documents": [f"<Document>: {d}{_RR_SUFFIX}" for d in documents],
    }
    if top_n:
        body["top_n"] = top_n
    r = httpx.post(f"{config.EMB_BASE_URL}/rerank", json=body, timeout=30)
    r.raise_for_status()
    res = [(x["index"], float(x["relevance_score"])) for x in r.json()["results"]]
    return sorted(res, key=lambda t: -t[1])
