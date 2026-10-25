"""문서 RAG: 작업 기준서·로봇 운용 설명서·시스템 안내서(10-22 세 문서) → 절 단위 조각 → Chroma(OVMS 임베딩)
→ (필요할 때만) OVMS 리랭크.

LangChain 부품으로 나눠 둔다(README "LangGraph 가 흐름, LangChain 이 부품"):
  HybridRetriever(BaseRetriever)            1단계: 임베딩 코사인 순위 + 글자 2-gram BM25 순위의 RRF
  QwenRerankCompressor(BaseDocumentCompressor) 2단계: OVMS Qwen3-Reranker 로 앞 몇 개만 다시 줄 세움(확실하면 생략)
  RerankRetriever(BaseRetriever)            1+2단계를 하나로(= LangChain ContextualCompressionRetriever 와 같은 일.
                                            그 클래스는 langchain 1.x 에서 langchain-classic 으로 빠졌는데 .venv 에 없어 core 부품으로)
리랭커 지연이 후보 수에 비례해서(10-21 실측)
- 임베딩 1위가 충분히 높고 2위와 벌어져 있으면 리랭크를 건너뛴다(RAG_SKIP_SCORE·RAG_SKIP_MARGIN)
- 리랭크해도 합친 순위 상위 RAG_RERANK_N 개만
- 보고(report)에 붙일 조항은 기준서가 바뀔 때 한 번 계산해 캐시한다(citations, 기준서에서만)
"""
import hashlib
import json
import math
import re
import threading
import time
from typing import Any, Optional

import numpy as np
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.retrievers import BaseRetriever

from . import config, llm

RAG_INSTRUCT = ("Given a question about a block-sorting robot's work standard, robot manual or system guide, "
                "retrieve passages that answer it")

# 말과 문서의 낱말 차이. 기준서는 로봇을 "팔(로봇팔)"로 쓴다 → "로봇이 안 멈추면?" 이 6.4(팔이 멈추지 않으면)를
# 못 찾았다(10-21: 임베딩 10위, 리랭크 0.35). 검색할 때만 넓히고 LLM 에 주는 질문은 원문 그대로다
SYNONYMS = [(re.compile(r"로봇(?!\s*(?:팔|운용))"), "로봇 팔"),
            # 10-21 밤 물체가 볼 → 스펀지 블록. 받아쓰기 "블럭", 버릇으로 남은 "볼·공"(명사)도 기준서 낱말 "블록"으로
            (re.compile(r"블럭"), "블록"),
            (re.compile(r"(?<![가-힣])(?:볼|공)(?=[은는이가을를도만의]?(?:\s|$|[?.!]))"), "블록"),
            # 10-21 밤 색 빨강·파랑 → 분홍·초록. 문서는 "분홍"·"초록"으로 쓴다
            (re.compile(r"핑크"), "분홍"), (re.compile(r"녹색|연두"), "초록"),
            # 기준서는 "처리 순서"로 쓴다. 세 문서로 넓히자 "작업 순서가 어떻게 돼요?" 가 로봇 설명서 8.3 "첫 실제 실행 순서"로
            # 갔다(10-22, 기준서 5.1 은 합친 순위 6위·리랭크 0.56)
            (re.compile(r"작업\s*순서"), "작업 처리 순서")]
# 세 문서 모두 1장 = "목적과 적용 범위", 맨 앞 = 머리말(docs/README "근거에서 빼도 된다"). 질문이 목적·범위·문서를 묻지 않으면
# 후보에서 뺀다: 99조각에서 이 6조각이 리랭크 자리를 차지해 "손은 어디에" 의 기준서 6.1 이 밀렸다(10-22)
META_CUES = ("목적", "범위", "적용", "기준서", "설명서", "안내서", "문서")


def is_meta(section: str) -> bool:
    return "목적" in section.split(" > ")[0] or " > " not in section

# 보고에 붙일 근거(기준서에서만): 상황 → 검색어. 기준서가 바뀌면 리랭크까지 한 번 돌려 캐시한다(시연 중 실시간 RAG 없음)
CITATION_QUERIES = {
    "report": "결과 보고에 넣을 항목",
    "mismatch": "작업 기록과 카메라로 센 개수가 다를 때",
    "check": "재시도도 실패해 사람에게 확인을 요청할 때",
    "stopped": "멈춘 뒤 다시 시작하는 방법",
    "defect": "불량 판정 기준과 불량을 담는 곳",
    "red": "정상 분홍 블록을 담는 곳과 처리 순서",
    "blue": "정상 초록 블록을 담는 곳과 처리 순서",
    "handoff": "손 전달 방법",
}


def expand_query(q: str) -> str:
    for pat, rep in SYNONYMS:
        q = pat.sub(rep, q)
    return q


def rerank_scores(query: str, docs: list[str]) -> list[tuple[int, float]]:
    """[(문서 번호, 점수)] 내림차순. 리랭크는 원문 질의로 한 번(넓힌 질의는 임베딩·글자 검색에만 쓴다).
    10-21 기준서 하나일 때는 원문·넓힌 질의 평균이 필요했지만("로봇이 안 멈추면" 6.4), 세 문서에서 1장 빼기·코사인 상위 3개
    넣기·anchor 조건을 더한 뒤로는 원문 한 번이 51문항 상위 3개 51/51 로 같고 리랭크가 반으로 줄었다
    (조각당 GPU 약 0.19초, 두 번 보내면 GPU 에서 차례로 돌아 동시에 보내도 이득이 없었다. 10-22 01:28 비교)."""
    return llm.rerank(query, docs, RAG_INSTRUCT)


def chunk_markdown(text: str, max_chars: int = 600) -> list[dict]:
    """제목(#) 경로를 붙여 절 단위로 자른다. 긴 절은 빈 줄 단위로 max_chars 안에서 묶는다."""
    chunks, stack, body = [], [], []

    def flush():
        paras = [p.strip() for p in "\n".join(body).split("\n\n") if p.strip()]
        # 문서 제목(#)은 빼고 "장 > 조항" 만 붙인다(docs/README: 조항 = ### 하나, 앞에 ## 장 제목)
        section = " > ".join(t for lv, t in stack if lv > 1) or (stack[0][1] if stack else "기준서")
        buf = ""
        for p in paras:
            if buf and len(buf) + len(p) > max_chars:
                chunks.append({"section": section, "text": buf})
                buf = ""
            buf = f"{buf}\n\n{p}" if buf else p
        if buf:
            chunks.append({"section": section, "text": buf})
        body.clear()

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            body.append(line)
    flush()
    return chunks


def _grams(text: str) -> list[str]:
    """한국어용 토큰: 낱말마다 글자 2-gram(낱말이 한 글자면 그 글자). 띄어쓰기·조사 차이에 덜 흔들린다."""
    out = []
    for w in re.findall(r"[0-9A-Za-z가-힣]+", text.lower()):
        out += [w[i:i + 2] for i in range(len(w) - 1)] or [w]
    return out


class _BM25:
    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs, self.lens, self.df = [], [], {}
        for d in docs:
            g = _grams(d)
            tf: dict[str, int] = {}
            for x in g:
                tf[x] = tf.get(x, 0) + 1
            self.tfs.append(tf)
            self.lens.append(len(g))
            for x in tf:
                self.df[x] = self.df.get(x, 0) + 1
        self.n, self.avg = len(docs), sum(self.lens) / max(1, len(docs))

    def scores(self, query: str) -> list[float]:
        qg = set(_grams(query))
        out = []
        for tf, ln in zip(self.tfs, self.lens):
            s = 0.0
            for g in qg:
                if g in tf:
                    idf = math.log(1 + (self.n - self.df[g] + 0.5) / (self.df[g] + 0.5))
                    s += idf * tf[g] * (self.k1 + 1) / (tf[g] + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


VEC_TOP = 3                  # 리랭크 후보에 늘 넣는 임베딩 코사인 상위 개수


def _with(d: Document, **meta) -> Document:
    return Document(page_content=d.page_content, metadata={**d.metadata, **meta})


class HybridRetriever(BaseRetriever):
    """1단계 검색(LangChain BaseRetriever): 세 문서 조각 전체를 임베딩 코사인 순위 + 글자 2-gram BM25 순위의 RRF 로
    줄 세워 앞 k 개. 메타데이터: doc·doc_name·section·clause·title·cos·vec_rank·lex_rank·top_cos·cos_margin."""
    manual: Any = None
    k: int = 8
    doc: Optional[str] = None

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        cands = self.manual.candidates(query, doc=self.doc)
        # 합친 순위 앞 k 개 + 임베딩 순위 앞 VEC_TOP 개(합친 순위에 묻힌 것). 기준서 6.1 은 코사인 2위인데 글자 순위가
        # 36위라 합친 순위 8위 밖이었다("로봇이 일할 때 손은 어디에", 10-22)
        out = cands[:self.k] + [h for h in cands[self.k:] if h["vec_rank"] < VEC_TOP]
        return [Document(page_content=h.pop("text"), metadata=h) for h in out]


class QwenRerankCompressor(BaseDocumentCompressor):
    """2단계(LangChain BaseDocumentCompressor): OVMS Qwen3-Reranker(/v3/rerank)로 앞 rerank_n 개만 다시 줄 세워 k 개.
    임베딩 1위가 확실하면(코사인 ≥ skip_score, 2위와 차 ≥ skip_margin) 리랭크를 건너뛴다. 합친 순위 1위(anchor)는
    리랭크 결과와 상관없이 남긴다(리랭커가 "작업 순서"에서 5.1 보다 1.1 목적을 올림)."""
    k: int = 3
    rerank_n: int = 4
    skip_score: float = 0.65
    skip_margin: float = 0.04

    def compress_documents(self, documents, query: str, callbacks=None) -> list[Document]:
        documents = list(documents)
        if not documents:
            return []
        best = documents[0]
        # 확실 = 임베딩 1위가 높고 2위와 벌어졌고, 합친 순위 1위도 그 조각일 때. 두 신호가 다르면 리랭크한다:
        # "불량을 분량으로 알아듣는 건 왜 그래?" 는 코사인 1위 기준서 3.4·합친 1위 3.2 인데 건너뛰어 정답 안내서 6.5 가
        # 근거에서 빠졌고 4B 가 이유를 지어냈다(10-22)
        sure = best.metadata["top_cos"] >= self.skip_score and best.metadata["cos_margin"] >= self.skip_margin \
            and best.metadata["vec_rank"] == 0
        if self.rerank_n <= 0 or sure:
            top = [_with(d, skipped="sure" if sure else "off") for d in documents[:self.k]]
        else:
            head = documents[:self.rerank_n] + [d for d in documents[self.rerank_n:] if d.metadata["vec_rank"] < VEC_TOP]
            rest = [d for d in documents if d not in head]
            try:
                order = rerank_scores(query, [d.page_content for d in head])
                ranked = [_with(head[i], rerank=round(s, 4)) for i, s in order] + rest
            except Exception as e:          # 리랭커가 없으면 합친 순서 그대로
                ranked = [_with(d, rerank_error=f"{type(e).__name__}: {e}") for d in documents]
            top = ranked[:self.k]
            # 합친 순위 1위(anchor)는 리랭크가 1위의 절반 이상일 때만 남긴다. 세 문서에서 리랭크 0.45 인 설명서 2.1 이
            # "손은 어디에" 의 기준서 6.1(0.95)을 3위 밖으로 밀어냈다(10-22). "작업 순서" 의 5.1 은 0.72 라 남는다
            b_doc = next((d for d in ranked if d.page_content == best.page_content), best)
            b_rr, t_rr = b_doc.metadata.get("rerank"), max((d.metadata.get("rerank") or 0 for d in top), default=0)
            if all(d.page_content != best.page_content for d in top) and (b_rr is None or b_rr >= 0.5 * t_rr):
                top[-1] = b_doc
        return [_with(d, anchor=d.page_content == best.page_content) for d in top]


class RerankRetriever(BaseRetriever):
    """1단계 + 2단계를 하나의 리트리버로(ContextualCompressionRetriever 와 같은 일)."""
    base: BaseRetriever
    compressor: BaseDocumentCompressor

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return list(self.compressor.compress_documents(self.base.invoke(query), query))


class Manual:
    def __init__(self):
        from chromadb.config import Settings
        from langchain_chroma import Chroma

        self.sources = config.rag_sources()
        self.mtimes = {str(p): p.stat().st_mtime for _, _, p in self.sources}
        self.chunks, texts = [], []
        for key, name, path in self.sources:
            text = path.read_text(encoding="utf-8")
            texts.append(text)
            for c in chunk_markdown(text):
                m = re.match(r"(\d+\.\d+)\s+(.*)", c["section"].split(" > ")[-1])
                self.chunks.append({**c, "doc": key, "doc_name": name, "source": path.name,
                                    "clause": m.group(1) if m else "", "title": m.group(2) if m else ""})
        self.sha = hashlib.sha1("\n".join([config.EMB_MODEL, RAG_INSTRUCT] + [f"{k}:{n}" for k, n, _ in self.sources]
                                          + texts).encode()).hexdigest()
        self.std_sha = hashlib.sha1(texts[0].encode()).hexdigest() if texts else ""    # 보고 근거 캐시 키(기준서만)
        persist = config.CACHE_DIR / "chroma"
        persist.mkdir(parents=True, exist_ok=True)
        stamp = persist / "docs.json"
        self.emb = llm.QwenEmbeddings(RAG_INSTRUCT)

        def open_store():
            return Chroma(collection_name="docs", embedding_function=self.emb, persist_directory=str(persist),
                          client_settings=Settings(anonymized_telemetry=False, is_persistent=True))

        self.store = open_store()
        old = json.loads(stamp.read_text()) if stamp.exists() else {}
        if old.get("sha") != self.sha:      # 문서가 바뀌었으면 다시 만든다
            self.store.delete_collection()
            self.store = open_store()
            self.store.add_texts(
                [f"[{c['doc_name']} > {c['section']}]\n{c['text']}" for c in self.chunks],
                metadatas=[{k: c[k] for k in ("doc", "doc_name", "section", "clause", "title", "source")}
                           for c in self.chunks],
            )
            stamp.write_text(json.dumps({"sha": self.sha, "sources": [str(p) for _, _, p in self.sources],
                                         "chunks": len(self.chunks)}, ensure_ascii=False))
        # Chroma 에 저장된 조각 임베딩을 한 번 읽어 두고 질의마다 numpy 로 코사인을 잰다(Chroma query 약 0.3초 → 1ms)
        got = self.store._collection.get(include=["embeddings", "documents", "metadatas"])
        self.texts = list(got["documents"])
        self.metas = list(got["metadatas"])
        m = np.asarray(got["embeddings"], dtype=np.float32)
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        self.bm25 = _BM25(self.texts)

    def changed(self) -> bool:
        srcs = config.rag_sources()
        return [str(p) for _, _, p in srcs] != list(self.mtimes) or any(
            p.stat().st_mtime != self.mtimes.get(str(p)) for _, _, p in srcs)

    def candidates(self, query: str, doc: str | None = None) -> list[dict]:
        """모든 조각(doc 를 주면 그 문서만, 1장·머리말은 빼고)을 벡터 코사인 순위 + 글자 2-gram BM25 순위의 RRF 로
        줄 세운다(리랭크 없음)."""
        q = expand_query(query)
        v = np.asarray(self.emb.embed_query(q), dtype=np.float32)
        keep_meta = any(c in query for c in META_CUES)
        idx = [i for i, md in enumerate(self.metas) if (doc is None or md.get("doc") == doc)
               and (keep_meta or not is_meta(md.get("section") or ""))]
        cos_all = self.matrix @ (v / np.linalg.norm(v))
        lex_all = self.bm25.scores(q)
        cos = {i: float(cos_all[i]) for i in idx}
        vec_order = sorted(idx, key=lambda i: -cos[i])
        vec_rank = {i: r for r, i in enumerate(vec_order)}
        lex_rank = {i: r for r, i in enumerate(sorted(idx, key=lambda i: -lex_all[i]))}
        fused = sorted(idx, key=lambda i: -(1 / (60 + vec_rank[i]) + 1 / (60 + lex_rank[i])))
        c1 = cos[vec_order[0]]
        c2 = cos[vec_order[1]] if len(vec_order) > 1 else 0.0
        return [{**{k: self.metas[i].get(k) for k in ("doc", "doc_name", "section", "clause", "title", "source")},
                 "text": self.texts[i], "cos": round(cos[i], 4), "rerank": None,
                 "vec_rank": vec_rank[i], "lex_rank": lex_rank[i], "top_cos": round(c1, 4),
                 "cos_margin": round(c1 - c2, 4)} for i in fused]

    def retriever(self, k: int | None = None, rerank_n: int | None = None, skip_score: float | None = None,
                  skip_margin: float | None = None, doc: str | None = None) -> RerankRetriever:
        """LangChain 리트리버(1단계 HybridRetriever → 2단계 QwenRerankCompressor). 값을 비우면 config."""
        k = config.RAG_TOP_K if k is None else k
        rerank_n = config.RAG_RERANK_N if rerank_n is None else rerank_n
        comp = QwenRerankCompressor(k=k, rerank_n=rerank_n,
                                    skip_score=config.RAG_SKIP_SCORE if skip_score is None else skip_score,
                                    skip_margin=config.RAG_SKIP_MARGIN if skip_margin is None else skip_margin)
        return RerankRetriever(base=HybridRetriever(manual=self, k=max(8, k, rerank_n), doc=doc), compressor=comp)

    def search(self, query: str, k: int | None = None, rerank_n: int | None = None,
               skip_score: float | None = None, skip_margin: float | None = None, doc: str | None = None) -> list[dict]:
        """retriever(...).invoke(query) 를 dict 로(도구·평가·보고가 쓰는 꼴)."""
        docs = self.retriever(k, rerank_n, skip_score, skip_margin, doc).invoke(query)
        return [{**d.metadata, "text": d.page_content} for d in docs]

    # ---- 보고용 근거 캐시(기준서에서만)
    def citations_path(self):
        key = hashlib.sha1((self.std_sha + json.dumps(CITATION_QUERIES, ensure_ascii=False)).encode()).hexdigest()[:12]
        return config.CACHE_DIR / f"citations_{key}.json"

    def citations(self) -> dict:
        """CITATION_QUERIES 마다 리랭크까지 해서 고른 기준서 조항 2개. 기준서·검색어가 같으면 캐시를 읽는다."""
        path = self.citations_path()
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for name, q in CITATION_QUERIES.items():
            hits = self.search(q, k=2, rerank_n=5, skip_score=2.0, doc="standard")   # 캐시라 느려도 된다: 항상 리랭크
            out[name] = [{"section": h["section"], "rerank": h["rerank"], "text": h["text"]} for h in hits]
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        return out


_manual: Manual | None = None
_lock = threading.Lock()
_cite_lock = threading.Lock()            # 근거 캐시를 만드는 동안(리랭크 여러 번) 질문 검색을 막지 않게 따로 둔다


def get_manual() -> Manual:
    global _manual
    with _lock:
        if _manual is None or _manual.changed():   # 문서 세션이 문서를 새로 쓰거나 고치면 다시 읽는다
            _manual = Manual()
        return _manual


def get_citations() -> dict:
    m = get_manual()
    with _cite_lock:                     # 두 스레드가 동시에 캐시를 만들지 않게
        return m.citations()


def quick_citations(keys: list[str]) -> tuple[dict, str]:
    """보고용 근거. 캐시가 있으면 그대로(0초), 아직 만드는 중이면 기다리지 않고 리랭크 없이 찾는다."""
    m = get_manual()
    path = m.citations_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), "cache"
    out = {}
    for name in keys:
        hits = m.search(CITATION_QUERIES[name], k=2, rerank_n=0, doc="standard")
        out[name] = [{"section": h["section"], "rerank": None, "text": h["text"]} for h in hits]
    return out, "fast(캐시 없음)"


def warmup() -> dict:
    """첫 질문·보고가 느리지 않게 미리 준비한다(Chroma 열기·조각 임베딩·보고 근거 캐시)."""
    t0 = time.perf_counter()
    m = get_manual()
    c = get_citations()
    return {"chunks": len(m.chunks), "docs": len(m.sources), "citations": len(c),
            "elapsed_s": round(time.perf_counter() - t0, 2)}
