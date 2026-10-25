"""의도 판별 정답률·지연 시험.

  python -m agent.eval docs/intents.csv           # 라우터 + LLM(실제 동작과 같음)
  python -m agent.eval docs/intents.csv --no-llm  # 임베딩 라우터만
  python -m agent.eval docs/intents.csv --sweep   # 임계값·여유값 조합별 정답률과 LLM 호출 비율
  python -m agent.eval --rag [--sweep]            # 기준서 검색 9문항(+ 리랭크 설정 조합표)
  python -m agent.eval --qa                       # 기준서 질문 끝까지(판별·검색·답) 단계별 지연
  python -m agent.eval --timeline                 # 발표용 응답 시간표(가짜 작업대 1→5 + 질문 + 멈춰)
  python -m agent.eval --sp                       # scene 8문항·plan 10문항 끝까지(가짜 작업대)
  python -m agent.eval --docs                     # 세 문서 질문 47문항(docs/rag_questions.csv): 검색·인용 적중·거절·지연
CSV 열: text,intent,count (count 비우면 null). 성공 기준: 25문장 중 23개 이상(92%).
"""
import argparse
import csv
import statistics
import sys
import time

from . import config, router
from .intent_examples import EXAMPLES


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            c = (r.get("count") or "").strip()
            rows.append({"text": r["text"].strip(), "intent": r["intent"].strip(),
                         "count": int(c) if c and c.lower() not in ("null", "none") else None})
        return rows


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


# 기준서 검색 시험: 질문 → 근거로 나와야 할 조항. 앞 5개는 docs/README.md 의 "시험 질문 예와 근거 조항"
# (10-21 밤 볼 → 블록: 크기 질문을, 빨강·파랑 → 분홍·초록: 첫 질문을 docs/README 의 새 문장으로)
# 10-22 세 문서: 정답 = (문서, 조항) 쌍. "로봇이 안 멈추면" 은 기준서 6.4 또는 로봇 운용 설명서 5.4(docs/rag_questions.csv 와 같음)
_S = "standard"
RAG_CASES = [
    ("초록 불량은 왜 불량이야?", ((_S, "3.1"), (_S, "3.2"))), ("실패하면 어떻게 해?", ((_S, "5.4"), (_S, "5.5"))),
    ("로봇이 안 멈추면 어떻게 해야 돼?", ((_S, "6.4"), ("robot_manual", "5.4"))), ("블록은 몇 센티로 잘라야 돼?", ((_S, "2.2"),)),
    ("작업 순서가 어떻게 돼요?", ((_S, "5.1"),)), ("불량 기준이 뭐예요?", ((_S, "3.1"),)),
    ("로봇이 일할 때 손은 어디에 두면 돼?", ((_S, "6.1"),)), ("X 표시는 어떻게 그려야 하나요?", ((_S, "3.3"),)),
    ("결과 보고에는 뭐가 들어가요?", ((_S, "7.2"),)),
]


OOD_CASES = ["오늘 점심 메뉴 뭐야?", "내일 날씨 어때?", "너 이름이 뭐야?", "주식 시장 요즘 어때?"]


def _hit(h: dict, want) -> bool:
    return (h.get("doc") or _S, _no(h)) in want


def _cited(text: str, want) -> bool:
    """답에 "[작업 기준서 3.1]" 처럼 정답 문서·조항 인용이 있는가."""
    return any(f"[{config.DOC_NAMES[d]} {c}]" in text for d, c in want)


def _want_str(want) -> str:
    return "/".join(f"{config.DOC_NAMES[d][:2]}{c}" for d, c in want)


def _no(h: dict) -> str:
    return h["section"].split(" > ")[-1].split(" ")[0]


def rag_eval(sweep: bool) -> int:
    """지금 설정으로 실제 검색(정답 조항이 상위 k 개에 드는지·지연). --sweep 이면 설정 조합표도."""
    from .rag import get_manual, rerank_scores

    m = get_manual()
    k = config.RAG_TOP_K
    print(f"원본 {', '.join(p.name for _, _, p in m.sources)} · 조각 {len(m.chunks)}개 · 리랭크 {config.RAG_RERANK_N}개 · "
          f"생략 조건 cos≥{config.RAG_SKIP_SCORE}·차≥{config.RAG_SKIP_MARGIN} · 상위 {k}")
    hits, lat = 0, []
    for q, want in RAG_CASES:
        t0 = time.perf_counter()
        rr = m.search(q)
        lat.append((time.perf_counter() - t0) * 1000)
        ok = any(_hit(h, want) for h in rr)
        hits += ok
        how = "생략" if rr[0].get("skipped") == "sure" else "리랭크"
        secs = " | ".join(f"{config.DOC_NAMES.get(h.get('doc'), '')[:2]} {h['section'].split(' > ')[-1]}" for h in rr)
        print(f"{'  ' if ok else 'XX'} {q:<24} 정답 {_want_str(want):<12} cos {rr[0]['top_cos']:.3f}·차 "
              f"{rr[0]['cos_margin']:.3f} {how:<3} → {secs}  {lat[-1]:.0f}ms")
    n = len(RAG_CASES)
    print(f"\n상위 {k}개에 정답 조항 {hits}/{n} · 검색 지연 평균 {statistics.mean(lat):.0f}ms "
          f"중앙 {statistics.median(lat):.0f}ms 최대 {max(lat):.0f}ms")
    print("기준서 밖 질문의 임베딩 1위 cos: " + ", ".join(
        f"{q} {m.candidates(q)[0]['top_cos']:.3f}" for q in OOD_CASES) + f" (RAG_OOD_COS={config.RAG_OOD_COS})")
    if not sweep:
        return 0 if hits == n else 1

    # 조합표: 질문마다 후보 상위 5개를 한 번 리랭크해 두고(쌍마다 점수가 따로라 부분집합도 같은 점수) 설정별로 계산
    print("\n[sweep] 리랭크 5개 점수를 한 번씩 구하는 중…")
    rows, per_doc = [], []
    for q, want in RAG_CASES:
        t0 = time.perf_counter()
        cands = m.candidates(q)
        t_emb = time.perf_counter() - t0
        t1 = time.perf_counter()
        rr = dict(rerank_scores(q, [h["text"] for h in cands[:5]]))   # search() 와 같게
        per_doc.append((time.perf_counter() - t1) / 5)
        rows.append({"want": want, "cands": cands, "rr": rr, "t_emb": t_emb})
    doc_s = statistics.median(per_doc)
    emb_s = statistics.median(r["t_emb"] for r in rows)
    print(f"리랭크 조각당 {doc_s:.2f}초(중앙) · 임베딩+후보 {emb_s * 1000:.0f}ms")

    def final(r, nr, ss, sm):
        c, best = r["cands"], r["cands"][0]
        if nr <= 0 or (best["top_cos"] >= ss and best["cos_margin"] >= sm):
            return c[:k], 0
        idx = sorted(range(nr), key=lambda i: -r["rr"][i])
        top = ([c[i] for i in idx] + c[nr:])[:k]
        if best not in top:
            top[-1] = best
        return top, nr

    results = []
    for nr in range(0, 6):
        for ss in ([9.0] if nr == 0 else [9.0, 0.80, 0.75, 0.70, 0.65, 0.60]):
            for sm in ([0.0] if ss == 9.0 else [0.0, 0.02, 0.04, 0.06]):
                got = [final(r, nr, ss, sm) for r in rows]
                h = sum(any(_hit(x, r["want"]) for x in top) for r, (top, _) in zip(rows, got))
                docs = sum(d for _, d in got)
                est = emb_s + doc_s * docs / len(rows)
                worst = emb_s + doc_s * max(d for _, d in got)
                results.append((h, est, worst, nr, ss, sm, sum(d > 0 for _, d in got)))
    print(f"{'정답':>5} {'예상평균':>8} {'최악':>6} {'리랭크N':>7} {'생략cos':>7} {'생략차':>6} {'리랭크한 질문':>10}")
    for h, est, worst, nr, ss, sm, used in sorted(results, key=lambda x: (-x[0], x[1]))[:15]:
        print(f"{h:>3}/{n} {est:>7.2f}s {worst:>5.2f}s {nr:>7} {('-' if ss > 2 else ss):>7} {sm:>6} {used:>6}/{n}")
    return 0


def qa_eval() -> int:
    """기준서 질문을 run_command 로 끝까지(판별 → 검색 → 답) 돌려 단계별 지연과 조항 인용을 본다(로봇 안 씀)."""
    import os
    from datetime import datetime

    os.environ["BALL_MOCK"] = "1"
    from . import graph

    w = graph.warmup()
    print(f"warmup {w} · LLM {router.llm.model_name()}")

    def t(ev):
        return datetime.fromisoformat(ev["ts"]).timestamp()

    rows = []
    for i, (q, want) in enumerate(RAG_CASES + [(q, ()) for q in OOD_CASES]):
        evs = list(graph.run_command(q, f"qa-{time.strftime('%H%M%S')}-{i}"))
        t0 = t(evs[0])
        cls = next((e for e in evs if e["type"] == "tool" and e["data"].get("name") == "intent_router"), None)
        srch = next((e for e in evs if e["type"] == "tool" and e["data"].get("name") == "search_manual"
                     and e["data"].get("status") == "end"), None)
        say = next((e for e in evs if e["type"] == "say" and not e["data"].get("ack")), None)
        ack = next((e for e in evs if e["type"] == "say" and e["data"].get("ack")), None)
        total = evs[-1]["data"].get("elapsed_s") or 0
        text = say["data"]["text"] if say else ""
        ok = _cited(text, want) if want else "그 내용이 없습니다" in text
        c_ms = cls["data"]["result"]["latency_ms"] if cls else 0
        s_ms = (t(srch) - t(cls)) * 1000 if srch and cls else 0
        a_ms = (t(say) - t(srch)) * 1000 if say and srch else (t(say) - t0) * 1000 if say else 0
        meth = cls["data"]["result"]["method"] if cls else "-"
        k_s = t(ack) - t0 if ack else None
        rows.append((q, ok, meth, c_ms, s_ms, a_ms, total, text, k_s))
        print(f"{'  ' if ok else 'XX'} {q:<24} 판별 {meth:<9} {c_ms:>6.0f}ms · 검색 {s_ms:>5.0f}ms · 답 {a_ms:>5.0f}ms"
              f" · 합계 {total:>5.2f}s · 첫마디 {'-' if k_s is None else f'{k_s:.2f}s'}  {text}")
    tot = [r[6] for r in rows]
    print(f"\n인용·판정 정답 {sum(r[1] for r in rows)}/{len(rows)} · 합계 중앙 {statistics.median(tot):.2f}s "
          f"최대 {max(tot):.2f}s · 3초 안 {sum(x <= 3.0 for x in tot)}/{len(rows)}")
    print(f"  판별 중앙 {statistics.median(r[3] for r in rows):.0f}ms · 검색 중앙 {statistics.median(r[4] for r in rows):.0f}ms"
          f" · 답 중앙 {statistics.median(r[5] for r in rows):.0f}ms")
    ks = [r[8] for r in rows if r[8] is not None]
    if ks:
        print(f"  첫마디 \"{graph.ASK_ACK}\" {len(ks)}/{len(rows)}문항 · 중앙 {statistics.median(ks):.2f}s 최대 {max(ks):.2f}s"
              f"(텍스트 도착 기준, 받아쓰기 시간 제외)")
    return 0


TIMELINE_Q = "불량 기준이 뭐예요?"      # docs/README 시험 질문 중 하나


def _measure(text: str, tid: str, graph, on_policy=None) -> dict:
    """run_command 한 번: 텍스트를 넘긴 때부터 이벤트가 UI 에 닿는 시각(받아쓰기 제외)."""
    t0 = time.perf_counter()
    r = {"text": text, "first": None, "last": None, "says": [], "policy": [], "hand_s": 0.0, "method": None}
    asked = None
    for ev in graph.run_command(text, tid):
        dt, d = time.perf_counter() - t0, ev["data"]
        if ev["type"] == "say":
            r["first"] = dt if r["first"] is None else r["first"]
            r["last"] = dt
            r["says"].append(d["text"])
            if d["text"] == "손을 내밀어 주세요.":
                asked = dt
        elif ev["type"] == "tool" and d.get("name") == "run_policy":
            if d.get("status") == "start":
                if asked is not None:           # 손을 기다린 시간(사람 몫)
                    r["hand_s"] += dt - asked
                    asked = None
                if on_policy:
                    on_policy()
            elif d.get("status") == "end":
                r["policy"].append(d["result"].get("duration_s") or 0)
        elif ev["type"] == "tool" and d.get("name") == "intent_router":
            r["method"] = d["result"]["method"]
        elif ev["type"] == "node" and ev["node"] == "end":
            r["method"] = r["method"] or d.get("method")
            r["end"] = dt
    r["agent_s"] = (r["last"] or 0) - sum(r["policy"]) - r["hand_s"]
    return r


DEMO_SCRIPT = ["불량품 골라내 줘", "지금 배치 어때?", "불량 빼고 나머지 다 정리해 줘", "분홍 블록 하나 내 손에 줘", "결과 알려 줘"]


def timeline_eval(policy_s: float, repeat: int, demo: bool = False) -> int:
    """발표용 응답 시간표: 가짜 작업대로 시연 1→5 + 질문 1개 + 멈춰(쉬는 중·정책 실행 중). 중앙값을 표로.
    demo=True 면 계약서 2절 발표 실연(demo4 장면 ①~⑤ + 멈춰)."""
    import os
    import threading

    os.environ["BALL_MOCK"] = "1"
    os.environ["BALL_MOCK_SCENE"] = "demo4" if demo else "full"
    os.environ.setdefault("BALL_MOCK_SNAP", "0")
    os.environ.setdefault("BALL_MOCK_FAIL_AT", "" if demo else "3")   # full: 3번째 시도에서 한 번 놓침(재시도 포함)
    os.environ.setdefault("BALL_MOCK_HAND_DELAY", "2")     # 손은 2초 뒤에 나타남
    from . import graph, mocks
    from .cli import SCENARIO

    w = graph.warmup()
    print(f"warmup {w} · LLM {router.llm.model_name()} · 가짜 정책 1회 {policy_s}초 · {repeat}회 반복")
    runs = []
    for k in range(repeat):
        os.environ["BALL_MOCK_POLICY_S"] = str(policy_s)
        mocks.workbench.reset()
        tid = f"timeline-{time.strftime('%H%M%S')}-{k}"
        rows = [_measure(t, tid, graph) for t in (DEMO_SCRIPT if demo else SCENARIO + [TIMELINE_Q]) + ["멈춰"]]
        # 정책 실행 중 멈춰: 긴 정책(5초)을 돌려 두고 1초 뒤 "멈춰" → 멈췄다는 말 / 돌던 명령이 끝날 때까지
        mocks.workbench.reset()
        os.environ["BALL_MOCK_POLICY_S"] = "5"
        started = threading.Event()
        box = {}
        th = threading.Thread(target=lambda: box.update(r=_measure("분홍 블록 옮겨 줘", tid + "-stop", graph,
                                                                    on_policy=started.set)))
        th.start()
        started.wait(30)
        time.sleep(1.0)
        t_stop = time.perf_counter()
        s = _measure("멈춰", tid + "-stop", graph)
        th.join(30)
        s["text"] = "멈춰 (정책 실행 중)"
        s["last"] = time.perf_counter() - t_stop if box.get("r") else None   # 돌던 명령의 마지막 말까지
        s["after"] = [f"(돌던 명령) {x}" for x in box.get("r", {}).get("says", [])[-1:]]
        s["policy"] = box.get("r", {}).get("policy", [])                     # 중단된 정책
        rows[-1]["text"] = "멈춰 (쉬는 중)"
        runs.append(rows + [s])
        print(f"[{k + 1}/{repeat}] " + " | ".join(f"{r['text'][:8]} {r['first'] or 0:.2f}/{r['last'] or 0:.2f}s"
                                                 for r in runs[-1]))

    def med(i, key):
        xs = [rs[i][key] for rs in runs if rs[i].get(key) is not None]
        return statistics.median(xs) if xs else None

    print(f"\n| # | 입력 | 판별 | 첫 응답 문장 | 최종 문장 | 첫 응답(초) | 최종(초) | 정책 실행 | 에이전트 몫(초) |")
    print("|---|---|---|---|---|---:|---:|---:|---:|")
    labels = ["①", "②", "③", "④", "⑤", "멈춤", "멈춤"] if demo else ["1", "2", "3", "4", "5", "질문", "멈춤", "멈춤"]
    for i, r in enumerate(runs[-1]):
        last_say = (r.get("after") or r["says"][-1:] or [""])[-1]
        first_say = r["says"][0] if r["says"] else ""
        n_pol = len(r["policy"])
        stop_run = i == len(runs[-1]) - 1
        agent = None if stop_run else med(i, "agent_s")
        print(f"| {labels[i]} | {r['text']} | {r['method'] or '-'} | {first_say} | "
              f"{last_say if last_say != first_say else '(같음)'} | {med(i, 'first') or 0:.2f} | "
              f"{(med(i, 'last') or 0):.2f} | {n_pol}회{'(중단)' if stop_run and n_pol else ''} | "
              f"{'' if agent is None else f'{agent:.2f}'} |")
    return 0


# scene: 바꿔 둔 작업대(아래 SP_SCENE_STATE)에서 답에 들어 있어야 할 말(하나라도). 숫자·칸 지어내기 검사도 통과해야 정답
SP_SCENE_STATE = {"input": {"red_ok": 2, "blue_ok": 3, "red_defect": 1}, "bowl_red": {"red_ok": 2},
                  "bowl_blue": {"blue_ok": 1}, "bowl_defect": {"red_defect": 1, "blue_defect": 1}, "outside": {"red_ok": 1}}
SP_SCENE = [("불량 몇 개 남았어?", ["1개"]), ("초록 그릇에 몇 개야?", ["1개"]), ("분홍 블록 몇 개 남았어?", ["2개"]),
            ("분홍 블록 어디 있어?", ["@red_ok"]), ("구역 밖에 떨어진 거 있어?", ["1개"]), ("손 자리에 손 보여?", ["없", "않"]),
            ("투입 구역에 뭐 남았어?", ["3개"]), ("지금 작업대 어때?", ["개"])]
# plan: 기본 작업대(분홍 4·초록 4·불량 3)에서 끝난 뒤 상태. None = 거절(로봇 0회)
SP_PLAN = [
    ("분홍 세 개 옮기고 불량 골라내 줘", {"bowl_red": 3, "bowl_defect": 3, "in_blue": 4}),
    ("초록색은 놔두고 나머지 다 정리해 줘", {"bowl_red": 4, "bowl_defect": 3, "in_blue": 4}),
    ("불량 먼저 빼고 초록 블록 옮겨 줘", {"bowl_defect": 3, "bowl_blue": 4, "in_red": 4}),
    ("분홍 하나는 내 손에 주고 나머지 분홍은 그릇에 넣어 줘", {"handed": 1, "bowl_red": 3, "in_red": 0}),
    ("초록 두 개만 옮기고 분홍 하나 내 손에 줘", {"bowl_blue": 2, "handed": 1, "in_red": 3}),
    ("전부 다 정리해 줘", {"bowl_red": 4, "bowl_blue": 4, "bowl_defect": 3}),
    ("분홍이랑 초록 다 옮겨 줘", {"bowl_red": 4, "bowl_blue": 4, "in_defect": 3}),
    ("블록으로 숫자 3 모양 만들어 줘", None), ("초록 블록을 분홍 그릇에 담아 줘", None), ("블록 색깔별로 섞어 줘", None),
]
# 규칙 계획이 읽지 못해 LLM 으로 가는 말(첫 말 = "네, 계획을 세울게요."). 기대 상태는 여러 개면 그중 하나
SP_PLAN_LLM = [
    ("분홍이랑 초록 하나씩 옮겨 줘", [{"bowl_red": 1, "bowl_blue": 1, "in_defect": 3}]),
    ("분홍이랑 초록 두 개씩 옮겨 줘", [{"bowl_red": 2, "bowl_blue": 2, "in_defect": 3}]),
    ("초록 하나 옮기고 끝내", [{"bowl_blue": 1, "in_red": 4, "in_defect": 3}]),
]


# 좌표 집기(arrange): 기본 작업대(정상 분홍 4·초록 4). 기대 끝 상태(확인 대기면 "응" 으로 이어 완료까지)
SP_ARRANGE = [("블록으로 7 만들어 줘", "done"), ("핑크 블록으로 숫자 7 만들어 줘", "refused"),
              ("블록으로 하트 만들어 줘", "refused"), ("블록으로 숫자 8 만들어 줘", "refused"), ("블록으로 12 만들어 줘", "refused")]


def sp_eval() -> int:
    """scene·plan 끝까지(판별 → detect → 답 / 계획·실행). 가짜 작업대, 정책 1회 0.05초, 손 0.3초 뒤."""
    import os

    os.environ["BALL_MOCK"] = "1"
    os.environ["BALL_MOCK_SCENE"] = "full"            # 기대 상태는 11개 장면 기준
    os.environ.setdefault("BALL_MOCK_SNAP", "0")
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    os.environ["BALL_MOCK_POLICY_S"] = "0.05"
    os.environ["BALL_MOCK_HAND_DELAY"] = "0.3"
    from . import graph, mocks

    config.HAND_POLL_S = 0.2
    print(f"warmup {graph.warmup()} · LLM {router.llm.model_name()}")
    w = mocks.workbench
    ok_s, lat_s = 0, []
    print("\n[scene]")
    for i, (q, want) in enumerate(SP_SCENE):
        w.reset({})
        for z, d in SP_SCENE_STATE.items():
            (w.outside if z == "outside" else w.zones[z]).update(d)
        t0 = time.perf_counter()
        evs = list(graph.run_command(q, f"sp-scene-{time.strftime('%H%M%S')}-{i}"))
        dt = time.perf_counter() - t0
        say = next((e["data"] for e in evs if e["type"] == "say"), {})
        text, facts = say.get("text", ""), say.get("facts")
        cells = {o["cell"] for o in mocks.layout(w.zones, w.outside) if o["cls"] == "red_ok" and o["zone"] == "input"}
        hit = any((any(c in text for c in cells) if x == "@red_ok" else x in text) for x in want)
        chk = bool(facts) and graph.scene_answer_ok(text, facts)[0]
        good = evs[-1]["data"].get("intent") == "scene" and hit and chk and w.attempts == 0
        ok_s += good
        lat_s.append(dt)
        print(f"{'  ' if good else 'XX'} {q:<20} {say.get('method', '-'):<8} {dt:5.2f}s  {text}")
    ok_p, ok_l, lat_first, lat_ann, lat_end, by = 0, 0, [], [], [], {}
    print("\n[plan]")
    for i, (q, want) in enumerate(SP_PLAN + SP_PLAN_LLM):
        if i == len(SP_PLAN):
            print("  -- 규칙으로 못 읽는 말(LLM 계획)")
        w.reset()
        t0, first, ann = time.perf_counter(), None, None
        evs = []
        for e in graph.run_command(q, f"sp-plan-{time.strftime('%H%M%S')}-{i}"):
            evs.append(e)
            if e["type"] == "say" and first is None:
                first = time.perf_counter() - t0
            if e["type"] == "say" and e["node"] == "planner" and e["data"]["text"] not in graph.ACCEPT_MSGS.values() and ann is None:
                ann = time.perf_counter() - t0          # 계획 알림(또는 거절·할 일 없음)
        dt = time.perf_counter() - t0
        z = w.zones
        got = {"bowl_red": z["bowl_red"]["red_ok"], "bowl_blue": z["bowl_blue"]["blue_ok"],
               "bowl_defect": z["bowl_defect"]["red_defect"] + z["bowl_defect"]["blue_defect"], "handed": w.handed,
               "in_red": z["input"]["red_ok"], "in_blue": z["input"]["blue_ok"],
               "in_defect": z["input"]["red_defect"] + z["input"]["blue_defect"]}
        end = evs[-1]["data"]
        if want is None:
            good = end.get("status") == "refused" and w.attempts == 0
        else:
            # LLM 경로 3문항은 끝난 작업대로만 채점("초록 하나 옮기고 끝내" 는 blue 한 동작으로 판별돼도 맞는 결과)
            good = (end.get("intent") == "plan" or i >= len(SP_PLAN)) and any(
                all(got[k] == v for k, v in wv.items()) for wv in (want if isinstance(want, list) else [want]))
        if i < len(SP_PLAN):
            ok_p += good
        else:
            ok_l += good
        lat_first.append(first or dt)
        lat_ann.append(ann or dt)
        lat_end.append(dt)
        mp = next((e["data"] for e in evs if e["type"] == "tool" and e["data"].get("name") == "make_plan"), {})
        plan, how = mp.get("steps"), (mp.get("result") or {}).get("by", "-")
        by[how] = by.get(how, 0) + 1
        says = [e["data"]["text"] for e in evs if e["type"] == "say"]
        print(f"{'  ' if good else 'XX'} {q:<30} 첫 말 {first or 0:5.2f}s · 알림 {ann or 0:5.2f}s · 끝 {dt:5.2f}s · "
              f"{how:<4} · 시도 {w.attempts:>2} · 계획 {[(s['intent'], s['count']) for s in plan] if plan else '-'}")
        print(f"      첫 말: {says[0] if says else ''}")
        if want is not None and not good:
            keys = (want if isinstance(want, list) else [want])[0]
            print(f"      기대 {want} · 실제 { {k: got[k] for k in keys} }")
    print(f"\nscene {ok_s}/{len(SP_SCENE)} · 지연 중앙 {statistics.median(lat_s):.2f}s 최대 {max(lat_s):.2f}s")
    ok_a, lat_a1, lat_a2 = 0, [], []
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    print("\n[arrange]")
    for i, (q, want) in enumerate(SP_ARRANGE):
        w.reset()
        tid = f"sp-arr-{time.strftime('%H%M%S')}-{i}"
        t0, first, plan_say = time.perf_counter(), None, None
        evs = []
        for e in graph.run_command(q, tid):
            evs.append(e)
            if e["type"] == "say":
                dt_ = time.perf_counter() - t0
                first = dt_ if first is None else first
                if not e["data"]["text"].startswith("네, 숫자"):
                    plan_say = dt_ if plan_say is None else plan_say
        st = evs[-1]["data"].get("status")
        if st == "waiting_confirm":
            evs += list(graph.run_command("응", tid))
            st = evs[-1]["data"].get("status")
        good = st == want and evs[-1]["data"].get("intent") == "arrange" and (want != "done" or len(w.arranged) == 7)
        ok_a += good
        lat_a1.append(first or 0)
        lat_a2.append(plan_say or 0)
        says = [e["data"]["text"] for e in evs if e["type"] == "say"]
        print(f"{'  ' if good else 'XX'} {q:<24} 첫 말 {first or 0:5.2f}s · 계획/거절 말 {plan_say or 0:5.2f}s · 끝 {st}")
        print(f"      {' / '.join(says)[:220]}")
    print(f"arrange {ok_a}/{len(SP_ARRANGE)} · 첫 말 최대 {max(lat_a1):.2f}s · 계획/거절 말 중앙 {statistics.median(lat_a2):.2f}s"
          f" 최대 {max(lat_a2):.2f}s")
    n_rule = len(SP_PLAN)
    print(f"plan(LLM 경로 {len(SP_PLAN_LLM)}문항) {ok_l}/{len(SP_PLAN_LLM)} · 첫 말 최대 "
          f"{max(lat_first[n_rule:]):.2f}s · 계획 알림 중앙 {statistics.median(lat_ann[n_rule:]):.2f}s 최대 {max(lat_ann[n_rule:]):.2f}s")
    lat_first, lat_ann, lat_end = lat_first[:n_rule], lat_ann[:n_rule], lat_end[:n_rule]
    print(f"plan {ok_p}/{len(SP_PLAN)} · 첫 말 중앙 {statistics.median(lat_first):.2f}s 최대 {max(lat_first):.2f}s"
          f" · 계획 알림(거절) 중앙 {statistics.median(lat_ann):.2f}s 최대 {max(lat_ann):.2f}s"
          f" · 끝까지 중앙 {statistics.median(lat_end):.2f}s(가짜 정책 0.05초·손 0.3초) · 계획 출처 {by}")
    return 0 if ok_s == len(SP_SCENE) and ok_p == len(SP_PLAN) else 1   # LLM 경로 3문항은 참고


RECORD_QUESTIONS = ["오늘 몇 번 실패했어?", "방금 몇 개 옮겼어?", "지금까지 성공한 거 몇 번이야?", "아까 실패한 거 있어?",
                    "오늘 불량 몇 개 골라냈어?"]


def _expected(doc: str, clause: str) -> tuple:
    """rag_questions.csv 의 정답 칸 → (문서, 조항) 쌍(docs/README: '|' 는 같은 자리끼리 짝, 문서가 하나면 그 문서의 조항 중 하나)."""
    if doc.strip() == "none":
        return ()
    ds, cs = doc.split("|"), clause.split("|")
    return tuple((ds[0], c) for c in cs) if len(ds) == 1 else tuple(zip(ds, cs))


def docs_eval(path: str = "docs/rag_questions.csv") -> int:
    """세 문서 질문 시험: 검색 적중(상위 k 에 정답 문서·조항) + 답 인용 적중 + 문서 밖 거절 + 지연. 가짜 작업대."""
    import os

    os.environ["BALL_MOCK"] = "1"
    from . import graph
    from .rag import get_manual

    w = graph.warmup()
    m = get_manual()
    print(f"warmup {w} · 조각 {len(m.chunks)}개(" + ", ".join(f"{n} {sum(c['doc'] == k for c in m.chunks)}"
                                                         for k, n, _ in m.sources) + f") · LLM {router.llm.model_name()}")
    with open(config.ROOT / path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    per: dict = {}
    lat, out, stages, acks, cut, pre = [], [], [], [], 0, 0
    for i, r in enumerate(rows):
        q, want = r["question"].strip(), _expected(r["expected_doc"], r["expected_clause"])
        key = "none" if not want else want[0][0]
        rr = m.search(q)
        r_hit = any(_hit(h, want) for h in rr) if want else None
        evs, t0, mk = [], time.perf_counter(), {}
        for e in graph.run_command(q, f"docs-{time.strftime('%H%M%S')}-{i}"):
            evs.append(e)
            if e["type"] == "tool" and (e.get("data") or {}).get("status") == "end":
                mk.setdefault(e.get("node"), time.perf_counter() - t0)
            if e["type"] == "say":
                mk.setdefault("ack" if (e.get("data") or {}).get("ack") else "say", time.perf_counter() - t0)
        end = evs[-1]["data"]
        if "ack" in mk:
            acks.append(mk["ack"])
        fin = next((e["data"].get("finish") for e in evs if e["type"] == "say" and not e["data"].get("ack")), None)
        cut += fin == "length"
        pre += any(e["type"] == "tool" and e["data"].get("prefetched") for e in evs)
        if {"classify", "answer", "say"} <= mk.keys():
            stages.append((mk["classify"], mk["answer"] - mk["classify"], mk["say"] - mk["answer"]))
        say = next((e["data"]["text"] for e in evs if e["type"] == "say" and not e["data"].get("ack")), "")
        a_hit = _cited(say, want) if want else "그 내용이 없습니다" in say
        lat.append(end.get("elapsed_s") or 0)
        d = per.setdefault(key, {"n": 0, "r": 0, "a": 0})
        d["n"] += 1
        d["r"] += bool(r_hit) if want else 0
        d["a"] += a_hit
        out.append((q, want, r_hit, a_hit, end.get("intent"), lat[-1], say, rr))
        top = " | ".join(f"{config.DOC_NAMES.get(h.get('doc'), '')[:2]}{_no(h)}" for h in rr)
        print(f"{'  ' if a_hit else 'XX'} {q:<28} 정답 {_want_str(want) if want else '문서 밖':<12} 검색 "
              f"{'-' if r_hit is None else ('맞음' if r_hit else '틀림'):<2} [{top}] · {end.get('intent'):<8} "
              f"{lat[-1]:5.2f}s  {say[:90]}")
    n_in = sum(1 for o in out if o[1])
    print(f"\n검색 적중(상위 {config.RAG_TOP_K}개에 정답 문서·조항) {sum(bool(o[2]) for o in out if o[1])}/{n_in} · "
          f"답 인용 적중 {sum(o[3] for o in out if o[1])}/{n_in} · 문서 밖 거절 {sum(o[3] for o in out if not o[1])}/"
          f"{len(out) - n_in}")
    for k, d in per.items():
        name = config.DOC_NAMES.get(k, "문서 밖")
        print(f"  {name:<10} {d['n']:>2}문항 · 검색 {d['r']}/{d['n'] if k != 'none' else '-'} · 답 {d['a']}/{d['n']}")
    print(f"질문 응답(판별 → 검색 → 답) 중앙 {statistics.median(lat):.2f}s · 최대 {max(lat):.2f}s · 3초 안 "
          f"{sum(x <= 3 for x in lat)}/{len(lat)}")
    if stages:
        print("  단계 중앙(최대): " + " · ".join(
            f"{n} {statistics.median(x):.2f}s({max(x):.2f}s)" for n, x in zip(("판별", "검색", "답"), zip(*stages))))
    if acks:
        print(f"  첫마디 \"{graph.ASK_ACK}\" {len(acks)}/{len(lat)}문항 · 중앙 {statistics.median(acks):.2f}s "
              f"최대 {max(acks):.2f}s(텍스트 도착 기준)")
    print(f"  답 max_tokens {config.ANSWER_MAX_TOKENS} · 잘린 답(finish=length) {cut}건 · 미리 검색 {pre}건"
          f"(QA_PREFETCH={int(config.QA_PREFETCH)})")
    print("\n작업 기록 질문 → report(runs/ 기록에서 답)인가:")
    for q in RECORD_QUESTIONS:
        o = router.classify(q)
        print(f"  {'  ' if o['intent'] == 'report' else 'XX'} {q:<24} → {o['intent']} ({o['method']}"
              f"{', ' + o['report_check'] if o.get('report_check') else ''})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--rag", action="store_true", help="기준서 검색 시험(정답 조항이 상위 k 개에 드는지)")
    ap.add_argument("--qa", action="store_true", help="기준서 질문을 끝까지(판별·검색·답) 돌려 단계별 지연")
    ap.add_argument("--timeline", action="store_true", help="발표용 응답 시간표(가짜 작업대 1→5 + 질문 + 멈춰)")
    ap.add_argument("--sp", action="store_true", help="scene·plan 끝까지(가짜 작업대) 정확도·지연")
    ap.add_argument("--docs", action="store_true", help="세 문서 질문 시험(docs/rag_questions.csv)")
    ap.add_argument("--policy-s", type=float, default=0.3, help="--timeline 의 가짜 정책 1회 시간")
    ap.add_argument("--repeat", type=int, default=3, help="--timeline 반복 횟수(표는 중앙값)")
    ap.add_argument("--demo", action="store_true", help="--timeline 을 발표 실연(demo4 장면 ①~⑤)으로")
    a = ap.parse_args()
    if a.rag:
        return rag_eval(a.sweep)
    if a.qa:
        return qa_eval()
    if a.timeline:
        return timeline_eval(a.policy_s, a.repeat, a.demo)
    if a.sp:
        return sp_eval()
    if a.docs:
        return docs_eval()
    if not a.csv:
        ap.error("CSV 경로가 필요하다(또는 --rag)")
    rows = load(a.csv)
    router.classify("준비")          # 첫 호출(예문 임베딩·연결)은 지연에서 뺀다
    if not a.no_llm:
        router.classify("준비 운동 겸 아무 말")

    results, lat, by_method = [], [], {}
    for r in rows:
        t0 = time.perf_counter()
        out = router.classify(r["text"], use_llm=not a.no_llm)
        ms = (time.perf_counter() - t0) * 1000
        ok_i, ok_c = out["intent"] == r["intent"], out["count"] == r["count"]
        results.append((r, out, ok_i, ok_c))
        lat.append(ms)
        by_method.setdefault(out["method"], []).append(ms)

    print(f"{'':2} {'문장':<28} {'정답':<14} {'판별':<14} {'방법':<10} {'점수':>6} {'여유':>6} {'ms':>7}")
    for r, out, ok_i, ok_c in results:
        mark = "  " if ok_i and ok_c else "XX"
        print(f"{mark} {r['text'][:28]:<28} {r['intent'] + '/' + str(r['count']):<14} "
              f"{out['intent'] + '/' + str(out['count']):<14} {out['method']:<10} "
              f"{out.get('score', 0):>6.3f} {out.get('margin', 0):>6.3f} {out['latency_ms']:>7.1f}")
    n = len(rows)
    ni = sum(x[2] for x in results)
    nc = sum(x[3] for x in results)
    nb = sum(x[2] and x[3] for x in results)
    print(f"\n문장 {n}개 | 의도 정답 {ni}/{n} ({ni / n:.0%}) | 개수 정답 {nc}/{n} | 둘 다 {nb}/{n} ({nb / n:.0%})")
    print(f"지연 ms: 평균 {statistics.mean(lat):.0f} · p50 {pct(lat, .5):.0f} · p95 {pct(lat, .95):.0f} · 최대 {max(lat):.0f}")
    for m, xs in sorted(by_method.items()):
        print(f"  {m:<20} {len(xs):>3}건  평균 {statistics.mean(xs):.0f}ms  최대 {max(xs):.0f}ms")
    print(f"설정: LLM={router.llm.model_name()} @ {config.LLM_BASE_URL}, 임계값 {config.ROUTER_THRESHOLD}, "
          f"여유 {config.ROUTER_MARGIN}")
    seen = {t.replace(" ", "") for ts in EXAMPLES.values() for t in ts}
    same = [r["text"] for r in rows if r["text"].replace(" ", "") in seen]
    rest = [x for x in results if x[0]["text"].replace(" ", "") not in seen]
    print(f"라우터 예문과 같은 문장(띄어쓰기 무시) {len(same)}개: {', '.join(same) or '없음'}")
    if rest:
        nr = sum(x[2] and x[3] for x in rest)
        print(f"  겹치지 않는 {len(rest)}문장만: 둘 다 정답 {nr}/{len(rest)} ({nr / len(rest):.0%})")
    need = -(-n * 23 // 25)
    print(f"성공 기준(23/25 비율 → {need}/{n}): {'통과' if nb >= need else '미달'}")

    if a.sweep:
        print("\n[sweep] 임계값×여유 → 정답률(의도) / LLM 호출 비율")
        llm_intent = {}
        for r, out, *_ in results:
            try:
                llm_intent[r["text"]] = router._llm_classify(r["text"], out if "top3" in out else None).intent
            except Exception as e:
                llm_intent[r["text"]] = f"err:{type(e).__name__}"
        print("  LLM 만 썼을 때: " + f"{sum(llm_intent[r['text']] == r['intent'] for r, *_ in results)}/{n}")
        for th in (0.70, 0.75, 0.80, 0.85, 0.90):
            cells = []
            for mg in (0.0, 0.02, 0.04, 0.06, 0.08):
                hit = calls = 0
                for r, out, *_ in results:
                    if out["method"] == "rule":
                        pred = "stop"
                    elif out.get("score", 0) >= th and out.get("margin", 0) >= mg:
                        pred = out["top3"][0][0]
                    else:
                        pred, calls = llm_intent[r["text"]], calls + 1
                    hit += pred == r["intent"]
                cells.append(f"m{mg:.2f} {hit:>2}/{n} llm{calls / n:>4.0%}")
            print(f"  th {th:.2f} | " + " | ".join(cells))
    return 0 if nb >= need else 1


if __name__ == "__main__":
    sys.exit(main())
