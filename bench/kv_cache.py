"""KV 캐시 크기에 따른 의도 판별·보고 지연 (OVMS qwen3-4b, graph 의 cache_size).

시연 흐름을 흉내 낸다: 의도 판별 → 바로 한 번 더 → 결과 보고(200토큰) → 기준서 답(근거 3조항) → 의도 판별.
의도 판별은 에이전트 코드(agent/router.py `_llm_classify`)를 그대로 부른다(읽기만, runs/ 에 안 씀).
보고·답 프롬프트는 agent/graph.py 의 report_node·answer_node 에서 복사(10-21)한 같은 꼴이다.

사용: .venv/bin/python bench/kv_cache.py <태그> [--rounds 6]
결과: bench/out/kv_<태그>.json (+ 화면 요약)
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "bench" / "out"

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from agent import llm, router  # noqa: E402

TEXTS = ["불량품 골라내 줘", "빨간 볼 옮겨 줘", "파란 볼 두 개만 옮겨 줘", "빨간 볼 하나 내 손에 줘", "결과 알려 줘",
         "X 표시된 파란 공 치워 줘"]
REPORT_SYS = ("너는 부직포볼 선별 로봇의 작업 보고 담당이다. [사실]만 써서 음성으로 읽을 결과 보고를 "
              "한국어 2문장 안팎으로 쓴다. [예시 문장]의 말투와 순서를 따른다. 규칙: 사실에 있는 숫자를 하나도 "
              "빠뜨리거나 바꾸지 않는다. 사실에 없는 평가·추측·권고를 만들지 않는다. 손에 건넨 개수는 그릇 개수와 "
              "따로 말한다. 마크다운·목록·따옴표 없이 문장만 쓴다.")
REPORT_USER = ("[사실]\n- 명령: 불량품 골라내기, 빨간 볼 옮기기, 파란 볼 옮기기, 손에 건네기, 결과 보고\n"
               "- 불량 그릇 2개, 빨강 그릇 3개, 파랑 그릇 4개, 손에 건넨 빨간 볼 1개\n- 정책 실행 10회 중 성공 10회\n"
               "- 투입 구역에 남은 볼 0개\n- 기록과 카메라 개수 일치\n\n[예시 문장]\n불량품 2개, 빨간 볼 3개, 파란 볼 4개를 "
               "그릇에 담았고 빨간 볼 1개를 손에 건넸습니다. 기록과 카메라 개수가 맞습니다.")
ANSWER_SYS = ("너는 부직포볼 선별 로봇의 안내 담당이다. [기준서 근거]와 [이번 대화의 작업 기록]만 보고 질문에 답한다.\n"
              "첫 줄: 답의 근거로 쓴 조항 번호만(예: 3.2). 기준서가 아니라 작업 기록으로 답하거나 근거에 답이 없으면 0.\n"
              "둘째 줄: 질문에 바로 답하는 한국어 한 문장(40자 안팎, '~입니다/~합니다'체). 조항 번호·제목은 쓰지 않는다. "
              "근거의 '~하지 않는다'·'~일 때만' 같은 금지·조건은 근거 낱말 그대로 써서 뜻을 뒤집지 않는다. "
              "근거에 없는 내용을 지어내지 않는다. 근거에도 기록에도 답이 없으면 "
              "'기준서에는 그 내용이 없습니다.'라고 쓴다. 음성으로 읽으므로 마크다운 없이 문장만 쓴다.")


def answer_user():
    """기준서에서 3.1·3.2·5.1 조항을 근거로 붙인 질문(answer_node 와 같은 꼴)."""
    md = (ROOT / "docs" / "standard.md").read_text(encoding="utf-8")
    secs = dict(re.findall(r"^### (\d\.\d) [^\n]*\n(.*?)(?=^#)", md, re.S | re.M))
    hits = "\n\n".join(f"[근거 {i + 1}] {k} {secs[k].strip()}" for i, k in enumerate(["3.1", "3.2", "5.1"]))
    return f"[질문] 찢어진 볼은 어디에 담아?\n\n[기준서 근거]\n{hits}\n\n[이번 대화의 작업 기록]\n없음"


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000, out


def main(tag, rounds):
    rt = router.get_router()
    hints = {t: rt.route(t) for t in TEXTS}  # 임베딩 라우터 결과(LLM 에 참고로 붙는 후보) — 시간 밖에서 미리
    ans_user = answer_user()
    # 예열: 컴파일·첫 요청 비용 빼기
    router._llm_classify(TEXTS[0], hints[TEXTS[0]])
    llm.chat(max_tokens=200).invoke([SystemMessage(REPORT_SYS), HumanMessage(REPORT_USER)])
    rows = {"intent_after_other": [], "intent_repeat": [], "report": [], "answer": []}
    wrong = []
    for i in range(rounds):
        a, b = TEXTS[i % len(TEXTS)], TEXTS[(i + 1) % len(TEXTS)]
        ms, r = timed(lambda: router._llm_classify(a, hints[a]))
        rows["intent_after_other"].append(ms)
        ms, r2 = timed(lambda: router._llm_classify(a, hints[a]))
        rows["intent_repeat"].append(ms)
        ms, rep = timed(lambda: llm.chat(max_tokens=200).invoke([SystemMessage(REPORT_SYS), HumanMessage(REPORT_USER)]))
        rows["report"].append(ms)
        ms, ans = timed(lambda: llm.chat(max_tokens=80).invoke([SystemMessage(ANSWER_SYS), HumanMessage(ans_user)]))
        rows["answer"].append(ms)
        ms, r3 = timed(lambda: router._llm_classify(b, hints[b]))
        rows["intent_after_other"].append(ms)
        if r.intent != r2.intent:
            wrong.append((a, r.intent, r2.intent))
        print(f"[{tag}] {i + 1}/{rounds} 판별 {rows['intent_after_other'][-2]:.0f}→반복 {rows['intent_repeat'][-1]:.0f}ms · "
              f"보고 {rows['report'][-1]:.0f}ms · 답 {rows['answer'][-1]:.0f}ms · 판별 {rows['intent_after_other'][-1]:.0f}ms "
              f"| {a}={r.intent}/{r.count} {b}={r3.intent}/{r3.count}", flush=True)
    summ = {k: {"median_ms": round(statistics.median(v)), "max_ms": round(max(v)), "n": len(v)} for k, v in rows.items()}
    print(f"[{tag}] 요약 " + " · ".join(f"{k} {v['median_ms']}ms(최대 {v['max_ms']})" for k, v in summ.items()))
    OUT.mkdir(exist_ok=True)
    (OUT / f"kv_{tag}.json").write_text(json.dumps({"tag": tag, "summary": summ, "rows": rows,
                                                    "unstable_intent": wrong,
                                                    "sample_report": rep.content, "sample_answer": ans.content},
                                                   ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--rounds", type=int, default=6)
    a = ap.parse_args()
    main(a.tag, a.rounds)
