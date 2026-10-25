"""텍스트로 명령을 넣어 이벤트를 본다.

  python -m agent.cli "불량품 골라내 줘"                 # 한 문장
  python -m agent.cli "불량품 골라내 줘" "분홍 블록 옮겨 줘"  # 여러 문장을 한 프로세스에서(가짜 작업대 상태 유지)
  python -m agent.cli --scenario                         # 시연 순서 1→5
  python -m agent.cli                                    # 대화형
옵션: --thread ID(기본 cli-<시각>), --json(이벤트 원문)
"""
import argparse
import json
import sys
from datetime import datetime

from . import run_command

SCENARIO = ["불량품 골라내 줘", "분홍 블록 옮겨 줘", "초록 블록 옮겨 줘", "분홍 블록 하나 내 손에 줘", "결과 알려 줘"]
SHORT = {"red_ok": "분", "blue_ok": "초", "red_defect": "분X", "blue_defect": "초X"}


def _zones(obs: dict) -> str:
    def z(name):
        return " ".join(f"{SHORT[k]}{n}" for k, n in obs["zones"][name].items() if n) or "-"
    hand = "손O" if obs["hand"]["present"] else "손X"
    return (f"투입[{z('input')}] 분홍[{z('bowl_red')}] 초록[{z('bowl_blue')}] 불량[{z('bowl_defect')}] "
            f"{hand} ({obs.get('device')} {obs.get('latency_ms')}ms)")


def show(ev: dict) -> None:
    ts, t, n, d = ev["ts"][11:23], ev["type"], ev["node"], ev["data"]
    if t == "say":
        print(f"{ts}  말> {d['text']}")
    elif t == "error":
        print(f"{ts}  !! [{n}] {d.get('message')}")
    elif t == "report":
        print(f"{ts}  보고서 {d['html_path']} (문장: {d['method']}, 기록 대조 {'일치' if d['facts']['match'] else '다름'})")
    elif t == "tool":
        name, res = d.get("name"), d.get("result")
        if d.get("status") == "start":
            if name == "run_policy":
                print(f"{ts}     run_policy({d['args']['name']}) 시작")
            return
        if name == "detect_objects":
            print(f"{ts}     detect  {_zones(res)}")
        elif name == "run_policy":
            print(f"{ts}     run_policy 끝 ok={res.get('ok')} rc={res.get('returncode')} {res.get('duration_s')}s")
        elif name == "intent_router":
            extra = f" score={res.get('score')} margin={res.get('margin')}" if "score" in res else ""
            print(f"{ts}     의도 {res['intent']} count={res['count']} [{res['method']}]{extra} "
                  f"{res['latency_ms']}ms" + (f" llm={res['llm_ms']}ms" if "llm_ms" in res else "")
                  + (f" ({res['error']})" if res.get("error") else ""))
        elif name == "search_manual":
            print(f"{ts}     search_manual " + " | ".join(f"{h['section']}({h['rerank']})" for h in res))
        else:
            print(f"{ts}     {name} {json.dumps(res, ensure_ascii=False)[:120] if res else ''}")
    elif n == "start":
        b = d["backend"]
        print(f"\n== {d['text']}   (대화 {d['thread_id']}, vision={b['vision']} robot={b['robot']}, llm={d['llm']})")
    elif n == "end":
        print(f"== 끝: {d.get('intent')} / {d.get('status')} / {d.get('elapsed_s')}s")
    else:
        rest = {k: v for k, v in d.items() if k not in ("text",)}
        print(f"{ts}  [{n}] {json.dumps(rest, ensure_ascii=False) if rest else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("texts", nargs="*")
    ap.add_argument("--scenario", action="store_true", help="시연 순서 1→5")
    ap.add_argument("--thread", default=f"cli-{datetime.now().strftime('%H%M%S')}")
    ap.add_argument("--json", action="store_true", help="이벤트를 JSON 한 줄씩")
    a = ap.parse_args()
    texts = SCENARIO if a.scenario else a.texts
    errors = 0

    def run(text):
        nonlocal errors
        for ev in run_command(text, a.thread):
            errors += ev["type"] == "error"
            print(json.dumps(ev, ensure_ascii=False)) if a.json else show(ev)

    if texts:
        for t in texts:
            run(t)
    else:
        print("명령을 입력하세요(빈 줄로 끝).")
        for line in sys.stdin:
            if not line.strip():
                break
            run(line.strip())
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
