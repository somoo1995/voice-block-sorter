"""발표 실연 시간 흉내 — 실연 ①~⑤ 를 처음부터 끝까지 돌려 전체·구간별 시간을 잰다(계약서 2절 끝 "발표 실연", 목표 3분).

    BALL_MOCK=1 BALL_MOCK_SCENE=demo4 BALL_MOCK_POLICY_S=19 .venv/bin/python -m ui.demo_timing   # → docs/demo_timing.md

①~⑤ 뒤에 선택 단계 "⑥ 질문 하나"(기본 ui/samples/demo 08 "그리퍼가 뜨거우면 어떻게 해?", --question none 이면 뺌)를 돌려
①~⑤ 만일 때와 ⑥ 까지일 때 둘 다 목표와 견준다. ⑥ 은 에이전트 측정의 질문 응답 최대값(Q_MAX)으로도 어림한다.
--fail-once 면 이어서 블록을 한 번 놓치는 조건(BALL_MOCK_FAIL_AT=3)으로 한 번 더 돌려 같은 문서 아래 절에 붙인다.

한 명령의 흐름(대시보드와 같은 줄):
  사람 말(3초, --speak-s) → 받아쓰기(ui/samples/demo 음성 → 실제 Whisper) → 에이전트(진짜 그래프 + 가짜 작업대:
  판단·정책·확인·손 기다림) → 에이전트 말(Supertonic 합성 → 소리 없이 음성 길이만큼 기다림, 차례대로 한 줄)
  → 말이 다 끝나면 다음 사람 말. 에이전트 말은 정책과 따로 돌아 정책 중에 나온다(겹침).
시각은 이벤트를 받은 순간의 벽시계다. 받아쓰기·판단·말은 실제 모델이라 실연과 같고, 정책(가짜 19초)·카메라(가짜)는 흉내다.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui import stt, tts  # noqa: E402
from ui.demo_samples import OUT as DEMO_DIR, SENTENCES as DEMO_SENTENCES  # noqa: E402

OUT = ROOT / "docs" / "demo_timing.md"
STEPS = [s for s in DEMO_SENTENCES if s[0] in ("01", "02", "03", "04", "05")]  # 실연 ①~⑤
TAGS = dict(zip(("01", "02", "03", "04", "05"), "①②③④⑤"))
Q_MED, Q_MAX = 3.13, 6.81  # 세 문서 질문 47문항의 판별 → 검색 → 답(agent/RESULT_20261021.md 7절)


class Speech:
    """대시보드의 말하기 줄(tts.enqueue, timed)과 같다. 재생 구간을 적는다."""

    def __init__(self):
        self.cv = threading.Condition()
        self.pending = 0
        self.spans: list[dict] = []

    def say(self, text: str):
        t_q = time.time()
        with self.cv:
            self.pending += 1
        rec = {"text": text, "queued": t_q}

        def on_play(_t, _p, secs):
            rec.update(start=time.time(), secs=secs)

        def on_done(_t, _p):
            rec["end"] = time.time()
            with self.cv:
                self.spans.append(rec)
                self.pending -= 1
                self.cv.notify_all()

        tts.enqueue(text, on_done=on_done, sound="timed", on_play=on_play)

    def drain(self):
        with self.cv:
            self.cv.wait_for(lambda: self.pending == 0)


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(x1, y1) - max(x0, y0)) for x0, x1 in a for y0, y1 in b)


def run_one(fn, no: str, text_ref: str, wav: Path, speak_s: float, sp: Speech, thread_id: str) -> dict:
    r = {"no": no, "ref": text_ref, "t0": time.time(), "events": []}
    time.sleep(speak_s)  # 사람이 말하는 시간
    r["speak"] = (r["t0"], time.time())
    t = time.time()
    text = stt.transcribe(str(wav))
    r["stt"] = (t, time.time())
    r["heard"] = text
    n_spans = len(sp.spans)
    pol, det, open_tools, hand = [], [], {}, []
    t_run = time.time()
    hand_t = None
    intent = status = None
    for ev in fn(text, thread_id=thread_id):
        now = time.time()
        r["events"].append([round(now - r["t0"], 2), ev.get("type"), ev.get("node"),
                            {k: v for k, v in (ev.get("data") or {}).items() if k in ("name", "status", "intent", "latency_ms", "elapsed_s", "text")}])
        kind, node, data = ev.get("type"), ev.get("node") or "", ev.get("data") or {}
        if kind == "node" and node == "hand_wait" and hand_t is None:
            hand_t = now
        if kind == "node" and node == "act" and hand_t is not None:
            hand.append((hand_t, now))
            hand_t = None
        if kind == "node" and node == "end":
            intent, status = data.get("intent"), data.get("status")
        if kind == "tool" and data.get("status") != "reuse":  # reuse = 앞 감지를 다시 씀(시작·끝 없음)
            name = data.get("name") or data.get("tool")
            res = data.get("result", data.get("output"))
            if res is None and data.get("status") != "end":
                open_tools[name] = now
            elif name in open_tools:
                (pol if name == "run_policy" else det if name == "detect_objects" else []).append((open_tools.pop(name), now))
        if kind == "say" and (data.get("text") or "").strip():
            sp.say(data["text"].strip())
    r["agent"] = (t_run, time.time())
    sp.drain()
    r["t1"] = time.time()
    r["spans"] = sp.spans[n_spans:]
    r["policy"], r["detect"], r["hand"], r["intent"], r["status"] = pol, det, hand, intent, status
    # 손 기다림 안의 카메라는 손 기다림으로 센다
    in_hand = [d for d in det if any(h0 <= d[0] <= h1 for h0, h1 in hand)]
    r["detect_only"] = [d for d in det if d not in in_hand]
    return r


def summarize(r: dict) -> dict:
    d = lambda iv: sum(b - a for a, b in iv)
    agent = r["agent"][1] - r["agent"][0]
    pol, det, hand = d(r["policy"]), d(r["detect_only"]), d(r["hand"])
    speech_iv = [(s["start"], s["end"]) for s in r["spans"] if "start" in s]
    talk = d(speech_iv)
    return {"speak": r["speak"][1] - r["speak"][0], "stt": r["stt"][1] - r["stt"][0],
            "think": max(0.0, agent - pol - det - hand), "policy": pol, "n_policy": len(r["policy"]), "detect": det,
            "hand": hand, "agent": agent, "talk": talk, "n_talk": len(speech_iv),
            "overlap": _overlap(speech_iv, r["policy"]), "tail": r["t1"] - r["agent"][1], "total": r["t1"] - r["t0"]}


def _mmss(sec: float) -> str:
    t = round(sec)  # 179.9 → "3분 0초"(따로 반올림하면 "2분 60초")
    return f"{t // 60}분 {t % 60}초"


def _goal_line(name: str, total: float, goal_s: float) -> str:
    return (f"{name}: 전체 {total:.0f}초 = {_mmss(total)} — 목표 {goal_s / 60:.0f}분 "
            f"{'안' if total <= goal_s else '넘음'}({'여유' if total <= goal_s else '초과'} {abs(goal_s - total):.0f}초)")


def _table(rows: list[dict]) -> list[str]:
    S = [summarize(r) for r in rows]
    f = lambda x: f"{x:.1f}"
    keys = ("speak", "stt", "think", "policy", "detect", "hand", "talk", "overlap", "tail", "total")
    out = ["| # | 말 | 사람 말 | 받아쓰기 | 판단 | 정책(횟수) | 확인 | 손 기다림 | 에이전트 말(문장) | 말·정책 겹침 | 말 끝 기다림 | 명령 전체 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def total_row(label, part):
        t = {k: sum(x[k] for x in part) for k in keys}
        return (f"| | **{label}** | {f(t['speak'])} | {f(t['stt'])} | {f(t['think'])} | {f(t['policy'])} | {f(t['detect'])} | "
                f"{f(t['hand'])} | {f(t['talk'])} | {f(t['overlap'])} | {f(t['tail'])} | **{f(t['total'])}** |")
    for i, (r, x) in enumerate(zip(rows, S)):
        if r["tag"] == "⑥":
            out.append(total_row("합 ①~⑤", S[:i]))
        ov = f"{f(x['overlap'])} ✓" if x["overlap"] > 0 else ("0 ✗" if x["n_policy"] else "- (정책 없음)")
        out.append(f"| {r['tag']}{'(선택)' if r['tag'] == '⑥' else ''} | {r['ref']} | {f(x['speak'])} | {f(x['stt'])} | {f(x['think'])} | "
                   f"{f(x['policy'])}({x['n_policy']}) | {f(x['detect'])} | {f(x['hand'])} | {f(x['talk'])}({x['n_talk']}) | {ov} | "
                   f"{f(x['tail'])} | **{f(x['total'])}** |")
    out.append(total_row("합 ①~⑥" if any(r["tag"] == "⑥" for r in rows) else "합 ①~⑤", S))
    return out


def _heads(rows: list[dict], ends: dict, goal_s: float) -> list[str]:
    out = [f"**{_goal_line('실연 ①~⑤', ends['main'], goal_s)}**", ""]
    q = next((r for r in rows if r["tag"] == "⑥"), None)
    if q is not None:
        sq = summarize(q)
        worst = ends["q"] + max(0.0, Q_MAX - sq["agent"])
        out += [f"**{_goal_line('선택 ⑥ 질문 하나까지', ends['q'], goal_s)}**", "",
                f"- ⑥ 의 판단(판별 → 검색 → 답)은 이번에 {sq['agent']:.1f}초였다. 에이전트 측정(세 문서 47문항)은 중앙 {Q_MED}초·최대 {Q_MAX}초"
                f"(agent/RESULT_20261021.md 7절)라, 가장 느린 질문이면 약 {worst:.0f}초 = {_mmss(worst)}"
                f"({'목표 안' if worst <= goal_s else '목표 넘음'}, 답이 길면 에이전트 말도 길어져 몇 초 더 붙을 수 있음).", ""]
    return out


def _first_sound(rows: list[dict]) -> list[float]:
    """말하기 줄이 비어 있을 때 들어온 말의 '에이전트 say → 첫 소리' 초(앞 말을 기다린 몫은 뺀다)."""
    spans = sorted((s for r in rows for s in r["spans"] if "start" in s), key=lambda s: s["queued"])
    return [s["start"] - s["queued"] for i, s in enumerate(spans) if i == 0 or spans[i - 1]["end"] <= s["queued"]]


def to_markdown(rows: list[dict], cond: str, ends: dict, goal_s: float, ref: tuple | None = None, note: str = "") -> str:
    """ends = {"main": ①~⑤ 끝 시각까지 초, "q": ⑥ 까지 초(없으면 None)}. ref = 놓침 1회 실행의 (rows, ends)."""
    fs = sorted(_first_sound(rows))
    out = [f"# 발표 실연 시간 흉내 — {datetime.now():%Y-%m-%d %H:%M}", "",
           f"> `python -m ui.demo_timing` 으로 다시 만든다. {cond}", ""]
    if note:
        out += [f"> {note}", ""]
    out += _heads(rows, ends, goal_s)
    if ref is not None:
        out += [f"**{_goal_line('블록을 한 번 놓쳤을 때 실연 ①~⑤', ref[1]['main'], goal_s)}**"
                + (f" · ⑥ 까지 {ref[1]['q']:.0f}초" if ref[1].get("q") else "") + "(아래 \"놓침 1회\" 절)", ""]
    out += _table(rows)
    out += ["",
            "- 명령 전체 = 사람 말 + 받아쓰기 + 에이전트(판단 + 정책 + 확인 + 손 기다림) + 말 끝 기다림. 에이전트 말은 정책과 따로 돌아 겹치는 몫은 시간에 더해지지 않는다.",
            "- 판단 = 에이전트 시간에서 정책·확인·손 기다림을 뺀 나머지(의도 판별·계획·답 문장 만들기 등 언어 모델과 그래프).",
            "- 말·정책 겹침 = 에이전트 말이 나오는 동안 정책이 도는 시간(✓ = 겹침, 로봇이 말하면서 움직임). 말 끝 기다림 = 에이전트가 끝난 뒤 남은 말이 끝날 때까지.",
            (f"- 말 → 첫 소리(에이전트 say 에서 소리 시작까지, 앞 말을 기다리지 않은 {len(fs)}문장): 중앙 {fs[len(fs) // 2]:.2f}초, 최대 {fs[-1]:.2f}초." if fs else ""),
            "- 받아쓰기 글: " + " / ".join(f"{r['tag']} “{r['heard']}”(의도 {r['intent']}, {r['status']})" for r in rows), "",
            "## 명령별 흐름(명령 시작 = 사람이 말하기 시작한 때)", ""]
    for r in rows:
        t0 = r["t0"]
        items = [(r["speak"][0], "사람 말 시작"), (r["stt"][0], "받아쓰기 시작"), (r["stt"][1], f"받아쓰기 끝 “{r['heard']}”"),
                 (r["agent"][0], "에이전트 시작"), (r["agent"][1], "에이전트 끝")]
        items += [(a, f"정책 시작({i + 1})") for i, (a, _) in enumerate(r["policy"])] + [(b, f"정책 끝({i + 1})") for i, (_, b) in enumerate(r["policy"])]
        items += [(s["start"], f"말 시작 “{s['text'][:40]}”") for s in r["spans"] if "start" in s] + [(s["end"], "말 끝") for s in r["spans"]]
        out.append(f"- **{r['tag']} {r['ref']}**: " + " → ".join(f"{t - t0:.1f} {w}" for t, w in sorted(items)))
    if ref is not None:
        out += ["", "## 놓침 1회(`BALL_MOCK_FAIL_AT=3`: 전체 3번째 시도에서 블록을 놓침 → \"놓쳤어요. 한 번 더 해 볼게요.\" 뒤 정책 한 번 더)", ""]
        out += _heads(*ref[:2], goal_s) + _table(ref[0])
    out.append("")
    return "\n".join(l for l in out if l is not None)


def _run_all(fn, plan: list, a, sp: Speech) -> tuple[list[dict], dict]:
    rows, ends = [], {"main": None, "q": None}
    thread_id = f"timing-{datetime.now():%H%M%S}"
    t_all = time.time()
    for no, text, tag in plan:
        wav = DEMO_DIR / f"{a.voice}_{no}.wav"
        r = run_one(fn, no, text, wav, a.speak_s, sp, thread_id)
        r["tag"] = tag
        print(f"{tag} {summarize(r)['total']:.1f}초 · “{r['heard']}” → {r['intent']}/{r['status']}", file=sys.stderr)
        rows.append(r)
        ends["q" if tag == "⑥" else "main"] = time.time() - t_all
    return rows, ends


def main():
    ap = argparse.ArgumentParser(description="발표 실연 시간 흉내")
    ap.add_argument("--speak-s", type=float, default=3.0, help="사람이 명령을 말하는 시간(초)")
    ap.add_argument("--voice", default="F1", help="ui/samples/demo 음성(F1·M1)")
    ap.add_argument("--goal", type=float, default=180)
    ap.add_argument("--question", default="08", help="선택 단계 ⑥ 질문의 ui/samples/demo 번호(none = 뺌)")
    ap.add_argument("--fail-once", action="store_true", help="이어서 BALL_MOCK_FAIL_AT=3(블록 한 번 놓침)으로 한 번 더 돌려 같은 문서에")
    ap.add_argument("--note", default="", help="문서 머리에 덧붙일 한 줄(지난 측정과 비교 등)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dump", help="명령별 이벤트(명령 시작 기준 초)를 JSON 으로(느린 구간 찾기)")
    a = ap.parse_args()
    if os.getenv("BALL_MOCK") != "1":
        sys.exit("BALL_MOCK=1 로 돌린다(실제 로봇 금지)")
    agent = importlib.import_module("agent")
    fn = agent.run_command
    if hasattr(agent, "warmup"):
        agent.warmup()
    bench = importlib.import_module("agent.mocks").workbench
    bench.reset()
    tts.available()
    tts.speak("준비")  # 첫 합성(모델 적재)을 재기 전에
    plan = [(no, text, TAGS[no]) for no, text, _ in STEPS]
    if a.question != "none":
        plan += [(no, text, "⑥") for no, text, _ in DEMO_SENTENCES if no == a.question]
    fail = os.getenv("BALL_MOCK_FAIL_AT", "3")  # agent/mocks.py: 기본 = 3번째 시도에서 블록을 놓침
    sp = Speech()
    rows, ends = _run_all(fn, plan, a, sp)
    ref = None
    if a.fail_once:
        os.environ["BALL_MOCK_FAIL_AT"] = "3"  # Workbench.reset 이 읽는다
        bench.reset()
        print("— 놓침 1회", file=sys.stderr)
        ref = _run_all(fn, plan, a, sp)
    cond = (f"가짜 작업대 `BALL_MOCK_SCENE={os.getenv('BALL_MOCK_SCENE', '(기본)')}`, 정책 1회 `BALL_MOCK_POLICY_S={os.getenv('BALL_MOCK_POLICY_S', '(기본)')}`초, "
            f"놓침 주입 `BALL_MOCK_FAIL_AT={fail}`({f'{fail}번째 시도에서 놓침' if fail.strip() else '없음'}), "
            f"사람 말 {a.speak_s:.0f}초, 받아쓰기 = `ui/samples/demo/{a.voice}_0N.wav` → OVMS Whisper(NPU), 판단 = 진짜 에이전트(qwen3-4b 등), "
            f"에이전트 말 = Supertonic {'문장 단위 합성(앞 조각 재생 중 다음 조각 합성)' if tts.SPLIT else '통째 합성'}"
            f"{', 앞뒤 무음 줄임' if tts.TRIM else ''} + 음성 길이만큼 기다림(소리 없음, 대시보드 `UI_TTS_OUT=timed` 와 같은 줄), 카메라·손은 가짜 작업대.")
    md = to_markdown(rows, cond, ends, a.goal, ref=ref, note=a.note)
    print(md)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
        print(f"→ {a.out}", file=sys.stderr)
    if a.dump:
        Path(a.dump).write_text(json.dumps([{"tag": r["tag"], "events": r["events"]} for r in rows + (ref[0] if ref else [])],
                                           ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
