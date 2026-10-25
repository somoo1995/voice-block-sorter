"""시연 연속 시험(기본 30분): 가짜 작업대(BALL_MOCK=1)로 에이전트 run_command 를 시연 순서대로 되풀이하면서
받아쓰기(STT, ui/samples 음성)·음성 합성(TTS)·YOLO NPU detect 를 주기적으로 겹친다.

시연 날 대시보드 프로세스 하나가 하는 일(에이전트·STT·TTS·비전)을 한 프로세스에 모았다. 인터넷 차단도 대시보드와
같은 ui/netguard.py 를 건다. 한 바퀴 = [초기화 → 명령 1→5] + [초기화 → scene → plan → scene → 질문] (thread_id 는 soak-*).
보는 것: 메모리 추이(가용·OVMS·이 프로세스), GPU HANG·NPU 오류(journalctl -k), 지연이 시간에 따라 느는지(5분 단위 중앙값), 오류.
다른 세션이 OVMS 와 주고받은 시각은 따로 남긴다(그 구간 지연은 섞인 값). OVMS 가 죽으면 거기서 멈춘다.

YOLO 는 vision/detect.py 의 detect_objects(사진 파일, NPU)를 그대로 부른다. 모델은 블록 모델(ball_int8)이 있으면 그것, 없으면
같은 5클래스 yolov8n 인 e2e_int8 → synth_int8 순. 구역은 vision/zones.json 이 아직 없으면 vision/_test/zones_test.json.
사용: .venv/bin/python bench/soak.py [--minutes 30] [--tag night] [--stt-every 12] [--tts-every 15] [--yolo-every 0.5]
결과: bench/out/soak_<tag>.json (+ 화면 요약)
"""
from __future__ import annotations

import os

os.environ["BALL_MOCK"] = "1"
os.environ.setdefault("BALL_MOCK_SNAP", "0")          # runs/ 에 가짜 작업대 그림을 쌓지 않는다
os.environ.setdefault("BALL_MOCK_POLICY_S", "0.5")    # 정책 1회 0.5초 — 실물(약 18초)보다 LLM 을 촘촘히 부른다
os.environ.setdefault("YOLO_OFFLINE", "1")

import argparse
import csv
import json
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
OUT = ROOT / "bench" / "out"

from ui import netguard  # noqa: E402  대시보드와 같이, 다른 것을 불러오기 전에 건다

netguard.install()

import agent  # noqa: E402
from agent import mocks  # noqa: E402
from npu_concurrency import foreign_now, ovms_alive, training_running  # noqa: E402
from ui import stt, tts  # noqa: E402
from ui.stt_check import SENTENCES, cer  # noqa: E402
from vision.detect import BALL_MODEL, detect_objects  # noqa: E402

YOLO_MODEL = next((m for m in (BALL_MODEL, *(ROOT / "vision" / "models" / f"{n}_int8_openvino_model" for n in ("e2e", "synth")))
                   if m.exists()), BALL_MODEL)
ZONES = ROOT / "vision" / "zones.json"
if not ZONES.exists():
    ZONES = ROOT / "vision" / "_test" / "zones_test.json"

# 시연 순서(계약서 2절, 사람이 듣는 말은 분홍·초록). 바퀴마다 다음 말투로 바꾼다
DEMO = [("defect", ["불량품 골라내 줘", "불량 블록 치워 줘"]),
        ("red", ["분홍 블록 옮겨 줘", "핑크 블록 옮겨 줘"]),
        ("blue", ["초록 블록 옮겨 줘", "녹색 블록 옮겨 줘", "연두 블록 옮겨 줘"]),
        ("handoff", ["분홍 블록 하나 내 손에 줘", "핑크 블록 하나 내 손에 줘"]),
        ("report", ["결과 알려 줘", "결과 보고해 줘"])]
QUESTIONS = ["초록 불량은 왜 불량이야?", "찢어진 블록은 어디에 담아?", "로봇이 움직일 때 손을 넣어도 돼?"]
SCENE_DEF = ["지금 작업대에 뭐가 있어?", "분홍 블록 몇 개 남았어?"]
PLAN_DEF = ["불량품 먼저 치우고 초록 블록 두 개만 옮겨 줘"]
TTS_TEXTS = ["불량품 두 개를 불량 그릇에 담았습니다.", "분홍 블록 세 개, 초록 블록 네 개를 옮겼습니다.",
             "기록과 카메라 개수가 맞습니다.", "기준서에는 그 내용이 없습니다."]
KERNEL_PAT = r"gpu hang|fence expiration|context reset|i915.*(error|fail)|intel_vpu.*(error|fail|timeout)|ivpu.*(error|fail|timeout)"


def intents_csv(intent: str) -> list[str]:
    """docs/intents.csv 에서 그 intent 문장(UI·문서 담당이 정한 말투)."""
    p = ROOT / "docs" / "intents.csv"
    try:
        with p.open(encoding="utf-8") as f:
            return [r["text"] for r in csv.DictReader(f) if r.get("intent") == intent and r.get("text")]
    except (OSError, KeyError):
        return []


class Rec:
    def __init__(self, t0: float):
        self.t0, self.rows, self.lock = t0, [], threading.Lock()

    def add(self, kind: str, ms: float | None, ok: bool, **kw):
        with self.lock:
            self.rows.append({"t": round(time.time() - self.t0, 1), "kind": kind, "ms": None if ms is None else round(ms),
                              "ok": ok, **kw})


def run_cmd(rec: Rec, text: str, want: str, tid: str) -> bool:
    t = time.perf_counter()
    errs, end, says = [], {}, 0
    try:
        for ev in agent.run_command(text, tid):
            if ev["type"] == "error":
                errs.append(ev["data"].get("message", "")[:160])
            elif ev["type"] == "say":
                says += 1
            elif ev["type"] == "node" and ev["node"] == "end":
                end = ev["data"]
    except Exception as e:
        errs.append(f"{type(e).__name__}: {e}"[:160])
    ms = (time.perf_counter() - t) * 1000
    got = end.get("intent")
    ok = not errs and got == want
    rec.add(f"cmd:{want}", ms, ok, text=text, intent=got, count=end.get("count"), status=end.get("status"),
            done=end.get("done"), says=says, errors=errs)
    return ok


def agent_loop(rec: Rec, end: float, stop: threading.Event):
    scenes = intents_csv("scene") or SCENE_DEF
    plans = intents_csv("plan") or PLAN_DEF
    qs = QUESTIONS + intents_csv("question")[:3]
    n = 0
    while time.time() < end and not stop.is_set():
        n += 1
        mocks.workbench.reset()
        tid = f"soak-{datetime.now():%H%M%S}-{n}a"
        for want, texts in DEMO:
            if time.time() >= end or stop.is_set():
                return
            run_cmd(rec, texts[n % len(texts)], want, tid)
        mocks.workbench.reset()
        tid = f"soak-{datetime.now():%H%M%S}-{n}b"
        for want, text in (("scene", scenes[n % len(scenes)]), ("plan", plans[n % len(plans)]),
                           ("scene", scenes[(n + 1) % len(scenes)]), ("question", qs[n % len(qs)])):
            if time.time() >= end or stop.is_set():
                return
            run_cmd(rec, text, want, tid)
        rec.add("cycle", None, True, n=n)


def stt_loop(rec: Rec, end: float, stop: threading.Event, every: float):
    ref = {num: s for num, s, _ in SENTENCES}
    wavs = sorted((ROOT / "ui" / "samples").glob("*.wav"))
    i = 0
    while time.time() < end and not stop.is_set() and wavs:
        w = wavs[i % len(wavs)]
        i += 1
        num = (re.findall(r"_(\d\d)\.wav$", w.name) or [""])[0]
        t = time.perf_counter()
        try:
            text = stt.transcribe(str(w))
            ms = (time.perf_counter() - t) * 1000
            c = cer(ref[num], text) if num in ref else None
            rec.add("stt", ms, True, file=w.name, text=text, cer=None if c is None else round(c, 3))
        except Exception as e:
            rec.add("stt", (time.perf_counter() - t) * 1000, False, file=w.name, errors=[f"{type(e).__name__}: {e}"[:160]])
        stop.wait(every)


def tts_loop(rec: Rec, end: float, stop: threading.Event, every: float):
    i = 0
    while time.time() < end and not stop.is_set():
        text = TTS_TEXTS[i % len(TTS_TEXTS)]
        i += 1
        t = time.perf_counter()
        try:
            path = tts.speak(text)
            rec.add("tts", (time.perf_counter() - t) * 1000, bool(path), chars=len(text))
        except Exception as e:
            rec.add("tts", (time.perf_counter() - t) * 1000, False, errors=[f"{type(e).__name__}: {e}"[:160]])
        stop.wait(every)


def yolo_loop(rec: Rec, end: float, stop: threading.Event, every: float, image: Path, snaps: Path):
    snaps.mkdir(parents=True, exist_ok=True)
    while time.time() < end and not stop.is_set():
        t = time.perf_counter()
        try:
            r = detect_objects(str(image), model=YOLO_MODEL, zones_path=ZONES, out_dir=snaps)
            rec.add("yolo", r["latency_ms"], r.get("device", "").startswith("NPU"), total_ms=r.get("total_ms"),
                    device=r.get("device"), n=len(r.get("objects") or r.get("detections") or []))
            Path(ROOT / r["image_path"]).unlink(missing_ok=True)  # 박스 그린 사진은 지운다(다 같은 장면)
        except Exception as e:
            rec.add("yolo", (time.perf_counter() - t) * 1000, False, errors=[f"{type(e).__name__}: {e}"[:160]])
        stop.wait(every)


def _unit_mem(unit: str) -> float | None:
    v = subprocess.run(["systemctl", "--user", "show", unit, "-p", "MemoryCurrent", "--value"],
                       capture_output=True, text=True).stdout.strip()
    return round(int(v) / 2**30, 2) if v.isdigit() else None


def _meminfo() -> dict:
    m = dict(re.findall(r"^(\w+):\s+(\d+)", Path("/proc/meminfo").read_text(), re.M))
    return {"avail_gb": round(int(m["MemAvailable"]) / 2**20, 2),
            "swap_used_gb": round((int(m["SwapTotal"]) - int(m["SwapFree"])) / 2**20, 2)}


def _rss_gb(pid: int | str) -> float | None:
    try:
        kb = re.search(r"VmRSS:\s+(\d+)", Path(f"/proc/{pid}/status").read_text()).group(1)
        return round(int(kb) / 2**20, 2)
    except (OSError, AttributeError):
        return None


def kernel_lines(since: str) -> list[str]:
    out = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager", "-o", "short-iso"],
                         capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if re.search(KERNEL_PAT, l, re.I)]


def monitor(rec: Rec, end: float, stop: threading.Event, since: str, mem: list, foreign: list):
    ui_pid = subprocess.run(["systemctl", "--user", "show", "ball-ui", "-p", "MainPID", "--value"],
                            capture_output=True, text=True).stdout.strip()
    me = {os.getpid()}
    last_mem = 0.0
    while time.time() < end and not stop.is_set():
        f = foreign_now(me)
        if f or training_running():
            foreign.append({"t": round(time.time() - rec.t0, 1), "who": list(f.values())[:2],
                            "train": training_running()})
        if time.time() - last_mem >= 10:
            last_mem = time.time()
            n = ovms_alive()
            row = {"t": round(time.time() - rec.t0, 1), **_meminfo(), "ovms_gb": _unit_mem("ball-ovms"),
                   "self_gb": _rss_gb(os.getpid()), "ui_gb": _rss_gb(ui_pid) if ui_pid not in ("", "0") else None,
                   "ovms_models": n, "load1": round(os.getloadavg()[0], 1)}
            mem.append(row)
            if n < 4:
                rec.add("ovms_down", None, False, models=n)
                stop.set()
                return
        stop.wait(1)


def summarize(rows: list, mem: list, minutes: float) -> dict:
    kinds = sorted({r["kind"] for r in rows if r["kind"] != "cycle"})
    out = {}
    nb = max(1, int(minutes // 5 + (minutes % 5 > 0)))
    for k in kinds:
        rs = [r for r in rows if r["kind"] == k]
        ms = sorted(r["ms"] for r in rs if r["ms"] is not None and r["ok"])
        buckets = []
        for b in range(nb):
            xs = [r["ms"] for r in rs if r["ok"] and r["ms"] is not None and b * 300 <= r["t"] < (b + 1) * 300]
            buckets.append(round(statistics.median(xs)) if xs else None)
        out[k] = {"n": len(rs), "fail": sum(not r["ok"] for r in rs), "errors": sum(bool(r.get("errors")) for r in rs),
                  "median_ms": round(statistics.median(ms)) if ms else None,
                  "p95_ms": ms[min(len(ms) - 1, int(len(ms) * 0.95))] if ms else None, "max_ms": ms[-1] if ms else None,
                  "median_by_5min": buckets}
    if mem:
        out["memory"] = {k: {"start": mem[0][k], "end": mem[-1][k],
                             "min" if k == "avail_gb" else "max":
                                 (min if k == "avail_gb" else max)(m[k] for m in mem if m[k] is not None)}
                         for k in ("avail_gb", "ovms_gb", "self_gb", "ui_gb", "swap_used_gb") if mem[0].get(k) is not None}
    return out


def main(a):
    OUT.mkdir(exist_ok=True)
    image = Path(a.image)
    t0 = time.time()
    since = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[soak] {since} 시작 · {a.minutes}분 · YOLO {YOLO_MODEL.name} · 구역 {ZONES.name} · 사진 {image.name} · "
          f"STT {a.stt_every}s TTS {a.tts_every}s YOLO {a.yolo_every}s · 인터넷 {netguard.status()['mode']}", flush=True)
    # 예열(시간 밖): 에이전트 라우터·기준서, TTS 엔진, YOLO NPU 컴파일, STT 첫 요청
    w = agent.warmup()
    tts.speak("예열")
    detect_objects(str(image), model=YOLO_MODEL, zones_path=ZONES, out_dir=OUT / f"soak_{a.tag}_snaps")
    stt.transcribe(str(sorted((ROOT / "ui" / "samples").glob("*.wav"))[0]))
    print(f"[soak] 예열 끝 {time.time() - t0:.0f}s · agent warmup {w.get('total_s')}s", flush=True)

    rec = Rec(time.time())
    end = rec.t0 + a.minutes * 60
    stop = threading.Event()
    mem, foreign = [], []
    th = [threading.Thread(target=stt_loop, args=(rec, end, stop, a.stt_every), daemon=True),
          threading.Thread(target=tts_loop, args=(rec, end, stop, a.tts_every), daemon=True),
          threading.Thread(target=yolo_loop, args=(rec, end, stop, a.yolo_every, image, OUT / f"soak_{a.tag}_snaps"),
                           daemon=True),
          threading.Thread(target=monitor, args=(rec, end, stop, since, mem, foreign), daemon=True)]
    for t in th:
        t.start()

    def progress():
        while not stop.wait(60):
            with rec.lock:
                rs = list(rec.rows)
            last = [r for r in rs if r["t"] >= rs[-1]["t"] - 60] if rs else []
            fails = [r for r in last if not r["ok"]]
            m = mem[-1] if mem else {}
            cmds = [r["ms"] for r in last if r["kind"].startswith("cmd:")]
            print(f"[soak] {rs[-1]['t'] / 60 if rs else 0:4.1f}분 · 최근 1분 명령 {len(cmds)}개 중앙 "
                  f"{statistics.median(cmds) / 1000 if cmds else 0:.2f}s · 실패 {len(fails)} · 가용 {m.get('avail_gb')}GB "
                  f"OVMS {m.get('ovms_gb')}GB 나 {m.get('self_gb')}GB · 다른 세션 {len(foreign)}회", flush=True)
            for r in fails[:3]:
                print(f"         실패 {r['kind']} {r.get('text', r.get('file', ''))} → {r.get('intent')} {r.get('errors')}",
                      flush=True)
            if time.time() >= end:
                return
    pt = threading.Thread(target=progress, daemon=True)
    pt.start()
    try:
        agent_loop(rec, end, stop)
    except Exception:
        rec.add("agent_crash", None, False, errors=[traceback.format_exc()[-400:]])
    stop.set()
    for t in th:
        t.join(timeout=120)
    minutes = (time.time() - rec.t0) / 60
    klines = kernel_lines(since)
    summ = summarize(rec.rows, mem, minutes)
    res = {"tag": a.tag, "started": since, "minutes": round(minutes, 1), "yolo_model": YOLO_MODEL.name, "zones": ZONES.name,
           "env": {k: os.environ.get(k) for k in ("BALL_MOCK_POLICY_S", "BALL_MOCK_FAIL_AT", "BALL_MOCK_SNAP")},
           "cycles": sum(r["kind"] == "cycle" for r in rec.rows), "summary": summ, "kernel": klines,
           "netguard": netguard.status(), "foreign": foreign, "memory": mem, "rows": rec.rows}
    (OUT / f"soak_{a.tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    snaps = OUT / f"soak_{a.tag}_snaps"
    for p in snaps.glob("*.jpg"):
        p.unlink()
    snaps.rmdir() if snaps.exists() and not any(snaps.iterdir()) else None

    print(f"\n[soak] 끝 {minutes:.1f}분 · 바퀴 {res['cycles']} · 커널 이상 {len(klines)}줄 · "
          f"인터넷 시도 {res['netguard']['blocked']} · 다른 세션 겹침 {len(foreign)}회")
    print(f"{'종류':<14}{'n':>5}{'실패':>5}{'중앙':>8}{'p95':>8}{'최대':>8}  5분 단위 중앙(ms)")
    for k, v in summ.items():
        if k == "memory":
            continue
        print(f"{k:<14}{v['n']:>5}{v['fail']:>5}{v['median_ms'] or 0:>8}{v['p95_ms'] or 0:>8}{v['max_ms'] or 0:>8}  "
              f"{v['median_by_5min']}")
    for k, v in summ.get("memory", {}).items():
        print(f"메모리 {k}: {v}")
    for l in klines[:10]:
        print("커널:", l)
    bad = [r for r in rec.rows if not r["ok"]]
    for r in bad[:15]:
        print(f"실패 {r['t']:>7}s {r['kind']} {r.get('text', r.get('file', ''))} → {r.get('intent')} {r.get('errors')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--tag", default="night")
    ap.add_argument("--stt-every", type=float, default=12)
    ap.add_argument("--tts-every", type=float, default=15)
    ap.add_argument("--yolo-every", type=float, default=0.5)
    ap.add_argument("--image", default=str(ROOT / "vision" / "_test" / "scene.jpg"))
    main(ap.parse_args())
