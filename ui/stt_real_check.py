"""받아쓰기 실물 점검 — 실물처럼 망가뜨린 음성(또는 대시보드가 남긴 실물 녹음)을 같은 파일로 다듬기·설정·모델만 바꿔 비교한다.

    python -m ui.stt_real_check                      # 합성 26개 × 실물 흉내 6조건 × 다듬기 6가지 + 말 없는 소리 → docs/stt_real_check.md
    python -m ui.stt_real_check --voice-log ui/voice_log/20261023   # 대시보드가 남긴 실물 녹음(.wav + voice.jsonl)을 다듬기별로 다시
        [--since 23:02 --refs demo --agent-db runs/agent_checkpoints_mock.sqlite]  # 그 시각 뒤만·정답 추정(실연 문장)·그때 의도
    python -m ui.stt_real_check --cpu-prompt         # + openvino_genai CPU(initial_prompt 있음/없음)로 가장 나쁜 조건 비교(느림)
    python -m ui.stt_real_check --quick              # 조건 3개·다듬기 3가지만

실물 흉내(합성 음성 ui/samples/tts_{F1,M1}_01~12 + demo/{F1,M1}_13, 44.1kHz):
  깨끗 / 1m 잔향(RT60 0.6초, 직접음 = 잔향) / 1m + 대화 10dB·5dB(다른 말 5개를 멀리서 겹침) / 1m + 소음 5dB(분홍 잡음 + 120Hz 웅웅)
  / 1m + 앞 잘림(첫 말소리 0.12초를 잃음 — 버튼을 누르며 바로 말했는데 마이크가 늦게 열림). 모두 말소리 -35dBFS, 브라우저처럼 opus(48kHz) → 44.1kHz wav.
다듬기(ui/stt.py STT_PREP): 없음(예전) · aa · aa+pad · aa+trim · aa+norm · 모두.
채점: 원문(바로잡기 전)을 ui/stt_check 처럼(띄어쓰기·문장부호·동의어·숫자 표기 무시). "의도 같음" = 에이전트 임베딩 라우터
(agent.router, LLM 안 부름)가 받아쓴 글을 정답 문장과 같은 의도로 보냄. 말 없는 소리는 ui/stt 거르기·agent/hearing 거르기를 함께 본다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from ui import stt
from ui.stt_check import SENTENCES, cer, norm

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "ui" / "samples"
SIM = SAMPLES / "sim"
OUT = ROOT / "docs" / "stt_real_check.md"
SR = 44100
LEVEL_DB = -35.0
ALL = [*SENTENCES, ("13", "블록으로 숫자 7 만들어 줘", "arrange")]
BABBLE_SRC = ("q1", "q2", "q3", "c1", "c2")  # 명령 낱말이 없는 말(질문·잡담)만 배경 대화로
CONDS = [("clean", "깨끗"), ("room", "1m 잔향"), ("babble10", "1m + 대화 10dB"), ("babble5", "1m + 대화 5dB"),
         ("noise5", "1m + 소음 5dB"), ("cut", "1m + 앞 잘림")]
VARIANTS = [("base", "없음(예전)", ()), ("aa", "aa", ("aa",)), ("pad", "aa+pad", ("aa", "pad")),
            ("trim", "aa+trim", ("aa", "trim")), ("norm", "aa+norm", ("aa", "norm")), ("all", "모두", stt.PREP_STEPS)]
QUICK_CONDS, QUICK_VARIANTS = ("clean", "babble10", "cut"), ("base", "aa", "all")


# ── 실물 흉내 ────────────────────────────────────────────────────────────
def _active_rms(a: np.ndarray) -> float:
    fr = int(SR * 0.02)
    n = len(a) // fr
    e = np.sqrt((a[:n * fr].reshape(n, fr) ** 2).mean(axis=1))
    on = e[e > e.max() * 0.1]
    return float(np.sqrt(np.mean(on**2))) if len(on) else 1e-9


def _onset(a: np.ndarray) -> int:
    fr = int(SR * 0.02)
    n = len(a) // fr
    e = np.sqrt((a[:n * fr].reshape(n, fr) ** 2).mean(axis=1))
    return int(np.flatnonzero(e > e.max() * 0.1)[0]) * fr


def _conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = 1 << int(np.ceil(np.log2(len(x) + len(h))))
    return np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(h, n), n)[:len(x)].astype("float32")


def _room_ir(rng, rt60=0.6, drr_db=0.0) -> np.ndarray:
    n = int(SR * rt60)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)
    tail[:int(0.003 * SR)] = 0
    for i in range(1, n):  # 벽에서 고음이 먼저 줄어드는 것을 한 극 저역 통과로
        tail[i] = 0.6 * tail[i - 1] + 0.4 * tail[i]
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.sum(tail**2))
    tail[0] += 1.0
    return tail.astype("float32")


def _pink(rng, n: int) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) / np.sqrt(np.maximum(f, 20))
    x = np.fft.irfft(spec, n)
    return (x / np.std(x)).astype("float32")


def _babble(rng, n: int, clips: list[np.ndarray]) -> np.ndarray:
    """다른 사람 말 여러 개를 멀리서(잔향 많이) 쉬지 않고 겹친다."""
    out = np.zeros(n, "float32")
    far = _room_ir(rng, 0.6, -6.0)
    for c in clips:
        c = _conv(np.concatenate([c, np.zeros(int(0.3 * SR), "float32")]), far)
        c /= _active_rms(c)
        off = int(rng.integers(0, len(c)))
        out += np.tile(c, n // len(c) + 2)[off:off + n]
    return out / max(np.std(out), 1e-9)


def _opus(a: np.ndarray, kbps: int = 64) -> np.ndarray:
    """브라우저 MediaRecorder(opus 48kHz) → Gradio 가 44.1kHz wav 로 푸는 길."""
    with tempfile.TemporaryDirectory() as d:
        src, enc, dec = f"{d}/a.wav", f"{d}/a.webm", f"{d}/b.wav"
        sf.write(src, np.clip(a, -1, 1), SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ar", "48000", "-c:a", "libopus", "-b:a", f"{kbps}k", enc],
                       check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", enc, "-ar", str(SR), "-ac", "1", dec], check=True)
        return sf.read(dec, dtype="float32")[0]


def _load44(path: Path) -> np.ndarray:
    a, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, f"{path} {sr}Hz"
    return a.mean(axis=1)


def _src(voice: str, no: str) -> Path:
    return SAMPLES / ("demo" if no == "13" else "") / (f"{voice}_13.wav" if no == "13" else f"tts_{voice}_{no}.wav")


def build_sims(voices: list[str], conds: list[str], nos: set[str] | None = None) -> list[dict]:
    """조건별 파일을 ui/samples/sim/<조건>/<목소리>_<번호>.wav 에 만든다(씨앗 고정이라 다시 만들어도 같다)."""
    items = []
    for v in voices:
        other = "M1" if v == "F1" else "F1"
        clips = [_load44(SAMPLES / f"tts_{other}_{k}.wav") for k in BABBLE_SRC]
        for k, (no, ref, intent) in enumerate(ALL):
            if nos and no not in nos:
                continue
            clean = _load44(_src(v, no))
            rng = np.random.default_rng(1000 + k + (0 if v == "F1" else 500))
            ir = _room_ir(rng)
            wet = _conv(clean, ir)
            floor = _pink(rng, len(wet)) * _active_rms(wet) * 10 ** (-30 / 20)
            for cond in conds:
                if cond == "clean":
                    a = clean.copy()
                else:
                    a = wet + floor
                    if cond.startswith("babble"):
                        a = a + _babble(rng, len(a), clips) * _active_rms(wet) * 10 ** (-int(cond[6:]) / 20)
                    elif cond == "noise5":
                        t = np.arange(len(a)) / SR
                        hum = 0.3 * np.sin(2 * np.pi * 120 * t) + 0.15 * np.sin(2 * np.pi * 240 * t)
                        a = a + (_pink(rng, len(a)) + hum.astype("float32")) * _active_rms(wet) * 10 ** (-5 / 20)
                    elif cond == "cut":
                        a = a[_onset(clean) + int(0.12 * SR):]  # 녹음이 늦게 열려 첫 말소리 0.12초를 잃음
                a = a / _active_rms(a) * 10 ** (LEVEL_DB / 20)
                a = _opus(a)
                p = SIM / cond / f"{v}_{no}.wav"
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(p, np.clip(a, -1, 1), SR, subtype="PCM_16")
                items.append({"voice": v, "no": no, "ref": ref, "intent": intent, "cond": cond, "wav": p})
    return items


# ── 말 없는 소리(환청) ────────────────────────────────────────────────────
def build_noise() -> list[dict]:
    rng = np.random.default_rng(7)
    out = []
    clips = [_load44(SAMPLES / f"tts_F1_{k}.wav") for k in BABBLE_SRC] + [_load44(SAMPLES / f"tts_M1_{k}.wav") for k in BABBLE_SRC]

    def add(name, a, desc):
        p = SIM / "noise" / f"{name}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, np.clip(_opus(a), -1, 1), SR, subtype="PCM_16")
        out.append({"name": name, "desc": desc, "wav": p})

    for i, sec in enumerate((1.3, 2.5)):
        n = int(SR * sec)
        room = _pink(rng, n) * 10 ** (-55 / 20)
        add(f"room{i}", room, f"조용한 방 {sec}초(-55dBFS)")
        click = room.copy()
        click[int(0.05 * SR):int(0.05 * SR) + 300] += np.hanning(300).astype("float32") * 0.3
        add(f"click{i}", click, f"마이크 켜는 딸깍 + 조용한 방 {sec}초")
        t = np.arange(n) / SR
        fan = (_pink(rng, n) + 0.3 * np.sin(2 * np.pi * 120 * t)) * 10 ** (-35 / 20)
        add(f"fan{i}", fan.astype("float32"), f"환풍기·프로젝터 소음 {sec}초(-35dBFS)")
        bab = _babble(rng, n, [clips[j] for j in rng.choice(len(clips), 5, replace=False)]) * 10 ** (-35 / 20)
        add(f"babble{i}", bab, f"멀리서 들리는 대화 {sec}초(-35dBFS)")
        motor = sum(np.sin(2 * np.pi * f * t) / (k + 1) for k, f in enumerate((310, 620, 930, 1240)))
        motor = (motor * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.5 * _pink(rng, n)) * 10 ** (-32 / 20)
        add(f"motor{i}", motor.astype("float32"), f"서보 모터 윙윙 {sec}초(-32dBFS)")
    return out


# ── 실행 ─────────────────────────────────────────────────────────────────
class Router:
    """에이전트 임베딩 라우터(LLM 안 부름). 못 불러오면 의도 칸을 비운다."""

    def __init__(self):
        try:
            from agent.router import get_router
            self.r = get_router()
            self.cache: dict[str, str] = {}
        except Exception as e:
            print(f"라우터 없음: {e}", file=sys.stderr)
            self.r = None

    def __call__(self, text: str) -> str | None:
        if self.r is None or not text:
            return None
        if text not in self.cache:
            self.cache[text] = self.r.route(text)["intent"]
        return self.cache[text]


def run_sims(items: list[dict], variants: list[tuple], router: Router) -> list[dict]:
    from agent import hearing
    rows = []
    for it in items:
        ref_intent = router(it["ref"])
        sp = hearing.speech_seconds(str(it["wav"]))  # 소리 기준 환청 거르기가 진짜 말을 거르지 않는지
        for key, _, steps in variants:
            try:
                stt.transcribe(str(it["wav"]), prompt="", prep=steps)
                hyp, ms, err = stt.last_info.get("raw", ""), stt.last_info.get("latency_ms"), None
            except Exception as e:
                hyp, ms, err = "", None, f"{e.__class__.__name__}: {e}"
            rows.append({**{k: v for k, v in it.items() if k != "wav"}, "variant": key, "hyp": hyp, "ms": ms, "err": err,
                         "cer": cer(it["ref"], hyp), "exact": norm(it["ref"]) == norm(hyp),
                         "intent_ok": None if ref_intent is None else router(hyp) == ref_intent,
                         "speech_s": sp, "hear": hearing.check(hyp, {"speech_s": sp}) if hyp else None,
                         "ui_speech_s": stt.last_info.get("speech_s"),
                         "hear_ui": hearing.check(hyp, {"speech_s": stt.last_info.get("speech_s")}) if hyp else None,
                         "wav": str(it["wav"].relative_to(ROOT))})
        print(f"{it['cond']} {it['voice']}_{it['no']} " + " · ".join(f"{r['variant']}={r['hyp']!r}" for r in rows[-len(variants):]),
              file=sys.stderr)
    return rows


def run_noise(items: list[dict]) -> list[dict]:
    from agent import hearing
    rows = []
    for it in items:
        for key, steps in (("base", ()), ("all", stt.PREP_STEPS)):
            fixed = stt.transcribe(str(it["wav"]), prompt="", prep=steps)
            raw = stt.last_info.get("raw", "")
            sp = hearing.speech_seconds(str(it["wav"]))
            rows.append({**{k: v for k, v in it.items() if k != "wav"}, "variant": key, "raw": raw, "ui": fixed,
                         "hear_text": hearing.check(fixed) if fixed else "ui 가 거름",
                         "hear_sound": hearing.check(fixed, {"speech_s": sp}) if fixed else "ui 가 거름", "speech_s": sp,
                         "ui_speech_s": stt.last_info.get("speech_s"),
                         "hear_ui": hearing.check(fixed, {"speech_s": stt.last_info.get("speech_s")}) if fixed else "ui 가 거름",
                         "rms": stt.last_info.get("rms")})
        print(f"noise {it['name']} " + " · ".join(f"{r['variant']}={r['raw']!r}" for r in rows[-2:]), file=sys.stderr)
    return rows


def run_cpu_prompt(items: list[dict], cond: str, threads: int) -> tuple[list[dict], str]:
    from ui.stt_prompt_check import GenaiCpu
    be = GenaiCpu(threads)
    rows = []
    for it in [i for i in items if i["cond"] == cond]:
        audio = stt.prepare(stt._load_16k_mono(str(it["wav"]), aa=True), stt.PREP_STEPS)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            sf.write(tmp.name, audio, 16000, subtype="PCM_16")
            for key, p in (("cpu", ""), ("cpu_prompt", stt.PROMPT)):
                hyp, ms = be(Path(tmp.name), p)
                rows.append({"voice": it["voice"], "no": it["no"], "ref": it["ref"], "cond": cond, "variant": key, "hyp": hyp,
                             "ms": ms, "cer": cer(it["ref"], hyp), "exact": norm(it["ref"]) == norm(hyp)})
        print(f"cpu {it['voice']}_{it['no']} " + " · ".join(f"{r['variant']}={r['hyp']!r}" for r in rows[-2:]), file=sys.stderr)
    return rows, be.desc


# ── 결과 문서 ─────────────────────────────────────────────────────────────
def _cell(rs: list[dict]) -> str:
    n = len(rs)
    ex = sum(r["exact"] for r in rs)
    io = [r["intent_ok"] for r in rs if r.get("intent_ok") is not None]
    s = f"{ex}/{n} · {sum(r['cer'] for r in rs) / max(1, n):.3f}"
    return s + (f" · {sum(io)}/{len(io)}" if io else "")


def to_markdown(rows, variants, conds, noise_rows, cpu, elapsed) -> str:
    vk = [v[0] for v in variants]
    vl = {v[0]: v[1] for v in variants}
    cl = dict(CONDS)
    ms = [r["ms"] for r in rows if r.get("ms")]
    out = [f"## {datetime.now():%Y-%m-%d %H:%M} — 실물 흉내 {len(conds)}조건 × 다듬기 {len(variants)}가지", "",
           f"- 받아쓰기 OVMS `{stt.BASE_URL}` `{stt.MODEL}`(NPU), 문맥 없음, 원문(바로잡기 전) 채점. 파일 {len(rows) // max(1, len(variants))}개 × "
           f"다듬기 {len(variants)} = {len(rows)}번, 평균 {np.mean(ms) if ms else 0:.0f} ms. {elapsed / 60:.0f}분 걸림.",
           "- 칸 = 완전 일치 / 평균 CER / 의도 같음(에이전트 임베딩 라우터가 정답 문장과 같은 의도로 보냄).", "",
           "| 조건 | " + " | ".join(vl[k] for k in vk) + " |", "|---|" + "---|" * len(vk)]
    for c in conds:
        out.append(f"| {cl[c]} | " + " | ".join(_cell([r for r in rows if r["cond"] == c and r["variant"] == k]) for k in vk) + " |")
    out.append("| **합** | " + " | ".join(f"**{_cell([r for r in rows if r['variant'] == k])}**" for k in vk) + " |")
    for key, lab in (("hear", "agent.speech_seconds(녹음)"), ("hear_ui", "ui/stt 말소리 길이(상대 문턱)")):
        wrong = [r for r in rows if r.get(key)]
        out += [f"- 소리 기준 환청 거르기(`agent.hearing.check(글, {{speech_s}})`, speech_s = {lab})가 진짜 말을 거른 것: "
                f"{len(wrong)}/{len(rows)}" + "".join(f" · {cl[r['cond']]} {r['voice']}_{r['no']} “{r['hyp']}” {r[key]}"
                                                       for r in wrong[:4])]
    first, last = vk[0], vk[-1]
    out += ["", f"### 틀린 글 ({vl[first]} → {vl[last]})", "", "| 조건 | 목소리 | 정답 | " + vl[first] + " | " + vl[last] + " |",
            "|---|---|---|---|---|"]
    for c in conds:
        for r0 in [r for r in rows if r["cond"] == c and r["variant"] == first]:
            r1 = next(r for r in rows if r["cond"] == c and r["variant"] == last and r["voice"] == r0["voice"] and r["no"] == r0["no"])
            if not (r0["exact"] and r1["exact"]):
                mark = lambda r: r["hyp"] if r["exact"] else f"**{r['hyp']}**" + ("" if r.get("intent_ok") is not False else " ✗의도")
                out.append(f"| {cl[c]} | {r0['voice']} | {r0['ref']} | {mark(r0)} | {mark(r1)} |")
    if noise_rows:
        out += ["", "### 말 없는 소리 (환청)", "", "- ui = `ui/stt.py` 거르기(음량 `STT_MIN_RMS`·앞부분 환청 문구) 뒤 글, "
                "글 기준 = `agent/hearing.check(글)`, 소리 기준 = 거기에 `speech_s`(agent.speech_seconds)를 넘긴 것.", "",
                "| 소리 | 다듬기 | Whisper 원문 | ui | 글 기준 | 소리 기준 agent(초) | 소리 기준 ui(초) |", "|---|---|---|---|---|---|---|"]
        for r in noise_rows:
            out.append(f"| {r['desc']} | {vl.get(r['variant'], r['variant'])} | {r['raw'] or '(빈 글)'} | {r['ui'] or '(거름)'} | "
                       f"{r['hear_text'] or '**통과**'} | {r['hear_sound'] or '**통과**'} ({r['speech_s']}) | "
                       f"{r['hear_ui'] or '**통과**'} ({r['ui_speech_s']}) |")
    if cpu:
        crow, desc = cpu
        out += ["", f"### CPU 문맥(initial_prompt) — {desc}", "",
                f"- 조건 {cl[crow[0]['cond']]}, 다듬기 모두, 문맥 “{stt.PROMPT}”.", "",
                "| | 완전 일치 / CER | 평균 ms |", "|---|---|---|"]
        for k, lab in (("cpu", "CPU 문맥 없음"), ("cpu_prompt", "CPU 문맥")):
            rs = [r for r in crow if r["variant"] == k]
            out.append(f"| {lab} | {sum(r['exact'] for r in rs)}/{len(rs)} · {sum(r['cer'] for r in rs) / len(rs):.3f} | "
                       f"{np.mean([r['ms'] for r in rs]):.0f} |")
    out.append("")
    return "\n".join(out)


# 실연 ①~⑤·멈춰 와 에이전트가 이 문장에 내는 의도(10-22 가짜 작업대·실물 기록). 라우터만 쓰는 칸은 라우터가 정답 문장에 낸 의도와 비교
DEMO_INTENTS = {"불량품 골라내 줘": "defect", "지금 배치 어때?": "scene", "불량 빼고 나머지 다 정리해 줘": "plan",
                "분홍 블록 하나 내 손에 줘": "handoff", "결과 알려 줘": "report", "멈춰": "stop"}
DEMO_REFS = list(DEMO_INTENTS)
OLD_FIXES = 3  # stt.FIXES 앞 3개 = 10-21 밤 것(그 뒤는 10-22 밤에 더함)


def _agent_history(db: Path, since: str) -> list[dict]:
    """에이전트 대화 기록(LangGraph sqlite, 읽기 전용 사본)에서 since(HH:MM) 뒤 명령의 {started, text, intent, status}."""
    import shutil
    import sqlite3

    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    with tempfile.TemporaryDirectory() as d:
        for suf in ("", "-wal", "-shm"):
            if Path(f"{db}{suf}").exists():
                shutil.copy(f"{db}{suf}", f"{d}/c.sqlite{suf}")
        c = sqlite3.connect(f"{d}/c.sqlite")
        ser = JsonPlusSerializer()
        out, seen = [], set()
        for t, cp in c.execute("select type, checkpoint from checkpoints order by checkpoint_id"):
            for h in (ser.loads_typed((t, cp)).get("channel_values", {}).get("history") or []):
                k = (h.get("started"), h.get("text"))
                if k in seen or not h.get("started") or h["started"][11:16] < since:
                    continue
                seen.add(k)
                out.append({"started": h["started"], "text": h.get("text"), "intent": h.get("intent"), "status": h.get("status")})
        c.close()
    return sorted(out, key=lambda h: h["started"])


def run_voice_log(folder: Path, variants: list[tuple], since: str = "", refs: list[str] | None = None,
                  db: Path | None = None, ref_map: dict | None = None) -> str:
    """대시보드가 남긴 실물 녹음: voice.jsonl 의 {ts, file, text(그때 받아쓴 글), ref(정답, 있으면)}.
    refs 를 주면 정답이 없는 녹음은 다시 받아쓴 글들과 가장 가까운 문장을 정답으로 잡는다(추정 — 표에 ~ 표시).
    db(에이전트 sqlite)를 주면 그때 에이전트가 정한 의도를 같이 적는다(대시보드 글과 같고 녹음 뒤 30초 안의 명령).
    ref_map = {파일 이름: {"ref": 정답 글, "intent": 맞는 의도}} — 사람이 확인한 정답(정답 글이 없으면 글 일치는 세지 않음).
    "지금" 칸 = 지금 에이전트 판별(agent.router.classify, LLM 포함)에 새 FIXES 글을 넣은 의도."""
    import re

    from agent import hearing

    def is_stop_text(t: str) -> bool:  # ui/app.py is_stop_text 와 같게(대시보드는 이 글이면 에이전트 전에 멈춘다)
        c = re.sub(r"[\W_]+", "", t or "")
        return 0 < len(c) <= 10 and any(w in c for w in ("멈춰", "스톱", "정지", "중지", "그만"))

    recs = [json.loads(l) for l in (folder / "voice.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    recs = [r for r in recs if r.get("ts", "")[11:16] >= since]
    silent = [r for r in recs if (r.get("peak") if r.get("peak") is not None else 1) <= 0]  # 마이크 입력 0(장치 문제) — 받아쓰기 비교에서 뺀다
    recs = [r for r in recs if r not in silent]
    hist = _agent_history(db, since) if db else []
    router = Router()
    try:
        from agent.router import classify
    except Exception:
        classify = None
    vl = [v[1] for v in variants]
    out = [f"## {datetime.now():%Y-%m-%d %H:%M} — 실물 녹음 `{folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder}`"
           f"{f' {since} 뒤' if since else ''} {len(recs)}개"
           + (f"(완전 무음 {len(silent)}개 {silent[0]['ts'][11:19]}~{silent[-1]['ts'][11:19]} 는 마이크 입력 문제라 뺌)" if silent else ""), "",
           "- 그때 = 대시보드가 녹음 직후 받아쓴 글(그때 설정)과 에이전트가 정한 의도. 다시 = 같은 파일을 다듬기별로 다시 받아쓴 원문, "
           "FIXES = 그 원문(첫 다듬기)에 10-22 밤 바로잡기를 더한 글. 의도 ✓ = 정답 문장과 같은 의도(멈춤은 멈춤 규칙).", "",
           "| 시각 | 길이·말소리·앞 무음·최대 | 정답 | 그때 받아쓴 글 | 그때 의도 | " + " | ".join(f"다시: {x}" for x in vl)
           + " | FIXES 옛→새 | 지금 의도(새 FIXES 글) |", "|---|---|---|---|---|" + "---|" * len(vl) + "---|---|"]
    tally = {"then_exact": 0, "then_intent": 0, "new_exact": 0, "new_intent": 0, "n": 0, "then_intent_n": 0}
    per_var = {x: [0, 0] for x in vl}

    def intent_of(t):
        return "stop" if is_stop_text(t) else router(t)

    for r in recs:
        wav = folder / r["file"]
        hyps = []
        for _, _, steps in variants:
            try:
                stt.transcribe(str(wav), prompt="", prep=steps)
                hyps.append(stt.last_info.get("raw", "") or "")
            except Exception as e:
                hyps.append(f"(오류 {e.__class__.__name__})")
        m = stt.measure(stt._load_16k_mono(str(wav)))
        ref, guessed = r.get("ref"), False
        fixed_want = (ref_map or {}).get(r["file"], {})
        ref = ref or fixed_want.get("ref")
        if not ref and refs and not fixed_want:
            pool = [h for h in hyps + [r.get("raw") or r.get("text") or ""] if h]
            ref = min(refs, key=lambda c: min(cer(c, h) for h in pool)) if pool else ""
            guessed = True
        old_fix = hyps[0]
        for pat, rp in stt.FIXES[:OLD_FIXES]:
            old_fix = pat.sub(rp, old_fix)
        new_fix = stt.fix_text(hyps[0])
        want = intent_of(ref) if ref else None                     # 라우터 기준
        want_agent = fixed_want.get("intent") or (DEMO_INTENTS.get(ref, want) if ref else None)  # 에이전트 기준
        want = want_agent or want
        then_text = r.get("text") or ""
        ts = r.get("ts", "")
        agent = next((h for h in hist if h["text"] == then_text and 0 <= _sec(h["started"]) - _sec(ts) <= 30), None)
        then_intent = "stop" if then_text and is_stop_text(then_text) else (agent or {}).get("intent")
        if not then_text:
            then_intent = "(못 들음)"
        elif then_intent is None and hearing.check(then_text):
            then_intent = "unheard"
        mk = lambda t: (t or "(빈 글)") if not ref or norm(t) == norm(ref) else f"**{t or '(빈 글)'}**"  # 정답 글이 없으면 굵게 안 함
        hit = lambda i, w: i is not None and w is not None and i in str(w).split("|")  # "question|none" = 둘 다 맞음
        ok = lambda i, w=want: "" if w is None or i is None else (" ✓" if hit(i, w) else " ✗")
        tally["n"] += 1
        tally["refs"] = tally.get("refs", 0) + bool(ref)
        tally["then_exact"] += bool(ref) and norm(then_text) == norm(ref)
        tally["new_exact"] += bool(ref) and norm(new_fix) == norm(ref)
        ni = (("stop" if is_stop_text(new_fix) else classify(new_fix).get("intent")) if classify else intent_of(new_fix)) \
            if new_fix else None
        tally["new_intent"] += hit(ni, want)
        if then_intent not in (None,):
            tally["then_intent_n"] += 1
            tally["then_intent"] += hit(then_intent, want_agent)
        for x, h in zip(vl, hyps):
            per_var[x][0] += bool(ref) and norm(h) == norm(ref)
            per_var[x][1] += hit(intent_of(h), want) if h else False
        out.append(f"| {ts[11:19]} | {m['seconds']}초·{m['speech_s']}초·{m['lead_s']}초·{m['peak']:.2f} | {'~' if guessed else ''}{ref or '-'} | "
                   f"{mk(then_text)} | {then_intent or '-'}{ok(then_intent, want_agent)} | " + " | ".join(mk(h) for h in hyps)
                   + f" | {old_fix} → {mk(new_fix)} | {ni}{ok(ni)} |")
    n = max(1, tally["n"])
    nr = tally.get("refs", 0)
    out += ["", f"- 그때: 받아쓴 글 정답 {tally['then_exact']}/{nr}(정답 글을 아는 것만), 의도 맞음 {tally['then_intent']}/{tally['then_intent_n']}. "
            f"다시(정답 글 / 임베딩 라우터 의도: " + ", ".join(f"{x} {a}/{nr}·{b}/{tally['n']}" for x, (a, b) in per_var.items())
            + f"). 새 FIXES 글 {tally['new_exact']}/{nr}, 지금 에이전트 의도 {tally['new_intent']}/{tally['n']}.", ""]
    return "\n".join(out)


def _sec(iso: str) -> float:
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return 0.0


def main():
    ap = argparse.ArgumentParser(description="받아쓰기 실물 점검")
    ap.add_argument("--voices", default="F1,M1")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nos", help="문장 번호만(쉼표, 예 01,05 — 빠른 확인용)")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--cpu-prompt", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--voice-log", help="대시보드 녹음 폴더(ui/voice_log/<날짜>)")
    ap.add_argument("--since", default="", help="--voice-log 에서 이 시각(HH:MM) 뒤 녹음만")
    ap.add_argument("--refs", help="정답 후보(|로 나눔). 'demo' = 실연 ①~⑤·멈춰. 녹음마다 가장 가까운 것을 정답으로(추정)")
    ap.add_argument("--agent-db", help="에이전트 대화 기록 sqlite(예 runs/agent_checkpoints_mock.sqlite) — 그때 의도")
    ap.add_argument("--ref-map", help="JSON {파일: {ref, intent}} — 사람이 확인한 정답·맞는 의도")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if not stt.available():
        sys.exit(f"OVMS 에 {stt.MODEL} 모델이 없습니다({stt.BASE_URL}/models)")
    variants = [v for v in VARIANTS if not a.quick or v[0] in QUICK_VARIANTS]
    t0 = time.time()
    if a.voice_log:
        refs = DEMO_REFS if a.refs == "demo" else a.refs.split("|") if a.refs else None
        md = run_voice_log(Path(a.voice_log).resolve(), variants, a.since, refs, Path(a.agent_db) if a.agent_db else None,
                           json.loads(Path(a.ref_map).read_text(encoding="utf-8")) if a.ref_map else None)
    else:
        conds = [c for c, _ in CONDS if not a.quick or c in QUICK_CONDS]
        items = build_sims([v.strip() for v in a.voices.split(",") if v.strip()], conds, set(a.nos.split(",")) if a.nos else None)
        rows = run_sims(items, variants, Router())
        noise_rows = [] if a.no_noise else run_noise(build_noise())
        cpu = run_cpu_prompt(items, "babble10" if "babble10" in conds else conds[-1], a.threads) if a.cpu_prompt else None
        md = to_markdown(rows, variants, conds, noise_rows, cpu, time.time() - t0)
        dump = ROOT / "ui" / "samples" / "sim" / f"rows_{datetime.now():%Y%m%d_%H%M}.json"
        dump.write_text(json.dumps({"rows": rows, "noise": noise_rows, "cpu": cpu[0] if cpu else []}, ensure_ascii=False, indent=1,
                                   default=str), encoding="utf-8")
    print(md)
    if a.out:
        p = Path(a.out)
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        prev = old[old.index("\n## ") + 1:] if "\n## " in old else ""
        head = ("# 받아쓰기 실물 점검\n\n> `python -m ui.stt_real_check` 로 다시 만든다(새 실측이 위). 굵은 글 = 정답과 다름"
                "(띄어쓰기·문장부호·동의어·숫자 표기 무시), ✗의도 = 에이전트가 다른 명령으로 알아들음.\n\n")
        p.write_text(head + md + "\n" + prev, encoding="utf-8")
        print(f"\n→ {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
