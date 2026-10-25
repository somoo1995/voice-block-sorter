"""문장 → 음성. Supertonic 3 한국어, CPU(onnxruntime).

    speak(text, voice=None, trim=False) -> str  # 계약서 3절. wav 경로. 모델이 없으면 "" (대체 경로: 화면 글자만)
    play(wav_path)                           # 노트북 스피커로 재생(pw-play → paplay → aplay), 끝날 때까지 기다림
    enqueue(text, on_done=None, sound=True)  # 합성+재생을 순서대로(앞 말이 끝나야 다음 말). sound=False 면 합성만, "timed" 면 소리 없이 음성 길이만큼 기다림

말하기 줄(enqueue)은 긴 답을 문장(긴 문장은 쉼표·"에 따르면" 뒤)으로 나눠 첫 조각부터 합성·재생하고, 앞 조각을 재생하는 동안
다음 조각을 합성한다(합성 스레드 → 재생 스레드). 조각마다 Supertonic 이 붙이는 앞 0.4~0.6초·뒤 0.55~0.75초 무음은
앞 0.08초·뒤 0.2초만 남긴다. clear_queue()("멈춰")는 남은 말을 버리고 재생 중인 조각도 끊는다.

모델은 내려받지 않는다(auto_download=False, 시연은 와이파이를 끈다). 찾는 순서:
  TTS_MODEL_DIR → <프로젝트>/models/supertonic-3 → ~/.cache/supertonic3 (supertonic 기본 위치)
환경변수: TTS_VOICE(기본 F1, M1~M5·F1~F5), TTS_SPEED(1.05), TTS_STEPS(8), TTS_NUM_KO(1 = 숫자를 한글로 읽기),
          SUPERTONIC_INTRA_OP_THREADS(기본 4)·SUPERTONIC_INTER_OP_THREADS(기본 1) — onnxruntime 스레드,
          TTS_SPLIT(1 = 말하기 줄을 문장 단위로, 0 = 통째), TTS_TRIM(1 = 말하기 줄의 앞뒤 무음 줄이기)
CLI: python -m ui.tts "불량 3개를 분리했습니다."
"""
from __future__ import annotations

import os

os.environ.setdefault("SUPERTONIC_INTRA_OP_THREADS", "4")  # 기본(자동, 22스레드)은 오히려 느리다: 한 문장 2.4~3.1초 → 0.74~1.2초
os.environ.setdefault("SUPERTONIC_INTER_OP_THREADS", "1")  # (인프라 실측 SETUP.md §6·7.4, 10-22 새벽 대시보드 문장으로 다시 잼)
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 모델은 로컬에서만 찾는다

import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "ui" / "tts_out"
VOICE = os.getenv("TTS_VOICE", "F1")
SPEED = float(os.getenv("TTS_SPEED", "1.05"))
STEPS = int(os.getenv("TTS_STEPS", "8"))
NUM_KO = os.getenv("TTS_NUM_KO", "1") == "1"
KEEP_FILES = 40
SPLIT = os.getenv("TTS_SPLIT", "1") == "1"
TRIM = os.getenv("TTS_TRIM", "1") == "1"
LONG, MIN_PIECE = 40, 12  # 이보다 긴 문장은 쉼표에서 더 나누고, 조각은 이 글자 수보다 짧지 않게 붙인다

_lock = threading.Lock()
_engine = None  # (TTS, style)
_styles: dict = {}  # 다른 목소리(voice=)의 style
_load_error: str | None = None
_last_try = 0.0  # 모델이 없으면 30초마다 다시 찾는다(인프라가 나중에 설치해도 재시작 없이 잡히게)
last_info: dict = {}  # 마지막 합성의 {latency_ms, seconds, engine}


# ── 숫자 읽기: "4개" → "네 개", "13회" → "십삼 회", "18.2초" → "십팔 점 이 초" ──────────────
_SINO = "영 일 이 삼 사 오 육 칠 팔 구".split()
_NATIVE_ONES = ["", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]
_NATIVE_TENS = ["", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔"]
_NATIVE_UNITS = ("개", "명", "번", "가지", "마리", "장", "살")
_UNIT_WORDS = {"%": "퍼센트", "cm": "센티미터", "ms": "밀리초"}
_WORDS = {"X": "엑스", "NPU": "엔피유", "GPU": "지피유", "CPU": "씨피유", "LLM": "엘엘엠"}


def _sino(n: int) -> str:
    if n == 0:
        return "영"
    out = ""
    for value, name in ((10**8, "억"), (10**4, "만")):
        if n >= value:
            head, n = divmod(n, value)
            out += ("" if head == 1 and name == "만" else _sino(head)) + name + " "
    for value, name in ((1000, "천"), (100, "백"), (10, "십")):
        if n >= value:
            d, n = divmod(n, value)
            out += ("" if d == 1 else _SINO[d]) + name
    if n:
        out += _SINO[n]
    return out.strip()


def _native(n: int) -> str:
    if n == 20:
        return "스무"
    return _NATIVE_TENS[n // 10] + _NATIVE_ONES[n % 10]


def _read_number(m: re.Match) -> str:
    num, unit = m.group(1), m.group(2) or ""
    word = _UNIT_WORDS.get(unit, unit)
    if "." in num:
        whole, frac = num.split(".")
        said = _sino(int(whole)) + " 점 " + " ".join(_SINO[int(d)] for d in frac)
    elif unit.startswith(_NATIVE_UNITS) and 0 < int(num) < 100:
        said = _native(int(num))
    else:
        said = _sino(int(num))
    return f"{said} {word}".strip() if word else said


def normalize_ko(text: str) -> str:
    for k, v in _WORDS.items():
        text = re.sub(rf"(?<![A-Za-z]){k}(?![A-Za-z])", v, text)
    if NUM_KO:
        # "(분홍 1, 초록 0)" 처럼 색 뒤 단위 없는 개수 → "분홍 1개" ("분홍 일"은 "분홍이"·"분홍 2"로 들린다)
        text = re.sub(r"(분홍|초록|빨강|파랑|불량) (\d+)(?=\s*[,)])", r"\1 \2개", text)
        text = re.sub(r"(\d+(?:\.\d+)?)\s*(번째|개|명|번|가지|마리|장|살|회|초|분|시간|%|cm|ms)?", _read_number, text)
    return text


# ── 엔진 ───────────────────────────────────────────────────────────────────
def _model_dir() -> Path | None:
    from supertonic.loader import get_cache_dir, has_all_onnx_modules

    cands = [os.getenv("TTS_MODEL_DIR"), ROOT / "models" / "supertonic-3", get_cache_dir("supertonic-3")]
    for c in cands:
        if c and has_all_onnx_modules(c):
            return Path(c)
    return None


def _get_engine():
    global _engine, _load_error, _last_try
    if _engine is not None or time.time() - _last_try < 30:
        return _engine
    _last_try = time.time()
    try:
        from supertonic import TTS

        d = _model_dir()
        if d is None:
            raise FileNotFoundError("Supertonic 3 모델 파일이 없습니다(TTS_MODEL_DIR·models/supertonic-3·~/.cache/supertonic3)")
        tts = TTS(model="supertonic-3", model_dir=d, auto_download=False)
        _engine = (tts, tts.get_voice_style(voice_name=VOICE))
        _load_error = None
    except Exception as e:  # 패키지·모델이 없으면 글자만 보여 주는 대체 경로로
        _load_error = f"{e.__class__.__name__}: {e}"
    return _engine


def available() -> bool:
    with _lock:
        return _get_engine() is not None


def status() -> str:
    """잠금 없이 상태만: ready | missing | loading (대시보드 머리줄용)."""
    return "ready" if _engine is not None else "missing" if _load_error else "loading"


def load_error() -> str | None:
    with _lock:
        _get_engine()
        return _load_error


def _trim(wav, sr: int, lead: float = 0.08, tail: float = 0.2, rel: float = 0.01):
    """앞뒤 무음을 lead·tail 초만 남기고 자른다(최대 진폭의 rel 보다 작은 소리 = 무음)."""
    a = np.abs(np.asarray(wav).reshape(-1))
    if not a.size or a.max() <= 0:
        return wav
    idx = np.flatnonzero(a > rel * a.max())
    return wav[..., max(0, idx[0] - int(lead * sr)):min(a.size, idx[-1] + int(tail * sr))]


def _synth(text: str, voice: str | None = None, trim: bool = False) -> tuple[str, float, float]:
    """(wav 경로, 합성 ms, 음성 초). 모델이 없으면 ("", 0, 0) — last_info 는 건드리지 않는다(말하기 줄이 따로 적음)."""
    with _lock:
        eng = _get_engine()
        if eng is None:
            return "", 0.0, 0.0
        tts, style = eng
        if voice and voice != VOICE:
            style = _styles.get(voice) or _styles.setdefault(voice, tts.get_voice_style(voice_name=voice))
        t0 = time.perf_counter()
        wav, dur = tts.synthesize(normalize_ko(text), voice_style=style, lang="ko", speed=SPEED, total_steps=STEPS)
        secs = float(np.asarray(dur).reshape(-1)[0])
        if trim:
            n = np.asarray(wav).reshape(-1).size
            wav = _trim(wav, round(n / secs) if secs > 0 else 44100)
            secs = secs * np.asarray(wav).reshape(-1).size / n if n else secs
        ms = (time.perf_counter() - t0) * 1000
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"say_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.wav"
        while path.exists():  # 조각을 1ms 안에 둘 만들 수 있다
            path = path.with_name(path.stem + "b.wav")
        tts.save_audio(wav, str(path))
    # 수정 시각 순으로 오래된 것만 지운다(이름의 시각 순이면 자정을 넘긴 새 파일이 "오래된 것"이 되어 바로 지워졌다)
    for old in sorted(OUT_DIR.glob("say_*.wav"), key=lambda f: f.stat().st_mtime)[:-KEEP_FILES]:
        if old != path:
            old.unlink(missing_ok=True)
    return str(path), ms, secs


def speak(text: str, voice: str | None = None, trim: bool = False) -> str:
    """voice: M1~M5·F1~F5 (없으면 TTS_VOICE). trim = 앞뒤 무음 줄이기(말하기 줄과 같게)."""
    text = (text or "").strip()
    if not text:
        return ""
    path, ms, secs = _synth(text, voice, trim)
    last_info.clear()
    if path:
        last_info.update(engine="supertonic-3", latency_ms=ms, seconds=secs)
    else:
        last_info.update(engine=None, error=_load_error)
    return path


def split_text(text: str) -> list[str]:
    """말하기 줄의 조각: 문장(. ? ! 뒤)으로 나누고, LONG 자보다 긴 문장은 괄호 밖 쉼표·"에 따르면" 뒤에서 더 나눈다.
    MIN_PIECE 자보다 짧은 조각은 다음 조각과 붙인다("4.3]"·"1)" 처럼 공백이 없는 점은 안 나눔)."""
    out = []
    for sent in re.split(r"(?<=[.?!])\s+", (text or "").strip()):
        if not sent:
            continue
        if len(sent) <= LONG:
            out.append(sent)
            continue
        parts, buf, depth = [], "", 0
        for i, ch in enumerate(sent):
            buf += ch
            depth += (ch in "([") - (ch in ")]")
            nxt = sent[i + 1:i + 2]
            if depth <= 0 and nxt == " " and (ch == "," or buf.endswith("에 따르면")):
                parts.append(buf.strip())
                buf = ""
        if buf.strip():
            parts.append(buf.strip())
        merged = []
        for p in parts:
            if merged and len(merged[-1]) < MIN_PIECE:
                merged[-1] += " " + p
            else:
                merged.append(p)
        if len(merged) > 1 and len(merged[-1]) < MIN_PIECE:
            tail = merged.pop()  # (merged[-2] += merged.pop() 은 pop 뒤 인덱스로 첫 조각을 덮어쓴다)
            merged[-1] += " " + tail
        out += merged
    return out


# ── 재생 ───────────────────────────────────────────────────────────────────
def player() -> str | None:
    for cmd in ("pw-play", "paplay", "aplay"):
        if shutil.which(cmd):
            return cmd
    return None


_proc: subprocess.Popen | None = None  # 지금 재생 중인 플레이어(clear_queue 가 끊는다)


def play(wav_path: str) -> bool:
    global _proc
    cmd = player()
    if not wav_path or cmd is None:
        return False
    args = [cmd, "-q", wav_path] if cmd == "aplay" else [cmd, wav_path]
    _proc = p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return p.wait(timeout=120) == 0
    except subprocess.TimeoutExpired:
        p.kill()
        return False
    finally:
        _proc = None


_q: queue.Queue = queue.Queue()        # 말(문장 덩어리): (세대, text, on_done, sound, on_play)
_play_q: queue.Queue = queue.Queue()   # 합성된 조각: (세대, 말 정보 dict, 조각 번호, 마지막?, 경로, 초, ms)
_gen = 0                               # clear_queue 마다 +1 → 앞 세대 조각은 재생하지 않는다
_cut = threading.Event()               # clear_queue 가 켠다 → timed 기다림을 끊는다
_workers: list[threading.Thread] = []


def _synth_loop():
    while True:
        gen, text, on_done, sound, on_play = _q.get()
        utt = {"text": text, "on_done": on_done, "sound": sound, "on_play": on_play, "path": "", "seconds": 0.0,
               "first_ms": None, "pieces": 0, "error": None}
        parts = (split_text(text) if SPLIT else [text]) or [text]  # 빈 글이어도 on_done 은 부르게
        for i, part in enumerate(parts):
            path, secs, ms = "", 0.0, 0.0
            if gen == _gen:
                try:
                    path, ms, secs = _synth(part, trim=TRIM)
                    if not path:
                        utt["error"] = _load_error
                except Exception as e:
                    utt["error"] = f"{e.__class__.__name__}: {e}"
            last = i == len(parts) - 1 or not path  # 끊겼거나 합성이 안 되면 이 조각이 마지막
            _play_q.put((gen, utt, i, last, path, secs, ms))
            if last:
                break


def _play_loop():
    while True:
        gen, utt, i, last, path, secs, ms = _play_q.get()
        try:
            if path and gen == _gen:
                if utt["first_ms"] is None:
                    utt["first_ms"], utt["path"] = ms, path
                    if utt["on_play"]:
                        utt["on_play"](utt["text"], path, secs)
                utt["seconds"] += secs
                utt["pieces"] += 1
                if utt["sound"] == "timed":  # 소리는 안 내고 음성 길이만큼 기다림(실연 시간 재기, UI_TTS_OUT=timed)
                    _cut.wait(secs)
                elif utt["sound"]:
                    play(path)
        except Exception as e:
            utt["error"] = f"{e.__class__.__name__}: {e}"
        finally:
            if last:
                last_info.clear()
                last_info.update(engine="supertonic-3" if utt["path"] else None, latency_ms=utt["first_ms"],
                                 seconds=utt["seconds"], pieces=utt["pieces"])
                if utt["error"]:
                    last_info["error"] = utt["error"]
                if utt["on_done"]:
                    utt["on_done"](utt["text"], utt["path"])


def enqueue(text: str, on_done=None, sound: bool | str = True, on_play=None) -> None:
    """차례대로 합성·재생(문장 단위로 미리 합성). sound: True = 스피커, False = 합성만, "timed" = 소리 없이 음성 길이만큼 기다림.
    on_play(text, path, seconds) = 첫 조각 재생을 시작하는 순간(시간 재기용), on_done(text, path) = 마지막 조각까지 끝남.
    on_done 때 last_info = {latency_ms: 첫 조각 합성(= 말 → 첫 소리), seconds: 음성 합, pieces: 조각 수}."""
    if not _workers:
        for fn, name in ((_synth_loop, "tts-synth"), (_play_loop, "tts-play")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            _workers.append(t)
    _cut.clear()
    _q.put((_gen, text, on_done, sound, on_play))


def clear_queue() -> None:
    """아직 말하지 않은 문장을 버리고 재생 중인 조각도 끊는다("멈춰" 때). 버린 말도 on_done 은 부른다."""
    global _gen
    _gen += 1
    _cut.set()
    p = _proc
    if p is not None and p.poll() is None:
        p.terminate()
    try:
        while True:
            _, text, on_done, _, _ = _q.get_nowait()
            if on_done:
                on_done(text, "")
    except queue.Empty:
        pass


if __name__ == "__main__":
    t = " ".join(sys.argv[1:]) or "불량 3개를 분리했습니다. 시도 13회 중 12회 성공했습니다."
    print("읽을 문장:", normalize_ko(t))
    p = speak(t)
    print(p or f"(대체 경로: 소리 없음) {load_error()}", last_info)
