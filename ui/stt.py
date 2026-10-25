"""음성 → 문장. OVMS 의 OpenAI 호환 `/v3/audio/transcriptions`(Whisper, NPU) 를 부른다.

    transcribe(wav_path) -> str      # 계약서 3절. 못 알아들으면 "" (조용함·환각 문구)

환경변수: LLM_BASE_URL(기본 http://127.0.0.1:8100/v3), STT_MODEL(기본 whisper), STT_LANGUAGE(기본 ko),
          STT_FIX(기본 1 = 아래 FIXES 로 이 작업의 잦은 오인식을 바로잡음, 0 = 끔),
          STT_PROMPT(기본 0 = 끔, 1 = 시연 낱말 문맥 PROMPT 를 보냄, 그 밖의 글 = 그 글을 문맥으로)
          ※ 10-21 밤 시험: OVMS 2026.4.1 의 NPU Whisper 는 prompt 를 받아도 무시한다(결과 같음, ui/stt_prompt_check.py).
          STT_PREP(보내기 전 다듬기, 쉼표로: aa = 16kHz 로 줄일 때 8kHz 위를 먼저 걸러 접힘 막기, pad = 말 앞뒤 무음 0.3초 확보,
                   trim = 말 끝 0.4초 뒤·말 앞 0.4초 앞 무음 자르기, norm = 말소리 크기 -20dBFS 로. 비우면 예전 그대로)
                   ※ 고른 기본값과 실측은 docs/stt_real_check.md(python -m ui.stt_real_check)
CLI: python -m ui.stt <파일.wav>
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import requests
import soundfile as sf

BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8100/v3").rstrip("/")
MODEL = os.getenv("STT_MODEL", "whisper")
LANGUAGE = os.getenv("STT_LANGUAGE", "ko")
TIMEOUT_S = float(os.getenv("STT_TIMEOUT", "30"))
MIN_RMS = float(os.getenv("STT_MIN_RMS", "0.004"))  # 이보다 조용하면 말이 없다고 본다
FIX = os.getenv("STT_FIX", "1") != "0"
# OpenAI 의 prompt(초기 문맥). openvino_genai 는 NPU 에서 initial_prompt 를 지원하지 않아 지금 서버(NPU)에선 효과가 없다
PROMPT = "분홍 블록, 초록 블록, 불량품, 작업대, 그릇, 손, 결과 알려 줘"
_p = os.getenv("STT_PROMPT", "0")
PROMPT_ON = "" if _p in ("", "0") else PROMPT if _p == "1" else _p

# 이 작업 말에서 Whisper 가 되풀이해 틀린 것(10-21 밤 TTS 실측, docs/stt_check.md) → 바로잡는다. 원문은 last_info["raw"]
FIXES = [(re.compile(r"분량"), "불량"),            # "불량 몇 개" → "분량 몇 개"(두 목소리 모두)
         (re.compile(r"작업\s*때"), "작업대"),     # "작업대"[작업때] → "작업 때"
         (re.compile(r"골라내죠"), "골라내 줘"),    # 남자 목소리 "골라내 줘" → "골라내죠"
         # 10-22 밤 실물 흉내(docs/stt_real_check.md, 1m 잔향·대화 소음·앞 잘림)에서 되풀이된 것 + 10-22 실물 "불량성불란해죠"·"초록보록"
         (re.compile(r"(분홍|초록)\s*(?:불로|불룩|불록|볼록|볼로|보록)"), r"\1 블록"),
         (re.compile(r"[곤혼골불]란(?:해|하)(?:죠|줘|져)(?:요|고)?|골라\s*(?:해|내)\s*줘|골라야죠"), "골라내 줘"),
         (re.compile(r"불량[성필기]\s*(?=골라내)"), "불량품 "),
         (re.compile(r"[곰공]겨\s*줘"), "옮겨 줘")]

# Whisper 가 조용한 소리에서 지어내는 한국어 문구(방송 자막 학습의 흔적)
HALLUCINATIONS = ("시청해 주셔서 감사합니다", "시청해주셔서 감사합니다", "구독과 좋아요", "MBC 뉴스", "자막 제공")

PREP_STEPS = ("aa", "pad", "trim", "norm")
PREP = tuple(s for s in (p.strip() for p in os.getenv("STT_PREP", "").split(",")) if s in PREP_STEPS)
PAD_S, KEEP_S, TARGET_RMS, MAX_GAIN = 0.3, 0.4, 0.1, 30.0  # 말 앞뒤 확보·남길 무음, -20dBFS, 최대 +29.5dB

last_info: dict = {}  # 마지막 호출의 {latency_ms, seconds, rms, peak, clip, speech_s, lead_s, raw, prep} — 대시보드·녹음 기록용


class STTError(RuntimeError):
    pass


def fix_text(text: str) -> str:
    for pat, rep in FIXES:
        text = pat.sub(rep, text)
    return text


def _lowpass(x: np.ndarray, sr: int, fc: float = 7600.0, taps: int = 161) -> np.ndarray:
    """해밍 창 sinc FIR. 44.1kHz 녹음을 16kHz 로 줄이기 전에 8kHz 위(16kHz 에 담을 수 없는 소리)를 걸러 낮은 대역으로 접히지 않게."""
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * fc / sr * n) * np.hamming(taps)
    return np.convolve(x, (h / h.sum()).astype("float32"), mode="same").astype("float32")


def _load_16k_mono(path: str, aa: bool | None = None) -> np.ndarray:
    """aa: None 이면 STT_PREP 을 따른다."""
    aa = "aa" in PREP if aa is None else aa
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:  # webm·m4a 등은 ffmpeg 로 wav 로 바꿔 다시 읽는다(ffmpeg 리샘플러가 걸러 준다)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000", tmp.name], check=True)
            audio, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000 and len(audio):
        if aa and sr > 16000:
            audio = _lowpass(audio, sr)
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype("float32")
    return audio


def measure(audio: np.ndarray, sr: int = 16000) -> dict:
    """녹음 상태: 최대 음량·클리핑 비율·잡음 바닥(floor)·말소리 길이·첫 말소리까지(lead_s)·마지막 말소리 뒤(tail_s)·말소리 크기.
    말소리 = 20ms 틀 세기가 문턱(잡음 바닥 p10 × 3, 가장 큰 틀 × 0.1, 0.003 중 큰 값)을 넘는 틀."""
    fr = int(sr * 0.02)
    n = len(audio) // fr
    out = {"seconds": round(len(audio) / sr, 2), "peak": round(float(np.abs(audio).max()) if len(audio) else 0.0, 4),
           "clip": round(float((np.abs(audio) >= 0.99).mean()) if len(audio) else 0.0, 5),
           "rms": float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0}
    if n == 0:
        return {**out, "floor": 0.0, "speech_s": 0.0, "lead_s": None, "tail_s": None, "speech_rms": 0.0, "first": None, "last": None}
    e = np.sqrt((audio[:n * fr].reshape(n, fr) ** 2).mean(axis=1))
    out["floor"] = float(np.percentile(e, 10))
    thr = max(0.003, 3 * out["floor"], 0.1 * float(e.max()))
    on = np.flatnonzero(e > thr)
    if not len(on):
        return {**out, "speech_s": 0.0, "lead_s": None, "tail_s": None, "speech_rms": 0.0, "first": None, "last": None}
    first, last = int(on[0]) * fr, (int(on[-1]) + 1) * fr
    return {**out, "speech_s": round(len(on) * 0.02, 2), "lead_s": round(first / sr, 2),
            "tail_s": round((len(audio) - last) / sr, 2), "speech_rms": float(np.sqrt(np.mean(e[on] ** 2))),
            "first": first, "last": last}


def prepare(audio: np.ndarray, steps=None, info: dict | None = None) -> np.ndarray:
    """16kHz 소리를 보내기 전에 다듬는다(steps 는 PREP_STEPS 중 pad·trim·norm, None 이면 STT_PREP)."""
    steps = PREP if steps is None else tuple(steps)
    m = info or measure(audio)
    if m["first"] is None:
        return audio
    first, last, sr = m["first"], m["last"], 16000
    if "trim" in steps:  # 말 앞뒤 무음은 KEEP_S 만 남긴다(긴 꼬리 무음에서 "감사합니다" 같은 글을 지어낸다)
        a, b = max(0, first - int(KEEP_S * sr)), min(len(audio), last + int(KEEP_S * sr))
        audio, first, last = audio[a:b], first - a, last - a
    if "norm" in steps and m["speech_rms"] > 0:
        gain = min(MAX_GAIN, TARGET_RMS / m["speech_rms"], 0.89 / max(m["peak"], 1e-6))
        audio = (audio * gain).astype("float32")
    if "pad" in steps:  # 버튼을 누르자마자 말하면 첫 소리가 녹음 맨 앞에 붙는다 → 앞뒤에 그 녹음의 방 소리 크기로 여유를 둔다
        head = max(0, int(PAD_S * sr) - first)
        tail = max(0, int(PAD_S * sr) - (len(audio) - last))
        if head or tail:
            fl = m["floor"] * (float(np.sqrt(np.mean(audio**2))) / max(m["rms"], 1e-9))  # norm 뒤면 같은 비율로
            rng = np.random.default_rng(0)
            audio = np.concatenate([(rng.standard_normal(head) * fl).astype("float32"), audio,
                                    (rng.standard_normal(tail) * fl).astype("float32")])
    return audio


def _post(wav_bytes: bytes, with_language: bool, prompt: str = "", extra: dict | None = None) -> requests.Response:
    data = {"model": MODEL, **(extra or {})}
    if with_language and LANGUAGE:
        data["language"] = LANGUAGE
    if prompt:
        data["prompt"] = prompt
    return requests.post(
        f"{BASE_URL}/audio/transcriptions",
        files={"file": ("speech.wav", wav_bytes, "audio/wav")},
        data=data,
        timeout=TIMEOUT_S,
    )


def transcribe(wav_path: str, prompt: str | None = None, prep=None, extra: dict | None = None) -> str:
    """prompt: None 이면 STT_PROMPT 를 따르고, "" 이면 안 보냄. prep: None 이면 STT_PREP(PREP_STEPS 중 고른 것).
    extra: 서버에 더 보낼 필드(예: {"temperature": 0.0} — 시험용)."""
    prompt = PROMPT_ON if prompt is None else prompt
    prep = PREP if prep is None else tuple(prep)
    audio = _load_16k_mono(wav_path, aa="aa" in prep)
    m = measure(audio)
    last_info.clear()
    last_info.update({k: v for k, v in m.items() if k not in ("first", "last")}, latency_ms=None, prep=",".join(prep))
    if len(audio) < 16000 * 0.3 or m["rms"] < MIN_RMS:
        return ""
    audio = prepare(audio, prep, m)

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, audio, 16000, subtype="PCM_16")
        wav_bytes = open(tmp.name, "rb").read()

    t0 = time.perf_counter()
    try:
        r = _post(wav_bytes, with_language=True, prompt=prompt, extra=extra)
        if r.status_code == 400 and "prompt" in r.text.lower() and prompt:  # 문맥을 못 받는 서버면 빼고 다시
            r = _post(wav_bytes, with_language=True, extra=extra)
        if r.status_code == 400 and "language" in r.text.lower():
            r = _post(wav_bytes, with_language=False, prompt=prompt, extra=extra)
    except requests.RequestException as e:
        raise STTError(f"음성 인식 서버({BASE_URL})에 연결하지 못했습니다: {e.__class__.__name__}") from e
    last_info["latency_ms"] = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        raise STTError(f"음성 인식 실패 HTTP {r.status_code}: {r.text[:200]}")

    text = (r.json().get("text") or "").strip()
    last_info["raw"] = text
    if any(text.startswith(h) for h in HALLUCINATIONS):
        return ""
    return fix_text(text) if FIX else text


def available() -> bool:
    """OVMS 가 STT 모델을 올려 두었는지(모델 목록에 있는지)."""
    try:
        r = requests.get(f"{BASE_URL}/models", timeout=2)
        return any(m.get("id") == MODEL for m in r.json().get("data", []))
    except Exception:
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("사용법: python -m ui.stt <파일.wav>")
    print(transcribe(sys.argv[1]))
    print(last_info, file=sys.stderr)
