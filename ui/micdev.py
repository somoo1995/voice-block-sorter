"""젠북 마이크 입력 점검 — PipeWire 기본 입력 장치(wpctl)·음소거·최근 대시보드 녹음 크기.

    default_source() -> {"id", "name", "nick", "volume", "muted"} | None
    check(listen_s=0) -> [(상태 "정상"|"주의"|"안됨", 항목, 설명), ...]   # python -m ui.mic_check --device 가 찍는다

10-22 밤: 재부팅 뒤 기본 입력이 이어폰 단자 마이크(Headphones Stereo Microphone)로 잡혀 크롬 녹음이 완전 무음(peak 0)이었다.
내장 마이크 = "Digital Microphone". 바꾸기: `wpctl status` 의 Sources 번호로 `wpctl set-default <번호>` → 대시보드 새로고침.
대시보드와 같은 PC 의 브라우저일 때만 맞다(다른 PC 에서 SSH 터널로 열면 그 PC 의 마이크다).
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = ROOT / "ui" / "voice_log"
JACK_WORDS = ("Headphones", "Headset")  # 이어폰 단자 — 꽂은 게 없으면 무음
_cache: dict = {"t": 0.0, "v": None}


def _run(*args: str) -> str:
    return subprocess.run(list(args), capture_output=True, text=True, timeout=2).stdout


def default_source(max_age: float = 10.0) -> dict | None:
    """기본 입력 장치(10초 캐시). wpctl 이 없거나 PipeWire 에 못 붙으면 None."""
    if time.time() - _cache["t"] < max_age:
        return _cache["v"]
    v = None
    try:
        info = _run("wpctl", "inspect", "@DEFAULT_AUDIO_SOURCE@")
        vol = _run("wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@")
        get = lambda k: (re.search(rf'{re.escape(k)} = "([^"]*)"', info) or [None, None])[1]
        m = re.match(r"id (\d+)", info.strip())
        if m:
            fv = re.search(r"Volume:\s*([\d.]+)", vol)
            v = {"id": int(m.group(1)), "name": get("node.description") or get("node.name"), "nick": get("node.nick"),
                 "volume": float(fv.group(1)) if fv else None, "muted": "MUTED" in vol}
    except Exception:
        v = None
    _cache.update(t=time.time(), v=v)
    return v


def label(src: dict | None = None) -> str:
    src = src if src is not None else default_source()
    if not src:
        return "알 수 없음(wpctl 안 됨)"
    return f"{src.get('nick') or src.get('name')} (wpctl {src['id']})" + (" · 음소거" if src.get("muted") else "")


def is_jack(src: dict | None) -> bool:
    return bool(src) and any(w in (src.get("name") or "") + (src.get("nick") or "") for w in JACK_WORDS)


def last_recordings(n: int = 3) -> list[dict]:
    """대시보드가 남긴 최근 녹음 정보(ui/voice_log/*/voice.jsonl, 새것이 앞)."""
    out = []
    for f in sorted(VOICE_DIR.glob("*/voice.jsonl"), reverse=True):
        lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        for l in reversed(lines):
            try:
                out.append(json.loads(l))
            except ValueError:
                continue
            if len(out) >= n:
                return out
    return out


def listen_peak(seconds: float, target: int | None = None) -> float | None:
    """기본 입력에서 seconds 초 받아 최대 크기만 돌려준다(파일은 /dev/shm 에 잠깐 두고 바로 지움)."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", dir="/dev/shm") as tmp:
        args = ["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16"]
        if target is not None:
            args += ["--target", str(target)]
        p = subprocess.Popen(args + [tmp.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(seconds)
        p.terminate()
        p.wait(timeout=3)
        try:
            a, _ = sf.read(tmp.name, dtype="float32")
        except Exception:
            return None
        return float(abs(a).max()) if len(a) else 0.0


def check(listen_s: float = 0.0) -> list[tuple[str, str, str]]:
    src = default_source(0)
    rows = []
    if not src:
        rows.append(("주의", "마이크 입력", "wpctl 로 기본 입력을 못 읽음 — 젠북 화면 세션에서 `wpctl status`"))
    else:
        st = "안됨" if src["muted"] else "주의" if is_jack(src) or (src["volume"] or 0) < 0.3 else "정상"
        note = (" — 음소거: `wpctl set-mute @DEFAULT_AUDIO_SOURCE@ 0`" if src["muted"] else
                " — 이어폰 단자 마이크(꽂은 게 없으면 무음). 내장 = Digital Microphone: `wpctl status` 의 Sources 번호로 "
                "`wpctl set-default <번호>` → 대시보드 새로고침" if is_jack(src) else "")
        rows.append((st, "마이크 입력", f"{label(src)} · 볼륨 {src['volume']}{note}"))
    recs = last_recordings(3)
    if recs:
        r = recs[0]
        peak = r.get("peak")
        when = (r.get("ts") or "")[5:19].replace("T", " ")
        if peak is not None and peak <= 0:
            rows.append(("안됨", "최근 녹음", f"{when} 녹음이 완전 무음(peak 0) — 입력 장치·음소거 확인 뒤 대시보드 새로고침"))
        elif peak is not None and all((x.get("peak") or 0) < 0.02 for x in recs):
            rows.append(("주의", "최근 녹음", f"최근 {len(recs)}개 모두 아주 작음(최대 {max(x.get('peak') or 0 for x in recs):.3f}) — 마이크 가까이"))
        else:
            rows.append(("정상", "최근 녹음", f"{when} · 최대 {peak} · 말소리 {r.get('speech_s')}초 · “{r.get('text') or ''}”"))
    else:
        rows.append(("주의", "최근 녹음", "대시보드 녹음 기록 없음(ui/voice_log) — 마이크 버튼으로 한 번 말해 보기"))
    if listen_s > 0:
        pk = listen_peak(listen_s, src["id"] if src else None)
        if pk is None:
            rows.append(("주의", "지금 소리", "pw-record 로 받지 못함"))
        else:
            rows.append(("안됨" if pk <= 0 else "주의" if pk < 0.002 else "정상", "지금 소리",
                         f"{listen_s:g}초 최대 {pk:.4f}" + (" — 완전 무음: 입력 장치 확인" if pk <= 0 else "")))
    return rows
