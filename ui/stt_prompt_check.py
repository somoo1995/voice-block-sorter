"""받아쓰기 초기 문맥(prompt) 시험 — 같은 음성을 문맥 없이/있게 받아써 원문(바로잡기 전) 일치·시간·끼워 넣기를 비교한다.

    python -m ui.stt_prompt_check                        # 지금 STT 서버(OVMS /v3/audio/transcriptions 의 prompt)
    python -m ui.stt_prompt_check --backend genai-cpu    # OVMS 를 거치지 않고 openvino_genai WhisperPipeline(CPU)의 initial_prompt
    python -m ui.stt_prompt_check --prompt "…" --repeat 3

음성: 시연 12문장(ui/samples/tts_{F1,M1}_01~12.wav, python -m ui.stt_check 가 만든 것) + 문맥 낱말이 없는 말 5개
(질문 3·잡담 2, 없으면 TTS 로 만들어 ui/samples/tts_{목소리}_q1… 로 남김). 채점은 ui/stt_check.py 와 같고(띄어쓰기·문장부호·
동의어·숫자 표기 무시) 바로잡기(ui/stt.py FIXES)는 안 거친 원문으로 센다. 끼워 넣기 = 정답에 없는 문맥 낱말이 받아쓴 글에 나옴.
결과는 docs/stt_prompt_check.md(새 실측이 위). 시간은 참고(다른 일과 서버를 나눠 쓰면 흔들린다).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from ui import stt, tts
from ui.stt_check import SENTENCES, cer, norm

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "ui" / "samples"
OUT = ROOT / "docs" / "stt_prompt_check.md"
SIDE = [("q1", "실패하면 어떻게 해?", "질문"), ("q2", "로봇이 안 멈추면 어떻게 해야 돼?", "질문"),
        ("q3", "X 표시는 어떻게 그려야 하나요?", "질문"), ("c1", "오늘 날씨 정말 좋네요.", "잡담"), ("c2", "점심은 뭐 먹을까?", "잡담")]
CONTEXT_WORDS = ("분홍", "초록", "블록", "불량", "작업대", "그릇", "손", "결과")
GENAI_MODEL = ROOT / "models" / "OpenVINO" / "whisper-large-v3-turbo-int8-ov"


def _inserted(ref: str, hyp: str) -> list[str]:
    r, h = norm(ref), norm(hyp)
    return [w for w in CONTEXT_WORDS if w in h and w not in r]


class Ovms:
    name = "OVMS"

    def __init__(self):
        if not stt.available():
            sys.exit(f"OVMS 에 {stt.MODEL} 모델이 없습니다({stt.BASE_URL}/models)")
        self.desc = f"OVMS `{stt.BASE_URL}` 모델 `{stt.MODEL}`(`prompt` 필드)"

    def __call__(self, wav: Path, prompt: str) -> tuple[str, float | None]:
        stt.transcribe(str(wav), prompt=prompt)
        return stt.last_info.get("raw", ""), stt.last_info.get("latency_ms")


class GenaiCpu:
    name = "genai-CPU"

    def __init__(self, threads: int):
        import openvino_genai as ov_genai

        props = {"INFERENCE_NUM_THREADS": threads} if threads else {}
        self.pipe = ov_genai.WhisperPipeline(str(GENAI_MODEL), "CPU", **props)
        self.desc = (f"openvino_genai {ov_genai.__version__} WhisperPipeline(CPU, 스레드 {threads or '기본'}) "
                     f"`{GENAI_MODEL.relative_to(ROOT)}`(`initial_prompt`)")

    def __call__(self, wav: Path, prompt: str) -> tuple[str, float | None]:
        audio = stt._load_16k_mono(str(wav)).tolist()
        kw = {"language": "<|ko|>", "task": "transcribe"}
        if prompt:
            kw["initial_prompt"] = prompt
        t0 = time.perf_counter()
        res = self.pipe.generate(audio, **kw)
        return str(res).strip(), (time.perf_counter() - t0) * 1000


def _samples(voices: list[str]) -> list[tuple[str, str, str, str, Path]]:
    """(목소리, 번호, 정답, 종류, 파일)"""
    out = []
    for v in voices:
        for no, ref, intent in SENTENCES:
            wav = SAMPLES / f"tts_{v}_{no}.wav"
            if not wav.exists():
                sys.exit(f"{wav} 없음 — 먼저 python -m ui.stt_check")
            out.append((v, no, ref, "시연", wav))
        for no, ref, kind in SIDE:
            wav = SAMPLES / f"tts_{v}_{no}.wav"
            if not wav.exists():
                shutil.copy(tts.speak(ref, voice=v), wav)
            out.append((v, no, ref, kind, wav))
    return out


def run(backend, prompt: str, voices: list[str], repeat: int) -> list[dict]:
    rows = []
    for v, no, ref, kind, wav in _samples(voices):
        r = {"voice": v, "no": no, "ref": ref, "kind": kind, "none": [], "ctx": []}
        for _ in range(repeat):  # 같은 파일을 문맥 없이·있게 번갈아
            for key, p in (("none", ""), ("ctx", prompt)):
                hyp, ms = backend(wav, p)
                r[key].append((hyp, ms))
        print(f"{v} {no} 없음 {r['none'][0][0]!r} · 문맥 {r['ctx'][0][0]!r}", file=sys.stderr)
        rows.append(r)
    return rows


def _stats(rows: list[dict], key: str, kinds: tuple[str, ...]) -> dict:
    sel = [(r, h) for r in rows if r["kind"] in kinds for h in r[key]]
    ms = [m for _, (_, m) in sel if m is not None]
    return {"n": len(sel), "exact": sum(norm(r["ref"]) == norm(h) for r, (h, _) in sel),
            "cer": sum(cer(r["ref"], h) for r, (h, _) in sel) / max(1, len(sel)),
            "ins": sum(bool(_inserted(r["ref"], h)) for r, (h, _) in sel), "ms": sum(ms) / max(1, len(ms))}


def to_markdown(rows: list[dict], backend, prompt: str, repeat: int, elapsed: float) -> str:
    same = sum(h1 == h2 for r in rows for (h1, _), (h2, _) in zip(r["none"], r["ctx"]))
    total = sum(len(r["none"]) for r in rows)
    out = [f"## {datetime.now():%Y-%m-%d %H:%M} — {backend.name}", "",
           f"- 받아쓰기: {backend.desc}. 문맥: “{prompt}”. 같은 파일을 문맥 없이·있게 번갈아 {repeat}번씩, 원문(바로잡기 전)으로 채점. {elapsed:.0f}초 걸림.",
           f"- 문맥 유무로 받아쓴 글이 똑같이 나온 경우: {same}/{total}", "",
           "| 묶음 | 조건 | 완전 일치 | 평균 CER | 끼워 넣기 | 평균 ms |", "|---|---|---|---|---|---|"]
    for label, kinds in (("시연 12문장", ("시연",)), ("질문 3·잡담 2", ("질문", "잡담"))):
        for key, cond in (("none", "문맥 없음"), ("ctx", "문맥")):
            s = _stats(rows, key, kinds)
            out.append(f"| {label} | {cond} | {s['exact']}/{s['n']} | {s['cer']:.3f} | {s['ins']} | {s['ms']:.0f} |")
    out += ["", "| 목소리 | # | 정답 | 문맥 없음(1회차) | 문맥(1회차) | 끼워 넣은 낱말 |", "|---|---|---|---|---|---|"]
    for r in rows:
        h0, h1 = r["none"][0][0], r["ctx"][0][0]
        ins = sorted({w for h, _ in r["ctx"] for w in _inserted(r["ref"], h)})
        mark = lambda h: h if norm(h) == norm(r["ref"]) else f"**{h}**"
        out.append(f"| {r['voice']} | {r['no']} | {r['ref']} | {mark(h0)} | {mark(h1)} | {', '.join(ins)} |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="받아쓰기 초기 문맥(prompt) 시험")
    ap.add_argument("--backend", choices=["ovms", "genai-cpu"], default="ovms")
    ap.add_argument("--prompt", default=stt.PROMPT)
    ap.add_argument("--voices", default="F1,M1")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4, help="genai-cpu 스레드(다른 일과 CPU 를 나눌 때 줄인다)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    backend = Ovms() if a.backend == "ovms" else GenaiCpu(a.threads)
    t0 = time.time()
    rows = run(backend, a.prompt, [v.strip() for v in a.voices.split(",") if v.strip()], a.repeat)
    md = to_markdown(rows, backend, a.prompt, a.repeat, time.time() - t0)
    print(md)
    if a.out:
        p = Path(a.out)
        old = p.read_text(encoding="utf-8") if p.exists() else ""
        prev = old[old.index("\n## ") + 1:] if "\n## " in old else ""
        head = ("# 받아쓰기 초기 문맥(prompt) 시험\n\n> `python -m ui.stt_prompt_check` 로 다시 만든다(새 실측이 위). "
                "굵은 글 = 정답과 다름(띄어쓰기·문장부호·동의어·숫자 표기 무시). 끼워 넣기 = 정답에 없는 문맥 낱말"
                f"({'·'.join(CONTEXT_WORDS)})이 받아쓴 글에 나온 횟수.\n\n")
        p.write_text(head + md + "\n" + prev, encoding="utf-8")
        print(f"\n→ {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
