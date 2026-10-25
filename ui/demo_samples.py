"""발표 실연 음성 백업 — 마이크가 안 될 때 대시보드 마이크 칸의 "파일 올리기"로 대신 넣을 음성 파일을 만든다.

    .venv/bin/python -m ui.demo_samples              # ui/samples/demo/{F1,M1}_NN.wav + README.md (받아쓰기 확인 표 포함)
    .venv/bin/python -m ui.demo_samples --voices F1  # 목소리 하나만

문장 = 계약서 2절 끝 "발표 실연" ①~⑤ + 멈춰 + 숫자 모양 명령(01~07), 덧붙여 문서 질문 3개(08·09·12, 문서마다 하나),
숫자 7 계획과 대답 "응"(10·11·13).
만든 뒤 Whisper(ui/stt.py, 바로잡기 포함)로 다시 받아써 맞는지 표로 남긴다. ui/screens.py 도 이 파일을 쓴다.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ui import stt, tts
from ui.stt_check import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ui" / "samples" / "demo"
TRIES = 4
SENTENCES = [  # (번호, 문장, 쓰임)
    ("01", "불량품 골라내 줘", "실연 ①"),
    ("02", "지금 배치 어때?", "실연 ②"),
    ("03", "불량 빼고 나머지 다 정리해 줘", "실연 ③"),
    ("04", "분홍 블록 하나 내 손에 줘", "실연 ④"),
    ("05", "결과 알려 줘", "실연 ⑤"),
    ("06", "멈춰", "멈춤"),
    ("07", "분홍으로 숫자 1 만들어 줘", "숫자 모양(좌표 집기)"),
    ("08", "그리퍼가 뜨거우면 어떻게 해?", "질문 — 로봇 운용 설명서 4.3"),
    ("09", "인터넷 없이 돌아가?", "질문 — 시스템 안내서 5.1"),  # "왜 YOLO 를 NPU 에 뒀어?"는 TTS 가 YOLO 를 "아이오올로"로 읽어 뺐다
    ("10", "분홍으로 숫자 7 만들어 줘", "숫자 모양(좌표 집기) — 분홍 정상 블록이 7개 있어야 계획, 모자라면 블록 부족 거절"),
    ("11", "그렇게 해", "숫자 배치 \"할까요?\" 대답(\"응\" 한 글자는 M1 목소리가 \"음\"으로 받아써져 이 말로)"),
    ("12", "초록 불량은 왜 불량이야?", "질문 — 작업 기준서 3.2"),
    ("13", "블록으로 숫자 7 만들어 줘", "숫자 모양(좌표 집기) — 색 없이(분홍 정상 블록이 7개 안 될 때)"),
]


def path(voice: str, no: str) -> Path:
    return OUT / f"{voice}_{no}.wav"


def make(voices: list[str]) -> list[dict]:
    if not tts.available():
        sys.exit(f"TTS 를 쓸 수 없습니다: {tts.load_error()}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in voices:
        for no, text, use in SENTENCES:
            dst = path(v, no)
            for tries in range(1, TRIES + 1):  # 합성은 할 때마다 조금씩 달라 받아쓰기가 맞는 것을 고른다
                shutil.copy(tts.speak(text, voice=v), dst)
                fixed = stt.transcribe(str(dst)) if stt.available() else ""
                raw = stt.last_info.get("raw", fixed)
                if norm(fixed) == norm(text):
                    break
            rows.append({"voice": v, "no": no, "text": text, "use": use, "raw": raw, "fixed": fixed, "tries": tries,
                         "ok": norm(fixed) == norm(text), "sec": stt.last_info.get("seconds")})
            print(f"{v} {no} {'O' if rows[-1]['ok'] else 'X'} {fixed!r}", file=sys.stderr)
    return rows


def readme(rows: list[dict], voices: list[str]) -> str:
    ok = sum(r["ok"] for r in rows)
    out = ["# ui/samples/demo — 발표 실연 음성 백업", "",
           "마이크가 안 될 때 대시보드 마이크 칸 아래 ⬆(파일 올리기)로 이 파일을 차례로 올린다. 녹음이 끝난 것과 같은 길(받아쓰기 → 🎙 → 실행)로 간다.",
           f"Supertonic 3 TTS 목소리 {', '.join(voices)}(F = 여자, M = 남자), {datetime.now():%Y-%m-%d %H:%M} 생성. 다시 만들기: `.venv/bin/python -m ui.demo_samples`.", "",
           "| 파일 | 문장 | 쓰임 |", "|---|---|---|"]
    for no, text, use in SENTENCES:
        out.append(f"| `{'`·`'.join(path(v, no).name for v in voices)}` | {text} | {use} |")
    out += ["", f"## 받아쓰기 확인 — {ok}/{len(rows)} 맞음 (`ui/stt.py`, 바로잡기 포함, 띄어쓰기·문장부호 무시)", "",
            "합성은 할 때마다 조금씩 달라서, 받아쓰기가 맞는 합성본이 나올 때까지 최대 4번 다시 만들었다(시도 열).", "",
            "| 파일 | 받아쓴 글(바로잡은 뒤) | Whisper 원문 | 맞음 | 시도 | 음성 초 |", "|---|---|---|---|---|---|"]
    for r in rows:
        raw = r["raw"] if r["raw"] != r["fixed"] else ""
        sec = f"{r['sec']:.2f}" if r["sec"] else "-"
        out.append(f"| `{path(r['voice'], r['no']).name}` | {r['fixed']} | {raw} | {'O' if r['ok'] else 'X'} | {r['tries']} | {sec} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="발표 실연 음성 백업 만들기")
    ap.add_argument("--voices", default="F1,M1")
    a = ap.parse_args()
    voices = [v.strip() for v in a.voices.split(",") if v.strip()]
    rows = make(voices)
    (OUT / "README.md").write_text(readme(rows, voices), encoding="utf-8")
    print(f"→ {OUT}  받아쓰기 {sum(r['ok'] for r in rows)}/{len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
