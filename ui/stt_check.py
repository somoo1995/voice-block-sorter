"""받아쓰기(STT) 실측 — 시연 명령을 음성 파일로 만들어(또는 사람이 녹음한 파일로) Whisper 에 넣고 문장별 CER·시간을 잰다.

    python -m ui.stt_check                       # TTS(F1·M1)로 음성을 만들어 시험 → docs/stt_check.md
    python -m ui.stt_check --voices F1,M1,F3     # 목소리 바꿔서
    python -m ui.stt_check --dir ~/rec --tag 사람  # 사람이 녹음한 파일 01.wav ~ 12.wav (번호 = 아래 SENTENCES) → docs/stt_check_사람.md

CER = 글자 편집 거리 / 정답 글자 수. 띄어쓰기·문장부호는 빼고 센다("골라내 줘" = "골라내줘"). 같은 말의 표기 흔들림("블럭" = "블록")과 색 동의어(핑크 = 분홍, 녹색·연두 = 초록), 숫자 표기("2개" = "두 개")도 같게 센다.
만든 음성은 ui/samples/ 에 남는다(대시보드 마이크 옆 "파일 올리기"로 그대로 시험할 수 있다).
결과 파일은 새 실측을 맨 위에 쓰고 앞 실측은 아래에 남긴다. CER 은 Whisper 원문으로 세고, ui/stt.py 가 바로잡은 글은 따로 적는다.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from ui import stt, tts

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "ui" / "samples"
SENTENCES = [  # (번호, 문장, 계약서 2절 intent)
    ("01", "불량품 골라내 줘", "defect"),
    ("02", "분홍 블록 옮겨 줘", "red"),
    ("03", "초록 블록 옮겨 줘", "blue"),
    ("04", "분홍 블록 하나 내 손에 줘", "handoff"),
    ("05", "결과 알려 줘", "report"),
    ("06", "멈춰", "stop"),
    ("07", "초록 불량은 왜 불량이야?", "question"),
    ("08", "초록색 거 두 개만 옮겨 줘", "blue"),  # 덤: 개수 말
    ("09", "지금 작업대 어때?", "scene"),
    ("10", "불량 몇 개 남았어?", "scene"),
    ("11", "분홍 두 개만 옮기고 불량 골라내 줘", "plan"),
    ("12", "초록은 그대로 두고 나머지 정리해 줘", "plan"),
]
# 받아쓰기 표기 흔들림·색 동의어(계약서: 분홍 = 핑크, 초록 = 녹색 = 연두) → 같은 말로 채점
SAME = {"블럭": "블록", "핑크": "분홍", "녹색": "초록", "연두": "초록"}
NUM_KO = {"1": "한", "2": "두", "3": "세", "4": "네", "5": "다섯"}  # Whisper 는 "두 개"를 "2개"로 쓰기도 한다
TITLE = "# 받아쓰기(STT) 실측"


def norm(s: str) -> str:
    s = re.sub(r"[\W_]+", "", s or "")
    s = re.sub(r"(?<!\d)([1-5])개", lambda m: NUM_KO[m.group(1)] + "개", s)
    for a, b in SAME.items():
        s = s.replace(a, b)
    return s


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(1, len(r))


def run_one(no: str, ref: str, intent: str, wav: Path, source: str, synth_ms: float | None) -> dict:
    t0 = time.perf_counter()
    try:
        fixed, err = stt.transcribe(str(wav)), None
    except Exception as e:
        fixed, err = "", f"{e.__class__.__name__}: {e}"
    total_ms = (time.perf_counter() - t0) * 1000
    hyp = stt.last_info.get("raw", fixed) if fixed else fixed  # 채점은 Whisper 원문, 바로잡은 글(ui/stt.py FIXES)은 따로
    return {"no": no, "source": source, "ref": ref, "intent": intent, "hyp": hyp, "fixed": fixed, "err": err,
            "cer": cer(ref, hyp), "exact": norm(ref) == norm(hyp), "exact_fixed": norm(ref) == norm(fixed),
            "audio_s": stt.last_info.get("seconds"),
            "server_ms": stt.last_info.get("latency_ms"), "total_ms": total_ms, "synth_ms": synth_ms,
            "wav": str(wav.relative_to(ROOT)) if wav.is_relative_to(ROOT) else str(wav)}


def from_tts(voices: list[str]) -> list[dict]:
    if not tts.available():
        sys.exit(f"TTS 를 쓸 수 없습니다: {tts.load_error()}")
    SAMPLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in voices:
        for no, ref, intent in SENTENCES:
            path = tts.speak(ref, voice=v)
            synth_ms = tts.last_info.get("latency_ms")
            dst = SAMPLES / f"tts_{v}_{no}.wav"
            shutil.copy(path, dst)
            rows.append(run_one(no, ref, intent, dst, f"TTS {v}", synth_ms))
            print(f"{v} {no} {rows[-1]['cer']:.2f} {rows[-1]['hyp']!r}", file=sys.stderr)
    return rows


def from_dir(d: Path, tag: str) -> list[dict]:
    rows = []
    for no, ref, intent in SENTENCES:
        cands = sorted(d.glob(f"{no}.*")) + sorted(d.glob(f"{no}_*"))
        if cands:
            rows.append(run_one(no, ref, intent, cands[0], tag, None))
            print(f"{tag} {no} {rows[-1]['cer']:.2f} {rows[-1]['hyp']!r}", file=sys.stderr)
    return rows


def _f(x, fmt):
    return "-" if x is None else format(x, fmt)


def to_markdown(rows: list[dict], note: str) -> str:
    """실측 한 번 = `## 시각 — note` 절 하나(안의 소스별 표는 ###)."""
    out = [f"## {datetime.now():%Y-%m-%d %H:%M} — {note}", "",
           f"Whisper = OVMS `{stt.BASE_URL}` 모델 `{stt.MODEL}` (language={stt.LANGUAGE}).", ""]
    for src in dict.fromkeys(r["source"] for r in rows):
        rs = [r for r in rows if r["source"] == src]
        ok = sum(r["exact"] for r in rs)
        ok_fixed = sum(r.get("exact_fixed", r["exact"]) for r in rs)
        mean_cer = sum(r["cer"] for r in rs) / len(rs)
        srv = [r["server_ms"] for r in rs if r["server_ms"] is not None]
        out += [f"### {src} — 완전 일치 {ok}/{len(rs)}(바로잡은 뒤 {ok_fixed}/{len(rs)}), 평균 CER {mean_cer:.3f}, "
                f"서버 평균 {sum(srv) / max(1, len(srv)):.0f} ms", "",
                "| # | 정답 | 받아쓴 글(Whisper 원문) | CER | 바로잡은 글 | 음성 초 | 서버 ms | 전체 ms | 합성 ms |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            hyp = r["hyp"] or (f"(오류: {r['err']})" if r["err"] else "(빈 결과)")
            fixed = r.get("fixed") if r.get("fixed") and r.get("fixed") != r["hyp"] else ""
            out.append(f"| {r['no']} | {r['ref']} | {hyp} | {r['cer']:.2f} | {fixed} | {_f(r['audio_s'], '.2f')} | "
                       f"{_f(r['server_ms'], '.0f')} | {r['total_ms']:.0f} | {_f(r['synth_ms'], '.0f')} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="STT 실측(CER·시간)")
    ap.add_argument("--voices", default="F1,M1", help="TTS 목소리(쉼표), 기본 F1,M1")
    ap.add_argument("--dir", help="녹음 파일 폴더(01.wav ~ 12.wav). 주면 TTS 대신 이것")
    ap.add_argument("--tag", default="녹음", help="--dir 결과 이름")
    ap.add_argument("--out", help="결과 Markdown(기본 docs/stt_check.md, --dir 이면 docs/stt_check_<tag>.md, 빈 값이면 안 씀)")
    a = ap.parse_args()
    if not stt.available():
        sys.exit(f"OVMS 에 {stt.MODEL} 모델이 없습니다({stt.BASE_URL}/models)")
    rows = from_dir(Path(a.dir).expanduser(), a.tag) if a.dir else from_tts([v.strip() for v in a.voices.split(",") if v.strip()])
    note = f"녹음 폴더 `{a.dir}`" if a.dir else f"Supertonic 3 TTS(목소리 {a.voices}, 속도 {tts.SPEED}, 단계 {tts.STEPS})"
    md = to_markdown(rows, note)
    print(md)
    out = a.out if a.out is not None else str(ROOT / "docs" / (f"stt_check_{a.tag}.md" if a.dir else "stt_check.md"))
    if out:
        old = Path(out).read_text(encoding="utf-8") if Path(out).exists() else ""
        prev = old[old.index("\n## ") + 1:] if "\n## " in old else ""  # 앞 실측들(맨 위 머리말 뒤 전부)
        head = (f"{TITLE}\n\n> `python -m ui.stt_check` 로 다시 만든다(새 실측이 맨 위, 앞 실측은 아래에 남는다). "
                "TTS 음성은 사람 목소리가 아니다 — 사람은 `--dir` 로 따로 잰다.\n> "
                "CER 은 띄어쓰기·문장부호를 빼고 센 글자 오류율이고, \"블럭\"=\"블록\", 핑크=분홍, 녹색·연두=초록, \"2개\"=\"두 개\"는 같은 말로 센다. "
                "서버 = OVMS 왕복, 전체 = 파일 읽기·16kHz 변환 포함.\n\n")
        Path(out).write_text(head + md + "\n" + prev, encoding="utf-8")
        print(f"\n→ {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
