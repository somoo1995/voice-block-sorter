"""Supertonic 이 만든 wav(bench/out/tts_*.wav)를 Whisper 로 받아써 지연·정확도(CER)를 잰다.

사용: .venv/bin/python bench/bench_stt.py [모델 ...] [--base http://127.0.0.1:8100/v3]
      (기본 모델: whisper — 본 서버의 NPU 판. GPU 비교는 whisper-gpu 를 올린 서버에서)
결과: bench/out/stt_<모델>.json
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parent / "out"


def norm(s):
    # 띄어쓰기·문장부호를 빼고 글자만 비교
    return re.sub(r"[\s\.,!?~·'\"]", "", s)


def cer(ref, hyp):
    a, b = norm(ref), norm(hyp)
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / max(len(a), 1)


def transcribe(base, model, path):
    with open(path, "rb") as f:
        t0 = time.perf_counter()
        r = requests.post(f"{base}/audio/transcriptions", files={"file": (path.name, f, "audio/wav")},
                          data={"model": model, "language": "ko"}, timeout=600)
        dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json()["text"].strip(), dt


def run(base, model):
    manifest = json.loads((OUT / "tts.json").read_text())
    rows = manifest["rows"]
    transcribe(base, model, OUT / rows[0]["file"])  # 예열
    res = []
    for row in rows:
        ts, text = [], ""
        for _ in range(3):
            text, dt = transcribe(base, model, OUT / row["file"])
            ts.append(dt)
        e = cer(row["text"], text)
        res.append({"file": row["file"], "ref": row["text"], "hyp": text, "cer": e,
                    "latency_s": statistics.median(ts), "audio_s": row["audio_s"]})
        print(f"[{model}] {statistics.median(ts)*1000:5.0f}ms (음성 {row['audio_s']:.1f}s) CER {e:.2f} | "
              f"{row['text']} → {text}")
    exact = sum(norm(r["ref"]) == norm(r["hyp"]) for r in res)
    summary = {"model": model, "base": base, "rows": res,
               "latency_median_s": statistics.median(r["latency_s"] for r in res),
               "cer_mean": sum(r["cer"] for r in res) / len(res), "exact": exact, "n": len(res)}
    print(f"[{model}] 지연 중앙값 {summary['latency_median_s']*1000:.0f}ms | 평균 CER {summary['cer_mean']:.3f} | "
          f"글자 완전일치 {exact}/{len(res)}")
    (OUT / f"stt_{model}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["whisper"])
    ap.add_argument("--base", default="http://127.0.0.1:8100/v3")
    a = ap.parse_args()
    for m in a.models:
        run(a.base, m)
