"""Supertonic-3 한국어 TTS 실측(CPU, onnxruntime). wav 는 Whisper 받아쓰기 시험에 그대로 쓴다.

사용: .venv/bin/python bench/bench_tts.py
모델: models/supertonic-3 (없으면 Hugging Face 에서 자동 다운로드)
결과: bench/out/tts_XX.wav, bench/out/tts.json
"""
import json
import resource
import time
from pathlib import Path

import onnxruntime as ort
import soundfile as sf
from supertonic import TTS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "out"
MODEL_DIR = ROOT / "models" / "supertonic-3"
VOICE = "F1"

SENTENCES = [
    "불량품 골라내 줘",
    "빨간 볼 옮겨 줘",
    "파란 공 다 담아",
    "빨간 거 하나 내 손에 줘",
    "결과 알려 줘",
    "빨간 공 세 개와 파란 공 두 개를 옮겼고, 불량품은 한 개 골라냈습니다.",
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("onnxruntime", ort.__version__, ort.get_available_providers())
    t0 = time.perf_counter()
    tts = TTS(model="supertonic-3", model_dir=MODEL_DIR, auto_download=True)
    load_s = time.perf_counter() - t0
    style = tts.get_voice_style(VOICE)
    print(f"[TTS] 로드 {load_s:.2f}s, 샘플레이트 {tts.sample_rate}Hz, 음성 {VOICE}")

    tts.synthesize("예열", voice_style=style, lang="ko")  # 예열
    rows = []
    for i, text in enumerate(SENTENCES):
        t0 = time.perf_counter()
        wav, dur = tts.synthesize(text, voice_style=style, lang="ko")
        dt = time.perf_counter() - t0
        audio_s = wav.shape[-1] / tts.sample_rate
        path = OUT / f"tts_{i:02d}.wav"
        sf.write(path, wav.squeeze(), tts.sample_rate)
        rows.append({"file": path.name, "text": text, "synth_s": dt, "audio_s": audio_s, "rtf": dt / audio_s})
        print(f"[TTS] {dt*1000:6.0f}ms  음성 {audio_s:4.2f}s  RTF {dt/audio_s:.3f}  {text}")

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    summary = {"load_s": load_s, "sample_rate": tts.sample_rate, "voice": VOICE, "peak_rss_mb": peak_mb,
               "rows": rows}
    (OUT / "tts.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[TTS] 프로세스 최대 RSS {peak_mb:.0f}MB")


if __name__ == "__main__":
    main()
