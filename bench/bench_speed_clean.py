"""다른 세션이 OVMS 를 같이 쓰는 중에도 '혼자 쓴' 속도를 얻는다.
측정 구간 동안 OVMS 로그에 동시 요청(All requests: 2 이상)이 한 줄도 없던 회차만 남긴다.

사용: .venv/bin/python bench/bench_speed_clean.py qwen3-8b qwen3-4b
결과: bench/out/speed_clean.json
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

from bench_llm import SPEED_PROMPT, stream_once

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "ball-ovms.log"
OUT = ROOT / "bench" / "out"
BUSY = re.compile(r"All requests: ([2-9]|\d\d)")


def busy_lines():
    return sum(1 for line in LOG.open(errors="replace") if BUSY.search(line))


def clean_runs(model, need=3, tries=25):
    stream_once(model, "안녕", max_tokens=8)
    clean, dirty = [], []
    for _ in range(tries):
        b0 = busy_lines()
        r = stream_once(model, SPEED_PROMPT)
        (clean if busy_lines() == b0 else dirty).append(r)
        if len(clean) >= need:
            break
        time.sleep(1)
    return clean, dirty


if __name__ == "__main__":
    res = {}
    for m in sys.argv[1:] or ["qwen3-8b", "qwen3-4b"]:
        clean, dirty = clean_runs(m)
        row = {"clean": clean, "dirty": dirty}
        if clean:
            row["ttft_s"] = statistics.median(x["ttft_s"] for x in clean)
            row["decode_tok_s"] = statistics.median(x["decode_tok_s"] for x in clean)
            print(f"{m}: 혼자 쓴 회차 {len(clean)}개 → TTFT {row['ttft_s']:.3f}s, 생성 {row['decode_tok_s']:.1f} tok/s "
                  f"(프롬프트 {clean[0]['prompt_tokens']}·생성 {clean[0]['completion_tokens']}토큰)", end="")
        else:
            print(f"{m}: 혼자 쓴 회차 없음", end="")
        if dirty:
            print(f" | 동시 사용 회차 {len(dirty)}개: 생성 {statistics.median(x['decode_tok_s'] for x in dirty):.1f} tok/s")
        else:
            print()
        res[m] = row
    OUT.mkdir(exist_ok=True)
    (OUT / "speed_clean.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
