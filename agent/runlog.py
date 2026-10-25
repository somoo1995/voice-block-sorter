"""작업 기록 runs/<YYYYMMDD>/log.jsonl — 정책 시도마다 한 줄 {ts, intent, policy, before, after, ok, ...}."""
import json
import os
import threading
from datetime import datetime
from pathlib import Path

from .config import RUNS_DIR

_lock = threading.Lock()


def day_dir(day: str | None = None) -> Path:
    """runs/<날짜>/ — 가짜 작업대(BALL_MOCK=1: selftest·eval·인프라 연속 시험)는 runs/<날짜>_mock/ 에 따로 쓴다.
    10-22 0시 연속 시험 911줄·내 시험 352줄이 실물과 같은 runs/20261022/log.jsonl 에 섞였다(보고는 thread 별이라
    세지 않지만 파일을 나눠 둔다)."""
    d = RUNS_DIR / ((day or datetime.now().strftime("%Y%m%d")) + ("_mock" if os.getenv("BALL_MOCK") == "1" else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_path(day: str | None = None) -> Path:
    return day_dir(day) / "log.jsonl"


def append(entry: dict) -> None:
    with _lock, log_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read(day: str | None = None, thread_id: str | None = None) -> list[dict]:
    p = log_path(day)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if thread_id is None or row.get("thread_id") == thread_id:
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    """정책별 시도·성공·옮긴 개수(클래스별)."""
    out = {"attempts": 0, "ok": 0, "policies": {}, "ended": {}}
    for r in rows:
        p = out["policies"].setdefault(r["policy"], {"attempts": 0, "ok": 0, "moved": 0, "by_class": {}})
        p["attempts"] += 1
        out["attempts"] += 1
        if r.get("ok"):
            p["ok"] += 1
            out["ok"] += 1
        p["moved"] += r.get("moved", 0)
        why = r.get("policy_reason") or ("duration_elapsed" if r.get("policy_ok") else "기록 없음")
        out["ended"][why] = out["ended"].get(why, 0) + 1          # 홈 복귀 감지(일찍 끝남)·시간 경과·timeout 을 따로 센다
        for k, n in (r.get("moved_by_class") or {}).items():
            p["by_class"][k] = p["by_class"].get(k, 0) + n
    return out
