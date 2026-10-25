"""GPU HANG 대조 — 커널 i915 사건 시각과 같은 시간대의 GPU 사용(정책 실행·OVMS 요청·Studio 작업)을 한 줄 시간표로 맞댄다.

읽기만 한다: robot/logs/<날짜>/*.log·*.runtime.yaml·*.ticks.jsonl, runs/<날짜>/log.jsonl, logs/ball-ovms.log,
journalctl -k -b <부팅>, docker logs(Studio). 사용: .venv/bin/python bench/hang_timeline.py [--boot -1] [--day 20261022]
"""
from __future__ import annotations

import argparse, datetime as dt, json, re, statistics, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KST = dt.timezone(dt.timedelta(hours=9))
PAT = re.compile(r"gpu hang|fence expiration|context reset|segfault|GUC: CT|wedged|suspend entry|suspend exit|PM: ", re.I)


def kernel(boot, day):
    out = subprocess.run(["journalctl", "-k", "-b", str(boot), "--no-pager", "-o", "short-iso", "--since", f"{day[:4]}-{day[4:6]}-{day[6:]} 00:00"],
                         capture_output=True, text=True).stdout
    ev = []
    for line in out.splitlines():
        if PAT.search(line):
            ts = dt.datetime.fromisoformat(line.split()[0])
            ev.append((ts, "커널", line.split("kernel: ", 1)[-1][:110]))
    return ev


def policy_runs(day):
    """정책 실행마다 (시작, 끝, 이름, 장치, 추론 수, 지연 중앙·최대 ms)."""
    runs = []
    for log in sorted((ROOT / "robot" / "logs" / day).glob("*.log")):
        stem = log.name[:-4]
        ticks = log.with_name(stem + ".ticks.jsonl")
        y = log.with_name(stem + ".runtime.yaml")
        dev = "?"
        if y.exists():
            m = re.search(r"backend:\s*\w+\s*\n\s*device:\s*(\w+)", y.read_text())  # 카메라의 device: 말고 InferenceModel 장치
            dev = m.group(1) if m else "?"
        t_start = t_end = None
        lat = []
        if ticks.exists():
            for line in ticks.read_text().splitlines():
                if not line.startswith("{"):
                    continue
                try:
                    j = json.loads(line)
                except ValueError:
                    continue
                ts = j.get("timestamp")
                if j.get("type") == "lifecycle" and j.get("event") == "start":
                    t_start = ts
                elif j.get("type") == "lifecycle":
                    t_end = ts
                elif j.get("type") == "inference":
                    lat.append(j.get("latency_s", 0) * 1000)
                    t_end = max(t_end or 0, ts)
        hh = stem.split("_", 1)
        base = dt.datetime.strptime(day + hh[0], "%Y%m%d%H%M%S").replace(tzinfo=KST)
        s = dt.datetime.fromtimestamp(t_start, KST) if t_start else base
        e = dt.datetime.fromtimestamp(t_end, KST) if t_end else None
        head = log.read_text(errors="replace")[:400]
        mm = re.search(r"model=(\S+)", head)
        runs.append({"start": s, "end": e, "name": hh[1] if len(hh) > 1 else stem, "dev": dev, "model": (mm.group(1)[:8] if mm else ""),
                     "n_inf": len(lat), "lat_med": round(statistics.median(lat), 1) if lat else None, "lat_max": round(max(lat), 1) if lat else None})
    return runs


def agent_events(day):
    p = ROOT / "runs" / day / "log.jsonl"
    ev = []
    if not p.exists():
        return ev
    for line in p.read_text(errors="replace").splitlines():
        try:
            j = json.loads(line)
        except ValueError:
            continue
        ts = j.get("ts")
        if not ts or str(j.get("thread_id", "")).startswith("soak"):
            continue
        try:
            t = dt.datetime.fromisoformat(ts)
        except ValueError:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=KST)
        ev.append((t, j))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", default="-1")
    ap.add_argument("--day", default="20261022")
    ap.add_argument("--window", type=int, default=120, help="사건 앞뒤 몇 초를 볼지")
    a = ap.parse_args()
    kev = kernel(a.boot, a.day)
    runs = policy_runs(a.day)
    agent = agent_events(a.day)
    print(f"정책 실행 {len(runs)}개(GPU {sum(r['dev'] == 'GPU' for r in runs)}·CPU {sum(r['dev'] == 'CPU' for r in runs)}·기타 {sum(r['dev'] not in ('GPU', 'CPU') for r in runs)}), 에이전트 기록 {len(agent)}줄, 커널 사건 {len(kev)}줄")
    inf = [r for r in runs if r["n_inf"]]
    if inf:
        allmed = [r["lat_med"] for r in inf]
        print(f"정책 추론 지연(실행별 중앙값의 중앙) {statistics.median(allmed)}ms, 실행별 최대의 최대 {max(r['lat_max'] for r in inf)}ms")
    for ts, _, msg in kev:
        if not re.search(r"hang|fence|segfault|GUC", msg, re.I):
            continue
        w = dt.timedelta(seconds=a.window)
        near = [r for r in runs if r["start"] - w <= ts <= (r["end"] or r["start"]) + w]
        ag = [(t, j) for t, j in agent if ts - w <= t <= ts + dt.timedelta(seconds=10)]
        print(f"\n■ {ts:%H:%M:%S} {msg}")
        for r in near:
            rel = (r["start"] - ts).total_seconds()
            dur = ((r["end"] or r["start"]) - r["start"]).total_seconds()
            print(f"   정책 {r['start']:%H:%M:%S}~{(r['end'] or r['start']):%H:%M:%S}({dur:.0f}s, 사건 기준 {rel:+.0f}s) {r['name']} {r['dev']} 추론 {r['n_inf']}회 중앙 {r['lat_med']}·최대 {r['lat_max']}ms")
        for t, j in ag[-8:]:
            keys = {k: j[k] for k in ("intent", "policy", "ok", "thread_id", "node", "type") if k in j}
            print(f"   에이전트 {t:%H:%M:%S}({(t - ts).total_seconds():+.0f}s) {json.dumps(keys, ensure_ascii=False)[:120]}")
    print("\n정책 실행 전체(시작·장치·추론 수):")
    for r in runs:
        print(f"  {r['start']:%H:%M:%S} {r['name']:<14} {r['dev']:<4} {r['model']} 추론 {r['n_inf']} 중앙 {r['lat_med']} 최대 {r['lat_max']}")


if __name__ == "__main__":
    main()
