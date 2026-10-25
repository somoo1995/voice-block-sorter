"""실물 IK vs ACT 집기 비교표 — 좌표 집기 실제 실행 기록(runs/<날짜>/coord_pick/*_sort_*_real.jsonl)을 읽기만.

    .venv/bin/python -m robot.compare.grasp_runs [--since 11:50] [--date 20261023]

줄마다: 시각·색·집기 방법·성공·시도 수·정책 끝(reason)·쥠·총 시간(요청 → 결과)·실패 이유. 끝에 방법별 성공·시간 중앙.
손목 실시간 맞춤(--servo, servo.py) 실행은 방법이 ik+servo 이고, 줄 끝에 시도마다 서보 결과(ok·초 또는 멈춘 이유).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent


def rows(date: str, since: str | None) -> list[dict]:
    out = []
    for f in sorted(glob.glob(str(ROOT / "runs" / date / "coord_pick" / "*_sort_*_real.jsonl"))):
        hm = Path(f).name[:4]
        if since and hm < since.replace(":", ""):
            continue
        ev = [json.loads(x) for x in Path(f).read_text(encoding="utf-8").splitlines() if x.strip()]
        req = next((e for e in ev if e["stage"] == "request"), None)
        res = next((e for e in ev if e["stage"] == "result"), None)
        if req is None or res is None:
            continue
        t = (dt.datetime.fromisoformat(res["ts"]) - dt.datetime.fromisoformat(req["ts"])).total_seconds()
        w = (res.get("wrist") or [{}])[0]
        at = w.get("attempts") or []
        acts = [a.get("act") or {} for a in at]
        svs = [a.get("servo") for a in at if a.get("servo")]
        out.append({"at": f"{hm[:2]}:{hm[2:]}", "color": "·".join(res.get("colors") or []),
                    "mode": (res.get("grasp_mode") or "") + ("+servo" if svs else ""),
                    "servo": [f"ok {v.get('time_s')}s" if v.get("ok") else f"X {str(v.get('why'))[:40]}" for v in svs],
                    "ok": bool(res.get("ok")), "tries": len(at), "time_s": round(t, 1),
                    "act": next((a.get("reason") or a.get("skipped") for a in acts if a), None),
                    "held": [a.get("held") for a in at], "why": (res.get("error") or w.get("why") or
                                                                 next((a.get("why") for a in at if a.get("why")), "")),
                    "file": Path(f).name})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.compare.grasp_runs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"))
    ap.add_argument("--since", help="이 시각(HH:MM) 뒤 실행만")
    a = ap.parse_args(argv)
    rs = rows(a.date, a.since)
    print(f"{'시각':5} {'색':7} {'방법':4} {'성공':4} {'시도':4} {'시간s':>6}  정책 끝 / 쥠 / 이유")
    for r in rs:
        print(f"{r['at']:5} {r['color']:7} {r['mode'] or '-':4} {'O' if r['ok'] else 'X':4} {r['tries']:4} {r['time_s']:6.1f}  "
              f"{r['act'] or '-'} / {r['held']} / {str(r['why'])[:90]}" + (f" / 서보 {r['servo']}" if r["servo"] else ""))
    for m in ("ik", "ik+servo", "act"):
        sub = [r for r in rs if r["mode"] == m]
        if sub:
            ok = [r for r in sub if r["ok"]]
            print(f"{m}: 성공 {len(ok)}/{len(sub)} · 첫 시도 성공 {sum(r['ok'] and r['tries'] == 1 for r in sub)} · 시간 중앙 "
                  f"{np.median([r['time_s'] for r in sub]):.1f}s(성공만 {np.median([r['time_s'] for r in ok]) if ok else float('nan'):.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
