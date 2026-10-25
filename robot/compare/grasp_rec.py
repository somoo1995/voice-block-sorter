"""집기 녹화(grasp 데이터셋) 분석 — 시작 자세·쥐는 순간·들기·끝 정지, 집기 끝 판정(grasp_stop)이 녹화에서 나는지.

    .venv/bin/python -m robot.compare.grasp_rec [--rec robot/logs/n6/grasp_rec.json] [--close-drop 6 …]

grasp_rec.json = 컨테이너 안에서 parquet 을 읽어 만든 에피소드별 상태·동작(robot/logs/n6, 읽기만).
쥐는 순간 = 그리퍼 동작값이 연 최댓값에서 떨어지기 시작한 틱(compare.common.closing), 블록 자리 = 그때 집게 가운데(FK).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from robot import grasp_stop
from robot.compare.common import closing, setup

ROOT = Path(__file__).resolve().parent.parent.parent
REC = ROOT / "robot" / "logs" / "n6" / "grasp_rec.json"


def pose(s, n) -> dict:
    """관절 6개(정규화) → 집게 가운데 작업대 xy·높이, 기울기(°)."""
    q = s.n_to_q(np.asarray(n, float))
    T = s.kin.fk(q)
    p = s.tcp_pos(q)
    return {"w": s.frames.r2w(p[:2]), "z": float(p[2]), "tilt": math.degrees(math.acos(max(-1.0, min(1.0, -float(T[2, 2])))))}


def analyze(rec: dict, det_kw: dict | None = None) -> list[dict]:
    s = setup()
    zf = grasp_stop.tcp_z_fn()
    out = []
    for ep, e in sorted(rec["episodes"].items(), key=lambda kv: int(kv[0])):
        S, A = np.asarray(e["s"], float), np.asarray(e["a"], float)
        gi = closing(A[:, 5], rest=float(np.median(A[:5, 5])) - 7)  # 시작부터 연 채라 쉬는 값 = 처음 − 7
        p0 = pose(s, S[0])
        row = {"ep": int(ep), "len_s": round(len(S) / 30, 2), "start_z": round(p0["z"], 1), "start_tilt": round(p0["tilt"], 1),
               "g_start": round(float(S[0, 5]), 1), "g_min": round(float(S[:, 5].min()), 1),
               "g_drop": round(float(S[:, 5].max() - S[:, 5].min()), 1), "a_min": round(float(A[:, 5].min()), 1)}
        if gi is not None:
            pg = pose(s, S[gi])
            zs = np.array([pose(s, x)["z"] for x in S])
            low = int(np.argmin(zs[gi:]) + gi)
            row.update(close_s=round(gi / 30, 2), grasp_w=np.round(pg["w"], 1).tolist(), grasp_z=round(pg["z"], 1),
                       grasp_tilt=round(pg["tilt"], 1), start_off_mm=round(float(np.linalg.norm(p0["w"] - pg["w"])), 1),
                       start_off_xy=np.round(p0["w"] - pg["w"], 1).tolist(), low_z=round(float(zs[low]), 1),
                       lift_mm=round(float(zs[low:].max() - zs[low]), 1), end_z=round(float(zs[-1]), 1),
                       end_still=round(float(np.ptp(S[-15:, :6], axis=0).max()), 2))
        d = grasp_stop.GraspStopDetector(zf, **(det_kw or {}))
        hit = next((k for k, x in enumerate(S) if d.feed(k / 30, x)), None)
        row.update(stop_kind=d.trigger.kind if d.trigger else None,
                   z_min_before_close=None if gi is None else round(min(zf(x) for x in S[:gi + 1]), 1),
                   stop_s=None if hit is None else round(hit / 30, 2),
                   stop_before_end_s=None if hit is None else round((len(S) - 1 - hit) / 30, 2),
                   stop_lift=d.trigger.lift_mm if d.trigger else None, closed=d.closed, lifted=d.lifted)
        out.append(row)
    return out


def summary(rows: list[dict]) -> dict:
    def q(k, ps=(10, 50, 90)):
        v = [r[k] for r in rows if r.get(k) is not None]
        return [round(float(np.percentile(v, p)), 1) for p in ps] if v else None
    return {"n": len(rows), "start_z": q("start_z"), "start_tilt": q("start_tilt"), "start_off_mm": q("start_off_mm"),
            "close_s": q("close_s"), "grasp_z": q("grasp_z"), "low_z": q("low_z"), "lift_mm": q("lift_mm"),
            "len_s": q("len_s"), "g_start": q("g_start"), "g_min": q("g_min"), "g_drop": q("g_drop", (0, 10, 50)),
            "end_still": q("end_still"),
            "stop_hit": sum(r["stop_s"] is not None for r in rows), "stop_s": q("stop_s"),
            "stop_before_end_s": q("stop_before_end_s"), "not_closed": [r["ep"] for r in rows if not r["closed"]],
            "floor_stops": [r["ep"] for r in rows if r.get("stop_kind") == "floor"],
            "z_min_before_close": q("z_min_before_close", (0, 10, 50)),
            "not_stopped": [r["ep"] for r in rows if r["stop_s"] is None]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.compare.grasp_rec", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", default=str(REC))
    ap.add_argument("--close-drop", type=float)
    ap.add_argument("--lift-mm", type=float)
    ap.add_argument("--still", type=float)
    ap.add_argument("--hold-s", type=float)
    ap.add_argument("--floor-mm", type=float)
    ap.add_argument("--floor-hold-s", type=float)
    ap.add_argument("--floor-hard-mm", type=float)
    ap.add_argument("--rows", action="store_true", help="에피소드별 줄도 찍는다")
    a = ap.parse_args(argv)
    kw = {k: v for k, v in (("close_drop", a.close_drop), ("lift_mm", a.lift_mm), ("still", a.still), ("hold_s", a.hold_s),
                            ("floor_mm", a.floor_mm), ("floor_hold_s", a.floor_hold_s), ("floor_hard_mm", a.floor_hard_mm))
          if v is not None}
    rows = analyze(json.loads(Path(a.rec).read_text()), kw)
    if a.rows:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
    print(json.dumps({"detector": {**grasp_stop.DEFAULTS, **kw}, **summary(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
