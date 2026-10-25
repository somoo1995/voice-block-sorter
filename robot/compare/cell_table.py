"""시연 배치 고르기 — 투입 구역 칸 1~9 가운데 × 블록 긴 축 0·45·90·135° 통과표(계획만, 로봇·카메라 0).

    .venv/bin/python -m robot.compare.cell_table [--jobs 8]      # → robot/logs/n7/cell_table.txt (+ .json)

칸마다:
  분홍·초록·불량 = 그 블록(red_ok·blue_ok·red_defect)을 IK 모드로 집어 제 그릇에 놓는 전체 경로(홈 → 블록 위 → hover →
                  내려감 → 닫기 → 들어 올림 → 운반 → 놓기 → 홈)가 계획(plan_sort)·작업 공간 검사(실행기와 같은 기준, 목표 그릇은
                  놓는 높이 − 여유까지 허용)를 통과하나. ✗ 면 실패 단계.
  들어올림 = 실행기의 hover → 집는 높이 → 집고 들어 올림을 블록 자리 그대로·±10mm(x·y 4방향, 손목 맞춤 여유) 5곳에서
             (정적 계산, 첫 hover 자세에서 시작). 앞 = 10-23 오전까지(기울기 고정), 뒤 = n29(안 되면 허용 안 가까운 기울기).
             11:37 먼 줄 실패는 실물 마지막 자세에서 다시 풀어 따로 보인다(replay_1137).
  ACT = 시작 자세(120mm · 블록에서 로봇 쪽 33 · 옆 −17mm · 22° 근처, grasp_max 안)가 풀리나.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "robot" / "logs" / "n7"
ANGLES = (0, 45, 90, 135)
TARGETS = (("분홍", "red_ok", "bowl_red"), ("초록", "blue_ok", "bowl_blue"), ("불량", "red_defect", "bowl_defect"))
JIT = ((0, 0), (10, 0), (-10, 0), (0, 10), (0, -10))
ACT = {"z": 120.0, "back": 33.0, "side": -17.0, "tilt": 22.0}
_W: dict = {}


def _init():
    from robot.coord_pick import executor as ex
    from robot.coord_pick.plan import Setup, load_config
    cfg = load_config()
    _W.update(cfg=cfg, s=Setup.load(cfg, use_studio=False), ex=ex, sf=ex.safety(cfg))


def _full(job) -> dict:
    """한 칸·각도·그릇: plan_sort + 작업 공간 검사."""
    from robot.coord_pick.plan import plan_sort, trajectory
    n, w, ang, (ko, cls, bowl) = job
    s, ex, sf, cfg = _W["s"], _W["ex"], _W["sf"], _W["cfg"]
    obj = {"cls": cls, "px": s.w_to_px(np.asarray(w, float))[0].tolist(), "conf": 0.9, "zone": "input",
           "yaw_w": math.radians(ang), "aspect": 1.8}
    p = plan_sort(s, [obj], classes=(cls,), max_blocks=1)
    if not p["order"] or p["failures"]:
        f = "; ".join(p["failures"])
        stage = ("집기" if "닿지 않는다" in f else "놓기" if "위에 안 닿음" in f or "그릇" in f else "경로") if f else "계획"
        return {"n": n, "ang": ang, "bowl": ko, "ok": False, "stage": stage, "why": f[:160]}
    traj = trajectory(p["steps"], cfg["motion"]["rate_hz"])
    polys, _ = ex.forbidden_polygons(s, sf)
    floor = p["heights"]["release"] - cfg["bowls"].get("release_margin_mm", 20)
    viol = ex.check_workspace(s, traj, sf, polys, {o["bowl"]: floor for o in p["order"]})
    if viol:
        return {"n": n, "ang": ang, "bowl": ko, "ok": False, "stage": "공간검사",
                "why": "; ".join(f"{v['kind']} {v['detail']}" for v in viol)[:160]}
    return {"n": n, "ang": ang, "bowl": ko, "ok": True, "stage": "", "why": "", "time_s": round(p["total_s"], 1)}


def lift_check(s, cfg, w, yaw_w, cls="red_ok") -> dict:
    """실행기 집고 들어 올림을 손목 맞춤 뒤 자리 5곳(블록 자리·±10mm)에서: 앞(기울기 고정)·뒤(가까운 기울기) 되는 곳 수.
    hover = 계획 자세 그대로(첫 시도), 집는 높이 = _grasp_z(같은 롤 갈래)."""
    from robot.coord_pick import executor as ex
    from robot.coord_pick.plan import _Path, plan_sort
    obj = {"cls": cls, "px": s.w_to_px(np.asarray(w, float))[0].tolist(), "conf": 0.9, "zone": "input",
           "yaw_w": yaw_w, "aspect": 1.8}
    p = plan_sort(s, [obj], classes=(cls,), max_blocks=1)
    if not p["order"]:
        return {"plan": False, "before": 0, "after": 0, "n": len(JIT)}
    o = p["order"][0]
    py, pt = o["pick_yaw_w"] - s.frames.yaw, o["pick_tilt"]
    zh = max(o["approach_z"], ex.WRIST_DEFAULTS["hover_mm"])
    g = cfg["gripper"]
    tmax = max(s.tilt["grasp"], s.tilt["release"])
    keep = math.radians(ex.WRIST_DEFAULTS["keep_roll_deg"])
    before = after = 0
    tilts = []
    for dx, dy in JIT:
        tw = np.asarray(w, float) + (dx, dy)
        qh, _ = s.solve(s.w_to_r3(tw, zh), py, o["q_approach"], strict=True, tilt=pt, only_prev=True)
        if qh is None:
            continue
        gz, qg = ex._grasp_z(s, cfg, tw, py, pt, qh)
        if qg is None:
            continue
        path = _Path(s, qg, g["open"], yaw=py, tilt=pt)
        path.gripper(g["close"], "집음")
        if path.line(s.tcp_pos(qg), s.w_to_r3(tw, zh), py, pt, "들어 올림", strict=True, keep_roll=keep):
            before += 1
            after += 1
            continue
        qz, iz = s.solve(s.w_to_r3(tw, zh), py, qg, strict=True, tilt=None, tilt_max=tmax, near=pt, grip_n=g["close"],
                         only_prev=True)
        path = _Path(s, qg, g["open"], yaw=py, tilt=pt)
        path.gripper(g["close"], "집음")
        if qz is not None and path.line(s.tcp_pos(qg), s.w_to_r3(tw, zh), py, float(iz["tilt_used"]), "들어 올림",
                                        strict=False, tilt_max=tmax, keep_roll=keep):
            after += 1
            tilts.append(round(math.degrees(float(iz["tilt_used"]) - pt)))
    return {"plan": True, "before": before, "after": after, "n": len(JIT), "tilt_change": tilts,
            "pick_tilt": round(math.degrees(pt)), "zh": round(zh, 1)}


LOG_1137 = ROOT / "runs" / "20261023" / "coord_pick" / "113558_coord_pick_sort_defect_real.jsonl"
POSE_1137 = [-7.4, 36.1, -39.9, 46.1, -17.4, 62.9]   # 11:37 '내려감' 끝 측정값(위 기록 마지막 step)
TARGET_1137 = [375.0, 79.0, 40.0]                     # 그때 들어 올림 목표(로봇 좌표 mm, 오류 메시지)


def replay_1137(s, cfg) -> dict:
    """11:37 실물 마지막 실제 자세에서 들어 올림을 다시 푼다: 앞(계획 기울기 15° 고정) vs 뒤(허용 안 가까운 기울기)."""
    from robot.coord_pick import executor as ex
    from robot.coord_pick.plan import _Path
    q = s.n_to_q(np.asarray(POSE_1137, float))
    T = s.kin.fk(q)
    py, pt = math.atan2(T[1, 0], T[0, 0]), math.radians(15.0)
    tw = s.frames.r2w(np.asarray(TARGET_1137[:2]))
    zh = TARGET_1137[2] - float(s.w_to_r3(tw, 0.0)[2])
    g = cfg["gripper"]
    keep = math.radians(ex.WRIST_DEFAULTS["keep_roll_deg"])
    tmax = max(s.tilt["grasp"], s.tilt["release"])
    path = _Path(s, q, g["close"], yaw=py, tilt=pt)
    before = path.line(s.tcp_pos(q), s.w_to_r3(tw, zh), py, pt, "들어 올림", strict=True, keep_roll=keep)
    why = "; ".join(path.failures)
    qz, iz = s.solve(s.w_to_r3(tw, zh), py, q, strict=True, tilt=None, tilt_max=tmax, near=pt, grip_n=g["close"],
                     only_prev=True)
    path = _Path(s, q, g["close"], yaw=py, tilt=pt)
    after = qz is not None and path.line(s.tcp_pos(q), s.w_to_r3(tw, zh), py, float(iz["tilt_used"]), "들어 올림",
                                         strict=False, tilt_max=tmax, keep_roll=keep)
    return {"before": bool(before), "why_before": why, "after": bool(after),
            "tilt_after": round(math.degrees(float(iz["tilt_used"]))) if qz is not None else None}


def _pick(job) -> dict:
    """한 칸·각도: 들어 올림(앞·뒤) + ACT 시작 자세."""
    from robot.coord_pick.plan import plan_sort
    n, w, ang = job
    s, cfg = _W["s"], _W["cfg"]
    lc = lift_check(s, cfg, w, math.radians(ang))
    act = None
    obj = {"cls": "red_ok", "px": s.w_to_px(np.asarray(w, float))[0].tolist(), "conf": 0.9, "zone": "input",
           "yaw_w": math.radians(ang), "aspect": 1.8}
    p = plan_sort(s, [obj], classes=("red_ok",), max_blocks=1)
    if p["order"]:
        o = p["order"][0]
        rb = s.frames.w2r(np.asarray(w, float))
        u = rb / np.linalg.norm(rb)
        ws = s.frames.r2w(rb - ACT["back"] * u + ACT["side"] * np.array([-u[1], u[0]]))
        q, _ = s.solve(s.w_to_r3(ws, ACT["z"]), o["pick_yaw_w"] - s.frames.yaw, o["q_approach"], strict=True, tilt=None,
                       tilt_max=s.tilt["grasp"], near=math.radians(ACT["tilt"]), only_prev=True)
        act = q is not None
    return {"n": n, "ang": ang, "lift": lc, "act": act}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.compare.cell_table", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)
    _init()
    from robot.coord_pick.plan import input_cells
    s, cfg = _W["s"], _W["cfg"]
    cells = input_cells(s)
    full_jobs = [(n, c.tolist(), ang, t) for n, c in cells for ang in ANGLES for t in TARGETS]
    pick_jobs = [(n, c.tolist(), ang) for n, c in cells for ang in ANGLES]
    with Pool(a.jobs, initializer=_init) as pool:
        full = pool.map(_full, full_jobs)
        pick = pool.map(_pick, pick_jobs)
    F = {(r["n"], r["ang"], r["bowl"]): r for r in full}
    P = {(r["n"], r["ang"]): r for r in pick}
    # 11:37 실물(초록X 작업대 145.8, 148.5, 위 카메라 긴 축 −174.2°)
    real = replay_1137(s, cfg)
    sm = {"safety.forbid_margin_mm": cfg["safety"]["forbid_margin_mm"], "heights": {k: cfg["heights_mm"][k] for k in
                                                                                   ("grasp", "approach", "carry")}}
    lines = ["칸 × 블록 긴 축 통과표 (계획만 — 로봇·카메라 0) " + json.dumps(sm, ensure_ascii=False),
             "칸 번호 = 위 카메라 화면 기준(1 왼쪽 위 · 3 오른쪽 위 · 7 왼쪽 아래 = 로봇 쪽). 긴 축 0° = 테이프 가로와 나란히.",
             "글자: 분/초/불 = 분홍·초록·불량 그릇까지 전체 경로 ✓·✗(실패 단계) | 들 앞→뒤 = 들어 올림 되는 곳/5(블록 자리·±10mm) "
             "| A = ACT 시작 자세(120mm·비킴 33/−17·22°)", ""]
    for n, c in cells:
        lines.append(f"칸 {n} (작업대 {np.round(c).astype(int).tolist()} mm)")
        for ang in ANGLES:
            parts = []
            for ko, _, _ in TARGETS:
                r = F[(n, ang, ko)]
                parts.append(f"{ko[0]}{'✓' if r['ok'] else '✗(' + r['stage'] + ')'}")
            pk = P[(n, ang)]
            lc = pk["lift"]
            lift = f"들 {lc['before']}→{lc['after']}/{lc['n']}" if lc["plan"] else "들 -"
            lines.append(f"   {ang:>3}°  {' '.join(parts):<28} {lift:<10} A{'✓' if pk['act'] else '✗' if pk['act'] is False else '-'}")
    # 추천: 세 그릇 모두 ✓ + 들어 올림(뒤) 5/5 + ACT ✓ 인 칸·각도 — 서로 다른 칸 4개, 로봇에 가까운 칸부터
    good = {}
    for n, c in cells:
        for ang in ANGLES:
            if all(F[(n, ang, ko)]["ok"] for ko, _, _ in TARGETS) and P[(n, ang)]["lift"]["after"] == len(JIT) and P[(n, ang)]["act"]:
                good.setdefault(n, []).append(ang)
    r_of = {n: float(np.linalg.norm(s.frames.w2r(c))) for n, c in cells}
    pick4 = sorted(good, key=lambda n: r_of[n])[:4]
    rec = ", ".join(f"칸 {n}({'·'.join(str(x) for x in good[n])}°)" for n in pick4)
    lines += ["", f"11:37 실물(초록X 작업대 145.8, 148.5) 마지막 실제 자세에서 들어 올림 다시 풀기: 앞(15° 고정) "
                  f"{'✓' if real['before'] else '✗ — ' + real['why_before'][:90]} → 뒤(가까운 기울기) "
                  f"{'✓ ' + str(real['tilt_after']) + '°' if real['after'] else '✗'}",
              "  (표의 '들' 은 정적 계산 — 첫 hover 자세에서 시작해 이 실패가 안 보인다. 실물은 손목 맞춤 이동 뒤 자세에서 IK 를 "
              "시작해 기울기 15° 가 안 풀렸다)",
              f"추천(세 그릇 모두 ✓ · 들어 올림 5/5 · ACT ✓, 로봇에 가까운 칸부터 4개): {rec or '없음'}",
              f"모두 되는 칸·각도: {json.dumps({n: good[n] for n in sorted(good)}, ensure_ascii=False)}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cell_table.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "cell_table.json").write_text(json.dumps({"full": full, "pick": pick, "real_1137": real}, ensure_ascii=False,
                                                    default=float), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
