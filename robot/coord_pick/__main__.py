"""좌표 집기 시제품 CLI — 전부 dry-run(로봇·카메라를 열지 않는다). 설명은 robot/COORD_PICK.md.

프로젝트 루트에서, 프로젝트 .venv 로:
    python -m robot.coord_pick table                        # 숫자 0~9 의 5×3 칸 그림과 블록 수
    python -m robot.coord_pick digit 5 --demo 13            # 가짜 장면(블록 13개 흩음)으로 계획·검사
    python -m robot.coord_pick digit 5 --detect det.json    # detect_objects() 결과 파일로 계획
    python -m robot.coord_pick sort --detect det.json       # 투입 구역 분홍·초록 → 제 그릇 계획(실행은 executor sort)
    python -m robot.coord_pick reach                        # 닿는 범위 지도(기울기) + 투입 구역·그릇 가운데 표
    python -m robot.coord_pick calib-camera det.json        # 테이프 4점 위 블록 4개의 픽셀 → 호모그래피·잔차
    python -m robot.coord_pick touches calibrate 0:27.4-29.5 1:40.1-42.2 2:52.2-55.0 3:11.4-13.2 -o touches.json
                                                            # Studio 데이터셋(짚기 녹화)의 정지 구간 → 팔로워 측정 관절값
    python -m robot.coord_pick calib-base touches.json      # 집게 끝으로 짚은 테이프 점 → 로봇 위치·방향
공통: --offline(Studio 보정 범위를 안 읽음) --config <파일> --save(30Hz 궤적을 logs/ 에 JSON 으로)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

from robot.coord_pick import digits
from robot.coord_pick.geometry import apply_h, fit_homography, fit_rigid2d
from robot.coord_pick.plan import (BOWL_KO, COLOR_KO, CONFIG_PATH, Setup, blocks_from_detection, bowl_targets,
                                   demo_scene, demo_sort_scene, load_config, plan_digit, plan_sort, trajectory, zones_w)

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
CONTACT_MM = {"fixed_tip": [3.0, 0.0, 6.3], "frame": [0.0, 0.0, 0.0]}  # 짚은 점(끝 좌표 mm): 고정 집게 끝 가운데 / 끝 자체


def _f(v, n=0):
    return f"{v:.{n}f}"


def _print_steps(p) -> None:
    print("\n경유점(구간 끝의 tcp 높이 mm · 기울기 · 관절 정규화 6값 · 구간 시간)")
    acc = 0.0
    for st in p["steps"]:
        acc += st.dur_s
        if st.what.endswith("…"):  # 직선 구간의 중간 점(10mm 간격)은 묶어서 한 줄
            continue
        pr = f"z{_f(st.p_r[2]):>4} {st.info['tilt_deg']:4.0f}°" if st.p_r is not None else "          "
        print(f"  {st.what:<34} {pr}  [{' '.join(f'{v:6.1f}' for v in st.n)}]  {acc:4.1f}s")
        acc = 0.0


def _print_checks(p) -> None:
    print("\n검사")
    for k, v in p["checks"].items():
        print(f"  {k:<26} {v}")
    for w in p["warnings"]:
        print(f"  [주의] {w}")
    for f in p["failures"]:
        print(f"  [안됨] {f}")


def _save(s, p, name) -> None:
    traj = trajectory(p["steps"], s.cfg["motion"]["rate_hz"])
    LOG_DIR.mkdir(exist_ok=True)
    out = LOG_DIR / f"coord_pick_{dt.datetime.now():%Y%m%d_%H%M%S}_{name}.json"
    out.write_text(json.dumps({"name": name, "rate_hz": s.cfg["motion"]["rate_hz"],
                               "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                               "wrist_roll", "gripper"],
                               "unit": "normalized", "targets": np.round(traj, 3).tolist()}, ensure_ascii=False))
    print(f"궤적 저장: {out}")


def cmd_digit(a, s: Setup) -> int:
    if a.detect:
        det = json.loads(Path(a.detect).read_text(encoding="utf-8"))
        blocks, src = blocks_from_detection(det), f"검출 파일 {a.detect}"
    else:
        blocks = demo_scene(s, a.demo, a.seed)
        n_ok = sum(b["cls"] in ("red_ok", "blue_ok") for b in blocks)
        src = (f"가짜 장면(입력 구역에 정상 {n_ok}개 + 불량 {len(blocks) - n_ok}개, seed {a.seed}"
               + (f" — 입력 구역이 좁아 {a.demo}개를 다 못 놓음" if n_ok < a.demo else "") + ")")
    p = plan_digit(s, a.digit, blocks, color=a.color)
    print(f"좌표 집기 dry-run — 숫자 {p['digit']} ({p['cells']}칸) · {src} · 관절 범위 {s.range_src}")
    print("로봇·카메라는 열지 않았다. 실제 실행은 python -m robot.coord_pick.executor (COORD_PICK.md 9절).\n")
    print(f"숫자 {p['digit']} 모양(위 = 맨 위 행):")
    for row in digits.FONT[p["digit"]]:
        print("   " + row.replace("#", "■ ").replace(".", "· "))
    print("\n순서  색        블록 픽셀      → 작업대 mm    → 칸(행,열)  칸 중심 mm     집기 여유  놓기 여유  집게 방향(집기/놓기)")
    for i, o in enumerate(p["order"], 1):
        b = o["block"]
        print(f"{i:>3}   {COLOR_KO[b['cls']]:<6} ({_f(b['px'][0]):>4},{_f(b['px'][1]):>4})  ({_f(b['w'][0]):>4},{_f(b['w'][1]):>4})"
              f"   {str(o['cell']):<9} ({_f(o['target_w'][0]):>4},{_f(o['target_w'][1]):>4})"
              f"   {o['pick_clear_mm']:>6.1f}mm  {o['place_clear_mm']:>6.1f}mm  "
              f"{math.degrees(o['pick_yaw_w']):.0f}°/{math.degrees(o['place_yaw_w']):.0f}°")
    if a.verbose:
        _print_steps(p)
    _print_checks(p)
    traj = trajectory(p["steps"], s.cfg["motion"]["rate_hz"])
    m, sec = divmod(int(round(p["total_s"])), 60)
    print(f"\n예상 시간 {m}분 {sec}초(블록 {len(p['order'])}개, 블록당 {p['total_s'] / max(1, len(p['order'])):.1f}초) · "
          f"목표값 {len(traj)}개({s.cfg['motion']['rate_hz']}Hz)")
    if a.save:
        _save(s, p, f"digit{p['digit']}")
    return 1 if p["failures"] else 0


def cmd_sort(a, s: Setup) -> int:
    if a.detect:
        blocks, src = blocks_from_detection(json.loads(Path(a.detect).read_text(encoding="utf-8"))), f"검출 파일 {a.detect}"
    else:
        blocks, src = demo_sort_scene(s), "시연 배치 가짜 장면(분홍 2·초록 1·불량 1)"
    classes = tuple(c for k in a.colors.split(",") for c in {"red": ("red_ok",), "blue": ("blue_ok",),
                                                            "defect": ("red_defect", "blue_defect")}[k])
    p = plan_sort(s, blocks, classes=classes, max_blocks=a.blocks, lift_mm=a.lift_mm)
    hz = p["heights"]
    print(f"그릇 옮기기 dry-run · {src} · 관절 범위 {s.range_src}")
    print(f"높이(tcp mm): 집기 {hz['grasp']:g} · 접근 {hz['approach']:g} · 나르기 {hz['carry']:g} · 놓기 {hz['release']:g}"
          f"{f' (공중 리허설 +{a.lift_mm:g})' if a.lift_mm else ''}\n")
    print("순서  색          블록 픽셀    → 작업대 mm     집기 기울기·집게 방향   → 그릇           놓는 점 mm     놓기 기울기  집기 여유")
    for i, o in enumerate(p["order"], 1):
        b = o["block"]
        print(f"{i:>3}   {COLOR_KO[b['cls']]:<8} ({_f(b['px'][0]):>4},{_f(b['px'][1]):>4})  ({_f(b['w'][0]):>4},{_f(b['w'][1]):>4})"
              f"   {math.degrees(o['pick_tilt']):>4.0f}° · {math.degrees(o['pick_yaw_w']):>4.0f}°"
              f"           {BOWL_KO[o['bowl']]:<8} ({_f(o['drop_w'][0]):>4},{_f(o['drop_w'][1]):>4})"
              f"   {math.degrees(o['drop_tilt']):>4.0f}°     {o['pick_clear_mm']:>5.1f}mm")
    if a.verbose:
        _print_steps(p)
    _print_checks(p)
    traj = trajectory(p["steps"], s.cfg["motion"]["rate_hz"])
    m, sec = divmod(int(round(p["total_s"])), 60)
    print(f"\n계획 속도(관절 {s.cfg['motion']['max_deg_s']}°/s) 예상 {m}분 {sec}초 · 목표값 {len(traj)}개 — 실제 송신 시간은 "
          f"executor 가 느린 속도(safety.exec_deg_s)로 다시 잰다")
    if a.save:
        _save(s, p, "sort")
    return 1 if p["failures"] else 0


def cmd_table(a, s) -> int:
    print("숫자 0~9 → 5행×3열 칸(■ = 블록 하나). 괄호 = 필요한 블록 수. 정상 블록은 분홍 8·초록 8 = 16개")
    print(digits.table())
    return 0


def cmd_reach(a, s: Setup) -> int:
    """작업대 좌표 격자에서 집기(approach ↔ grasp 를 같은 기울기로)가 되는 최소 기울기 지도 + 투입 구역·그릇 표."""
    hz, bw = s.cfg["heights_mm"], s.cfg.get("bowls", {})
    tmax = s.tilt["grasp"]
    heights = (hz["approach"], hz["grasp"])
    z_rel = bw.get("rim_mm", 0) + bw.get("release_above_mm", 50)
    carry = max(hz.get("carry", z_rel), z_rel)
    t_any = max(tmax, s.tilt["release"])
    xs = np.arange(a.x0, a.x1 + 1, a.step)
    ys = np.arange(a.y1, a.y0 - 1, -a.step)
    seed = s.n_to_q(np.array(s.cfg["home_n"]))
    zw = zones_w(s)
    print(f"작업대 좌표(mm) — 열 x {a.x0:g}→{a.x1:g}, 행 y {a.y1:g}→{a.y0:g}, {a.step:g}mm 간격. tcp 높이 {heights} 를 같은 "
          f"기울기로 오르내릴 수 있는 최소 기울기(작업대 축 방향 집게 0°·90° 중 되는 것)")
    print(f"  0~7 = 기울기 0·5·…·35°(숫자 × {math.degrees(s.tilt['step']):g}°)  . = {math.degrees(tmax):g}° 까지 안 됨  "
          "R = 로봇 받침  [ ] = 투입 구역 안쪽  ( ) = 그릇 가운데 근처")
    base = s.frames.t
    bowls = bowl_targets(s, zw)

    def mark(x, y):
        if any(math.dist((x, y), c) < a.step * 0.75 for c in bowls.values()):
            return "(", ")"
        if "input" in zw and _in(zw["input"], (x, y)):
            return "[", "]"
        return " ", " "

    for y in ys:
        row = ""
        for x in xs:
            l, r = mark(x, y)
            if math.dist((x, y), base) < 45:
                row += f"{l}R{r}"
                continue
            col = s.column((x, y), 0.0, heights, seed, tilt_max=tmax)
            ch = "." if col is None else str(int(round(col["tilt"] / s.tilt["step"])))
            row += f"{l}{ch}{r}"
        print(f"{y:5.0f} {row}")
    # 표: 투입 구역 안쪽(블록 가운데가 올 수 있는 곳 = 안쪽 테이프에서 블록 반 변 15mm 안) + 그릇 가운데
    (x0, y0), (x1, y1) = s.cfg["table"]["input_area_mm"]
    half = s.cfg.get("block_mm", 30) / 2
    gx = np.linspace(x0 + half, x1 - half, a.cols)
    gy = np.linspace(y1 - half, y0 + half, a.rows)
    print(f"\n투입 구역 안쪽 블록 자리 {a.rows}×{a.cols} (작업대 mm, 블록 가운데 x {x0 + half:g}~{x1 - half:g}, y {y0 + half:g}~{y1 - half:g})"
          f" — 칸 = 필요한 기울기°(집게 방향 0°/90°), × = {math.degrees(tmax):g}° 까지 안 됨")
    head = "".join(f"{x:>12.0f}" for x in gx)
    print(f"   y \\ x{head}")
    bad = []
    for y in gy:
        cells = []
        for x in gx:
            r = s.frames.w2r((x, y))
            opts = []
            for yw in (0.0, math.pi / 2):
                col = s.column((x, y), yw, heights, seed, tilt_max=tmax, yaws=(yw,))
                if col is not None:  # 집은 뒤 나르는 높이로(기울기는 바뀌어도 됨)
                    up, _ = s.solve(s.w_to_r3((x, y), carry), yw - s.frames.yaw, col["qs"][0], strict=True, tilt=None,
                                    tilt_max=t_any, grip_n=s.cfg["gripper"]["close"])
                    if up is None:
                        col, s.last_why = None, f"들어 올리는 높이 {carry:g}mm 에 안 닿음"
                opts.append("×" if col is None else f"{math.degrees(col['tilt']):.0f}")
                if col is None:
                    bad.append(((x, y), yw, s.last_why, float(np.linalg.norm(r))))
            cells.append(f"{opts[0]:>4}/{opts[1]:<4}")
        print(f"{y:7.0f} " + "".join(f"{c:>12}" for c in cells))
    if bad:
        print("  안 되는 곳 이유:")
        seen = set()
        for (x, y), yw, why, rr in bad:
            key = (why.split("(")[0], round(math.degrees(yw)))
            if key in seen:
                continue
            seen.add(key)
            print(f"   ({x:.0f},{y:.0f}) 집게 {math.degrees(yw):.0f}° 반경 {rr:.0f}mm: {why}")
    tr = s.tilt["release"]
    src = "config bowls.center_mm(녹화에서 사람이 놓은 자리)" if s.cfg.get("bowls", {}).get("center_mm") else "zones.json 가운데"
    print(f"\n그릇 놓는 점({src}) — 놓기 tcp {z_rel:g}mm·나르기 {carry:g}mm, 기울기 {math.degrees(tr):g}° 까지")
    for name, c in bowls.items():
        r = s.frames.w2r(c)
        yw = s.tangent_yaw(s.w_to_r3(c, z_rel)) + s.frames.yaw
        col = s.column(c, yw, (carry, z_rel), seed, tilt_max=tr, grip_n=s.cfg["gripper"]["close"], yaws=(yw,))
        res = (f"됨 — 기울기 {math.degrees(col['tilt']):.0f}°" if col else f"안 됨 — {s.last_why}")
        print(f"  {BOWL_KO[name]:<6} 작업대 ({c[0]:.0f}, {c[1]:.0f}) · 로봇 반경 {np.linalg.norm(r):.0f}mm: {res}")
    return 0


def _in(poly, p) -> bool:
    from robot.coord_pick.plan import _inside
    return _inside(p, poly)


def cmd_calib_camera(a, s: Setup) -> int:
    det = json.loads(Path(a.file).read_text(encoding="utf-8"))
    blocks = blocks_from_detection(det)
    mm = np.array(s.cfg["table"]["markers_mm"], float)
    if len(blocks) != len(mm):
        print(f"[안됨] 블록 {len(blocks)}개가 보인다 — 테이프 점 {len(mm)}개에 하나씩만 올리고 다시 찍는다")
        return 1
    px = np.array([b["px"] for b in blocks])
    # 사진 속 블록을 테이프 점에 짝짓기: 예시 호모그래피로 옮긴 위치에서 가장 가까운 점
    guess = apply_h(s.H, px)
    pairs = [int(np.argmin(np.linalg.norm(mm - g, axis=1))) for g in guess]
    if len(set(pairs)) != len(mm):
        print("[안됨] 블록과 테이프 점 짝이 겹친다 — markers_px 에 대략 값을 먼저 손으로 넣고 다시")
        return 1
    order = np.argsort(pairs)
    H = fit_homography(px[order], mm)
    res = np.linalg.norm(apply_h(H, px[order]) - mm, axis=1)
    scale = np.linalg.norm(apply_h(H, px[order][:1] + [[1, 0]]) - apply_h(H, px[order][:1]))
    print(f"호모그래피 잔차 {np.round(res, 2).tolist()} mm (4점이면 0 이 정상 — 5점 이상이면 검증이 된다), 픽셀당 약 {scale:.2f}mm")
    print('config.json table.markers_px 에 넣을 값:', json.dumps(np.round(px[order], 1).tolist()))
    return 0


def cmd_touches(a, s: Setup) -> int:
    """Studio 데이터셋(짚기 녹화)의 정지 구간 평균 → touches.json. 팔로워 측정값(observation.state)을 쓴다 —
    리더 명령(action)은 탁자를 누르는 만큼·처지는 만큼 다르다(10-22: 먼 점에서 끝 z 18mm 차)."""
    from robot import home_stop_replay as hsr
    from robot import run_policy as rp
    data = hsr.load(rp.load_config(), [a.dataset])
    if not data:
        print(f"[안됨] 데이터셋 없음: {a.dataset}")
        return 1
    d = next(iter(data.values()))
    states = d["episodes"][str(a.episode)] if str(a.episode) in d["episodes"] else d["episodes"][a.episode]
    fps = d["fps"]
    out = []
    for w in a.windows:
        m, rng = w.split(":")
        t0, t1 = (float(v) for v in rng.split("-"))
        seg = np.array(states[int(t0 * fps):int(t1 * fps)])
        if len(seg) < 3:
            print(f"[안됨] 구간 {w} 이 비었다")
            return 1
        out.append({"marker": int(m), "n": seg.mean(0).round(2).tolist(), "std": seg.std(0).round(2).tolist()})
        print(f"  점 {m}: {t0:g}~{t1:g}s {len(seg)}프레임 평균 {out[-1]['n']} 흔들림(표준편차 최대) {seg[:, :5].std(0).max():.2f}")
    out.sort(key=lambda t: t["marker"])
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {a.out} (팔로워 측정값 observation.state)")
    return 0


def cmd_calib_base(a, s: Setup) -> int:
    """touches.json = [{"marker": 0, "n": [정규화 6값]}, ...] (3점 이상). 집게 끝을 테이프 점에 대고 멈춘 구간의 팔로워 측정값."""
    touches = json.loads(Path(a.file).read_text(encoding="utf-8"))
    mm = np.array(s.cfg["table"]["markers_mm"], float)
    off = np.array(CONTACT_MM[a.contact]) / 1000
    r_pts, w_pts, zs = [], [], []
    for t in touches:
        T = s.kin.fk(s.n_to_q(np.array(t["n"])))
        p = (T[:3, 3] + T[:3, :3] @ off) * 1000
        r_pts.append(p[:2])
        w_pts.append(mm[t["marker"]])
        zs.append(p[2])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -T[2, 2]))))
        print(f"  점 {t['marker']}: 짚은 점 로봇 좌표 {np.round(p, 1).tolist()} mm (z 는 0 근처여야 함) · 기울기 {tilt:.0f}°")
    if len(r_pts) < 3:
        print("[안됨] 3점 이상 필요")
        return 1
    tx, ty, yaw = fit_rigid2d(np.array(r_pts), np.array(w_pts))
    c, si = math.cos(yaw), math.sin(yaw)
    pred = np.array(r_pts) @ np.array([[c, si], [-si, c]]) + [tx, ty]
    res = np.linalg.norm(pred - np.array(w_pts), axis=1)
    print(f"로봇 base: x {tx:.1f}, y {ty:.1f} mm, yaw {math.degrees(yaw):.1f}° · 잔차 {np.round(res, 1).tolist()} mm"
          f" · 짚은 점 z {np.round(zs, 1).tolist()} mm")
    print("  잔차가 5mm 넘으면 각도 변환(부호·영점) 또는 짚은 자리를 의심한다. 점 사이 거리가 줄자 값과 같아야 한다.")
    print('config.json table.base 에 넣을 값:', json.dumps({"x": round(tx, 1), "y": round(ty, 1), "yaw_deg": round(math.degrees(yaw), 1)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.coord_pick", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--offline", action="store_true", help="Studio 보정 범위를 읽지 않고 기본값")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("digit")
    d.add_argument("digit", choices=list(digits.FONT))
    d.add_argument("--detect")
    d.add_argument("--demo", type=int, default=9)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--color", choices=["any", "red", "blue"], default="any", help="red = 분홍, blue = 초록(코드 식별자)")
    d.add_argument("--save", action="store_true")
    d.add_argument("-v", "--verbose", action="store_true")
    so = sub.add_parser("sort")
    so.add_argument("--detect", help="det.json(없으면 시연 배치 가짜 장면)")
    so.add_argument("--colors", default="red,blue")
    so.add_argument("--blocks", type=int)
    so.add_argument("--lift-mm", type=float, default=0.0)
    so.add_argument("--save", action="store_true")
    so.add_argument("-v", "--verbose", action="store_true")
    sub.add_parser("table")
    r = sub.add_parser("reach")
    r.add_argument("--x0", type=float, default=-100)
    r.add_argument("--x1", type=float, default=480)
    r.add_argument("--y0", type=float, default=-180)
    r.add_argument("--y1", type=float, default=220)
    r.add_argument("--step", type=float, default=40)
    r.add_argument("--rows", type=int, default=5)
    r.add_argument("--cols", type=int, default=7)
    c = sub.add_parser("calib-camera")
    c.add_argument("file")
    t = sub.add_parser("touches")
    t.add_argument("dataset", help="Studio 데이터셋 이름 또는 id(짚기 녹화)")
    t.add_argument("windows", nargs="+", help="점번호:시작초-끝초 (예 0:27.4-29.5)")
    t.add_argument("--episode", type=int, default=0)
    t.add_argument("-o", "--out", default="touches.json")
    b = sub.add_parser("calib-base")
    b.add_argument("file")
    b.add_argument("--contact", choices=list(CONTACT_MM), default="fixed_tip",
                   help="짚은 점: fixed_tip = 고정 집게 끝(기본, 기울여 짚으면 가장 낮은 점) / frame = URDF 끝")
    a = ap.parse_args(argv)
    if a.cmd == "table":
        return cmd_table(a, None)
    s = Setup.load(load_config(a.config), use_studio=not a.offline)
    return {"digit": cmd_digit, "sort": cmd_sort, "reach": cmd_reach, "calib-camera": cmd_calib_camera,
            "touches": cmd_touches, "calib-base": cmd_calib_base}[a.cmd](a, s)


if __name__ == "__main__":
    sys.exit(main())
