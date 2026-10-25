"""좌표 집기 실행기 — 계획(plan.py) → 느린 속도로 시간 다시 맞추기 → 작업 공간 검사 → 가짜 로봇 모의 실행 →
(문이 전부 열렸을 때만) Studio 컨테이너에서 stream.py 로 SharedRobot 에 송신.

에이전트 도구(COORD_PICK.md 9절):
    from robot.coord_pick.executor import arrange_digit, go_home, sort_blocks
    sort_blocks("red")                           # dry-run: 투입 구역의 분홍 → 분홍 그릇(계획·시간·경고만)
    sort_blocks("red", max_blocks=1, dry_run=False)   # 실제 실행 — BALL_ROBOT=1 일 때만
    go_home(dry_run=False)                       # 지금 자세 → 홈(그리퍼 그대로, open_gripper=True 면 벌림)
    arrange_digit("7", "red")                    # 숫자 모양(dry-run)

CLI(프로젝트 루트, 프로젝트 .venv):
    python -m robot.coord_pick.executor sort --color red,blue --demo         # dry-run(시연 배치 가짜 장면: 분홍 2·초록 1·불량 1)
    python -m robot.coord_pick.executor sort --color red --detect det.json   # det.json = detect_objects()·detect_hsv 결과
    BALL_ROBOT=1 python -m robot.coord_pick.executor sort --color red --lift-mm 40 --confirm   # 공중 리허설(빈손, 입회 때만)
    BALL_ROBOT=1 python -m robot.coord_pick.executor sort --color red --blocks 1 --confirm     # 실제(입회 때만, 카메라로 검출)
    python -m robot.coord_pick.executor 7 --demo 9                     # 숫자 dry-run(가짜 장면)
    BALL_ROBOT=1 python -m robot.coord_pick.executor home --confirm [--open]   # 지금 자세 → 홈(8°/s, 60° 넘게 멀면 거부)
    python -m robot.run_policy --stop                                   # 멈춤 = 정책과 같은 stop()

실제로 보내는 조건(하나라도 빠지면 보내지 않고 이유와 함께 dry-run 결과를 돌려준다):
BALL_ROBOT=1 · --confirm(함수는 dry_run=False) · config calibrated · 계획 실패 0 · 작업 공간 위반 0 · 금지 구역 자료 있음 ·
모의 실행 통과 · run_policy 와 같은 잠금(이 PC 에서 하나만)·사전 점검(컨테이너·Studio 세션·실행 중·장치 잠금) 통과.
단 공중 리허설(sort --lift-mm ≥ safety.rehearsal_lift_min_mm, 블록·그릇에 안 닿는 높이)은 calibrated=false 여도 보낸다 —
보정 확정(calibrated=true)은 그 리허설을 사용자 입회로 1회 통과한 뒤 총괄이 켠다(r15).
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import functools
import json
import math
import os
import signal as _signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from robot import run_policy as rp
from robot.coord_pick import fake, grasp_act, stream, wrist
from robot.coord_pick.kinematics import ARM, finger_points_mm
from robot.coord_pick.plan import (BOWL_KO, BOWL_ZONES, COLOR_KO, UNSORT_DEFAULTS, UNSORT_TRAVEL_EXTRA_MM, Setup, _Path,
                                   blocks_from_detection, bowl_targets, demo_scene, demo_sort_scene, demo_unsort_scene, drop_points, free_cells,
                                   load_config, plan_digit, plan_sort, pose_hits, select_unsort, select_which, trajectory,
                                   which_error,
                                   travel, zones_path, zones_w)

HERE = Path(__file__).resolve().parent
ZONES_JSON = HERE.parent.parent / "vision" / "zones.json"
FORBID_ZONES = ("bowl_red", "bowl_blue", "bowl_defect", "hand_spot")
CAMERA_SIZE = (640, 480)  # vision/camera.py 가 주는 사진 크기 = 호모그래피 픽셀 좌표
SAFETY_DEFAULTS = {
    "exec_deg_s": 15, "exec_deg_s_max": 45, "grip_units_s": 20,
    "track_max_deg": 8, "track_ticks": 3, "lag_ticks": 3, "stale_s": 0.5, "start_tol_deg": 10,
    "floor_mm": 10, "link_floor_mm": 20, "radius_mm": [110, 300], "radius_below_mm": 120,
    "forbid_below_mm": 100, "forbid_margin_mm": 20, "forbid_extra_mm": [],
    "grip_min_n": 50, "grip_close_min_n": 50, "rehearsal_lift_min_mm": 30, "home_deg_s": 8, "home_max_deg": 60,
    # 공중 구간(계획 걸음 fast — 블록을 들고 그릇으로·놓은 뒤 올라감·다음 블록 위로·홈으로) 관절 속도·가감속.
    # None = 공중도 exec_deg_s(예전 그대로). 내려가 쥐기·놓기 직전은 언제나 exec_deg_s(10-23).
    "air_deg_s": None, "air_acc_deg_s2": 150,
}
KST = dt.timezone(dt.timedelta(hours=9))


def safety(cfg: dict) -> dict:
    return {**SAFETY_DEFAULTS, **{k: v for k, v in cfg.get("safety", {}).items() if not k.startswith("_")}}


class StageLog:
    """단계마다 한 줄(JSON). 실제 실행·dry-run 모두 남긴다."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, stage: str, **data) -> None:
        self.f.write(json.dumps({"ts": dt.datetime.now(KST).isoformat(timespec="milliseconds"), "stage": stage, **data},
                                ensure_ascii=False, default=_jsonable) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.round(3).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ------------------------------------------------------------------ 속도·한도

def limits(s: Setup, sf: dict, speed_deg_s: float) -> dict:
    """stream() 인자(정규화 단위). 몸통 틱당 한도 = 속도/주기, 추종 한도 = track_max_deg."""
    rate = float(s.cfg["motion"]["rate_hz"])
    dpu = np.array([s.jm.deg_per_unit(k) for k in ARM])
    return {"rate_hz": rate, "step_body": (speed_deg_s / rate / dpu).tolist(), "step_grip": sf["grip_units_s"] / rate,
            "track_max": (sf["track_max_deg"] / dpu).tolist(), "track_ticks": int(sf["track_ticks"]),
            "lag_ticks": int(sf["lag_ticks"]), "stale_s": float(sf["stale_s"]),
            "start_tol": float(sf["start_tol_deg"] / dpu.max()), "settle_s": 0.5}


def _tail_settle(sf: dict, steps) -> dict:
    """구간이 이미 멈춤(motion.settle_s)으로 끝나면 송신기 끝 대기(기본 0.5초)를 safety.hold_tail_s 로 — 겹치는 대기.
    끝 대기 동안에도 추종 검사는 한다. 값이 없으면(기본) 그대로."""
    if sf.get("hold_tail_s") is not None and steps and steps[-1].what.endswith("멈춤"):
        return {"settle_s": float(sf["hold_tail_s"])}
    return {}


def air_limits(s: Setup, sf: dict, speed_deg_s: float) -> dict | None:
    """공중 구간 한도(정규화 단위): 틱당 관절 변화 step(safety.air_deg_s, exec_deg_s_max 넘지 않게)와 가속(진행 단위/틱²).
    safety.air_deg_s 가 없거나 보통 속도 이하면 None(공중도 보통 속도)."""
    air = sf.get("air_deg_s")
    if not air:
        return None
    air = float(min(air, sf["exec_deg_s_max"]))
    if air <= speed_deg_s:
        return None
    rate = float(s.cfg["motion"]["rate_hz"])
    dpu = np.array([s.jm.deg_per_unit(k) for k in ARM])
    return {"deg_s": air, "step": (air / rate / dpu).tolist(), "acc": float(sf["air_acc_deg_s2"]) / (air * rate)}


def _smooth(seg: np.ndarray, step: np.ndarray, acc: float) -> np.ndarray:
    """seg[0] 에서 시작하는 공중 구간 틱들(seg[1:])을 같은 경로 위에서 사다리꼴 속도(최고 = 틱당 step, 가속 acc 진행/틱²,
    처음·끝 속도 0)로 다시 나눈다. 반환 = 새 틱들(seg[0] 빼고, 마지막 = seg[-1])."""
    d = np.abs(np.diff(seg[:, :5], axis=0)) / step
    inc = d.max(axis=1)
    cum = np.r_[0.0, np.cumsum(inc)]
    total = float(cum[-1])
    if total < 1e-9:
        return seg[1:]
    ramp_d = 1.0 / (2 * acc)                   # 0 → 최고 속도까지 가는 거리(진행 단위)
    if 2 * ramp_d >= total:                    # 최고 속도에 못 미침(삼각형)
        t_acc = math.sqrt(total / acc)
        T = 2 * t_acc
        vmax = acc * t_acc
    else:
        t_acc, vmax = 1.0 / acc, 1.0
        T = 2 * t_acc + (total - 2 * ramp_d) / vmax
    n = max(1, int(math.ceil(T - 1e-9)))
    out = []
    for k in range(1, n + 1):
        t = T * k / n
        if t <= t_acc:
            u = 0.5 * acc * t * t
        elif t >= T - t_acc:
            u = total - 0.5 * acc * (T - t) ** 2
        else:
            u = 0.5 * acc * t_acc * t_acc + vmax * (t - t_acc)
        u = min(max(u, 0.0), total)
        i = int(np.searchsorted(cum, u, side="right")) - 1
        i = min(max(i, 0), len(seg) - 2)
        f = 0.0 if cum[i + 1] - cum[i] < 1e-12 else (u - cum[i]) / (cum[i + 1] - cum[i])
        out.append(seg[i] + (seg[i + 1] - seg[i]) * f)
    out[-1] = seg[-1]
    return np.array(out)


def retime(targets: np.ndarray, marks: list, step_body, step_grip: float, *, fast=None,
           air: dict | None = None) -> tuple[np.ndarray, list]:
    """계획 궤적을 틱당 한도 안으로 늘린다(구간마다 모든 관절을 같은 비율로 → 경로 모양은 그대로).
    fast(틱마다 공중 구간 여부) + air(air_limits) 가 있으면 공중 구간은 공중 한도로 늘린 뒤, 이어진 공중 구간마다 사다리꼴
    가감속으로 다시 나눈다(처음·끝 속도 0 — 블록을 든 채 갑자기 출발·정지하지 않게). 그리퍼가 움직이는 공중 구간은 그대로."""
    sb = np.broadcast_to(np.asarray(step_body, float), (5,))
    if air is None or fast is None or not np.any(fast):
        out, where = [targets[0]], {0: 0}
        for i in range(1, len(targets)):
            out.extend(stream.ramp(targets[i - 1], targets[i], sb, step_grip))
            where[i] = len(out) - 1
        return np.array(out), [(where[i], label) for i, label in marks]
    sa = np.asarray(air["step"], float)
    out, where, i = [targets[0]], {0: 0}, 1
    while i < len(targets):
        if not fast[i]:
            out.extend(stream.ramp(targets[i - 1], targets[i], sb, step_grip))
            where[i] = len(out) - 1
            i += 1
            continue
        j = i
        while j + 1 < len(targets) and fast[j + 1]:
            j += 1
        seg = targets[i - 1:j + 1]
        if np.ptp(seg[:, 5]) > 1e-6:           # 그리퍼가 움직이는 구간은 보통 방식으로(공중 한도만)
            for k in range(i, j + 1):
                out.extend(stream.ramp(targets[k - 1], targets[k], sa, step_grip))
                where[k] = len(out) - 1
        else:
            dense = [targets[i - 1]]
            idx = {}
            for k in range(i, j + 1):
                dense.extend(stream.ramp(targets[k - 1], targets[k], sa, step_grip))
                idx[k] = len(dense) - 1
            dense = np.array(dense)
            sm = _smooth(dense, sa, air["acc"])
            # 표시 위치: 원래 틱의 경로 진행 비율 → 새 틱 번호
            inc = (np.abs(np.diff(dense[:, :5], axis=0)) / sa).max(axis=1)
            cum = np.r_[0.0, np.cumsum(inc)]
            inc2 = (np.abs(np.diff(np.vstack([dense[:1], sm])[:, :5], axis=0)) / sa).max(axis=1)
            cum2 = np.cumsum(inc2)
            base = len(out) - 1
            out.extend(sm)
            for k, di in idx.items():
                where[k] = base + 1 + min(int(np.searchsorted(cum2, cum[di] - 1e-9)), len(sm) - 1)
        i = j + 1
    return np.array(out), [(where[i_], label) for i_, label in marks]


# ------------------------------------------------------------------ 작업 공간

def forbidden_polygons(s: Setup, sf: dict) -> tuple[list[tuple[str, np.ndarray]], str]:
    """금지 구역(작업대 mm 다각형): 비전 zones.json 의 그릇 3·손 자리(픽셀 → 호모그래피) + config 의 추가 다각형."""
    polys, src = [], []
    zpath = zones_path(s.cfg)
    if zpath is not None and zpath.exists():
        d = json.loads(zpath.read_text(encoding="utf-8"))
        w, h = d.get("image_size", CAMERA_SIZE)
        for name in FORBID_ZONES:
            if name in d.get("zones", {}):
                px = np.array(d["zones"][name], float) * (CAMERA_SIZE[0] / w, CAMERA_SIZE[1] / h)
                polys.append((name, s.px_to_w(px)))
        src.append(f"vision/zones.json({len(polys)}개)")
    extra = sf.get("forbid_extra_mm") or []
    for i, poly in enumerate(extra):
        polys.append((f"extra{i + 1}", np.array(poly, float)))
    if extra:
        src.append(f"config safety.forbid_extra_mm({len(extra)}개)")
    return polys, " + ".join(src) or "없음"


def _inside(p: np.ndarray, poly: np.ndarray) -> bool:
    x, y, inside = p[0], p[1], False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _dist(p: np.ndarray, poly: np.ndarray) -> float:
    best = math.inf
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab, ap = b - a, p - a
        t = float(np.clip(ap @ ab / max(ab @ ab, 1e-9), 0, 1))
        best = min(best, float(np.linalg.norm(ap - t * ab)))
    return best


def check_workspace(s: Setup, targets: np.ndarray, sf: dict, polys: list,
                    allow: dict[str, float] | None = None) -> list[dict]:
    """보내기 전에 모든 목표를 검사한다. 종류마다 {kind, count, first_tick, detail}.

    바닥 = 집게 끝(고정·움직이는 집게 모서리, 그 틱의 그리퍼 값) 최저점. allow = {구역 이름: 이 높이(mm)까지는 내려가도 됨}
    — 그릇 옮기기에서 목표 그릇 위로 놓는 높이까지 내려가는 것(다른 그릇·손 자리는 forbid_below_mm 그대로).
    """
    found: dict[str, dict] = {}
    allow = allow or {}

    def add(kind: str, i: int, detail: str) -> None:
        f = found.setdefault(kind, {"kind": kind, "count": 0, "first_tick": i, "detail": detail})
        f["count"] += 1

    bad = np.nonzero(~np.isfinite(targets).all(axis=1))[0]
    if len(bad):
        return [{"kind": "not_finite", "count": len(bad), "first_tick": int(bad[0]), "detail": "NaN·inf 목표"}]
    lim = np.degrees(s.kin.limits())
    rmin, rmax = sf["radius_mm"]
    for i, n in enumerate(targets):
        q = s.n_to_q(n)
        pts = s.kin.link_points(q) * 1000  # 어깨 이후 관절 원점들 + 끝(마지막)
        T = s.kin.fk(q)
        fingers = finger_points_mm(n[5]) @ T[:3, :3].T + T[:3, 3] * 1000
        low = fingers[int(np.argmin(fingers[:, 2]))]
        if low[2] < sf["floor_mm"]:
            add("floor", i, f"집게 끝 높이 {low[2]:.0f}mm < {sf['floor_mm']}mm")
        if pts[:-1, 2].min() < sf["link_floor_mm"]:
            add("link_floor", i, f"관절 높이 {pts[:-1, 2].min():.0f}mm < {sf['link_floor_mm']}mm")
        r = math.hypot(low[0], low[1])
        if low[2] < sf["radius_below_mm"] and not rmin <= r <= rmax:
            add("radius", i, f"집게 끝 반경 {r:.0f}mm (허용 {rmin}~{rmax}, 높이 {low[2]:.0f}mm)")
        qd = np.degrees(q)
        if ((qd < lim[:, 0] - 1) | (qd > lim[:, 1] + 1)).any():
            j = int(np.argmax(np.maximum(lim[:, 0] - qd, qd - lim[:, 1])))
            add("joint_limit", i, f"{ARM[j]} {qd[j]:.0f}° (범위 {lim[j, 0]:.0f}~{lim[j, 1]:.0f})")
        if n[5] < sf["grip_min_n"]:
            add("gripper", i, f"그리퍼 {n[5]:.1f} < {sf['grip_min_n']}")
        for name, poly in polys:
            below = allow.get(name, sf["forbid_below_mm"])
            for p in (low, pts[-2]):  # 집게 끝 최저점과 손목
                if p[2] < below:
                    pw = s.frames.r2w(p[:2])
                    if _inside(pw, poly) or _dist(pw, poly) < sf["forbid_margin_mm"]:
                        add(f"forbidden:{name}", i, f"{name} 위 {p[2]:.0f}mm < {below:g} (작업대 {np.round(pw).tolist()} mm)")
                        break
    return list(found.values())


def simulate(targets: np.ndarray, marks: list, lim: dict, **fault) -> dict:
    """가짜 로봇·가짜 시계로 stream() 을 끝까지 돌린다(실제 기다림 없음). fault = FakeSharedRobot 오차 주입."""
    clock = fake.FakeClock()
    robot = fake.FakeSharedRobot(targets[0], clock, **fault)
    events: list[dict] = []
    res = stream.stream(robot, targets, marks, **lim, emit=events.append, clock=clock.now, sleep=clock.sleep)
    return {**res, "events": events, "robot": robot}


# ------------------------------------------------------------------ 컨테이너 송신

def build_script(payload: dict, fake_kwargs: dict | None = None, entry: str = "container_main") -> str:
    """stream.py 소스 + 실행 묶음 + 진입점(container_main = 궤적 송신, container_return = 지금 자세에서 목표로).
    fake_kwargs 가 있으면 fake.py 를 붙여 가짜 로봇으로(호스트 시험)."""
    parts = [(HERE / "stream.py").read_text(encoding="utf-8")]
    if fake_kwargs is not None:
        parts.append((HERE / "fake.py").read_text(encoding="utf-8").replace("from __future__ import annotations\n", ""))
    parts.append(f"PAYLOAD = json.loads({json.dumps(json.dumps(payload, default=_jsonable))})\n")
    if fake_kwargs is None:
        parts.append(f"sys.exit({entry}(PAYLOAD))\n")
    else:
        start = "PAYLOAD['targets'][0]" if entry == "container_main" else "PAYLOAD['fake_start']"
        parts.append(f"FAKE = json.loads({json.dumps(json.dumps(fake_kwargs))})\n"
                     "class _RealClock:\n    now = staticmethod(time.monotonic)\n"
                     f"sys.exit({entry}(PAYLOAD, robot_factory=lambda: "
                     f"FakeSharedRobot({start}, _RealClock(), **FAKE)))\n")
    return "\n".join(parts)


def supervise(argv: list[str], script: str, *, wall_s: float, log: StageLog, send_signal, env: dict | None = None) -> dict:
    """송신 프로세스를 띄우고 끝날 때까지 본다. run_policy._run 과 같은 멈춤 규칙:
    stop() → kill 이면 SIGKILL, home 이면 SIGINT(그 자리 유지 후 끝) / 벽시계 상한 → SIGINT → 1초마다 SIGKILL.
    send_signal(sig, proc): 실제는 run_policy._signal_runs(컨테이너 안 프로세스), 시험은 로컬 os.kill."""
    events: list[dict] = []
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=env)

    def reader() -> None:
        for line in proc.stdout:
            line = line.rstrip("\n")
            try:
                ev = json.loads(line)
                events.append(ev)
                log("stream", **ev)
            except ValueError:
                log("stream_text", text=line[-500:])

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        proc.stdin.write(script)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    deadline = time.monotonic() + wall_s
    stop_at = kill_at = None
    stopped = timed_out = False
    while True:
        try:
            rc = proc.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        mode = rp._stop_flag()
        kind, _, delivered = (mode or "").partition(":")
        if mode and stop_at is None:
            stopped, stop_at = True, now
            log("stop_seen", mode=mode)
            if kind == "kill" or delivered != "sent":
                send_signal("SIGINT" if kind == "home" else "SIGKILL", proc)
        elif not mode and not timed_out and now > deadline:
            timed_out, stop_at = True, now
            log("wall_limit", wall_s=wall_s)
            send_signal("SIGINT", proc)
        grace = 1.0 if kind == "kill" else 3.0
        if stop_at is not None and now - stop_at > grace and (kill_at is None or now - kill_at > 1.0):
            kill_at = now
            send_signal("SIGKILL", proc)
    th.join(timeout=2)
    end = next((e for e in reversed(events) if e.get("ev") == "end"), None)
    measured = next((e for e in events if e.get("ev") == "measured"), None)
    last_step = next((e for e in reversed(events) if e.get("ev") == "step"), None)
    return {"returncode": rc, "stopped": stopped, "timed_out": timed_out, "end": end, "measured": measured,
            "last_step": last_step, "steps_done": sum(e.get("ev") == "step" for e in events)}


def _studio_env(rcfg: dict, name: str | None) -> tuple[str, str]:
    """(project_id, environment_id): config 의 studio_environment 이름, 없으면 팔로워가 있는 환경이 하나일 때 그것."""
    def items(d):
        return d.get("items", d) if isinstance(d, dict) else d
    found = []
    for p in items(rp._get(rcfg, "/api/projects")):
        for e in items(rp._get(rcfg, f"/api/projects/{p['id']}/environments")):
            if name is None or e.get("name") == name:
                found.append((p["id"], e["id"]))
    if len(found) != 1:
        raise rp.PolicyError("environment", f"Studio 환경을 하나로 못 정함({len(found)}개) — config.json studio_environment 에 이름")
    return found[0]


def _run_real(s: Setup, targets: np.ndarray | None, marks: list, lim: dict, log: StageLog, *, name: str,
              entry: str = "container_main", extra: dict | None = None, wall_s: float | None = None) -> dict:
    """잠금 → Studio 환경·팔로워 → 이전 실행 기다림 → 사전 점검 → stop 확인 → 컨테이너에서 송신(감독). entry =
    container_main(targets 궤적) 또는 container_return(지금 자세 → extra["target"], go_home)."""
    rcfg = rp.load_config()
    rp.STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_f = open(rp.STATE_DIR / "run.lock", "w")
    try:
        try:
            fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return {"reason": "refused", "error": "already_running: 이 PC 에서 run_policy·좌표 집기가 실행 중"}
        pid, env_id = _studio_env(rcfg, s.cfg.get("studio_environment"))
        _env, ia, follower, notes = rp.environment_robot(rcfg, pid, env_id, require_devices=True)
        log("robot_config", follower=follower, notes=notes)
        rp._stop_requested.clear()
        (rp.STATE_DIR / "stop").unlink(missing_ok=True)
        prev = rp.wait_previous(rcfg)  # 결과를 먼저 돌려준 이전 정책 실행이 끝나는 중일 수 있다(run_policy 와 같은 규칙)
        log("wait_previous", **prev)
        if not prev["ok"]:
            return {"reason": "refused", "error": f"previous_not_released: {prev['busy']}"}
        problems, info = rp.robot_preflight(rcfg, follower["port"], [follower["port"]])
        log("preflight", problems=problems, info=info)
        if problems:
            return {"reason": "refused", "error": "; ".join(f"{c}: {m}" for c, m in problems)}
        if rp._stop_flag():
            return {"reason": "stopped", "error": "시작 전에 stop 요청"}
        run_id = f"{dt.datetime.now(KST):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        payload = {"run_id": run_id, "robot_name": rcfg["robot_name"], "driver": ia["robot"]["init_args"]["robot"],
                   "limits": lim, **({"targets": targets, "marks": marks} if entry == "container_main" else {}),
                   **(extra or {})}
        argv = (rp._docker(rcfg) + ["exec", "-i", "-u", "appuser", "-e", f"{rp.MARKER}={run_id}", "-w", "/tmp",
                                    rcfg["container"], rp.CPY, "-I", "-", "coord_pick", run_id])
        (rp.STATE_DIR / "current.json").write_text(json.dumps(
            {"run_id": run_id, "name": f"coord_pick:{name}", "started": dt.datetime.now(KST).isoformat(timespec="seconds"),
             "host_pid": os.getpid(), "log_path": str(log.path)}), encoding="utf-8")
        wall = wall_s or len(targets) / lim["rate_hz"] * 1.5 + 30
        log("launch", run_id=run_id, entry=entry, ticks=None if targets is None else len(targets), wall_s=round(wall, 1))
        try:
            sup = supervise(argv, build_script(payload, entry=entry), wall_s=wall, log=log,
                            send_signal=lambda sig, _proc: rp._signal_runs(rcfg, sig))
        finally:
            (rp.STATE_DIR / "current.json").unlink(missing_ok=True)
        end = sup["end"] or {}
        if not sup["end"]:  # 10-23 13:29:56: 송신기가 end 없이 끊김 → 원인을 다음에 볼 수 있게 종료 코드를 남긴다
            log("stream_lost", returncode=sup.get("returncode"), stopped=sup["stopped"], timed_out=sup["timed_out"])
        reason = ("stopped" if sup["stopped"] else "timeout" if sup["timed_out"] else end.get("reason", "error"))
        return {"reason": reason, "run_id": run_id, "run": {**sup, "end": end}, "error": end.get("error"),
                "measured": sup.get("measured")}
    finally:
        lock_f.close()


# ------------------------------------------------------------------ 에이전트 도구

def detect_live(s: Setup) -> dict:
    """위 카메라 한 장(원본은 robot/logs/<날짜>/front_*.jpg 로 남김) → detect_objects(그 사진) + 블록 긴 축 방향(yaw_w)."""
    import cv2

    from robot.coord_pick import detect_hsv
    from vision import camera
    from vision.detect import detect_objects
    frame, _src = camera.grab_frame()
    d = rp.LOG_DIR / f"{dt.datetime.now(KST):%Y%m%d}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"front_{dt.datetime.now(KST):%H%M%S}.jpg"
    cv2.imwrite(str(path), frame)
    det = detect_objects(str(path))
    det["objects"] = detect_hsv.measure_yaw(frame, det.get("objects", []), s)
    det["front_image"] = str(path)
    return det


def _detect(detection: dict | None, demo, s: Setup, demo_fn) -> tuple[list[dict], str]:
    if detection is not None:
        return blocks_from_detection(detection), "detection"
    if demo is not None:
        return demo_fn(s, demo), f"demo({demo})"
    from vision.detect import detect_objects  # 위 카메라를 잠깐 연다(비전 담당 함수)
    return blocks_from_detection(detect_objects()), "detect_objects()"


def _execute(p: dict, s: Setup, cfg: dict, sf: dict, speed: float, log: StageLog, out: dict, *, dry_run: bool,
             name: str, allow: dict[str, float] | None = None, rehearsal: bool = False,
             floor_from_start: bool = False) -> dict:
    """계획 p → 느린 속도로 늘림 → 작업 공간 검사 → 모의 실행 → (문이 다 열렸으면) 실제 송신. out 을 채워 돌려준다.
    floor_from_start: 시작 자세가 이미 floor_mm 아래면(집기 정책이 바닥 가까이 끝남) 바닥 한도를 시작 높이로 —
    더 내려가지만 않으면 거기서 빠져나오는 구간을 보낸다(10-23 하이브리드)."""
    traj, marks, fast = trajectory(p["steps"], cfg["motion"]["rate_hz"], with_marks=True, with_fast=True)
    lim = {**limits(s, sf, speed), **_tail_settle(sf, p["steps"])}
    air = air_limits(s, sf, speed)
    targets, marks = retime(traj, marks, lim["step_body"], lim["step_grip"], fast=fast, air=air)
    if air is not None:  # 컨테이너 송신기의 틱당 검사는 하나 — 공중 한도(큰 쪽)로. 보통 구간은 위에서 보통 한도로 늘렸다
        lim = {**lim, "step_body": np.maximum(lim["step_body"], air["step"]).tolist()}
        log("air", deg_s=air["deg_s"], acc_deg_s2=sf["air_acc_deg_s2"], fast_ticks=int(np.sum(fast)))
    polys, zsrc = forbidden_polygons(s, sf)
    sf_ws = sf
    if floor_from_start:
        T0 = s.kin.fk(s.n_to_q(targets[0]))
        f0 = float((finger_points_mm(targets[0][5]) @ T0[:3, :3].T + T0[:3, 3] * 1000)[:, 2].min())
        if f0 < sf["floor_mm"]:
            sf_ws = {**sf, "floor_mm": f0 - 0.5}
            log("floor_from_start", start_finger_mm=round(f0, 1), floor_mm=sf["floor_mm"])
    viol = check_workspace(s, targets, sf_ws, polys, allow)
    if cfg["gripper"]["close"] < sf["grip_close_min_n"]:
        viol.append({"kind": "gripper_close", "count": 1, "first_tick": -1,
                     "detail": f"gripper.close {cfg['gripper']['close']} < safety.grip_close_min_n {sf['grip_close_min_n']}"
                               " (블록에 닿는 값보다 더 쥐면 계속 민다)"})
    sim = simulate(targets, marks, lim)
    time_s = round(len(targets) / lim["rate_hz"], 1)
    out.update({"time_s": time_s, "plan_time_s": round(p["total_s"], 1), "ticks": len(targets),
                "warnings": p["warnings"] + ([] if polys else [f"금지 구역 자료 없음({zsrc}) — 실제 송신은 거부된다"]),
                "failures": p["failures"], "violations": viol, "forbid_zones": zsrc, "checks": p["checks"],
                "sim": {k: sim[k] for k in ("ok", "reason", "sent", "total", "max_err")}})
    log("plan", order=out.get("plan"), warnings=out["warnings"], failures=p["failures"], checks=p["checks"],
        plan_time_s=out["plan_time_s"])
    log("retime", ticks=len(targets), time_s=time_s, limits=lim)
    log("workspace", violations=viol, forbid_zones=zsrc, allow=allow)
    log("simulate", **out["sim"])
    plan_ok = not p["failures"] and not viol and sim["ok"]
    if dry_run:
        out["ok"] = plan_ok
        if not plan_ok:
            out["reason"] = "refused"
        return out
    gates = []
    if os.environ.get("BALL_ROBOT") != "1":
        gates.append("BALL_ROBOT=1 이 아님")
    if not cfg.get("calibrated") and not rehearsal:
        gates.append("config.json calibrated=false(보정 확정 전 — 공중 리허설 --lift-mm "
                     f"{sf['rehearsal_lift_min_mm']:g} 이상만 보낼 수 있다)")
    if not plan_ok:
        gates.append("계획 실패·작업 공간 위반·모의 실행 실패 중 하나")
    if not polys:
        gates.append("금지 구역 자료 없음(vision/zones.json 또는 safety.forbid_extra_mm)")
    log("gate", refused=gates)
    if gates:
        out.update(reason="refused", error="실제로 보내지 않음: " + " · ".join(gates))
        return out
    out["mode"] = "real"
    r = _run_real(s, targets, marks, lim, log, name=name)
    out.update(executed="run_id" in r, reason=r["reason"], run=r.get("run"), run_id=r.get("run_id"), error=r.get("error"))
    out["ok"] = r["reason"] == "done"
    return out


def _start(cfg: dict | None, speed_deg_s, tag: str, dry_run: bool, log_dir) -> tuple[dict, dict, float, StageLog, dt.datetime]:
    started = dt.datetime.now(KST)
    cfg = cfg or load_config()
    sf = safety(cfg)
    speed = float(min(speed_deg_s or sf["exec_deg_s"], sf["exec_deg_s_max"]))
    log = StageLog((log_dir or rp.LOG_DIR / f"{started:%Y%m%d}") / f"{started:%H%M%S}_coord_pick_{tag}{'' if dry_run else '_real'}.jsonl")
    return cfg, sf, speed, log, started


def arrange_digit(digit: str | int, color: str = "any", dry_run: bool = True, *, detection: dict | None = None,
                  small: bool = False, max_blocks: int | None = None, speed_deg_s: float | None = None,
                  demo: int | None = None, config: dict | None = None, log_dir: Path | None = None) -> dict:
    """흩어진 블록으로 숫자 모양을 만든다(COORD_PICK.md 9절).

    digit: "0"~"9". color: "red"(분홍)·"blue"(초록)·"any". dry_run=True(기본) 이면 로봇을 건드리지 않는다.
    detection: detect_objects() 결과(없으면 지금 찍는다 — 위 카메라를 잠깐 연다). demo: 가짜 장면 블록 수(시험용).
    반환 dict — ok·mode·executed·reason·summary·plan·time_s·warnings·failures·violations·sim·run·log_path·error.
    """
    digit = str(digit)
    cfg, sf, speed, log, _ = _start(config, speed_deg_s, f"{digit}{'s' if small else ''}", dry_run, log_dir)
    out: dict = {"ok": False, "mode": "dry_run", "executed": False, "reason": "planned", "digit": digit, "color": color,
                 "small": small, "speed_deg_s": speed, "log_path": str(log.path), "error": None}
    try:
        log("request", digit=digit, color=color, dry_run=dry_run, small=small, max_blocks=max_blocks, speed=speed,
            ball_robot=os.environ.get("BALL_ROBOT"))
        s = Setup.load(cfg, use_studio=True)
        blocks, src = _detect(detection, demo, s, demo_scene)
        p = plan_digit(s, digit, blocks, color=color, small=small, max_blocks=max_blocks)
        if len(p["order"]) < p["needed"]:  # 모양을 다 못 채우는 계획은 성공으로 치지 않는다
            p["failures"].append(f"숫자 {digit} 은 블록 {p['needed']}개가 필요한데 {len(p['order'])}개만 놓을 수 있다"
                                 f"({'색 ' + COLOR_KO.get(color + '_ok', color) + ' ' if color != 'any' else ''}블록을 더 놓는다)")
        out.update({"blocks": len(p["order"]), "source": src, "pattern": p["pattern"],
                    "plan": [{"order": i + 1, "color": COLOR_KO[o["block"]["cls"]],
                              "from_mm": np.round(o["block"]["w"], 1).tolist(), "to_cell": list(o["cell"]),
                              "to_mm": np.round(o["target_w"], 1).tolist()} for i, o in enumerate(p["order"])]})
        _execute(p, s, cfg, sf, speed, log, out, dry_run=dry_run, name=digit)
        m, sec = divmod(int(round(out["time_s"])), 60)
        out["summary"] = (f"숫자 {digit}{'(작은 글꼴)' if small else ''}: 블록 {len(p['order'])}개, 약 {m}분 {sec}초"
                          f"(관절 {speed:g}°/s). 실패 {len(p['failures'])} · 공간 위반 {len(out['violations'])} · "
                          f"경고 {len(out['warnings'])}")
        return out
    except Exception as e:  # noqa: BLE001 — 도구는 예외 대신 결과로
        out.update(reason="error", error=f"{type(e).__name__}: {e}")
        return out
    finally:
        log("result", **{k: v for k, v in out.items() if k not in ("plan",)})
        log.close()


SORT_CLASSES = {"red": ("red_ok",), "blue": ("blue_ok",), "defect": ("red_defect", "blue_defect"),
                "red_defect": ("red_defect",), "blue_defect": ("blue_defect",)}  # 색별 불량(10-23 11:35 "초록 불량만")


def sort_blocks(color="red", dry_run: bool = True, *, detection: dict | None = None,
                max_blocks: int | None = None, speed_deg_s: float | None = None, lift_mm: float = 0.0,
                demo: bool | None = None, config: dict | None = None, log_dir: Path | None = None,
                wrist_check: bool | None = None, redetect=None, front_image=None, grasp_mode: str | None = None,
                act_runner=None, keeper_factory=None, which: str | None = None) -> dict:
    """투입 구역의 블록을 색마다 제 그릇에 넣는다(좌표 집기 — ACT 정책 대신, COORD_PICK.md 9절).

    color: "red"(분홍 → 분홍 그릇)·"blue"(초록 → 초록 그릇)·"defect"(불량 → 불량 그릇), 또는 그 묶음("red,blue"·튜플).
    dry_run=True(기본)는 로봇을 건드리지 않는다. detection: detect_objects() 결과(objects[].cls·px·zone — 없으면 지금 위
    카메라로 찍는다). 대상 = zone "input" 의 그 색 블록, 로봇에 가까운 것부터(이웃과 집게 여유 3mm 이상인 것 먼저).
    max_blocks: 몇 개만. 놓는 점 = zones.json 그릇 다각형 가운데(호모그래피, 블록 높이) 위 bowls 높이에서 놓기(그릇 안으로
    내려가지 않음), 닿지 않으면 그릇 안에서 로봇 쪽으로 옮긴 점. lift_mm: 공중 리허설(모든 높이 +lift_mm, 빈손 궤적).
    demo=True: 시연 배치 가짜 장면. wrist_check: 손목 카메라 확인·보정(r18·r19 — 블록마다 hover·맞춤·닫기 전 판정·쥐었나·
    재시도, 실제 실행만. 기본 = config wrist.enabled, 공중 리허설이면 끔). redetect: 재시도 때 위 카메라 다시 찍기(기본
    vision.detect.detect_objects + 블록 방향). front_image: detection 과 같은 때 찍은 위 카메라 원본(경로·배열) — 주면 블록
    긴 축 방향을 재서 집게를 긴 축에 수직으로(짧은 변을 끼움). detection 도 없으면 위 카메라를 직접 찍어 둘 다 한다.
    반환 dict — arrange_digit 와 같은 키(ok·mode·executed·reason·summary·plan·time_s·warnings·failures·violations·sim·run·
    log_path·error). 안전 문도 같다(BALL_ROBOT=1·확인·calibrated·계획 실패 0·공간 위반 0·금지 구역·모의 실행) — 목적지 그릇만
    금지 구역 높이 제한에서 빼고 놓는 높이 − bowls.release_margin_mm 까지 허용. 공중 리허설은 calibrated 문을 면제.
    grasp_mode: "ik"(지금) | "act"(블록 위까지 IK → 집기 정책 → IK 나르기, grasp_act.py). 없으면 config grasp.mode(기본 ik).
    act 는 손목 확인 실행(실제)에서만 — dry-run 계획·시간은 IK 집기와 같다(정책 구간 시간은 grasp_act.block_time_estimate).
    act_runner·keeper_factory: 시험용(가짜 정책 실행·가짜 유지 구독자).
    which(10-23): 위치 "left"·"right"·"middle"(위 카메라 박스 가운데 u)·"top"·"bottom"(v — top = 로봇에서 먼 쪽)·
    "near"·"far"(로봇 베이스까지 작업대 거리), 관계 {"next_to": "red"|"blue"|"red_defect"|"blue_defect"}(그 색 블록에 가장
    가까운 것) — 같은 색 후보를 그 순서로 max_blocks 개(없으면 1개) 집는다. 안 고른 같은 색은 이웃으로만 본다. 모르는 값·
    옆에 둘 블록이 없으면 거부. 결과 r["which"] = 기준(rule)·chosen_px(·ref_px). 사용자 시점과 반대면 config which_flip: true
    (left·right 만, plan.select_which 한 곳).
    """
    colors = tuple(c for c in color.split(",")) if isinstance(color, str) else tuple(color)
    classes = tuple(c for k in colors for c in SORT_CLASSES[k])
    tag = "sort_" + "_".join(colors) + (f"_lift{lift_mm:g}" if lift_mm else "")
    cfg, sf, speed, log, _ = _start(config, speed_deg_s, tag, dry_run, log_dir)
    out: dict = {"ok": False, "mode": "dry_run", "executed": False, "reason": "planned", "colors": list(colors),
                 "lift_mm": lift_mm, "speed_deg_s": speed, "log_path": str(log.path), "error": None}
    try:
        log("request", colors=colors, dry_run=dry_run, max_blocks=max_blocks, lift_mm=lift_mm, speed=speed,
            ball_robot=os.environ.get("BALL_ROBOT"))
        s = Setup.load(cfg, use_studio=True)
        if detection is None and not demo:
            detection = detect_live(s)
        elif detection is not None and front_image is not None:  # 검출 파일 + 그때 원본 사진 → 블록 방향
            import cv2

            from robot.coord_pick import detect_hsv
            img = cv2.imread(str(front_image)) if not hasattr(front_image, "shape") else front_image
            detection = {**detection, "objects": detect_hsv.measure_yaw(img, detection.get("objects", []), s)}
        blocks, src = _detect(detection, True if demo else None, s, lambda s_, _n: demo_sort_scene(s_))
        out["block_yaws"] = [{"cls": b["cls"], "yaw_w_deg": round(math.degrees(b["yaw_w"]), 1), "aspect": b.get("aspect")}
                             for b in blocks if b.get("yaw_w") is not None]
        if which:
            if which_error(which):  # 모르는 값이면 엉뚱한 블록을 고르지 않게 거부
                out.update(ok=False, reason="refused", error=which_error(which))
                return out
            n_w = 1 if max_blocks is None else max_blocks
            flip = bool(cfg.get("which_flip", False))
            blocks, chosen, winfo = select_which(
                blocks, classes, which, n_w, flip=flip,
                dist_fn=lambda b: float(np.linalg.norm(s.frames.w2r(s.px_to_w(b["px"])[0]))),
                w_fn=lambda b: s.px_to_w(b["px"])[0])
            out["which"] = {"which": which, "flip": flip, "n": n_w, **winfo, "chosen_px": [b["px"] for b in chosen],
                            "chosen_cls": [b["cls"] for b in chosen]}
            log("which", **out["which"])
            if not chosen:
                ref = f"(옆에 둘 {which['next_to']} 블록이 투입 구역에 없음)" if isinstance(which, dict) else ""
                out.update(ok=False, reason="refused",
                           error=f"which={which}: 투입 구역에 고를 {'·'.join(colors)} 블록이 없음{ref}")
                return out
            max_blocks = len(chosen)
        p = plan_sort(s, blocks, classes=classes, max_blocks=max_blocks, lift_mm=lift_mm)
        if len(p["order"]) < p["needed"]:
            p["failures"].append(f"옮길 블록 {p['needed']}개 중 {len(p['order'])}개만 계획됨")
        bw = cfg["bowls"]
        # 목표 그릇 위는 놓는 높이 − 여유까지 내려가도 된다(집게 끝 기준). 다른 그릇·손 자리는 forbid_below_mm 그대로.
        floor = p["heights"]["release"] - bw.get("release_margin_mm", 20)
        allow = {o["bowl"]: floor for o in p["order"]}
        out.update({"blocks": len(p["order"]), "source": src, "heights_mm": p["heights"],
                    "plan": [{"order": i + 1, "color": COLOR_KO[o["block"]["cls"]],
                              "from_mm": np.round(o["block"]["w"], 1).tolist(), "to": BOWL_KO.get(o["bowl"], o["bowl"]),
                              "to_mm": np.round(o["drop_w"], 1).tolist(), "pick_tilt_deg": round(math.degrees(o["pick_tilt"])),
                              "drop_tilt_deg": round(math.degrees(o["drop_tilt"])), "drop_shift_mm": o["drop_shift_mm"]}
                             for i, o in enumerate(p["order"])]})
        use_wrist = (cfg.get("wrist", {}).get("enabled", False) if wrist_check is None else wrist_check) and not lift_mm
        out["wrist_check"] = use_wrist
        mode = grasp_mode or cfg.get("grasp", {}).get("mode", "ik")
        out["grasp_mode"] = mode
        if mode == "act" and not use_wrist and not dry_run:
            out.update(ok=False, reason="refused", error="grasp.mode act 는 손목 확인 실행(wrist.enabled)에서만 된다")
            return out
        if use_wrist and not dry_run:
            # 전체 계획이 실행 가능한지(문·공간·모의) 먼저 dry-run 으로 보고, 되면 손목 확인 단계 실행
            _execute(p, s, cfg, sf, speed, log, out, dry_run=True, name=tag, allow=allow)
            if not out["ok"]:
                return out
            gates = [] if os.environ.get("BALL_ROBOT") == "1" else ["BALL_ROBOT=1 이 아님"]
            if not cfg.get("calibrated"):
                gates.append("config.json calibrated=false")
            if gates:
                out.update(ok=False, reason="refused", error="실제로 보내지 않음: " + " · ".join(gates))
                return out
            out["mode"] = "real"
            if redetect is None:
                redetect = lambda: detect_live(s)  # noqa: E731
            _sort_wrist(s, cfg, sf, speed, log, [dict(b) for b in blocks], classes, max_blocks, out, redetect,
                        grasp_mode=mode, act_runner=act_runner, keeper_factory=keeper_factory)
        else:
            _execute(p, s, cfg, sf, speed, log, out, dry_run=dry_run, name=tag, allow=allow,
                     rehearsal=lift_mm >= sf["rehearsal_lift_min_mm"])
        m, sec = divmod(int(round(out["time_s"])), 60)
        out["summary"] = (f"그릇 옮기기({'·'.join(colors)}{f', 공중 +{lift_mm:g}mm' if lift_mm else ''}): 블록 "
                          f"{len(p['order'])}개, 약 {m}분 {sec}초(관절 {speed:g}°/s). 실패 {len(p['failures'])} · "
                          f"공간 위반 {len(out['violations'])} · 경고 {len(out['warnings'])}")
        return out
    except Exception as e:  # noqa: BLE001 — 도구는 예외 대신 결과로
        out.update(reason="error", error=f"{type(e).__name__}: {e}")
        return out
    finally:
        log("result", **{k: v for k, v in out.items() if k not in ("plan",)})
        log.close()


def unsort_blocks(dry_run: bool = True, colors=None, max_blocks: int | None = None, *, detection: dict | None = None,
                  speed_deg_s: float | None = None, demo: bool | None = None, config: dict | None = None,
                  log_dir: Path | None = None, wrist_check: bool | None = None, redetect=None, front_image=None) -> dict:
    """그릇 → 투입 구역 되돌리기(리허설·시연 사이 초기화, 10-23). sort_blocks 와 같은 꼴의 dict + skipped·free_cells.

    colors: None = 모든 블록(red·blue·defect), 또는 sort_blocks 와 같은 이름("blue"·"blue_defect"·"red,defect" …).
    집기 = 위 카메라에서 그릇(bowl_red·bowl_blue·bowl_defect) 안에 가운데가 있는 블록(plan.select_unsort):
      겹치거나 붙은 것(서로 unsort.overlap_mm 안·검출 박스가 다른 블록의 1.8배 넘음)·그릇 가장자리에 걸친 것(edge_mm 안)은
      건너뛰고 skipped 에 이유를 남긴다. 집는 높이 = 지금 집기 높이 + unsort.floor_mm(종이 접시 바닥). 집는 그 그릇은
      그 블록 집기·떠나기 구간에서만 금지 구역 예외(바닥까지). 먼 그릇에서 기울여야 해 집게 끝이 블록 옆까지 못 내려가면
      (바닥 위 unsort.max_grasp_mm 넘음) 안 꺼낸다. 손목 확인·'집은 뒤' 판정은 그릇 옮기기와 같다(집기는 IK 만).
    놓기 = 투입 구역 3×3 칸 가운데 중 빈 칸(위 카메라 검출 블록과 unsort.cell_gap_mm 이상) — 로봇에 가까운 칸부터,
      집게축 unsort.place_yaw_deg(90 = 블록 긴 축 0°, 테이프 가로와 나란히), 높이 heights_mm.place(기울면 집게 끝이 탁자에
      안 닿게 올림). 칸이 모자라면 남은 블록은 두고 끝(warnings). 끝나면 홈. 거부·실패 때는 그릇 옮기기와 같은 규칙
      (보내기 전 거부면 수직으로 올라가 홈, 쥔 채면 집게를 열지 않음).
    """
    colors = (("red", "blue", "defect") if colors is None else
              tuple(c for c in colors.split(",")) if isinstance(colors, str) else tuple(colors))
    classes = tuple(dict.fromkeys(c for k in colors for c in SORT_CLASSES[k]))
    tag = "unsort_" + "_".join(colors)
    cfg, sf, speed, log, _ = _start(config, speed_deg_s, tag, dry_run, log_dir)
    out: dict = {"ok": False, "mode": "dry_run", "executed": False, "reason": "planned", "colors": list(colors),
                 "speed_deg_s": speed, "log_path": str(log.path), "error": None, "skipped": [], "free_cells": []}
    try:
        log("request", kind="unsort", colors=colors, dry_run=dry_run, max_blocks=max_blocks, speed=speed,
            ball_robot=os.environ.get("BALL_ROBOT"))
        s = Setup.load(cfg, use_studio=True)
        if detection is None and not demo:
            detection = detect_live(s)
        elif detection is not None and front_image is not None:
            import cv2

            from robot.coord_pick import detect_hsv
            img = cv2.imread(str(front_image)) if not hasattr(front_image, "shape") else front_image
            detection = {**detection, "objects": detect_hsv.measure_yaw(img, detection.get("objects", []), s)}
        blocks, src = _detect(detection, True if demo else None, s, lambda s_, _n: demo_unsort_scene(s_))
        boxes = {}  # 검출 박스 넓이(px²) — 붙은 덩어리 판정
        for d in (detection or {}).get("detections", []):
            if d.get("box"):
                x1, y1, x2, y2 = d["box"]
                boxes[(round((x1 + x2) / 2), round((y1 + y2) / 2))] = (x2 - x1) * (y2 - y1)
        for b in blocks:
            if (round(b["px"][0]), round(b["px"][1])) in boxes:
                b["box_area"] = boxes[(round(b["px"][0]), round(b["px"][1]))]
        zw = zones_w(s)
        u = {**UNSORT_DEFAULTS, **cfg.get("unsort", {})}
        blocks = select_unsort(s, blocks, classes, zw)
        cells = free_cells(s, blocks, u["cell_gap_mm"], zw)
        out["skipped"] = [{"block": COLOR_KO[b["cls"]], "bowl": BOWL_KO.get(b["zone"], b["zone"]),
                           "from_mm": np.round(b["w"], 1).tolist(), "why": b["skip"]}
                          for b in blocks if b.get("skip") and b["cls"] in classes]
        out["free_cells"] = [n for n, _ in cells]
        p = plan_sort(s, blocks, classes=classes, max_blocks=max_blocks, unsort=True, cells=cells)
        out["skipped"] += [{"block": COLOR_KO[x["block"]["cls"]], "bowl": BOWL_KO.get(x["block"]["zone"], x["block"]["zone"]),
                            "from_mm": np.round(x["block"]["w"], 1).tolist(), "why": x["why"]} for x in p["unreachable"]]
        allow = {o["src"]: 0.0 for o in p["order"]}   # 집는 그릇은 바닥까지(그 블록 구간 — 전체 계획 검사에서는 계획 전체)
        out.update({"blocks": len(p["order"]), "source": src, "heights_mm": p["heights"],
                    "plan": [{"order": i + 1, "color": COLOR_KO[o["block"]["cls"]], "from": BOWL_KO.get(o["src"], o["src"]),
                              "from_mm": np.round(o["block"]["w"], 1).tolist(), "to": f"칸 {o['cell']}",
                              "to_mm": np.round(o["drop_w"], 1).tolist(), "grasp_z": round(float(o["grasp_z"]), 1),
                              "place_z": round(float(o["place_z"]), 1), "pick_tilt_deg": round(math.degrees(o["pick_tilt"])),
                              "place_tilt_deg": round(math.degrees(o["drop_tilt"]))} for i, o in enumerate(p["order"])]})
        if not p["order"]:
            why = "; ".join([x["why"] for x in out["skipped"]] + p["failures"] + p["warnings"][:2]) or "그릇에 블록이 없음"
            out.update(ok=not out["skipped"] and not p["failures"], reason="nothing", time_s=0.0,
                       summary=f"되돌릴 블록 없음 — {why}", warnings=p["warnings"], failures=p["failures"], violations=[])
            return out
        use_wrist = cfg.get("wrist", {}).get("enabled", False) if wrist_check is None else wrist_check
        out["wrist_check"] = use_wrist
        if use_wrist and not dry_run:
            _execute(p, s, cfg, sf, speed, log, out, dry_run=True, name=tag, allow=allow)
            if not out["ok"]:
                return out
            gates = [] if os.environ.get("BALL_ROBOT") == "1" else ["BALL_ROBOT=1 이 아님"]
            if not cfg.get("calibrated"):
                gates.append("config.json calibrated=false")
            if gates:
                out.update(ok=False, reason="refused", error="실제로 보내지 않음: " + " · ".join(gates))
                return out
            out["mode"] = "real"
            if redetect is None:
                redetect = lambda: detect_live(s)  # noqa: E731
            _sort_wrist(s, cfg, sf, speed, log, [dict(b) for b in blocks], classes, max_blocks, out, redetect,
                        unsort=True, cells=list(cells))
        else:
            _execute(p, s, cfg, sf, speed, log, out, dry_run=dry_run, name=tag, allow=allow)
        m, sec = divmod(int(round(out.get("time_s") or 0)), 60)
        out["summary"] = (f"되돌리기({'·'.join(colors)}): 블록 {len(p['order'])}개 → 칸 "
                          f"{', '.join(str(o['cell']) for o in p['order'])}, 약 {m}분 {sec}초(관절 {speed:g}°/s). "
                          f"못 꺼냄 {len(out['skipped'])} · 실패 {len(p['failures'])} · 공간 위반 {len(out.get('violations') or [])}")
        return out
    except Exception as e:  # noqa: BLE001 — 도구는 예외 대신 결과로
        out.update(reason="error", error=f"{type(e).__name__}: {e}")
        return out
    finally:
        log("result", **{k: v for k, v in out.items() if k not in ("plan",)})
        log.close()


WRIST_DEFAULTS = {"hover_mm": 45, "align_iters": 2, "align_tol_mm": 3, "align_step_max_mm": 25, "align_gain": 0.6,
                  "near_mm": 10, "near_iters": 5, "near_step_max_mm": 8, "near_gain": 0.7, "near_tol_frac": 0.5,
                  "shots": 2,
                  "yaw_aspect_min": 1.2, "yaw_ok_deg": 62,
                  "deeper_mm": 4, "deeper_max": 2, "realign_max": 1, "retries": 2, "held_margin_n": 0.8,
                  "keep_roll_deg": 8,
                  "after_area_scale": 1.0,  # '집은 뒤' 면적 하한 = 녹화 하위 10% × 이 값(고정 노출 등으로 덩어리가 작아지면 낮춘다)
                  "close_area_scale": 1.0,
                  "retry_put_down": False,
                  "reuse_dir_photo": False,  # 방향 사진을 hover 맞춤 첫 장으로(같은 자세일 때만, IK 만)  # 못 쥠으로 다시 할 때 쥔 채면 제자리에 내려놓고 벌림(기본: 그 높이에서 벌림)
                  "after_tol_min_px": 0.0,  # '집은 뒤' 가운데 허용 하한(px) — 녹화가 좁게 몰린 축 보완(10-23 13:27: 분홍 v ±14)
                  "ref": None}  # 손목 기준 파일(coord_pick 안 이름, None = wrist_ref.json). 고정 노출 = wrist_ref_fixed.json  # '닫기 직전' 면적 하한도 같은 식(10-23 손목 고정 노출에서 초록이 7.2~8.3천 < 하한 17.6천)
REF_DS = {"red_ok": "pick_red", "blue_ok": "pick_green", "red_defect": "pick_defect", "blue_defect": "pick_defect"}


def _segment(s: Setup, cfg: dict, sf: dict, speed: float, log: StageLog, path: _Path, name: str, *,
             allow: dict | None = None, rehearsal: bool = False, floor_from_start: bool = False) -> dict:
    """구간 하나(경로 path)를 실행기 문·검사 그대로 실제 송신. 결과 dict(_execute 와 같은 키)."""
    checks, total = path.checks()
    p = {"steps": path.steps, "failures": path.failures, "warnings": [], "checks": checks, "total_s": total}
    seg = {"ok": False, "mode": "dry_run", "executed": False, "reason": "planned", "error": None, "plan": None}
    log("segment", name=name, steps=len(path.steps))
    return _execute(p, s, cfg, sf, speed, log, seg, dry_run=False, name=name, allow=allow, rehearsal=rehearsal,
                    floor_from_start=floor_from_start)


def _jac(sm: "wrist.ScreenModel", dz: float) -> np.ndarray:
    return np.array([[sm.cu[1] + sm.cu[4] * dz, sm.cu[2] + sm.cu[5] * dz],
                     [sm.cv[1] + sm.cv[4] * dz, sm.cv[2] + sm.cv[5] * dz]])


def _wrist_look(s: Setup, q: np.ndarray, color: str, sm, log: StageLog, tag: str, near=None,
                center_off=(0.0, 0.0), shots: int = 1, want=None, merge: bool = False, pale: bool = False) -> dict | None:
    """손목 카메라 한 장 → 목표 색 덩어리와, 지금 자세에서 '가운데'면 보일 자리(화면 모형 + center_off)와의 차이·보정량(mm).
    center_off = 녹화 '닫기 직전' 중심 − 화면 모형의 가운데(dz 0) — 맞춤 목표를 닫기 판정 기준과 같은 점으로.
    merge = 불량 블록(검은 X 조각을 한 덩어리로, wrist.blob)."""
    T = s.kin.fk(q)
    tcp = (T[:3, 3] + T[:3, :3] @ s.tcp) * 1000
    dz = float((wrist.BLOCK_Z_MM - tcp[2]) * T[2, 2])  # tcp 바로 아래 블록 가운데(높이 15)까지 끝 z 방향 거리
    want = (np.asarray(want(dz), float) if want is not None
            else sm.predict([[0.0, 0.0, dz]])[0] + np.asarray(center_off, float))
    blobs = [wrist.blob(wrist.capture(), color, near=near if near is not None else want, merge=merge, pale=pale)
             for _ in range(max(1, shots))]
    b = blobs[0]
    spread = None
    if shots > 1:  # 여러 장 평균(한 장이라도 없으면 없음)
        if any(x is None for x in blobs):
            b = None
        else:
            P = np.array([x["px"] for x in blobs])
            spread = round(float(np.ptp(P, axis=0).max()), 1)
            b = {**blobs[-1], "px": np.round(P.mean(0), 1).tolist(), "area": int(np.mean([x["area"] for x in blobs]))}
    rec = {"tag": tag, "dz": round(dz, 1), "want_px": np.round(want, 1).tolist(), "blob": b, "spread_px": spread}
    if b is not None:
        e = np.array(b["px"]) - want
        d = np.linalg.solve(_jac(sm, dz), e)  # 끝 좌표 (dx, dy) mm — 블록이 tcp 에서 이만큼
        rec.update(err_px=np.round(e, 1).tolist(), d_mm=np.round(d, 1).tolist(),
                   dw=np.round(s.frames.R @ (T[:2, :2] @ d), 1).tolist())
    log("wrist", **rec)
    return rec


def _grasp_z(s: Setup, cfg: dict, tw, yaw_r: float, tilt: float, q: np.ndarray, z_add: float = 0.0):
    """hover 와 같은 집게 방향·기울기·롤 갈래(시작값 = 지금 자세만)로 블록 tw 위 집기 높이: heights_mm.grasp 에서 시작해
    집게 끝(벌림 → 닫힘 사이 모든 값)이 grasp_finger 보다 낮으면 올린다. z_add = 블록 바닥 높이(되돌리기의 접시 바닥).
    (높이, 자세) 또는 (None, None)."""
    hz, g = cfg["heights_mm"], cfg["gripper"]
    target, z = hz.get("grasp_finger", 0) + z_add, float(hz["grasp"]) + z_add
    for _ in range(4):
        qg, info = s.solve(s.w_to_r3(tw, z), yaw_r, q, strict=True, tilt=tilt, only_prev=True, finger_check=False)
        if qg is None or abs(float(qg[4] - q[4])) > math.radians(WRIST_DEFAULTS["keep_roll_deg"]):
            return None, None
        fl = min(s.finger_low(qg, v) for v in np.linspace(g["close"], g["open"], 7))
        if fl >= target - 0.5:
            return (round(z, 1), qg) if fl >= hz.get("finger_min", 0) else (None, None)
        z += target - fl
    return None, None


ARRIVE_STEP_MM = 20  # 블록 위 approach_safe 높이 자세가 안 풀리면 이만큼씩 낮춰 본다(90 → 70 → hover)


def _path_mark(path: _Path) -> tuple:
    return path.q.copy(), path.yaw, path.tilt, path.grip, len(path.steps), len(path.failures), len(path.flips)


def _path_reset(path: _Path, m: tuple) -> None:
    """_path_mark 뒤에 쌓은 걸음·실패를 지운다(길을 해 보고 안 되면 되돌리기)."""
    path.q, path.yaw, path.tilt, path.grip = m[0].copy(), m[1], m[2], m[3]
    del path.steps[m[4]:], path.failures[m[5]:], path.flips[m[6]:]


def _arrive_above(s: Setup, path: _Path, tw, py: float, pt: float, zh: float, qh: np.ndarray, z_want: float,
                  tmax: float, tag: str, go) -> dict:
    """블록 위 z_want(grasp.approach_safe_mm)로 와서 수직으로 hover(zh, 기울기 pt)까지 — 옆에서 쓸고 들어오지 않게.
    10-23 09:46 실물: 먼 칸(작업대 x 72)은 90mm 에서 hover 기울기 15° 가 안 풀려(손목 한계) 거부됐다.
    높이마다 ① hover 기울기 그대로 ② tilt_max 안에서 그 기울기에 가까운 것으로 자세를 풀고, go(path, z, 기울기, q) 로 그 높이까지
    온 뒤 hover 까지 수직으로 내려가는 길이 되면 쓴다(기울기가 다르면 내려가며 hover 기울기로 바꾼다). 안 되면 ARRIVE_STEP_MM 씩
    낮추고, hover 높이까지 내려오면 used_mm = None(부르는 쪽이 hover 로 바로 간다). 해 본 길은 path 에서 되돌린다.
    반환 {want_mm, used_mm, tilt_deg, plan_tilt_deg, tried}."""
    rec = {"want_mm": round(float(z_want), 1), "used_mm": None, "plan_tilt_deg": round(math.degrees(pt), 1), "tried": []}
    z = float(z_want)
    while z > zh + 0.5:
        for free in (False, True):
            qa, ia = s.solve(s.w_to_r3(tw, z), py, qh, strict=True, tilt=None if free else pt,
                             tilt_max=tmax if free else None, near=pt if free else None, only_prev=True)
            if qa is None:
                rec["tried"].append({"z": round(z, 1), "tilt": f"≤{math.degrees(tmax):.0f}" if free else rec["plan_tilt_deg"],
                                     "why": ia.get("why")})
                continue
            ta = float(ia["tilt_used"])
            same = abs(ta - pt) < 1e-6
            if free and same:  # hover 기울기 그대로는 ① 에서 이미 해 봤다
                continue
            m = _path_mark(path)
            ok = (go(path, z, ta, qa) and len(path.failures) == m[5]
                  and path.line(s.w_to_r3(tw, z), s.w_to_r3(tw, zh), py, pt, f"{tag} hover 로(수직)", strict=same,
                                tilt_max=None if same else tmax, q_goal=None if same else qh))
            if ok:
                rec.update(used_mm=round(z, 1), tilt_deg=round(math.degrees(ta), 1))
                return rec
            rec["tried"].append({"z": round(z, 1), "tilt": round(math.degrees(ta), 1),
                                 "why": "; ".join(path.failures[m[5]:])[:160] or "길 안 됨"})
            _path_reset(path, m)
        z -= ARRIVE_STEP_MM
    return rec


ZONE_KO = {**BOWL_KO, "hand_spot": "손 자리"}


def _path_hits(s: Setup, path: _Path, zones: dict, sf: dict) -> set:
    """경로(걸음 사이 관절 보간 4점씩 — 제자리 돌림도 쓸고 지나가는 자리를 본다)에서 집게 끝·손목이 금지 구역(zones) 위
    forbid_below_mm 아래인 구역 이름들(plan.pose_hits = 실행기 작업 공간 검사와 같은 기준)."""
    qs = [s.n_to_q(np.asarray(st.n, float)) for st in path.steps]
    out: set = set()
    for a, b in zip(qs, qs[1:]):
        for u in (0.25, 0.5, 0.75, 1.0):
            out |= set(pose_hits(s, a + (b - a) * u, (path.grip,), zones, sf["forbid_below_mm"], sf["forbid_margin_mm"]))
    return out


def _recover_home(s: Setup, cfg: dict, sf: dict, speed: float, log: StageLog, q: np.ndarray, grip: float,
                  zones: dict, z_high: float) -> dict:
    """손목 확인 실행이 보내기 전 거부·계획 실패로 멈췄을 때: 지금 자세(마지막 구간 끝)에서 수직으로 올라간 뒤 홈으로.
    10-23 09:50 실물: 초록 그릇 위에 남은 팔을 go_home 이 too_far(어깨 81°)로 거부해 사람이 max_deg 150 으로 돌렸다.
    올라갈 높이 = z_high(over_zone) → 지금 +60 → +30 중 처음 되는 것(기울기는 지금에 가까운 것). 지금 자세가 걸친 금지 구역
    (방금 놓은 그릇 등)은 지금 집게 끝 높이까지 허용한다. 실행기 검사는 그대로(안 되면 거부하고 그 자리)."""
    T = s.kin.fk(q)
    yaw_c = math.atan2(T[1, 0], T[0, 0])
    tilt_c = math.acos(max(-1.0, min(1.0, -float(T[2, 2]))))
    tmax = max(s.tilt["grasp"], s.tilt["release"])
    cur = s.tcp_pos(q)
    # 집게가 닫힌 값(close 근처 이하)이면 블록을 쥐고 있을 수 있다 → 홈에서도 집게를 열지 않는다(떨어뜨리지 않게, 10-23 11:37)
    holding = float(grip) <= float(cfg["gripper"]["close"]) + 1.5
    rec = {"ok": False, "from_mm": np.round(cur, 1).tolist(), "lift_to_mm": None, "why": "올라갈 길 없음",
           "grip": round(float(grip), 1), "holding": holding}
    path = None
    for zl in sorted({max(float(cur[2]), z) for z in (z_high, cur[2] + 60, cur[2] + 30)}, reverse=True):
        p = _Path(s, q, grip, yaw=yaw_c, tilt=tilt_c)
        if zl > cur[2] + 0.5:
            top = np.r_[cur[:2], zl]
            qz, iz = s.solve(top, yaw_c, q, strict=True, tilt=None, tilt_max=tmax, near=tilt_c, grip_n=grip,
                             jaw_limit=False)
            if qz is None or not p.line(cur, top, yaw_c, float(iz["tilt_used"]), "멈춘 자리에서 올라감", strict=False,
                                        tilt_max=tmax):
                rec["why"] = (p.failures or [iz.get("why", "IK 안 풀림") if qz is None else "올라갈 길 없음"])[-1]
                continue
        p.finish(zones, sf["forbid_below_mm"], sf["forbid_margin_mm"], z_high, keep_grip=holding)
        if p.failures:
            rec["why"] = p.failures[-1]
            continue
        path, rec["lift_to_mm"] = p, round(zl, 1)
        break
    if path is not None:
        fl = s.finger_low(q, grip)
        allow = {h: max(0.0, round(fl - 2, 1)) for h in pose_hits(s, q, (grip,), zones, sf["forbid_below_mm"],
                                                                  sf["forbid_margin_mm"])}
        r = _segment(s, cfg, sf, speed, log, path, "거부 뒤 올라가 홈으로", allow=allow or None, floor_from_start=True)
        rec.update(ok=r["reason"] == "done", reason=r["reason"], why=r.get("error"), allow=allow, time_s=r.get("time_s"))
    log("recover", **rec)
    return rec


def _sort_wrist(s: Setup, cfg: dict, sf: dict, speed: float, log: StageLog, blocks: list[dict], classes: tuple,
                max_blocks: int | None, out: dict, redetect, *, grasp_mode: str = "ik", act_runner=None,
                keeper_factory=None, unsort: bool = False, cells: list | None = None) -> dict:
    """손목 카메라 확인·보정 실행(r18·r19·r23): 블록마다 hover → 방향 확인·돌림 → hover 에서 대강 맞춤(align_iters) →
    닫기 직전 높이 + near_mm 로 내려가 녹화 '닫기 직전' 중심에 직접 맞춤(2장 평균·한 번 8mm 까지·near_iters) → 내려감 →
    닫기 직전 사진이 녹화 기준(사람이 닫기 직전) 안이고 충분히 크면 닫음(작으면 deeper_mm 씩 더, 벗어나면 다시 맞춤) →
    들어 올려 손목 사진(녹화 "집은 뒤" 기준, 없으면 그리퍼 측정값)으로 쥐었나 확인 → 놓기. 못 쥐었으면 놓고 홈 → 위 카메라로 다시 찾아 retries 번까지.

    grasp_mode "act"(10-23 하이브리드): 블록마다 첫 시도는 시작 높이(grasp.start_z_mm 90)에서 방향 맞춤·대강 맞춤까지 IK →
    유지 구독자(공중 넘겨주기 토크 유지) → 집기 정책(복귀 없음, 닫고·들고·멈춤에서 끝) → 끝 감지 + '집은 뒤' 손목 사진으로 쥐었나 →
    쥐었으면 그 자리부터 IK 로 나르기. 못 쥐었으면 벌리고 hover 로 올라가 그 시도 안에서 IK 집기(grasp.fallback_ik).

    unsort=True(되돌리기, 10-23): 그릇 안 블록(select_unsort 의 skip 은 뺌)을 같은 손목 확인으로 집어 cells(빈 칸) 차례로
    놓는다 — 집는 높이 + unsort.floor_mm, 집는 그 그릇은 그 블록의 구간에서만 금지 구역 예외(바닥까지), 놓을 때 집게축
    unsort.place_yaw_deg·높이 heights_mm.place(기울면 집게 끝이 탁자에 안 닿게 올림). 집기는 IK 만."""
    W = {**WRIST_DEFAULTS, **cfg.get("wrist", {})}
    G = {**grasp_act.GRASP_DEFAULTS, **cfg.get("grasp", {})}
    keepers: list = []
    ref = wrist.load_ref(HERE / W["ref"]) if W.get("ref") else wrist.load_ref()
    if ref is None:
        raise RuntimeError("robot/coord_pick/wrist_ref.json 없음 — python -m robot.coord_pick.wrist fit")
    sm = wrist.screen_model(ref)
    g, hz, bw = cfg["gripper"], cfg["heights_mm"], cfg["bowls"]
    rel_floor = bw["rim_mm"] + bw["release_above_mm"] - bw.get("release_margin_mm", 20)  # 목표 그릇 위 집게 끝 하한(놓기와 같음)
    zw = zones_w(s)
    zones_forbid = {k: v for k, v in zw.items() if k in ("bowl_red", "bowl_blue", "bowl_defect", "hand_spot")}
    tmax = max(s.tilt["grasp"], s.tilt["release"])
    z_high = hz.get("over_zone", 130)
    for b in blocks:
        b["w"] = s.px_to_w(b["px"])[0]
    U = {**UNSORT_DEFAULTS, **cfg.get("unsort", {})}
    if unsort:
        grasp_mode = "ik"
        cells = list(cells or [])
        todo = [b for b in blocks if b["cls"] in classes and b.get("zone") in BOWL_ZONES and not b.get("skip")]
    else:
        todo = [b for b in blocks if b["cls"] in classes and b.get("zone", "input") == "input"]
    todo.sort(key=lambda b: (b.get("which_rank", 0), float(np.linalg.norm(s.frames.w2r(b["w"])))))  # which 면 그 순서
    if max_blocks is not None:
        todo = todo[:max_blocks]
    q, grip = s.n_to_q(np.array(cfg["home_n"])), float(cfg["home_n"][5])
    at_home, prev_bowl, results = True, None, []
    blk_allow: dict = {}   # 이 블록 구간에서만 허용하는 금지 구역(되돌리기 = 집는 그 그릇, 바닥까지)

    def run(path, name, allow=None, floor_from_start=False):
        nonlocal q, grip
        allow = {**blk_allow, **(allow or {})} or None
        r = _segment(s, cfg, sf, speed, log, path, name, allow=allow, floor_from_start=floor_from_start)
        if not r.get("executed") and r["reason"] != "done":
            zs = sorted({v["kind"].split(":", 1)[1] for v in r.get("violations") or [] if v["kind"].startswith("forbidden:")})
            if zs:  # 10-23 10:07: 손 자리 41mm 옆으로 떨어진 블록 — 그 자리에서 끝내고(자동으로 홈) 사람에게 알린다
                where = "·".join(ZONE_KO.get(z, z) for z in zs)
                raise _Abort(f"{name}: {where} 가까이라 보내지 않음 — 블록이 {where} 가까이 있으면 사람이 투입 구역 안쪽으로 "
                             f"옮겨 주세요", r)
            raise _Abort(f"{name}: {r['reason']} {r.get('error') or ''}".strip(), r)
        if r["reason"] != "done":
            raise _Abort(f"{name}: {r['reason']} {r.get('error') or ''}".strip(), r)
        q, grip = path.q, path.grip
        return r

    class _Abort(Exception):
        def __init__(self, msg, r):
            super().__init__(msg)
            self.r = r

    def _place_cell(s, cfg, sf, speed, log, b, tw, q, grip, py, pt, cells, last, tag, zones_forbid, z_high, allow, U, res):
        """되돌리기 놓기: 빈 칸(로봇에 가까운 칸부터) 차례로 — 나르는 높이로 올라가 칸 위로(금지 구역은 높이 올려 건너고,
        집은 그릇에서 떠나는 구간은 그 그릇 허용) → 집게축 place_yaw_deg 로 놓는 높이(기울면 집게 끝이 탁자에 안 닿게 올림) →
        벌림 → 올라감(마지막 블록이면 홈). 놓으면 그 칸을 cells 에서 뺀다. 반환 ((q, grip) 또는 None, 이유, '칸 N')."""
        hz_, g_ = cfg["heights_mm"], cfg["gripper"]
        tm = max(s.tilt["grasp"], s.tilt["release"])
        zc_ = hz_["carry"]
        yaw_pl = math.radians(U["place_yaw_deg"])
        qc, ic = s.solve(s.w_to_r3(tw, zc_), py, q, tilt=None, tilt_max=tm, near=pt, grip_n=g_["close"], jaw_limit=False)
        if qc is None:
            return None, f"나르는 높이 {zc_:g}mm 에 안 닿음({ic.get('why')})", None
        why = "빈 칸 없음(위 카메라 기준 다른 블록과 cell_gap_mm 안)"
        for cell in list(cells):
            num, cw = cell
            dcol = s.column(cw, yaw_pl, (zc_, hz_.get("place", 10)), qc, tilt_max=s.tilt["release"], grip_n=g_["close"],
                            yaws=(yaw_pl,), finger_target=hz_.get("grasp_finger", 0), close_n=g_["close"])
            if dcol is None:
                why = f"칸 {num} 에 안 닿음({s.last_why})"
                continue
            hits = sorted({h for qq in dcol["qs"] for h in pose_hits(s, qq, (g_["close"], g_["release"]), zones_forbid,
                                                                      sf["forbid_below_mm"], sf["forbid_margin_mm"])})
            if hits:  # 로봇 쪽 칸은 손가락이 손 자리 여유에 걸릴 수 있다 → 다음 칸
                why = f"칸 {num} 은 놓을 때 집게가 {'·'.join(ZONE_KO.get(h, h) for h in hits)} 여유에 걸림"
                continue
            dy, dt_, z_pl = dcol["yaw_w"] - s.frames.yaw, dcol["tilt"], dcol["zs"][-1]
            path = _Path(s, q, grip, yaw=py, tilt=pt)
            ok = (path.line(s.tcp_pos(q), s.w_to_r3(tw, zc_), ic["yaw_used"], ic["tilt_used"], f"{tag} 나르는 높이로",
                            strict=False, tilt_max=tm)
                  and travel(s, path, tw, zc_, cw, zc_, dy, dt_, f"{tag} → 칸 {num} 위", zones=zones_forbid,
                             skip={b.get("zone")}, z_high=z_high, margin=sf["forbid_margin_mm"] + UNSORT_TRAVEL_EXTRA_MM, tilt_max=tm,
                             q_goal=dcol["qs"][0], fast=True)
                  and path.line(s.w_to_r3(cw, zc_), s.w_to_r3(cw, z_pl), dy, dt_, f"{tag} 놓는 높이로", strict=True))
            if not ok:
                why = "; ".join(path.failures)[:200]
                continue
            path.gripper(g_["release"], f"{tag} 놓음", settle=True)
            path.line(s.w_to_r3(cw, z_pl), s.w_to_r3(cw, zc_), dy, dt_, f"{tag} 올라감", strict=False, tilt_max=tm, fast=True)
            if last:
                path.finish(zones_forbid, sf["forbid_below_mm"], sf["forbid_margin_mm"], z_high)
            if path.failures:
                why = "; ".join(path.failures)[:200]
                continue
            r = _segment(s, cfg, sf, speed, log, path, f"{tag} 칸 {num} 에 놓기", allow=dict(allow) or None)
            if r["reason"] == "refused" and not r.get("executed"):  # 공간 검사 등에서 걸림 → 다음 칸
                why = r.get("error") or "거부"
                log("place_retry", tag=tag, why=why, cell=num)
                continue
            if r["reason"] != "done":
                raise _Abort(f"{tag} 칸 {num} 에 놓기: {r['reason']} {r.get('error') or ''}".strip(), r)
            cells.remove(cell)
            res.update(cell=num, place_mm=np.round(cw, 1).tolist(), place_z=round(float(z_pl), 1))
            return (path.q, path.grip), "", f"칸 {num}"
        return None, why, None

    try:
        for bi, b in enumerate(todo):
            color = "red" if b["cls"].startswith("red") else "blue"
            # 불량은 X 조각을 묶어 보고, 초록은 기본 마스크에 안 잡히면 연한 초록(Lab)으로 한 번 더(높은 시작 자세 — 10-23)
            wlook = functools.partial(_wrist_look, merge=b["cls"].endswith("defect"), pale=color == "blue")
            rd = ref["close"]["by_dataset"][REF_DS[b["cls"]]]
            cref = {"close": {"u": {"med": rd["u_med"]}, "v": {"med": rd["v_med"]}, "tol_px": rd["tol_px"],
                              "area": {"p10": int(rd["area_p10"] * W["close_area_scale"])}}}
            center_off = np.array([rd["u_med"], rd["v_med"]]) - sm.predict([[0.0, 0.0, 0.0]])[0]
            tw = np.array(b["w"], float)
            others = [dict(o, cls="obstacle") for o in blocks if o is not b and (unsort or o.get("zone", "input") == "input")]
            blk_allow.clear()
            if unsort:
                blk_allow[b["zone"]] = 0.0
            # 홈으로 갈 때 '그 위를 낮게 지나면 먼저 올라감'을 볼 구역 — 되돌리기는 지금 집던 그릇을 뺀다(먼 그릇 위 130mm 는
            # 안 닿아 계획이 실패한다. 그 그릇 위에서 떠나는 것이라 높이 올릴 까닭도 없다)
            zf_home = {k: v for k, v in zones_forbid.items() if not (unsort and k == b.get("zone"))}
            gz_add = float(U["floor_mm"]) if unsort else 0.0
            res = {"block": COLOR_KO[b["cls"]], "from_mm": np.round(tw, 1).tolist(), "attempts": [], "ok": False,
                   "top_yaw_deg": None if b.get("yaw_w") is None else round(math.degrees(b["yaw_w"]), 1)}
            results.append(res)
            for attempt in range(W["retries"] + 1):
                tag = f"[{bi + 1}/{len(todo)}·{attempt + 1}] {COLOR_KO[b['cls']]}"
                att = {"start_mm": np.round(tw, 1).tolist(), "align": [], "close": [], "held": None}
                res["attempts"].append(att)
                last = bi == len(todo) - 1

                def choose(tw_):
                    px = s.w_to_px(tw_)[0].round(1).tolist()
                    p = plan_sort(s, [{"cls": b["cls"], "px": px, "zone": b["zone"] if unsort else "input",
                                       **{k: b[k] for k in ("yaw_w", "aspect") if b.get(k) is not None}}] + others,
                                  classes=(b["cls"],), max_blocks=1, unsort=unsort, cells=cells if unsort else None)
                    if not p["order"]:
                        raise _Abort(f"{tag}: 계획 안 됨 — {'; '.join(p['failures'])}", {})
                    return p["order"][0]

                o = choose(tw)
                py, pt = o["pick_yaw_w"] - s.frames.yaw, o["pick_tilt"]
                act_try = grasp_mode == "act" and not res.get("act_tried")
                zh_ik = max(o["approach_z"], W["hover_mm"])
                zh = max(zh_ik, float(G["start_z_mm"])) if act_try else zh_ik   # act = 정책 시작 자세 높이
                z_arrive = max(zh, float(G["approach_safe_mm"] or 0))            # 블록 위 이 높이로 와서 수직으로 hover 로
                # 1) hover
                path = _Path(s, q, grip)
                path.gripper(g["open"], f"{tag} 집게 벌림")
                ph = pt   # hover 높이(act 는 정책 시작 높이)의 집게 기울기 — IK 는 계획 기울기 그대로
                tw_block = tw.copy()   # 블록 자리(act 시작 자세는 녹화처럼 블록에서 비켜 선다 — 정책이 못 쥐면 여기로 돌아온다)
                if act_try and (G["start_back_mm"] or G["start_side_mm"]):
                    # 녹화 사람 시작 자세(10-23 grasp 39개): 집게 가운데가 블록보다 로봇 쪽 33mm·옆 −17mm(중앙), 높이 137, 기울기 22°
                    rb = s.frames.w2r(tw)
                    u_r = rb / max(float(np.linalg.norm(rb)), 1e-6)
                    tw = s.frames.r2w(rb - float(G["start_back_mm"]) * u_r + float(G["start_side_mm"]) * np.array([-u_r[1], u_r[0]]))
                t_pref = math.radians(G["start_tilt_deg"]) if act_try and G.get("start_tilt_deg") is not None else pt
                qh, ih = s.solve(s.w_to_r3(tw, zh), py, o["q_approach"], strict=True, tilt=t_pref, only_prev=True)
                if qh is None and act_try and zh > zh_ik:
                    # 5축 팔에 tcp 가 옆으로 떨어져 있어 높이가 바뀌면 맞출 수 있는 기울기가 3~8° 달라진다 — 시작 높이 90mm 에서는
                    # 계획 기울기(approach 50mm 기준)가 안 되는 칸이 대부분이다(10-23 표). 시작 자세는 grasp_max 안 가까운 기울기로
                    qh, ih = s.solve(s.w_to_r3(tw, zh), py, o["q_approach"], strict=True, tilt=None,
                                     tilt_max=s.tilt["grasp"], near=t_pref, only_prev=True)
                if qh is not None and act_try and zh > zh_ik:
                    ph = float(ih["tilt_used"])
                    att["act_start"] = {"tilt_deg": round(math.degrees(ph), 1), "plan_tilt_deg": round(math.degrees(pt), 1),
                                        "start_mm": np.round(tw, 1).tolist(), "block_mm": np.round(tw_block, 1).tolist(),
                                        "z_mm": zh}
                    log("act_start", tag=tag, **att["act_start"])
                if qh is None and act_try and zh > zh_ik:
                    # 집기 정책 시작 자세(start_z_mm)가 이 칸에서 안 풀림(먼 칸 — 손목 한계) → 이 블록은 IK 집기로
                    if not G["fallback_ik"]:
                        raise _Abort(f"{tag}: 집기 정책 시작 자세({zh:g}mm) IK 실패({ih.get('why')}) — grasp.fallback_ik 꺼짐", {})
                    att["act"] = {"skipped": f"시작 자세 {zh:g}mm·기울기 {math.degrees(pt):.0f}° 안 풀림({ih.get('why')}) "
                                             f"— 이 블록은 IK 집기"}
                    log("act_skip", tag=tag, **att["act"])
                    res["act_tried"] = True
                    act_try, zh, tw = False, zh_ik, tw_block
                    z_arrive = max(zh, float(G["approach_safe_mm"] or 0))
                    qh, ih = s.solve(s.w_to_r3(tw, zh), py, o["q_approach"], strict=True, tilt=pt, only_prev=True)
                if qh is None:
                    raise _Abort(f"{tag}: hover 자세 IK 실패({ih.get('why')})", {})
                arr = None
                if at_home:
                    if path.joint_hits(qh, zones_forbid, sf["forbid_below_mm"], sf["forbid_margin_mm"]):
                        qhh, ihh = s.solve(s.w_to_r3(tw, z_high), py, qh, strict=True, tilt=None, tilt_max=tmax, near=pt,
                                           only_prev=True)
                        if qhh is None:
                            raise _Abort(f"{tag}: 블록 위 {z_high:g}mm 자세 IK 실패({ihh.get('why')})", {})
                        path.joint_move(qhh, f"{tag} 홈 → 블록 위(높이 {z_high:g})", yaw=py, tilt=ihh["tilt_used"],
                                        fast=True)
                        path.line(s.w_to_r3(tw, z_high), s.w_to_r3(tw, zh), py, ph, f"{tag} hover 로", strict=False,
                                  tilt_max=tmax, q_goal=qh)
                    else:
                        if z_arrive > zh:
                            def leg(p, z, ta, qa):
                                p.joint_move(qa, f"{tag} 홈 → 블록 위({z:g})", yaw=py, tilt=ta, fast=True)
                                return True
                            arr = _arrive_above(s, path, tw, py, ph, zh, qh, z_arrive, tmax, tag, leg)
                        if arr is None or arr["used_mm"] is None:
                            path.joint_move(qh, f"{tag} 홈 → hover", yaw=py, tilt=ph, fast=True)
                else:
                    cur = s.tcp_pos(q)

                    def leg(p, z, ta, qa):
                        return travel(s, p, s.frames.r2w(cur[:2]), cur[2], tw, z, py, ta, f"{tag} → hover",
                                      zones=zones_forbid, skip={prev_bowl}, z_high=z_high, margin=sf["forbid_margin_mm"],
                                      tilt_max=tmax, q_goal=qa, fast=True)
                    if z_arrive > zh:
                        arr = _arrive_above(s, path, tw, py, ph, zh, qh, z_arrive, tmax, tag, leg)
                    if arr is None or arr["used_mm"] is None:
                        leg(path, zh, ph, qh)
                if arr is not None:  # approach_safe 높이·기울기를 바꿨거나 못 썼으면 로그에
                    att["arrive"] = arr
                    if arr["tried"] or arr["used_mm"] != arr["want_mm"]:
                        log("arrive", tag=tag, **arr)
                # 방금 놓은 그릇 위에서 떠나는 구간: 그 그릇은 놓을 때와 같은 높이까지 허용(10-23 09:50 실물 — 놓은 자리
                # 집게 끝 54mm 가 금지 구역 60mm 아래라 다음 블록 hover 가 첫 틱부터 거부됐다)
                run(path, f"{tag} hover", allow=None if at_home or prev_bowl is None else {prev_bowl: rel_floor})
                at_home = False
                # 1b) hover 손목 사진의 블록 긴 축 → 집게축과 |각| 이 사람 범위(62° 이상) 밖이면 hover 에서만 돌린다
                #     (이후 내려감·집기는 keep_roll 로 그 롤 그대로)
                def _start_axis(bl0) -> str | None:
                    """① 시작 장면: 손목 화면 블록 긴 축이 start_axis_deg 에서 start_axis_tol_deg 넘게 벗어나면 손목 롤을 돌린다.
                    화면이 롤과 같은 쪽·반대쪽 어느 쪽으로 도는지는 화면 모형(이 높이에선 범위 밖)으로 믿기 어려우니, 한 번 돌려
                    실제로 바뀐 각으로 배율을 재서 한 번 더(최대 두 번). 못 돌리면 그대로. 반환 = 정책을 건너뛸 이유(없으면 None)."""
                    nonlocal py, ph
                    off = ((bl0["axis_deg"] - G["start_axis_deg"] + 90) % 180) - 90
                    rec = {"axis_deg": bl0["axis_deg"], "off_deg": round(off, 1), "turns": []}
                    att["start_align"] = rec
                    if abs(off) <= G["start_axis_tol_deg"] or bl0.get("aspect", 1) < W["yaw_aspect_min"]:
                        return None
                    k = 1.0   # 화면 각 변화 ÷ 롤 변화(부호를 모름 — 첫 번째로 잰다)
                    for n_turn in range(2):
                        turn = max(-90.0, min(90.0, -off / k))
                        ny = py + math.radians(turn)
                        qn, inn = s.solve(s.w_to_r3(tw, zh), ny, q, strict=True, tilt=ph)
                        if qn is None:
                            qn, inn = s.solve(s.w_to_r3(tw, zh), ny, q, strict=True, tilt=None, tilt_max=s.tilt["grasp"],
                                              near=ph)
                        if qn is None:
                            rec["why"] = f"롤 {turn:+.0f}° 자세 IK 안 풀림"
                            break
                        tn = float(inn["tilt_used"])
                        same = abs(tn - ph) < 1e-6
                        pth = _Path(s, q, grip, yaw=py, tilt=ph)
                        if (not pth.line(s.tcp_pos(q), s.w_to_r3(tw, zh), ny, tn, f"{tag} 시작 장면 롤 {turn:+.0f}°", strict=same,
                                         tilt_max=None if same else s.tilt["grasp"]) or _path_hits(s, pth, zones_forbid, sf)):
                            rec["why"] = "롤 돌림 길이 안 됨(IK·금지 구역)"
                            break
                        run(pth, f"{tag} 시작 장면 롤 {turn:+.0f}°")
                        py, ph = ny, tn
                        lk = wlook(s, q, color, sm, log, f"{tag} 시작 장면 긴 축 {n_turn + 1}", center_off=center_off)
                        b2 = lk.get("blob")
                        if b2 is None:
                            log("start_axis", tag=tag, **rec)
                            return "손목 롤을 돌린 뒤 블록이 안 보임 — 정책을 켜지 않고 IK 집기"
                        off2 = ((b2["axis_deg"] - G["start_axis_deg"] + 90) % 180) - 90
                        rec["turns"].append({"turn_deg": round(turn, 1), "axis_deg": b2["axis_deg"], "off_deg": round(off2, 1)})
                        if abs(turn) > 3 and abs((off2 - off) / turn) > 0.3:
                            k = (off2 - off) / turn
                        off = off2
                        if abs(off) <= G["start_axis_tol_deg"]:
                            break
                    log("start_axis", tag=tag, **rec)
                    return None

                look = wlook(s, q, color, sm, log, f"{tag} 방향", center_off=center_off)
                # 방향 사진의 목표점(화면 모형 + center_off)은 hover 맞춤 목표점(hover_want)과 같다 → 집게를 안 돌렸으면 첫 장으로
                dir_look = (look, q.copy()) if look.get("err_px") is not None else None
                bl = look.get("blob")
                att["standing_hint"] = grasp_act.standing_hint(bl, G.get("standing_aspect_max"))
                if att["standing_hint"] and grasp_mode != "act":
                    att["why"] = f"선 블록 의심(손목 사진 가로세로비 {bl.get('aspect')}) — IK 로는 집지 않음(사람이 눕힘)"
                    raise _Abort(f"{tag}: {att['why']}", {})
                img_start = act_try and G.get("start_align") == "image"
                act_skip_why = None
                if img_start:
                    if bl is None:
                        act_skip_why = "시작 자세 손목 사진에 블록이 안 보임 — 정책을 켜지 않고 IK 집기"
                    else:
                        act_skip_why = _start_axis(bl)
                if not img_start and bl is not None and bl.get("aspect", 1) >= W["yaw_aspect_min"]:
                    Tq = s.kin.fk(q)
                    beta = wrist.axis_in_robot(sm, Tq, look["dz"], bl["axis_deg"])
                    gam = math.atan2(Tq[1, 0], Tq[0, 0])
                    rho = (beta - gam + math.pi / 2) % math.pi - math.pi / 2
                    yaw_rec = {"rho_deg": round(math.degrees(rho), 1), "aspect": bl["aspect"], "rotated": 0.0}
                    if abs(math.degrees(rho)) < W["yaw_ok_deg"]:
                        want = beta + math.pi / 2
                        turn = (want - gam + math.pi / 2) % math.pi - math.pi / 2
                        blocked: set = set()   # 금지 구역 때문에 못 쓴 돌림(10-23 10:07: 손 자리 41mm 옆 블록 → 실행기 거부)
                        for extra_t in (0.0, math.radians(15), -math.radians(15), math.radians(30), -math.radians(30)):
                            ny = py + turn + extra_t
                            qn, inn = s.solve(s.w_to_r3(tw, zh), ny, q, strict=True, tilt=ph)
                            if qn is None and act_try:  # act 시작 높이: 돌린 방향에서 그 기울기가 안 되면 가까운 기울기로
                                qn, inn = s.solve(s.w_to_r3(tw, zh), ny, q, strict=True, tilt=None,
                                                  tilt_max=s.tilt["grasp"], near=ph)
                            if qn is None:
                                continue
                            tn = float(inn["tilt_used"])
                            gz_try, _ = _grasp_z(s, cfg, tw_block if act_try else tw, ny, pt, qn, z_add=gz_add)
                            if gz_try is None:
                                continue
                            path = _Path(s, q, grip, yaw=py, tilt=ph)
                            same = abs(tn - ph) < 1e-6
                            if path.line(s.tcp_pos(q), s.w_to_r3(tw, zh), ny, tn, f"{tag} 집게 돌림", strict=same,
                                         tilt_max=None if same else s.tilt["grasp"]):
                                hits = _path_hits(s, path, zones_forbid, sf)   # 실행기 작업 공간 검사와 같은 기준으로 미리
                                if hits:
                                    blocked |= hits
                                    continue
                                run(path, f"{tag} 집게 돌림")
                                yaw_rec["rotated"] = round(math.degrees(turn + extra_t), 1)
                                py, ph = ny, tn
                                break
                        else:
                            if blocked:  # 돌려야 하는데(짧은 변이 아님) 돌리면 금지 구역 → 긴 변을 쥐지 않고 끝(자동으로 홈)
                                where = "·".join(sorted(ZONE_KO.get(h, h) for h in blocked))
                                att["why"] = (f"{where} 가까이 있어 집게를 돌릴 수 없어요 — 사람이 블록을 투입 구역 안쪽으로 "
                                              f"옮겨 주세요(작업대 {np.round(tw).tolist()} mm)")
                                log("yaw_blocked", tag=tag, zones=sorted(blocked), rho_deg=yaw_rec["rho_deg"])
                                raise _Abort(f"{tag}: {att['why']}", {})
                    att["yaw"] = yaw_rec
                    log("yaw", tag=tag, **yaw_rec)
                # 2) hover 에서 대강 맞춤(최대 align_iters) — hover(45mm) 화면은 내려간 뒤 위치를 잘 예측하지 못한다(17:32 실물)
                keep = math.radians(W["keep_roll_deg"])
                gain = {"k": 1.0}  # 화면 감도 실측 배율(모형 대비) — 움직인 만큼 화면이 얼마나 움직였나로 고친다

                def align(level: str, z_now: float, want_fn, iters: int, tol_px, step_max: float, gain_c: float,
                          shots: int, tilt: float | None = None, tilt_free: bool = False, first_look=None) -> bool:
                    """손목 사진(shots 장 평균)으로 want_fn(dz) 화면 점에 블록 덩어리를 맞춘다. 맞으면 True.
                    tilt = 그 높이의 집게 기울기(없으면 계획 기울기 pt). tilt_free(act 시작 높이): 옮긴 자리에서 그 기울기가
                    안 되면 grasp_max 안 가까운 기울기로 옮긴다(바뀐 기울기는 ph 에)."""
                    nonlocal tw, q, ph
                    t_al = pt if tilt is None else tilt
                    prev = None
                    for it in range(iters):
                        # first_look = (사진, 찍은 자세): 같은 자세·같은 목표점이면 첫 장으로 다시 쓴다(10-23 속도: 사진 약 0.8초)
                        if it == 0 and first_look is not None and np.allclose(first_look[1], q):
                            look = first_look[0]
                            log("wrist_reuse", tag=tag, level=level, err_px=look.get("err_px"))
                        else:
                            look = wlook(s, q, color, sm, log, f"{tag} {level} 맞춤 {it + 1}", shots=shots,
                                         want=want_fn)
                        if look["blob"] is None:
                            raise _Abort(f"{tag}: 손목 사진에 블록이 안 보임({level})", {})
                        e = np.array(look["err_px"])
                        if prev is not None:  # 예측 화면 변화 = −J·d(블록이 tcp 기준으로 −d 만큼 옮겨 보임)
                            pred = -prev[1] @ prev[2]
                            if float(pred @ pred) > 25:
                                gain["k"] = float(np.clip((e - prev[0]) @ pred / (pred @ pred), 0.5, 3.0))
                        J = _jac(sm, look["dz"])
                        d = np.linalg.solve(J, e) / gain["k"]
                        Tq = s.kin.fk(q)
                        dw = s.frames.R @ (Tq[:2, :2] @ d)
                        ok = abs(e[0]) <= tol_px[0] and abs(e[1]) <= tol_px[1]
                        step = dw * gain_c
                        if np.linalg.norm(step) > step_max:
                            step = step / np.linalg.norm(step) * step_max
                        rec = {"level": level, "it": it + 1, "err_px": look["err_px"], "spread_px": look.get("spread_px"),
                               "d_mm": np.round(d, 1).tolist(), "step_mm": None if ok else np.round(step, 1).tolist(),
                               "gain_k": round(gain["k"], 2), "ok": ok}
                        att["align"].append(rec)
                        log("align", tag=tag, **rec)
                        if ok:
                            return True
                        prev = (e, J, np.linalg.solve(Tq[:2, :2], s.frames.R.T @ step))  # 실제 명령한 끝 좌표 이동
                        tw = tw + step
                        what = f"{tag} {level} 맞춤 이동 {it + 1}"
                        path = _Path(s, q, grip, yaw=py, tilt=t_al)
                        if not path.line(s.tcp_pos(q), s.w_to_r3(tw, z_now), py, t_al, what, strict=True, keep_roll=keep):
                            qn, inn = (s.solve(s.w_to_r3(tw, z_now), py, q, strict=True, tilt=None, tilt_max=s.tilt["grasp"],
                                               near=t_al, only_prev=True) if tilt_free else (None, {}))
                            why = "; ".join(path.failures)
                            path = _Path(s, q, grip, yaw=py, tilt=t_al)
                            if qn is None or not path.line(s.tcp_pos(q), s.w_to_r3(tw, z_now), py, float(inn["tilt_used"]),
                                                           what, strict=False, tilt_max=s.tilt["grasp"], keep_roll=keep):
                                raise _Abort(f"{tag}: {why or '; '.join(path.failures)}", {})
                            t_al = float(inn["tilt_used"])
                            if tilt is not None:
                                ph = t_al
                        run(path, what)
                    return False

                hover_want = (lambda dz: sm.predict([[0.0, 0.0, dz]])[0] + center_off)
                tol_h = (W["align_tol_mm"] * 2.4, W["align_tol_mm"] * 2.4)
                # act 시작을 녹화처럼 블록에서 비켜 세우면(start_back·side) 손목 맞춤(블록을 화면 가운데로)은 그걸 되돌리므로 안 한다
                offset_start = act_try and bool(G["start_back_mm"] or G["start_side_mm"])
                sv = None
                if img_start and act_skip_why is None:
                    # ② 블록 가운데를 녹화 시작 프레임 자리(start_px)로 — 화면 모형은 이 높이에선 범위 밖이라 둘째 걸음부터
                    #    실제로 움직인 만큼 화면이 움직인 배율(gain_k)로 고친다
                    ok_c = align("start", zh, lambda dz: np.asarray(G["start_px"], float), int(G["start_align_iters"]),
                                 (float(G["start_tol_px"]),) * 2, W["align_step_max_mm"], W["align_gain"], 1, tilt=ph,
                                 tilt_free=True)
                    att["start_align"] = {**(att.get("start_align") or {}), "centered": ok_c,
                                          "err_px": (att["align"][-1]["err_px"] if att["align"] else None)}
                    log("start_align", tag=tag, **att["start_align"])
                else:
                    if G.get("servo") and not act_try:
                        # 시험판(servo.py, 10-23): 손목 실시간 맞춤으로 hover → 닫기 직전 높이를 한 번에. 안 되면 hover
                        # 높이로 올라와 있고 아래 지금 방식(hover 맞춤·닫기 직전 위로·near 맞춤)으로 잇는다
                        from robot.coord_pick import servo
                        sv = servo.ik_descend(s=s, ex=sys.modules[__name__], cfg=cfg, sf=sf, W=W, log=log, tag=tag, q=q,
                                              grip=grip, yaw=py, tilt=ph, tilt_plan=pt, tilt_max=tmax, tw=tw, zh=zh,
                                              gz_add=gz_add, color=color, cls=b["cls"], sm=sm, cref=cref, allow=blk_allow,
                                              speed=speed)
                        att["servo"] = {k: sv.get(k) for k in ("ok", "why", "time_s", "z_to", "skipped")}
                        if sv.get("stopped"):
                            raise _Abort(f"{tag}: 손목 실시간 맞춤 중 멈춤(stop)", {"reason": "stopped"})
                        q, tw, ph = sv["q"], np.asarray(sv["tw"], float), float(sv.get("tilt", ph))
                    if not (sv and sv["ok"]):
                        # hover 맞춤 이동은 IK 에서도 그 기울기가 안 되면 가까운 기울기로(10-23 13:30: 기울기 5° 고정 IK 실패
                        # → 중단). hover 높이(집게 끝 40mm 위)라 블록에 닿지 않고, 바뀐 기울기는 ph 로 다음 구간이 이어받는다
                        align("hover", zh, hover_want, 0 if offset_start else int(G["act_align_iters"]) if act_try else
                              W["align_iters"], tol_h, W["align_step_max_mm"], W["align_gain"], 1, tilt=ph, tilt_free=True,
                              first_look=None if act_try or not W["reuse_dir_photo"] else dir_look)
                def _act_grasp() -> bool:
                    """집기 정책 한 번(유지 구독자로 공중 넘겨주기). 쥐었으면 True — q·grip·tw 를 정책 끝 자세로."""
                    nonlocal q, grip, tw
                    if act_skip_why:
                        att["act"] = {"skipped": act_skip_why}
                        log("act_skip", tag=tag, **att["act"])
                        return False
                    kp = (keeper_factory or (lambda: grasp_act.OwnerKeeper(log, max_s=float(G["keeper_max_s"]))))()
                    if not kp.start():
                        att["act"] = {"skipped": "유지 구독자를 못 붙임(소유 프로세스 없음) — 정책 안 돌림(재연결 토크 풀림 위험)"}
                        log("act_skip", tag=tag, **att["act"])
                        return False
                    keepers.append(kp)
                    ar = grasp_act.run_grasp_policy(str(G["policy"]), float(G["timeout_s"]), log, runner=act_runner)
                    att["act"] = {k: ar.get(k) for k in ("ok", "reason", "time_s", "error", "early_stop", "log_path")}
                    if ar["reason"] == "stopped":
                        raise _Abort(f"{tag}: 집기 정책 중 멈춤(stop)", {"reason": "stopped"})
                    if ar.get("end_q") is None:
                        if ar["reason"] in ("refused", "error"):  # 정책이 안 움직였다 — 그 자리(시작 자세)에서 IK 로
                            return False
                        raise _Abort(f"{tag}: 집기 정책 끝 자세를 모름({ar['reason']}) — 팔 위치 불명", {"pose_unknown": True})
                    q, g_end = grasp_act.end_pose(s, ar, q)
                    grip = max(float(g_end), float(sf["grip_min_n"]))  # 정책이 더 쥐었어도 송신 한도(grip_min_n) 안으로
                    img_ok = None
                    ad = ref.get("after", {}).get("by_dataset", {}).get(REF_DS[b["cls"]])
                    if ad:
                        lk = wlook(s, q, color, sm, log, f"{tag} 집은 뒤(정책)", near=(ad["u_med"], ad["v_med"]))
                        va = wrist.judge(lk["blob"], ref, "after", REF_DS[b["cls"]], area_scale=W["after_area_scale"],
                                     tol_min=W["after_tol_min_px"])
                        att["after"] = va
                        img_ok = va["ok"]
                    es_ = ar.get("early_stop") or {}
                    # 제어 시간이 먼저 끝나도 닫고 들었으면(끝 정지만 못 봄) 쥔 것으로 본다 — 녹화 39개 중 17개는 끝에 0.5초
                    # 가만히 있지 않았다(10-23). 쥐었는지는 아래 손목 사진이 가린다
                    pol_ok = bool(ar["ok"]) or (ar.get("reason") == "duration_elapsed" and bool(es_.get("closed"))
                                                and bool(es_.get("lifted")))
                    held = pol_ok and img_ok is not False
                    att["held"] = held
                    if held:  # 든 자리(정책이 맞춰 내려간 곳)부터 나른다
                        tw = s.frames.r2w(s.tcp_pos(q)[:2])
                    if not held:
                        att["why"] = f"집기 정책 {ar['reason']}" + (
                            f" · 집은 뒤 손목 사진 기준 밖({att['after']['why']})" if img_ok is False else "")
                    log("act_judge", tag=tag, held=held, img_ok=img_ok, reason=ar["reason"])
                    return held

                # 2b) act: 집기 정책 — 쥐었으면 3)·4)(IK 내려가 쥐기)를 건너뛴다
                act_held = False
                py2, pt2 = py, pt
                if act_try:
                    res["act_tried"] = True
                    act_held = _act_grasp()
                    if act_held:
                        py2, pt2 = py, pt
                        es_ = (att.get("act") or {}).get("early_stop") or {}
                        gz = float(es_.get("z_low_mm") or hz["grasp"])   # '도로 내려놓기' 높이 기준
                    elif G["fallback_ik"]:
                        # 정책 끝 자세의 집게 방향·기울기 그대로 먼저 수직으로 올라가(낮은 곳에서 벌리면 집게 끝이 바닥 아래로)
                        # 벌리고, IK hover 높이·방향으로 가서 아래 IK 집기로(같은 시도 안에서 한 번)
                        zh, tw = zh_ik, tw_block
                        Tq = s.kin.fk(q)
                        yaw_c = math.atan2(Tq[1, 0], Tq[0, 0])
                        tilt_c = math.acos(max(-1.0, min(1.0, -float(Tq[2, 2]))))
                        path = _Path(s, q, grip, yaw=yaw_c, tilt=tilt_c)
                        cur = s.tcp_pos(q)
                        if s.finger_low(q, grip) < sf["floor_mm"]:
                            # 정책이 바닥을 누른 채 끝남(12:59 실물: 집게 끝 −16mm) — 계획 IK 는 집게 끝이 탁자 아래인 자세를
                            # 못 지나므로, 그 자세 그대로 수직 25mm 먼저 빼고(관절 공간, 실행기는 floor_from_start 로 허용) 잇는다
                            qu, _ = s.solve(np.r_[cur[:2], cur[2] + 25.0], yaw_c, q, strict=True, tilt=tilt_c,
                                            finger_check=False, only_prev=True, jaw_limit=False, grip_n=grip)
                            if qu is not None:
                                path.joint_move(qu, f"{tag} 바닥에서 빼기(수직 25mm)", yaw=yaw_c, tilt=tilt_c, links_only=True)
                                cur = s.tcp_pos(qu)
                        # 올라간 높이에서는 지금 기울기가 안 풀릴 수 있다(5축 — 높이마다 되는 기울기가 3~8° 다름) → 가까운 기울기로
                        top = np.r_[cur[:2], max(zh, cur[2])]
                        _, it_ = s.solve(top, yaw_c, path.q, strict=True, tilt=None, tilt_max=tmax, near=tilt_c, grip_n=grip,
                                         jaw_limit=False)
                        path.line(cur, top, yaw_c, float(it_["tilt_used"]) if it_.get("ok") else tilt_c,
                                  f"{tag} 올라감(정책 못 쥠)", strict=False, tilt_max=tmax)
                        path.gripper(g["open"], f"{tag} 벌림(정책 못 쥠)")
                        path.line(s.tcp_pos(path.q), s.w_to_r3(tw, zh), py, pt, f"{tag} hover 로(IK 집기)", strict=False,
                                  tilt_max=tmax)
                        run(path, f"{tag} 정책 못 쥠 → IK 집기", floor_from_start=True)
                if not act_held and (not act_try or G["fallback_ik"]):
                    # 3)·4) IK 로 내려가 쥐기(ik 모드, 또는 정책이 못 쥐어 대신)
                    # 집기 높이: hover 와 같은 집게 방향·기울기·롤 갈래로(다시 고르지 않는다 — 17:29 실물에서 다시 고른
                    # 방향이 90° 달라 내려가며 롤이 돌았다)
                    py2, pt2 = py, pt
                    gz, qg = _grasp_z(s, cfg, tw, py2, pt2, q, z_add=gz_add)
                    if qg is None:
                        raise _Abort(f"{tag}: 같은 롤 갈래로 집기 높이에 못 내려감(작업대 {np.round(tw).tolist()})", {})
                    # 3) 닫기 직전 높이 + near_mm 로 내려가 녹화 '닫기 직전' 기준에 직접 맞춘다(작은 이동·2장 평균, r23)
                    dz_g = float((wrist.BLOCK_Z_MM - s.tcp_pos(qg)[2]) * s.kin.fk(qg)[2, 2])
                    close_px = np.array([cref["close"]["u"]["med"], cref["close"]["v"]["med"]])
                    near_want = (lambda dz: close_px + sm.predict([[0.0, 0.0, dz]])[0] - sm.predict([[0.0, 0.0, dz_g]])[0])
                    tol_n = (cref["close"]["tol_px"][0] * W["near_tol_frac"], cref["close"]["tol_px"][1] * W["near_tol_frac"])
                    zn = gz + W["near_mm"]
                    if not (sv and sv["ok"]):   # 손목 실시간 맞춤(시험판)이 이미 닫기 직전 높이에 맞춰 내려왔으면 건너뜀
                        path = _Path(s, q, grip, yaw=py, tilt=ph)
                        same_t = abs(ph - pt2) < 1e-6   # hover 맞춤이 기울기를 바꿨으면 내려가며 계획 기울기로(가운데는 허용 안)
                        if not path.line(s.tcp_pos(q), s.w_to_r3(tw, zn), py2, pt2, f"{tag} 닫기 직전 위로", strict=same_t,
                                         tilt_max=None if same_t else tmax, keep_roll=keep):
                            raise _Abort(f"{tag}: {'; '.join(path.failures)}", {})
                        path.hold(cfg["motion"].get("settle_s", 0), f"{tag} 닫기 직전 위 멈춤")
                        run(path, f"{tag} 닫기 직전 위로")
                    verdict = {"ok": False, "why": "맞춤 안 됨"}
                    for rnd in range(W["realign_max"] + 1):
                        if not align("near", zn, near_want, W["near_iters"], tol_n, W["near_step_max_mm"], W["near_gain"],
                                     W["shots"]):
                            verdict = {"ok": False, "why": f"닫기 직전 높이에서 {W['near_iters']}번에 기준 안으로 못 맞춤"}
                            att["close"].append({**verdict, "z": round(zn, 1)})
                            break
                        gz, qg = _grasp_z(s, cfg, tw, py2, pt2, q, z_add=gz_add)
                        if qg is None:
                            raise _Abort(f"{tag}: 같은 롤 갈래로 집기 높이에 못 내려감", {})
                        path = _Path(s, q, grip, yaw=py2, tilt=pt2)
                        if not path.line(s.tcp_pos(q), s.w_to_r3(tw, gz), py2, pt2, f"{tag} 내려감", strict=True,
                                         keep_roll=keep):
                            raise _Abort(f"{tag}: {'; '.join(path.failures)}", {})
                        path.hold(cfg["motion"].get("settle_s", 0), f"{tag} 내려감 뒤 멈춤")
                        run(path, f"{tag} 내려감")
                        for deeper in range(W["deeper_max"] + 1):
                            look = wlook(s, q, color, sm, log, f"{tag} 닫기 전", near=close_px)
                            verdict = wrist.judge_close(look["blob"], cref)
                            att["close"].append({**verdict, "z": round(gz, 1)})
                            log("judge", tag=tag, **verdict)
                            if verdict["ok"] or "덜 내려감" not in verdict["why"] or deeper == W["deeper_max"]:
                                break
                            nz = gz - W["deeper_mm"]
                            qd, idd = s.solve(s.w_to_r3(tw, nz), py2, q, strict=True, tilt=pt2, only_prev=True,
                                              grip_n=g["open"])
                            if qd is None or idd["finger_low_mm"] < sf["floor_mm"] + 1:
                                break
                            path = _Path(s, q, grip, yaw=py2, tilt=pt2)
                            if not path.line(s.tcp_pos(q), s.w_to_r3(tw, nz), py2, pt2, f"{tag} 더 내려감", strict=True,
                                             keep_roll=keep):
                                break
                            run(path, f"{tag} 더 내려감")
                            gz = nz
                        if verdict["ok"] or rnd == W["realign_max"]:
                            break
                        path = _Path(s, q, grip, yaw=py2, tilt=pt2)
                        if not path.line(s.tcp_pos(q), s.w_to_r3(tw, zn), py2, pt2, f"{tag} 닫기 직전 위로 다시", strict=True,
                                         keep_roll=keep):
                            raise _Abort(f"{tag}: {'; '.join(path.failures)}", {})
                        run(path, f"{tag} 닫기 직전 위로 다시")
                    if not verdict["ok"]:
                        att["held"] = False
                        att["why"] = f"닫기 전 판정 실패: {verdict['why']}"
                    else:
                        # 4) 닫고 조금 들어 올려 쥐었나(그리퍼 측정값)
                        path = _Path(s, q, grip, yaw=py2, tilt=pt2)
                        path.gripper(g["close"], f"{tag} 집음")
                        if not path.line(s.tcp_pos(q), s.w_to_r3(tw, zh), py2, pt2, f"{tag} 들어 올림", strict=True,
                                         keep_roll=keep):
                            # 10-23 11:37: 먼 줄에서 hover 높이 계획 기울기가 안 풀림(5축 — 높이마다 되는 기울기가 3~8° 다름).
                            # 쥔 채로 기울기를 몇 도 바꾸는 건 괜찮다(블록은 집게 사이) → 허용 안 가까운 기울기로 올라간다
                            why_l = "; ".join(path.failures)
                            qz, iz = s.solve(s.w_to_r3(tw, zh), py2, q, strict=True, tilt=None, tilt_max=tmax, near=pt2,
                                             grip_n=g["close"], only_prev=True)
                            path = _Path(s, q, grip, yaw=py2, tilt=pt2)
                            path.gripper(g["close"], f"{tag} 집음")
                            if qz is None or not path.line(s.tcp_pos(q), s.w_to_r3(tw, zh), py2, float(iz["tilt_used"]),
                                                           f"{tag} 들어 올림", strict=False, tilt_max=tmax, keep_roll=keep):
                                raise _Abort(f"{tag}: {why_l}", {})
                            att["lift_tilt_deg"] = round(math.degrees(float(iz["tilt_used"])), 1)
                            log("lift_tilt", tag=tag, plan_tilt_deg=round(math.degrees(pt2), 1), tilt_deg=att["lift_tilt_deg"])
                        r = run(path, f"{tag} 집고 들어 올림")
                        meas = ((r.get("run") or {}).get("last_step") or {}).get("measured")
                        att["grip_measured"] = None if meas is None else meas[5]
                        grip_ok = meas is not None and meas[5] > g["close"] + W["held_margin_n"]
                        # 집은 뒤 손목 사진: 블록이 사람이 쥐고 들어 올릴 때처럼 두 날 사이에 그 크기로 보이나(r20)
                        ad = ref.get("after", {}).get("by_dataset", {}).get(REF_DS[b["cls"]])
                        img_ok = None
                        if ad:
                            look = wlook(s, q, color, sm, log, f"{tag} 집은 뒤", near=(ad["u_med"], ad["v_med"]))
                            va = wrist.judge(look["blob"], ref, "after", REF_DS[b["cls"]], area_scale=W["after_area_scale"],
                                             tol_min=W["after_tol_min_px"])
                            att["after"] = va
                            img_ok = va["ok"]
                            log("judge_after", tag=tag, grip_ok=grip_ok, **va)
                        att["held"] = img_ok if img_ok is not None else grip_ok
                        if not att["held"]:
                            att["why"] = (f"집은 뒤 손목 사진 기준 밖({att['after']['why']}) · 그리퍼 {meas[5] if meas else '?'}"
                                          if img_ok is not None else f"그리퍼가 {meas[5] if meas else '?'} 까지 닫힘(쥔 게 없음)")
                elif not act_held:
                    att["held"] = False
                if att["held"]:
                    # 5) 그릇에 놓기 — 블록을 든 뒤로는 롤 고정·같은 갈래를 풀고 지금 자세에서 다시 푼다(17:54 실물: 맞추기
                    #    전 계획의 나르는 기울기 10° 가 맞춘 자리에서 안 풀려 거부 → 블록을 든 채 멈춤)
                    zc = max(hz["carry"], bw["rim_mm"] + bw["release_above_mm"])
                    z_rel = bw["rim_mm"] + bw["release_above_mm"]
                    floor = z_rel - bw.get("release_margin_mm", 20)
                    bowls = bowl_targets(s, zw)
                    bowl = None if unsort else bw["targets"].get(b["cls"])
                    done_drop, why_drop = False, ""
                    if unsort:  # 되돌리기: 빈 칸 차례로 낮게 놓기(집게축 고정 = 블록 긴 축 0°)
                        done_drop, why_drop, to = _place_cell(s, cfg, sf, speed, log, b, tw, q, grip, py2, pt2, cells, last,
                                                              tag, zones_forbid, z_high, blk_allow, U, res)
                        if done_drop:
                            q, grip = done_drop
                    for zc_try in (() if unsort else (zc, zc + 15, zc + 30)):
                        qc, ic = s.solve(s.w_to_r3(tw, zc_try), py2, q, tilt=None, tilt_max=tmax, near=pt2,
                                         grip_n=g["close"], jaw_limit=False)
                        if qc is None:
                            why_drop = f"나르는 높이 {zc_try:g}mm 에 안 닿음({ic.get('why')})"
                            continue
                        for drop_w in drop_points(s, bowl, bowls, zw, 0.0, bw.get("edge_mm", 25)):
                            yd = s.tangent_yaw(s.w_to_r3(drop_w, z_rel)) + s.frames.yaw
                            dcol = s.column(drop_w, yd, (zc_try, z_rel), qc, tilt_max=s.tilt["release"], grip_n=g["close"],
                                            yaws=(ic["yaw_used"] + s.frames.yaw, yd))
                            if dcol is None:
                                why_drop = f"{BOWL_KO[bowl]} 위에 안 닿음({s.last_why})"
                                continue
                            dy_, dt_ = dcol["yaw_w"] - s.frames.yaw, dcol["tilt"]
                            path = _Path(s, q, grip, yaw=py2, tilt=pt2)
                            ok = (path.line(s.tcp_pos(q), s.w_to_r3(tw, zc_try), ic["yaw_used"], ic["tilt_used"],
                                            f"{tag} 나르는 높이로", strict=False, tilt_max=tmax)
                                  and travel(s, path, tw, zc_try, drop_w, zc_try, dy_, dt_, f"{tag} → {BOWL_KO[bowl]} 위",
                                             zones=zones_forbid, skip={bowl}, z_high=z_high, margin=sf["forbid_margin_mm"],
                                             tilt_max=tmax, q_goal=dcol["qs"][0], fast=True)
                                  and path.line(s.w_to_r3(drop_w, zc_try), s.w_to_r3(drop_w, z_rel), dy_, dt_,
                                                f"{tag} 놓는 높이로", strict=True))
                            if not ok:
                                why_drop = "; ".join(path.failures)[:200]
                                continue
                            path.gripper(g["release"], f"{tag} 놓음", settle=True)
                            path.line(s.w_to_r3(drop_w, z_rel), s.w_to_r3(drop_w, zc_try), dy_, dt_, f"{tag} 올라감",
                                      strict=False, tilt_max=tmax, fast=True)
                            if last:
                                path.finish({k: v for k, v in zones_forbid.items() if k != bowl}, sf["forbid_below_mm"],
                                            sf["forbid_margin_mm"], z_high)
                            if path.failures:
                                why_drop = "; ".join(path.failures)[:200]
                                continue
                            r = _segment(s, cfg, sf, speed, log, path, f"{tag} 그릇에 놓기", allow={bowl: floor},
                                         floor_from_start=act_held)
                            if r["reason"] == "refused" and not r.get("executed"):  # 공간 검사 등에서 걸림 → 다른 후보
                                why_drop = r.get("error") or "거부"
                                log("drop_retry", tag=tag, why=why_drop, drop_w=drop_w, zc=zc_try)
                                continue
                            if r["reason"] != "done":
                                raise _Abort(f"{tag} 그릇에 놓기: {r['reason']} {r.get('error') or ''}".strip(), r)
                            q, grip = path.q, path.grip
                            done_drop = True
                            res.update(drop_mm=np.round(drop_w, 1).tolist(), carry_mm=zc_try)
                            break
                        if done_drop:
                            break
                    if not done_drop:
                        # 놓을 길이 없으면 든 블록을 그 자리에 도로 내려놓고 멈춘다(든 채 홈으로 가지 않는다)
                        path = _Path(s, q, grip, yaw=py2, tilt=pt2)
                        path.line(s.tcp_pos(q), s.w_to_r3(tw, gz + 3), py2, pt2, f"{tag} 도로 내려놓기", strict=True,
                                  keep_roll=keep)
                        path.gripper(g["open"], f"{tag} 벌림(내려놓음)", settle=True)
                        path.line(s.w_to_r3(tw, gz + 3), s.w_to_r3(tw, zh_ik), py2, pt2, f"{tag} 올라감", strict=True,
                                  keep_roll=keep)  # IK hover 높이(act 시작 90mm 는 계획 기울기가 안 되는 칸이 많다)
                        path.finish(zf_home, sf["forbid_below_mm"], sf["forbid_margin_mm"], z_high)
                        run(path, f"{tag} 도로 내려놓고 홈")
                        raise _Abort(f"{tag}: {'놓을 빈 칸' if unsort else '그릇에 놓을 길'}을 못 찾아 블록을 도로 내려놓음 — "
                                     f"{why_drop}", {})
                    prev_bowl, at_home = bowl, last
                    res.update(ok=True, to=to if unsort else BOWL_KO[bowl], final_mm=np.round(tw, 1).tolist())
                    break
                # 못 쥠 → 벌리고 홈 → 위 카메라로 다시 찾기
                path = _Path(s, q, grip, yaw=py2, tilt=pt2)
                p_up = s.tcp_pos(q)
                try:
                    z_down = gz + 3
                except NameError:
                    z_down = None
                if W["retry_put_down"] and grip < g["release"] and z_down is not None:
                    # 쥔 채 들어 올렸으면 그 높이에서 벌리지 않고 제자리에 내려놓고 벌린다(10-23 13:27: 쥔 분홍을 '집은 뒤' 판정이
                    # 못 쥠으로 봐 들어 올린 높이에서 세 번 벌림 → 떨어뜨림). 내려가는 길이 안 되면 예전처럼 그 자리에서 벌린다
                    if path.line(p_up, s.w_to_r3(tw, z_down), py2, pt2, f"{tag} 도로 내려놓기(다시)", strict=False,
                                 tilt_max=tmax, keep_roll=keep):
                        p_up = s.w_to_r3(tw, z_down)
                    else:
                        path = _Path(s, q, grip, yaw=py2, tilt=pt2)
                path.gripper(g["open"], f"{tag} 벌림(다시)")
                path.line(p_up, s.w_to_r3(tw, zh_ik), py2, pt2, f"{tag} 올라감(다시)", strict=False, tilt_max=tmax)
                path.finish(zf_home, sf["forbid_below_mm"], sf["forbid_margin_mm"], z_high)
                run(path, f"{tag} 홈으로(다시)")
                at_home, prev_bowl = True, None
                if attempt == W["retries"]:
                    break
                found = [x for x in blocks_from_detection(redetect()) if x["cls"] == b["cls"]
                         and x.get("zone", "input") == (b["zone"] if unsort else "input")]
                if not found:
                    att["redetect"] = "안 보임"
                    break
                nb = min(found, key=lambda x: float(np.linalg.norm(s.px_to_w(x["px"])[0] - tw)))
                tw = s.px_to_w(nb["px"])[0]
                b = {**b, **{k: nb.get(k) for k in ("yaw_w", "aspect")}}
                att["redetect"] = np.round(tw, 1).tolist()
        out.update(ok=all(r["ok"] for r in results) and len(results) > 0, reason="done" if results else "planned",
                   executed=True, wrist=results)
    except _Abort as e:
        er = e.r or {}
        out.update(ok=False, reason=er.get("reason", "refused") if e.r else "refused", error=str(e),
                   executed=bool(results), wrist=results)
        if not at_home and er.get("reason") not in ("stopped",):
            if er.get("executed") or er.get("pose_unknown"):
                # 보내던 중 멈춤(추종 오차·상태 끊김 등)이나 정책 뒤 자세를 모름 — 저절로 움직이지 않는다
                out["error"] += " — 팔은 그 자리(go_home 으로 홈에)"
            else:
                # 보내기 전에 거부(검사·계획 실패) — 팔은 마지막 구간 끝(q)에 그대로 있다 → 수직으로 올라가 홈으로(10-23 09:50)
                rc = _recover_home(s, cfg, sf, speed, log, q, grip, zones_forbid, z_high)
                out["recover"] = rc
                if rc["ok"]:
                    at_home = True
                    out["error"] += (f" — 그 자리에서 {rc.get('lift_to_mm')}mm 로 올라가 홈으로 돌아감"
                                     + (" · ⚑ 블록을 쥔 채로 홈에 왔어요(집게는 열지 않음 — 사람이 받아 주세요)"
                                        if rc.get("holding") else ""))
                else:
                    out["error"] += f" — 팔은 그 자리(저절로 홈 복귀도 안 됨: {rc.get('why')} — go_home 으로 홈에)"
    finally:
        for kp in keepers:
            kp.stop()
    return out


def _fake_sort(color, det, cfg, grasp_mode, which) -> dict:
    """CLI sort --fake: 가짜 로봇(실행 대신 목표 자세로 옮김)·가짜 손목 카메라(화면 모형으로 그린 블록)로 손목 단계를 끝까지.
    로봇·카메라·Studio 정책은 안 쓴다(정책은 '안 움직임' = refused → 실행기가 IK 집기로 잇는다). 블록 1개."""
    from robot.coord_pick.wrist_fake import FakeWorld
    cfg = copy.deepcopy(cfg or load_config())
    cfg["calibrated"] = True
    s = Setup.load(cfg, use_studio=False)
    colors = tuple(c for c in color.split(",")) if isinstance(color, str) else tuple(color)
    classes = tuple(c for k in colors for c in SORT_CLASSES[k])
    blocks = blocks_from_detection(det) if det else demo_sort_scene(s)
    cand = [b for b in blocks if b["cls"] in classes and b.get("zone", "input") == "input"]
    if not cand:
        return {"ok": False, "mode": "fake", "reason": "refused", "error": f"{color} 블록 없음", "summary": ""}
    tb = min(cand, key=lambda b: float(np.linalg.norm(s.frames.w2r(s.px_to_w(b["px"])[0]))))
    yaw = tb.get("yaw_w", math.radians(20.0))
    objs = [dict(b) for b in blocks]
    for o in objs:
        if o is not None and o.get("px") == tb.get("px"):
            o.update(yaw_w=yaw, aspect=tb.get("aspect", 1.8))
    fw = FakeWorld(s, s.px_to_w(tb["px"])[0], "red" if tb["cls"].startswith("red") else "blue", true_yaw_w=yaw,
                   aspect=tb.get("aspect", 1.8))

    class _Keeper:
        def start(self):
            return True

        def stop(self):
            pass
    fw.install(sys.modules[__name__])
    old = os.environ.get("BALL_ROBOT")
    os.environ["BALL_ROBOT"] = "1"
    try:
        r = sort_blocks(colors, dry_run=False, detection={"objects": objs}, max_blocks=1, config=cfg, wrist_check=True,
                        log_dir=rp.LOG_DIR / f"{dt.datetime.now(KST):%Y%m%d}" / "fake", redetect=lambda: {"objects": objs},
                        grasp_mode=grasp_mode, which=which,
                        act_runner=lambda name, timeout_s=None, return_home=None: {"reason": "refused", "end_q": None},
                        keeper_factory=_Keeper)
    finally:
        fw.uninstall()
        if old is None:
            os.environ.pop("BALL_ROBOT", None)
        else:
            os.environ["BALL_ROBOT"] = old
    r["mode"] = "fake"
    r["fake_sends"] = fw.sends
    return r


def go_home(dry_run: bool = True, open_gripper: bool = False, *, speed_deg_s: float | None = None,
            max_deg: float | None = None, config: dict | None = None, log_dir: Path | None = None) -> dict:
    """지금 측정 자세에서 config home_n 까지 관절 공간 직선으로 천천히(기본 safety.home_deg_s = 8°/s) 보낸다.

    그리퍼는 지금 값 그대로(open_gripper=True 면 gripper.open). 시작 자세 검사(not_at_home)는 이 명령만 면제하되, 어느
    관절이든 max_deg(기본 safety.home_max_deg = 60°)보다 멀면 하나도 안 보내고 거부(too_far). 이미 홈이면(1.8° 안) 안 보낸다.
    문: BALL_ROBOT=1 · dry_run=False · 잠금 · 이전 실행 기다림 · 사전 점검 · stop() — sort_blocks 와 같다(calibrated 는
    안 본다: 작업대 좌표를 안 쓰는 관절 이동). 측정 자세는 컨테이너에서 연결한 뒤에 읽으므로 dry-run 은 계획만 돌려준다.
    반환 dict — ok·mode·executed·reason·summary·measured·target·time_s·warnings·failures·violations·sim·run·log_path·error.
    """
    cfg, sf, speed, log, _ = _start(config, speed_deg_s or safety(config or load_config())["home_deg_s"],
                                    "home" + ("_open" if open_gripper else ""), dry_run, log_dir)
    max_deg = float(max_deg or sf["home_max_deg"])
    out: dict = {"ok": False, "mode": "dry_run", "executed": False, "reason": "planned", "speed_deg_s": speed,
                 "max_deg": max_deg, "open_gripper": open_gripper, "plan": [], "warnings": [], "failures": [],
                 "violations": [], "sim": None, "run": None, "measured": None, "log_path": str(log.path), "error": None}
    try:
        log("request", kind="home", dry_run=dry_run, open_gripper=open_gripper, speed=speed, max_deg=max_deg,
            ball_robot=os.environ.get("BALL_ROBOT"))
        s = Setup.load(cfg, use_studio=True)
        lim = limits(s, sf, speed)
        dpu = max(s.jm.deg_per_unit(k) for k in ARM)
        target = np.array(cfg["home_n"], float)
        if open_gripper:
            target[5] = cfg["gripper"]["open"]
        extra = {"target": target.tolist(), "keep_gripper": not open_gripper, "already_tol": 1.8 / dpu,
                 "max_off": round(max_deg / dpu, 2)}
        out.update(target=target.round(2).tolist(), time_s=round(max_deg / speed, 1))
        out["summary"] = (f"홈 자세로(관절 {speed:g}°/s, 그리퍼 {'벌림 ' + str(cfg['gripper']['open']) if open_gripper else '그대로'})"
                          f" — 어느 관절이든 {max_deg:g}° 넘게 멀면 거부, 최대 약 {max_deg / speed:.0f}초")
        log("plan", extra=extra, limits=lim)
        if dry_run:
            out["ok"] = True
            return out
        if os.environ.get("BALL_ROBOT") != "1":
            out.update(reason="refused", error="실제로 보내지 않음: BALL_ROBOT=1 이 아님")
            log("gate", refused=[out["error"]])
            return out
        out["mode"] = "real"
        r = _run_real(s, None, [], lim, log, name="home", entry="container_return", extra=extra,
                      wall_s=max_deg / speed * 1.5 + 30)
        end = (r.get("run") or {}).get("end") or {}
        out.update(executed="run_id" in r, reason=r["reason"], run=r.get("run"), run_id=r.get("run_id"),
                   error=r.get("error") or (f"홈에서 {end.get('off')}(정규화) 떨어짐 > {extra['max_off']}"
                                           if r["reason"] == "too_far" else None),
                   measured=(r.get("measured") or {}).get("q"))
        out["ok"] = r["reason"] in ("done", "already_home")
        return out
    except Exception as e:  # noqa: BLE001 — 도구는 예외 대신 결과로
        out.update(reason="error", error=f"{type(e).__name__}: {e}")
        return out
    finally:
        log("result", **out)
        log.close()


def _print_result(r: dict) -> None:
    print(f"[{r['mode']}] {r.get('summary', '')}")
    print(f"결과 ok={r['ok']} reason={r['reason']}" + (f" · {r['error']}" if r.get("error") else ""))
    for o in r.get("plan", []):
        if "to_cell" in o:
            print(f"  {o['order']:>2}. {o['color']:<6} {o['from_mm']} → 칸 {tuple(o['to_cell'])} {o['to_mm']}")
        elif "place_z" in o:  # 되돌리기 — CLI unsort 가 따로 찍는다
            continue
        else:
            print(f"  {o['order']:>2}. {o['color']:<6} {o['from_mm']} (기울기 {o['pick_tilt_deg']}°) → {o['to']} "
                  f"{o['to_mm']} (기울기 {o['drop_tilt_deg']}°"
                  + (f", 가운데에서 로봇 쪽 {o['drop_shift_mm']:g}mm" if o.get("drop_shift_mm") else "") + ")")
    for v in r.get("violations", []):
        print(f"  [위반] {v['kind']} ×{v['count']} (첫 목표 {v['first_tick']}): {v['detail']}")
    for w in r.get("warnings", []):
        print(f"  [주의] {w}")
    for f in r.get("failures", []):
        print(f"  [안됨] {f}")
    if r.get("checks"):
        print("  검사: " + " · ".join(f"{k} {v}" for k, v in r["checks"].items()))
    if r.get("sim"):
        print(f"  모의 실행: {r['sim']['reason']} {r['sim']['sent']}/{r['sim']['total']} · 최대 추종 오차(정규화) {r['sim']['max_err']}")
    if r.get("run"):
        print(f"  실행: {json.dumps(r['run'].get('end'), ensure_ascii=False)[:300]}")
    print(f"  기록: {r['log_path']}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["home"]:
        ap = argparse.ArgumentParser(prog="python -m robot.coord_pick.executor home", description=go_home.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--open", action="store_true", help="그리퍼도 gripper.open 으로(기본은 그대로)")
        ap.add_argument("--speed", type=float, help="관절 속도 °/s (기본 safety.home_deg_s = 8)")
        ap.add_argument("--confirm", action="store_true", help="실제 송신(BALL_ROBOT=1 도 있어야 함)")
        a = ap.parse_args(argv[1:])
        if a.confirm and os.environ.get("BALL_ROBOT") != "1":
            print("[주의] --confirm 이 있지만 BALL_ROBOT=1 이 아니다 → 보내지 않는다")
        r = go_home(dry_run=not a.confirm, open_gripper=a.open, speed_deg_s=a.speed)
        print(f"[{r['mode']}] {r.get('summary', '')}")
        print(f"결과 ok={r['ok']} reason={r['reason']}" + (f" · {r['error']}" if r.get("error") else ""))
        if r.get("measured"):
            print(f"  측정 자세(시작) {r['measured']} → 목표 {r['target']}")
        if r.get("run"):
            print(f"  실행: {json.dumps(r['run'].get('end'), ensure_ascii=False)[:300]}")
        print(f"  기록: {r['log_path']}")
        return 0 if r["ok"] else 1
    if argv[:1] == ["unsort"]:
        ap = argparse.ArgumentParser(prog="python -m robot.coord_pick.executor unsort", description=unsort_blocks.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--color", help="red·blue·defect·red_defect·blue_defect, 여러 개는 쉼표로(없으면 모두)")
        ap.add_argument("--blocks", type=int, help="로봇에 가까운 것부터 몇 개만")
        ap.add_argument("--detect", help="det.json(detect_objects() 결과). 없고 --demo 도 없으면 위 카메라로 지금 찍는다")
        ap.add_argument("--image", help="--detect 와 같은 때 찍은 위 카메라 원본 사진(블록 방향을 잰다)")
        ap.add_argument("--demo", action="store_true", help="가짜 장면(그릇마다 1~2개 + 투입 구역 1개)")
        ap.add_argument("--speed", type=float, help="관절 속도 °/s (기본 safety.exec_deg_s)")
        ap.add_argument("--config", help="config 파일(기본 robot/coord_pick/config.json)")
        ap.add_argument("--wrist", dest="wrist", action="store_true", default=None, help="손목 카메라 확인·보정")
        ap.add_argument("--no-wrist", dest="wrist", action="store_false", help="손목 확인 없이")
        ap.add_argument("--confirm", action="store_true", help="실제 송신(BALL_ROBOT=1 도 있어야 함)")
        a = ap.parse_args(argv[1:])
        if a.confirm and os.environ.get("BALL_ROBOT") != "1":
            print("[주의] --confirm 이 있지만 BALL_ROBOT=1 이 아니다 → 보내지 않는다")
        det = json.loads(Path(a.detect).read_text(encoding="utf-8")) if a.detect else None
        r = unsort_blocks(not a.confirm, a.color, a.blocks, detection=det, speed_deg_s=a.speed,
                          demo=a.demo if det is None else None, config=load_config(a.config) if a.config else None,
                          wrist_check=a.wrist, front_image=a.image)
        _print_result(r)
        for x in r.get("plan") or []:
            print(f"  {x['order']}. {x['color']} {x['from']} {x['from_mm']} → {x['to']} {x['to_mm']} "
                  f"(집는 높이 {x['grasp_z']} · 놓는 높이 {x['place_z']} · 기울기 {x['pick_tilt_deg']}/{x['place_tilt_deg']}°)")
        for x in r.get("skipped") or []:
            print(f"  못 꺼냄: {x['block']} {x['bowl']} {x['from_mm']} — {x['why']}")
        print(f"  빈 칸(로봇에 가까운 순): {r.get('free_cells')}")
        if r.get("recover"):
            print(f"  거부 뒤 홈 복귀: {r['recover']}")
        return 0 if r["ok"] else 1
    if argv[:1] == ["sort"]:
        ap = argparse.ArgumentParser(prog="python -m robot.coord_pick.executor sort", description=sort_blocks.__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("color_pos", nargs="?", metavar="color", help="--color 와 같음(sort red …)")
        ap.add_argument("--color", default="red", help="red(분홍)·blue(초록)·defect(불량), 여러 개는 쉼표로")
        ap.add_argument("--detect", help="det.json(detect_objects()·detect_hsv 결과). 없고 --demo 도 없으면 위 카메라로 지금 찍는다")
        ap.add_argument("--image", help="--detect 와 같은 때 찍은 위 카메라 원본 사진(블록 방향을 잰다)")
        ap.add_argument("--demo", action="store_true", help="시연 배치 가짜 장면(분홍 2·초록 1·불량 1)")
        ap.add_argument("--blocks", type=int, help="로봇에 가까운 것부터 몇 개만(입회 시험: 1)")
        ap.add_argument("--which", choices=("left", "right", "middle", "top", "bottom", "near", "far"),
                        help="위 카메라 사진에서 왼쪽(u 작음)·오른쪽·가운데·위(v 작음 = 로봇에서 먼 쪽)·아래, 로봇에 가까운·먼 것부터 "
                             "--blocks 개(없으면 1개)")
        ap.add_argument("--next-to", choices=("red", "blue", "red_defect", "blue_defect"),
                        help="그 색 블록에 가장 가까운 것부터(--which 대신)")
        ap.add_argument("--lift-mm", type=float, default=0.0, help="공중 리허설: 모든 높이를 이만큼 올림(첫 구동은 40)")
        ap.add_argument("--speed", type=float, help="관절 속도 °/s (기본 safety.exec_deg_s)")
        ap.add_argument("--config", help="config 파일(기본 robot/coord_pick/config.json)")
        ap.add_argument("--wrist", dest="wrist", action="store_true", default=None, help="손목 카메라 확인·보정(r19)")
        ap.add_argument("--no-wrist", dest="wrist", action="store_false", help="손목 확인 없이(예전 한 번에 보내기)")
        ap.add_argument("--grasp", choices=("ik", "act"), help="집기 방법(기본 config grasp.mode, 없으면 ik). act = 블록 위까지 IK → "
                                                           "집기 정책(policies.json grasp) → IK 나르기(GRASP_ACT.md)")
        ap.add_argument("--start-align", choices=("none", "image"),
                        help="act 시작 장면 맞춤(이번 실행만 config grasp.start_align 대신): image = 손목 화면 긴 축·가운데를 "
                             "녹화 시작 프레임처럼 맞춘 뒤 정책")
        ap.add_argument("--servo", action="store_true",
                        help="시험판: 손목 실시간 맞춤(servo.py)으로 hover → 닫기 직전 높이를 한 번에(이번 실행만 config grasp.servo "
                             "대신, IK 만). 안 되면 지금 방식으로")
        ap.add_argument("--fake", action="store_true",
                        help="가짜 로봇·가짜 손목 카메라로 손목 단계(시작 장면 맞춤 포함)까지 돌려 본다 — 로봇·카메라·정책 0, "
                             "블록 1개(--detect 나 --demo 장면에서 로봇에 가까운 것). 정책 자리는 '안 움직임'으로 대신해 IK 로 이어진다. "
                             "기록 robot/logs/<날짜>/fake/")
        ap.add_argument("--confirm", action="store_true", help="실제 송신(BALL_ROBOT=1 도 있어야 함)")
        a = ap.parse_args(argv[1:])
        a.color = a.color_pos or a.color
        if a.next_to:
            a.which = {"next_to": a.next_to}
        if a.confirm and os.environ.get("BALL_ROBOT") != "1":
            print("[주의] --confirm 이 있지만 BALL_ROBOT=1 이 아니다 → 보내지 않는다")
        det = json.loads(Path(a.detect).read_text(encoding="utf-8")) if a.detect else None
        cfg_cli = load_config(a.config) if a.config else None
        if a.start_align:
            cfg_cli = copy.deepcopy(cfg_cli or load_config())
            cfg_cli.setdefault("grasp", {})["start_align"] = a.start_align
        if a.servo:
            cfg_cli = copy.deepcopy(cfg_cli or load_config())
            cfg_cli.setdefault("grasp", {})["servo"] = True
        if a.fake:
            r = _fake_sort(a.color, det, cfg_cli, a.grasp, a.which)
        else:
            r = sort_blocks(a.color, dry_run=not a.confirm, detection=det, max_blocks=a.blocks,
                            lift_mm=a.lift_mm, speed_deg_s=a.speed, demo=a.demo if det is None else None,
                            config=cfg_cli, wrist_check=a.wrist, front_image=a.image, grasp_mode=a.grasp, which=a.which)
        _print_result(r)
        if r.get("which"):
            print(f"  고른 블록({r['which'].get('rule')}): {r['which']['chosen_px']}"
                  + (f" · 옆 기준 {r['which'].get('ref_px')}" if r['which'].get('ref_px') is not None else ""))
        if r.get("recover"):
            print(f"  거부 뒤 홈 복귀: {r['recover']}")
        for w in r.get("wrist") or []:
            print(f"  손목: {w['block']} {w['from_mm']} (위 카메라 긴 축 {w.get('top_yaw_deg')}°) → "
                  f"{'성공 ' + w.get('to', '') if w['ok'] else '실패'} · 시도 {len(w['attempts'])}번")
            for at in w["attempts"]:
                print(f"    방향 {at.get('yaw')} · 맞춤 {[x.get('dw') for x in at['align']]} · 닫기 전 {[c.get('why') for c in at['close']]} · "
                      f"집은 뒤 {(at.get('after') or {}).get('why')} · 쥠 {at.get('held')} (그리퍼 {at.get('grip_measured')}) "
                      f"{at.get('why', '')}")
                ar = at.get("arrive") or {}
                if ar.get("tried") or ar.get("used_mm") != ar.get("want_mm"):
                    print(f"    블록 위 높이: {ar.get('want_mm')} → {ar.get('used_mm') or 'hover 로 바로'}mm "
                          f"(기울기 {ar.get('plan_tilt_deg')}° → {ar.get('tilt_deg', ar.get('plan_tilt_deg'))}°, "
                          f"안 된 것 {len(ar.get('tried') or [])})")
                if at.get("act_start") or at.get("start_align"):
                    print(f"    시작 자세: {at.get('act_start')} · 시작 장면 맞춤: {at.get('start_align')}")
                if at.get("act"):
                    ac = at["act"]
                    print(f"    집기 정책: {ac.get('reason') or ac.get('skipped')} {ac.get('time_s', '')}s · 끝 감지 {ac.get('early_stop')}")
        return 0 if r["ok"] else 1
    ap = argparse.ArgumentParser(prog="python -m robot.coord_pick.executor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("digit")
    ap.add_argument("--color", choices=["any", "red", "blue"], default="any", help="red = 분홍, blue = 초록")
    ap.add_argument("--small", action="store_true", help="3행 시험 글꼴(1 = 블록 3개, 7 = 블록 5개)")
    ap.add_argument("--blocks", type=int, help="앞의 칸 몇 개만(입회 시험: 1)")
    ap.add_argument("--detect", help="detect_objects() 결과 JSON 파일")
    ap.add_argument("--demo", type=int, help="가짜 장면 블록 수(시험용)")
    ap.add_argument("--speed", type=float, help="관절 속도 °/s (기본 safety.exec_deg_s)")
    ap.add_argument("--confirm", action="store_true", help="실제 송신(BALL_ROBOT=1 도 있어야 함)")
    a = ap.parse_args(argv)
    det = json.loads(Path(a.detect).read_text(encoding="utf-8")) if a.detect else None
    if a.confirm and os.environ.get("BALL_ROBOT") != "1":
        print("[주의] --confirm 이 있지만 BALL_ROBOT=1 이 아니다 → 보내지 않는다")
    r = arrange_digit(a.digit, a.color, dry_run=not a.confirm, detection=det, small=a.small, max_blocks=a.blocks,
                      speed_deg_s=a.speed, demo=a.demo if det is None else None)
    _print_result(r)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
