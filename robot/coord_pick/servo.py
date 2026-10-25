"""손목 카메라 실시간 맞춤(비주얼 서보잉) 시험판 — 블록 위(hover)에서 닫기 직전 높이(near)까지 한 번에 내려가며
손목 사진으로 목표를 계속 고친다(10-23 14:02 사용자 결정: 시연은 기존 코드, 이건 따로).

config grasp.servo(기본 없음 = 꺼짐) 또는 CLI --servo 일 때만 executor._sort_wrist(IK) 가 부른다. 꺼져 있으면 실행기는
예전과 같은 계획을 낸다(selftest). 바뀌는 구간은 'hover 맞춤 → 닫기 직전 위로' 하나다. 그 아래(near 확인·내려감·
닫기 전 판정·닫기)와 운반·놓기·홈은 그대로다.

    호스트(descend, 10Hz)                                   컨테이너(servo_main, 30Hz)
    손목 사진 → 블록 자리 추정 → 다음 목표(IK·검사)  ── 목표 관절값(줄 JSON, stdin) ─→  틱 한도 안에서 따라감
                                                   ←─ 측정 관절값(10Hz, stdout) ────  추종 오차·상태 끊김 검사

안전(지금 실행기와 같은 기준):
- 틱마다 관절 변화 ≤ safety.exec_deg_s(컨테이너가 자른다), 추종 오차·상태 끊김 검사(stream.py 와 같은 규칙).
- 목표마다 check_workspace(바닥·링크 바닥·반경·금지 구역) — 위반이면 멈춘다. 가장 낮은 목표 = 닫기 직전 높이.
- 손목 사진에 블록이 lost_max 장 연속 없거나, 추정 자리가 jump_mm 넘게 튀거나, 자세 계산이 안 되거나, 시간이 넘으면
  그 자리에서 멈추고 처음 높이(hover)로 수직으로 올라간 뒤 끝낸다 → 실행기가 지금 방식(hover 맞춤·near 맞춤)으로 잇는다.
- 호스트가 조용하면(host_timeout_s) 컨테이너는 그 자리에 선다. 그 다섯 배면 끝낸다. stop() 은 MARKER 로 죽인다(지금과 같음).
- 마지막 slow_mm 는 느리게(vz_slow_mm_s) — 움직이는 중 사진 번짐(노출 25ms).
"""
from __future__ import annotations

import collections
import dataclasses
import inspect
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from robot import run_policy as rp
from robot.coord_pick import wrist
from robot.coord_pick.stream import Abort, emit_json  # 컨테이너 쪽 함수가 쓴다(그쪽은 stream.py 소스로)

HERE = Path(__file__).resolve().parent

SERVO_DEFAULTS = {
    "rate_hz": 10.0,        # 호스트 고침 주기(손목 사진 → 목표)
    "vz_mm_s": 20.0,        # 내려가는 속도
    "vz_slow_mm_s": 8.0,    # 마지막 slow_mm 안
    "slow_mm": 25.0,
    "vxy_mm_s": 25.0,       # 옆으로 고치는 속도 상한
    "gate_mm": 8.0,         # 옆 오차가 이보다 크면 내려가지 않고 옆만 고친다
    "tol_mm": 2.5,          # 닫기 직전 높이에서 이 안이 두 장 이어지면 끝
    "alpha": 0.6,           # 블록 자리 추정 걸러내기(새 값 몫)
    "lost_max": 3,          # 블록 없는 사진 연속 이만큼이면 멈춤
    "jump_mm": 25.0,        # 추정 자리가 한 번에 이만큼 넘게 튀면 멈춤
    "max_s": 12.0,          # 이 시간 안에 못 끝나면 멈춤
    "host_timeout_s": 0.6,  # 컨테이너: 호스트 목표가 이만큼 없으면 그 자리
    "settle_s": 0.3,        # 끝날 때 마지막 목표에서 기다리며 한 번 더 추종 검사
    "warmup": 3,            # 손목 카메라 열고 버리는 장 수
}

# 스크립트는 UTF-8 바이트 길이 + 본문(한글 소스 — 컨테이너 로케일과 상관없게), 그 뒤 stdin 줄 = 목표 JSON
BOOT = ("import sys;b=sys.stdin.buffer;n=int(b.readline());"
        "exec(compile(b.read(n).decode('utf-8'),'<servo>','exec'))")


# ------------------------------------------------------------------ 컨테이너 쪽(stream.py 소스와 함께 보낸다)

def servo_loop(robot, inbox, *, rate_hz: float, step_body, step_grip: float, track_max, track_ticks: int,
               lag_ticks: int, stale_s: float, start_tol: float, start, max_s: float, host_timeout_s: float,
               settle_s: float, emit, clock=time.monotonic, sleep=time.sleep) -> dict:
    """inbox() → {"target": 6값} | {"end": 1} | None(새 것 없음). 마지막 목표를 향해 틱마다 step 한도 안에서 보낸다.
    반환·end 이벤트 = stream() 과 같은 꼴 {ok, reason, sent, max_err, elapsed_s}."""
    step_body = np.broadcast_to(np.asarray(step_body, float), (5,))
    track_max = np.broadcast_to(np.asarray(track_max, float), (5,))
    period = 1.0 / rate_hz
    obs = robot.get_observation()
    q = np.asarray(obs.joint_positions, float)
    off = float(np.max(np.abs(q[:5] - np.asarray(start, float)[:5])))
    base = {"sent": 0, "max_err": [0.0] * 5}
    if off > start_tol:
        res = {**base, "ok": False, "reason": "not_at_start", "detail": {"off": round(off, 2), "measured": q.round(2).tolist()}}
        emit({"ev": "end", **res})
        return res
    cmd, target = q.copy(), q.copy()
    emit({"ev": "start", "q": q.round(3).tolist()})
    sent: list = []
    max_err = np.zeros(5)
    over = 0
    t0 = clock()
    last_msg = t0
    last_ts, last_change = obs.timestamp, t0
    t_next = t0
    ending = None
    i = 0
    try:
        while True:
            now = clock()
            while True:
                m = inbox()
                if m is None:
                    break
                last_msg = now
                if m.get("end"):
                    ending = ending or now + settle_s
                elif m.get("target") is not None and ending is None:
                    t_ = np.asarray(m["target"], float)
                    if t_.shape == (6,) and np.isfinite(t_).all():
                        target = t_
            if ending is not None:
                target = cmd
                if now >= ending:
                    break
            elif now - last_msg > 5 * host_timeout_s:
                raise Abort("host_silent", {"since_s": round(now - last_msg, 2)})
            elif now - last_msg > host_timeout_s:
                target = cmd   # 호스트가 조용하면 그 자리
            if now - t0 > max_s:
                raise Abort("servo_wall", {"max_s": max_s})
            d = target - cmd
            d[:5] = np.clip(d[:5], -step_body, step_body)
            d[5] = float(np.clip(d[5], -step_grip, step_grip))
            cmd = cmd + d
            robot.send_action(cmd, goal_time=period)
            sent.append(cmd.copy())
            t_next += period
            dt_ = t_next - clock()
            if dt_ > 0:
                sleep(dt_)
            obs = robot.get_observation()
            q_meas = np.asarray(obs.joint_positions, float)
            now = clock()
            if obs.timestamp != last_ts:
                last_ts, last_change = obs.timestamp, now
            elif now - last_change > stale_s:
                raise Abort("stale_state", {"tick": i, "since_s": round(now - last_change, 2)})
            ref = sent[max(0, len(sent) - 1 - lag_ticks)]
            err = np.abs(q_meas[:5] - ref[:5])
            max_err = np.maximum(max_err, err)
            over = over + 1 if (err > track_max).any() else 0
            if over >= track_ticks:
                raise Abort("tracking", {"tick": i, "joint": int(np.argmax(err / track_max)), "err": err.round(2).tolist()})
            if i % 3 == 0:
                emit({"ev": "q", "t": round(now - t0, 3), "q": q_meas.round(3).tolist()})
            i += 1
        res = {**base, "ok": True, "reason": "done", "sent": len(sent), "max_err": max_err.round(2).tolist(),
               "elapsed_s": round(clock() - t0, 2)}
    except Abort as e:
        res = {**base, "ok": False, "reason": e.reason, "detail": e.detail, "sent": len(sent),
               "max_err": max_err.round(2).tolist(), "elapsed_s": round(clock() - t0, 2)}
    emit({"ev": "end", **res})
    return res


def servo_main(payload: dict, robot_factory=None) -> int:
    """컨테이너 안 진입점(container_main 과 같은 연결·종료 규칙). stdin 줄마다 목표 JSON."""
    import collections as _c
    import threading as _t
    emit_json({"ev": "boot", "run_id": payload.get("run_id"), "python": sys.version.split()[0]})
    q_in = _c.deque()

    def _reader():
        for line in sys.stdin:
            try:
                q_in.append(json.loads(line))
            except ValueError:
                pass
        q_in.append({"end": 1})   # 호스트가 닫음 → 끝
    _t.Thread(target=_reader, daemon=True).start()
    robot = robot_factory() if robot_factory is not None else None
    if robot is None:
        from physicalai.robot import SharedRobot
        robot = SharedRobot.from_config(payload["driver"], name=payload["robot_name"])
    try:
        robot.connect()
    except Exception as e:  # noqa: BLE001
        emit_json({"ev": "end", "ok": False, "reason": "connect_refused", "error": f"{type(e).__name__}: {e}"})
        return 3
    emit_json({"ev": "connected", "robot": payload.get("robot_name")})
    try:
        res = servo_loop(robot, lambda: q_in.popleft() if q_in else None, emit=emit_json, **payload["limits"],
                         **payload["servo"])
    except Exception as e:  # noqa: BLE001
        emit_json({"ev": "end", "ok": False, "reason": "error", "error": f"{type(e).__name__}: {e}"})
        return 4
    finally:
        robot.disconnect()
    return 0 if res["ok"] else 2


def build_script(payload: dict) -> str:
    """stream.py 소스(Abort·emit_json) + 위 두 함수 + 실행 묶음."""
    parts = [(HERE / "stream.py").read_text(encoding="utf-8"), "import collections, threading\n",
             inspect.getsource(servo_loop), inspect.getsource(servo_main),
             f"PAYLOAD = json.loads({json.dumps(json.dumps(payload))})\n", "sys.exit(servo_main(PAYLOAD))\n"]
    return "\n".join(parts)


# ------------------------------------------------------------------ 호스트 쪽 연결

class RealLink:
    """컨테이너 servo_main 과 줄 단위로 주고받는다. q_now() = 마지막 측정(정규화 6값)."""

    def __init__(self, argv: list[str], script: str, log, rcfg: dict):
        self.log, self.rcfg = log, rcfg
        self.q, self.end, self.started = None, None, threading.Event()
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.proc.stdin.write(f"{len(script.encode('utf-8'))}\n{script}")
        self.proc.stdin.flush()
        self.t0 = time.monotonic()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            line = line.rstrip("\n")
            try:
                ev = json.loads(line)
            except ValueError:
                self.log("servo_text", text=line[-300:])
                continue
            if ev.get("ev") == "q":
                with self.lock:
                    self.q = np.asarray(ev["q"], float)
                continue
            if ev.get("ev") == "start":
                with self.lock:
                    self.q = np.asarray(ev["q"], float)
                self.started.set()
            if ev.get("ev") == "end":
                self.end = ev
                self.started.set()
            self.log("servo_stream", **ev)

    def ready(self, timeout: float = 8.0) -> bool:
        return self.started.wait(timeout) and self.end is None

    def alive(self) -> bool:
        return self.end is None and self.proc.poll() is None

    def q_now(self):
        with self.lock:
            return None if self.q is None else self.q.copy()

    def send(self, n6) -> bool:
        try:
            self.proc.stdin.write(json.dumps({"target": np.round(np.asarray(n6, float), 4).tolist()}) + "\n")
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def wait(self, until: float) -> None:
        dt_ = until - time.monotonic()
        if dt_ > 0:
            time.sleep(dt_)

    def now(self) -> float:
        return time.monotonic() - self.t0

    def close(self, timeout: float = 4.0) -> dict:
        try:
            self.proc.stdin.write(json.dumps({"end": 1}) + "\n")
            self.proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            rp._signal_runs(self.rcfg, "SIGKILL")
            self.proc.kill()
            self.proc.wait(5)
        if self.end is None:
            self.log("servo_lost", returncode=self.proc.returncode)
        return self.end or {"ok": False, "reason": "error"}


class FakeLink:
    """가짜 로봇(시험·--fake): 보낸 목표를 틱 한도 안에서 따라가 FakeWorld 의 자세(fw.q)로 둔다. 시간은 가짜 시계."""

    def __init__(self, s, fw, q0, grip, lim: dict):
        self.s, self.fw, self.grip = s, fw, grip
        self.cmd = s.q_to_n(q0, grip)
        self.target = self.cmd.copy()
        self.step = np.broadcast_to(np.asarray(lim["step_body"], float), (5,))
        self.rate = float(lim["rate_hz"])
        self.t = 0.0
        self.end = None
        fw.q = s.n_to_q(self.cmd)

    def ready(self, timeout: float = 8.0) -> bool:
        return True

    def alive(self) -> bool:
        return self.end is None

    def q_now(self):
        return self.cmd.copy()

    def send(self, n6) -> bool:
        self.target = np.asarray(n6, float)
        return True

    def wait(self, until: float) -> None:
        while self.t < until:
            d = self.target - self.cmd
            d[:5] = np.clip(d[:5], -self.step, self.step)
            self.cmd = self.cmd + d
            self.t += 1.0 / self.rate
        self.fw.q = self.s.n_to_q(self.cmd)

    def now(self) -> float:
        return self.t

    def close(self, timeout: float = 4.0) -> dict:
        self.wait(self.t + 0.3)
        self.end = {"ok": True, "reason": "done", "fake": True}
        return self.end


class FakeCam:
    def __init__(self, fw):
        self.fw = fw

    def read(self):
        return self.fw.capture()

    def close(self) -> None:
        pass


class RealCam:
    """손목 카메라를 한 번 열어 두고 뒤에서 계속 읽는다 — read() = 가장 새 장(버퍼에 쌓인 옛 장을 쓰지 않게)."""

    def __init__(self, warmup: int = 3):
        import sys
        sys.path.insert(0, str(HERE.parent.parent))
        from vision import camera
        self.st = camera.DeviceStream(dev=wrist.WRIST_CAM, warmup=warmup)
        self.frame, self.n, self.stop = None, 0, False
        self.cv = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while not self.stop:
            f = self.st.read()
            with self.cv:
                self.frame, self.n = f, self.n + 1
                self.cv.notify_all()

    def read(self, timeout: float = 1.0):
        with self.cv:
            n0 = self.n
            self.cv.wait_for(lambda: self.n > n0, timeout)   # 이 호출 뒤에 들어온 장
            return self.frame

    def close(self) -> None:
        self.stop = True
        time.sleep(0.1)
        self.st.close()


def _fake_world():
    """FakeWorld.install 이 wrist.capture 를 바꿔 두었으면 그 FakeWorld(가짜 실행·시험) — 아니면 None."""
    fw = getattr(wrist.capture, "__self__", None)
    return fw if type(fw).__name__ == "FakeWorld" else None


def open_link(s, ex, sf: dict, lim: dict, log, q0, grip, P: dict):
    """(link, None) 또는 (None, 거부 이유). 실제는 실행기 _run_real 과 같은 문: 잠금 → 환경·팔로워 → 이전 실행 →
    사전 점검 → stop 확인."""
    fw = _fake_world()
    if fw is not None:
        return FakeLink(s, fw, q0, grip, lim), None
    import os
    if os.environ.get("BALL_ROBOT") != "1":   # 실제 송신 문(실행기와 같음) — 가짜 판별이 빗나가도 로봇을 안 연다
        return None, "BALL_ROBOT=1 이 아님"
    import datetime as dt
    import fcntl
    import uuid
    rcfg = rp.load_config()
    rp.STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_f = open(rp.STATE_DIR / "run.lock", "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_f.close()
        return None, "already_running: 이 PC 에서 run_policy·좌표 집기가 실행 중"
    try:
        pid, env_id = ex._studio_env(rcfg, s.cfg.get("studio_environment"))
        _env, ia, follower, notes = rp.environment_robot(rcfg, pid, env_id, require_devices=True)
        rp._stop_requested.clear()
        (rp.STATE_DIR / "stop").unlink(missing_ok=True)
        prev = rp.wait_previous(rcfg)
        if not prev["ok"]:
            raise RuntimeError(f"previous_not_released: {prev['busy']}")
        problems, _info = rp.robot_preflight(rcfg, follower["port"], [follower["port"]])
        if problems:
            raise RuntimeError("; ".join(f"{c}: {m}" for c, m in problems))
        if rp._stop_flag():
            raise RuntimeError("시작 전에 stop 요청")
    except Exception as e:  # noqa: BLE001
        lock_f.close()
        return None, f"{type(e).__name__}: {e}"
    run_id = f"{dt.datetime.now(ex.KST):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    payload = {"run_id": run_id, "robot_name": rcfg["robot_name"], "driver": ia["robot"]["init_args"]["robot"],
               "limits": {k: lim[k] for k in ("rate_hz", "step_body", "step_grip", "track_max", "track_ticks",
                                              "lag_ticks", "stale_s", "start_tol")},
               "servo": {"start": s.q_to_n(q0, grip).tolist(), "max_s": float(P["max_s"]) + 10.0,
                         "host_timeout_s": float(P["host_timeout_s"]), "settle_s": float(P["settle_s"])}}
    argv = rp._docker(rcfg) + ["exec", "-i", "-u", "appuser", "-e", f"{rp.MARKER}={run_id}", "-w", "/tmp",
                               rcfg["container"], rp.CPY, "-I", "-c", BOOT, "coord_pick", run_id]
    log("servo_launch", run_id=run_id, follower=follower)
    (rp.STATE_DIR / "current.json").write_text(json.dumps(
        {"run_id": run_id, "name": "coord_pick:servo", "started": dt.datetime.now(ex.KST).isoformat(timespec="seconds")}),
        encoding="utf-8")
    link = RealLink(argv, build_script(payload), log, rcfg)
    link.lock_f = lock_f
    return link, None


def open_cam(P: dict):
    import os
    fw = _fake_world()
    if fw is not None:
        return FakeCam(fw)
    if os.environ.get("BALL_ROBOT") != "1":
        raise RuntimeError("BALL_ROBOT=1 이 아님 — 손목 카메라 안 엶")
    return RealCam(int(P["warmup"]))


# ------------------------------------------------------------------ 호스트 제어 루프

@dataclasses.dataclass
class Ctx:
    """실행기가 넘기는 것(그 블록의 자세·목표·검사)."""
    s: object
    ex: object
    cfg: dict
    sf: dict
    log: object
    tag: str
    grip: float
    yaw: float
    tilt: float            # 지금(hover) 기울기
    tilt_plan: float       # 계획 집기 기울기(내려가며 이쪽으로)
    tilt_max: float
    z_from: float          # hover 높이(집게 가운데 mm)
    z_to: float            # 닫기 직전 높이
    color: str
    sm: object
    want_fn: object        # dz → 화면 목표점(px) — 실행기 near_want(닫기 직전 기준)
    blob_fn: object        # (img, near_px) → 덩어리 | None
    check_fn: object       # (N,6) 정규화 목표 → 위반 목록
    speed_deg_s: float


def descend(c: Ctx, q0: np.ndarray, tw0) -> dict:
    """{ok, q(끝 자세), tw(블록 작업대 xy 추정), why, rec}. 실패면 처음 높이로 올라가 끝낸 자세를 q 로."""
    s, ex = c.s, c.ex
    P = {**SERVO_DEFAULTS, **(c.cfg.get("servo") or {})}
    lim = ex.limits(s, c.sf, c.speed_deg_s)
    rec = {"frames": 0, "lost": 0, "steps": [], "params": {k: P[k] for k in ("vz_mm_s", "vz_slow_mm_s", "slow_mm", "tol_mm")}}
    out = {"ok": False, "q": q0, "tw": np.asarray(tw0, float), "why": "", "rec": rec}
    link, why = open_link(s, ex, c.sf, lim, c.log, q0, c.grip, P)
    lock_f = getattr(link, "lock_f", None)
    if link is None:
        out["why"] = f"송신기 못 띄움: {why}"
        c.log("servo", tag=c.tag, ok=False, why=out["why"])
        return out
    cam = None
    q_cmd = np.asarray(q0, float).copy()
    try:
        if not link.ready():
            raise _Stop(f"송신기 시작 안 됨: {(link.end or {}).get('reason')}")
        cam = open_cam(P)
        J_ = ex._jac
        tw_est = np.asarray(tw0, float).copy()
        z = float(s.tcp_pos(q_cmd)[2])
        tilt_cur = float(c.tilt)
        lost = good = 0
        first = True
        dt_ = 1.0 / float(P["rate_hz"])
        t_next = link.now()
        while True:
            t = link.now()
            if t > float(P["max_s"]):
                raise _Stop(f"{P['max_s']:g}초 안에 못 맞춤")
            if not link.alive():
                raise _Stop(f"송신기 끝남({(link.end or {}).get('reason')})")
            if rp._stop_flag():
                raise _Stop("stop 요청", stopped=True)
            img = cam.read()
            n_ph = link.q_now()
            q_ph = q_cmd if n_ph is None else s.n_to_q(n_ph)
            T = s.kin.fk(q_ph)
            tcp = (T[:3, 3] + T[:3, :3] @ s.tcp) * 1000
            dz = float((wrist.BLOCK_Z_MM - tcp[2]) * T[2, 2])
            want = np.asarray(c.want_fn(dz), float)
            b = None if img is None else c.blob_fn(img, want)
            rec["frames"] += 1
            if b is None:
                lost += 1
                rec["lost"] += 1
                if lost >= int(P["lost_max"]):
                    raise _Stop(f"손목 사진에 블록이 {lost}장 연속 없음")
                link.send(s.q_to_n(q_cmd, c.grip))   # 그 자리
                t_next += dt_
                link.wait(t_next)
                continue
            lost = 0
            e = np.asarray(b["px"], float) - want
            d = np.linalg.solve(J_(c.sm, dz), e)
            est = s.frames.r2w(tcp[:2]) + s.frames.R @ (T[:2, :2] @ d)
            if first:   # 첫 장은 튐 검사 없이 추정 시작(실제 화면 감도가 모형의 1.4~1.9배 — 첫 추정이 위 카메라 자리와 멀 수 있다)
                tw_est, first = est, False
            else:
                jump = float(np.linalg.norm(est - tw_est))
                if jump > float(P["jump_mm"]):
                    raise _Stop(f"블록 추정 자리가 {jump:.0f}mm 튐(사진 {rec['frames']})")
                tw_est = tw_est + float(P["alpha"]) * (est - tw_est)
            cur_w = s.frames.r2w(s.tcp_pos(q_cmd)[:2])
            off = tw_est - cur_w
            e_mm = float(np.linalg.norm(off))
            rec["steps"].append([round(t, 2), round(z, 1), round(e_mm, 1), np.round(e, 0).tolist()])
            if z <= c.z_to + 0.5 and e_mm <= float(P["tol_mm"]):
                good += 1
                if good >= 2:
                    break
            else:
                good = 0
            vz = 0.0 if e_mm > float(P["gate_mm"]) else (
                float(P["vz_slow_mm_s"]) if z - c.z_to < float(P["slow_mm"]) else float(P["vz_mm_s"]))
            step = off * min(1.0, float(P["vxy_mm_s"]) * dt_ / e_mm) if e_mm > 1e-6 else np.zeros(2)
            z_new = max(float(c.z_to), z - vz * dt_)
            xy_new = cur_w + step
            # 기울기는 내려가며 계획 기울기 쪽으로(되면 그대로, 안 되면 grasp 한도 안 가까운 것)
            qn, info = s.solve(s.w_to_r3(xy_new, z_new), c.yaw, q_cmd, strict=True, tilt=c.tilt_plan, only_prev=True)
            if qn is None:
                qn, info = s.solve(s.w_to_r3(xy_new, z_new), c.yaw, q_cmd, strict=True, tilt=None, tilt_max=c.tilt_max,
                                   near=tilt_cur, only_prev=True)
            if qn is None:
                raise _Stop(f"자세 계산 실패(높이 {z_new:.0f}mm)")
            viol = c.check_fn(np.vstack([s.q_to_n(q_cmd, c.grip), s.q_to_n(qn, c.grip)]))
            if viol:
                raise _Stop("작업 공간: " + "; ".join(str(v.get("kind")) for v in viol))
            if not link.send(s.q_to_n(qn, c.grip)):
                raise _Stop("송신기에 목표를 못 보냄")
            q_cmd, z, tilt_cur = qn, z_new, float(info.get("tilt_used", tilt_cur))
            t_next += dt_
            link.wait(t_next)
        end = link.close()
        link = None
        # tw = 지금 집게 가운데(맞춤이 tw + 이동으로 고치므로) — 블록 추정과 tol_mm 안
        out.update(ok=bool(end.get("ok")), q=q_cmd, tw=s.frames.r2w(s.tcp_pos(q_cmd)[:2]), tilt=tilt_cur,
                   why="" if end.get("ok") else f"송신기 {end.get('reason')}", time_s=round(t, 2), block_w=tw_est)
    except Exception as st:  # noqa: BLE001 — _Stop(멈춤 규칙) 또는 카메라·계산 오류: 그 자리에서 올라가 지금 방식으로
        out["why"] = str(st) if isinstance(st, _Stop) else f"{type(st).__name__}: {st}"
        out["stopped"] = getattr(st, "stopped", False)
        if link is not None and link.alive() and not out["stopped"]:
            # 그 자리에서 처음 높이로 수직으로 올라가 끝낸다 → 실행기가 지금 방식(hover 맞춤)으로 잇는다
            cur = s.tcp_pos(q_cmd)
            qu, _ = s.solve(np.r_[cur[:2], max(float(c.z_from), float(cur[2]))], c.yaw, q_cmd, strict=True, tilt=None,
                            tilt_max=c.tilt_max, near=c.tilt, only_prev=True)
            if qu is not None and not c.check_fn(np.linspace(s.q_to_n(q_cmd, c.grip), s.q_to_n(qu, c.grip), 8)):
                link.send(s.q_to_n(qu, c.grip))
                t_up = link.now() + 0.5 + float(np.degrees(np.max(np.abs(qu - q_cmd)))) / max(1.0, c.speed_deg_s) * 1.3
                while link.now() < t_up and link.alive():
                    link.send(s.q_to_n(qu, c.grip))
                    link.wait(link.now() + 0.1)
                q_cmd = qu
        if link is not None:
            end = link.close()
            link = None
            if not end.get("ok"):
                out["why"] += f" · 송신기 {end.get('reason')}"
        out["q"] = q_cmd
        out["tilt"] = float(math.acos(max(-1.0, min(1.0, -float(s.kin.fk(q_cmd)[2, 2])))))
        out["tw"] = s.frames.r2w(s.tcp_pos(q_cmd)[:2])   # 그 자리(hover 맞춤이 여기서부터 사진으로 고친다)
    finally:
        if cam is not None:
            cam.close()
        if link is not None:
            link.close()
        if lock_f is not None:
            (rp.STATE_DIR / "current.json").unlink(missing_ok=True)
            lock_f.close()
    c.log("servo", tag=c.tag, ok=out["ok"], why=out["why"], frames=rec["frames"], lost=rec["lost"],
          time_s=out.get("time_s"), steps=rec["steps"][-12:])
    return out


class _Stop(Exception):
    def __init__(self, why: str, stopped: bool = False):
        super().__init__(why)
        self.stopped = stopped


def ik_descend(*, s, ex, cfg: dict, sf: dict, W: dict, log, tag: str, q, grip: float, yaw: float, tilt: float,
               tilt_plan: float, tilt_max: float, tw, zh: float, gz_add: float, color: str, cls: str, sm, cref: dict,
               allow: dict, speed: float) -> dict:
    """실행기(IK, grasp.servo)가 hover 맞춤 자리에서 부른다. 목표 높이·화면 기준은 실행기 '닫기 직전' 과 같은 계산.
    skipped = 내려가지 않음(팔 그대로). 그 밖의 실패는 hover 높이로 올라와 있다."""
    gz, qg = ex._grasp_z(s, cfg, tw, yaw, tilt_plan, q, z_add=gz_add)
    if qg is None:
        return {"ok": False, "skipped": True, "why": "같은 롤 갈래로 집기 높이 자세 없음", "q": q, "tw": tw, "tilt": tilt}
    dz_g = float((wrist.BLOCK_Z_MM - s.tcp_pos(qg)[2]) * s.kin.fk(qg)[2, 2])
    close_px = np.array([cref["close"]["u"]["med"], cref["close"]["v"]["med"]], float)
    near_want = (lambda dz: close_px + sm.predict([[0.0, 0.0, dz]])[0] - sm.predict([[0.0, 0.0, dz_g]])[0])
    polys, _ = ex.forbidden_polygons(s, sf)
    c = Ctx(s=s, ex=ex, cfg=cfg, sf=sf, log=log, tag=tag, grip=grip, yaw=yaw, tilt=tilt, tilt_plan=tilt_plan,
            tilt_max=tilt_max, z_from=zh, z_to=gz + W["near_mm"], color=color, sm=sm, want_fn=near_want,
            blob_fn=lambda img, near: wrist.blob(img, color, near=near, merge=cls.endswith("defect"), pale=color == "blue"),
            check_fn=lambda n: ex.check_workspace(s, np.asarray(n, float), sf, polys, dict(allow) or None),
            speed_deg_s=speed)
    r = descend(c, q, tw)
    r["z_to"] = round(c.z_to, 1)
    return r
