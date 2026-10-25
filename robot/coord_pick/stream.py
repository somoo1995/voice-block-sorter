"""좌표 집기 실행 루프 — 관절 목표(정규화 6값)를 일정 주기로 보내며 안전장치를 건다.

이 파일은 두 곳에서 돈다(그래서 표준 라이브러리 + numpy 만 쓴다):
- Studio 컨테이너 안: executor.py 가 이 소스 + 실행 묶음(JSON)을 `python -I - coord_pick <run_id>` 의 stdin 으로 넣는다.
  컨테이너에 파일을 쓰지 않는다. 로봇 = physicalai.robot.SharedRobot(정책 실행과 같은 소유 프로세스·장치 잠금).
- 호스트 시험: fake.py 의 가짜 로봇·가짜 시계로 같은 루프를 돌린다(selftest).

안전장치(넘으면 그 틱에서 멈추고 한 번 보내 그 자리에 둔다 — 몸통은 측정 자세(상태 끊김이면 마지막 명령), 그리퍼는 마지막 명령):
- 시작 자세: 측정 자세가 궤적 첫 목표(홈)에서 start_tol 넘게 떨어져 있으면 아무것도 안 보내고 거부.
- 틱당 변화: 보내기 전에 전체(시작 이음 포함)를 검사 → 한도를 넘는 틱이 하나라도 있으면 아무것도 안 보내고 거부.
- 추종 오차: 측정값과 lag_ticks 전에 보낸 명령의 차가 track_max 를 track_ticks 틱 연속으로 넘으면 중단(그리퍼 제외 —
  블록을 쥐면 목표에 못 가는 게 정상).
- 상태 끊김: 소유 프로세스의 관측 시각이 stale_s 동안 안 바뀌면 중단.
- SIGINT(stop(home=True)) → 그 자리 유지 후 끝. SIGKILL(stop()) → 프로세스가 바로 죽고 소유 프로세스가 마지막 목표를 유지.
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np


class Abort(Exception):
    def __init__(self, reason: str, detail: dict):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def emit_json(ev: dict) -> None:
    print(json.dumps(ev, ensure_ascii=False), flush=True)


def ramp(a: np.ndarray, b: np.ndarray, step_body: np.ndarray, step_grip: float) -> list[np.ndarray]:
    """a → b 직선(관절 공간), 모든 관절을 같은 비율로 — 틱당 한도 안에서 가장 짧게. a 는 빼고 b 는 넣는다."""
    d = b - a
    k = max(1, int(math.ceil(max(float(np.max(np.abs(d[:5]) / step_body)), abs(float(d[5])) / step_grip) - 1e-9)))
    return [a + d * i / k for i in range(1, k + 1)]


def check_steps(seq: np.ndarray, step_body: np.ndarray, step_grip: float) -> int | None:
    """틱당 변화 한도를 넘는 첫 번호(seq[i] − seq[i−1]), 없으면 None. 숫자가 아닌 값(NaN·inf)도 걸린다."""
    if not np.isfinite(seq).all():
        return int(np.nonzero(~np.isfinite(seq).all(axis=1))[0][0])
    d = np.abs(np.diff(seq, axis=0))
    bad = np.nonzero((d[:, :5] > step_body * 1.001 + 1e-9).any(axis=1) | (d[:, 5] > step_grip * 1.001 + 1e-9))[0]
    return int(bad[0]) + 1 if len(bad) else None


def stream(robot, targets, marks=(), *, rate_hz: float, step_body, step_grip: float, track_max, track_ticks: int,
           lag_ticks: int, stale_s: float, start_tol: float, settle_s: float = 0.5, emit=emit_json,
           clock=time.monotonic, sleep=time.sleep) -> dict:
    """targets (N,6) 정규화를 rate_hz 로 보낸다. step_body·track_max 는 몸통 5관절(정규화 단위) 배열 또는 수.

    반환 {ok, reason, sent, total, max_err, ...}. reason: done | not_at_home | step_limit | tracking | stale_state |
    interrupted | error.
    """
    targets = np.asarray(targets, float)
    step_body = np.broadcast_to(np.asarray(step_body, float), (5,))
    track_max = np.broadcast_to(np.asarray(track_max, float), (5,))
    period = 1.0 / rate_hz
    obs = robot.get_observation()
    q = np.asarray(obs.joint_positions, float)
    off = float(np.max(np.abs(q[:5] - targets[0, :5])))
    base = {"total": len(targets), "sent": 0, "max_err": [0.0] * 5}
    if off > start_tol:
        res = {**base, "ok": False, "reason": "not_at_home", "detail": {"off": round(off, 2), "measured": q.round(2).tolist()}}
        emit({"ev": "end", **res})
        return res
    pre = ramp(q, targets[0], step_body, step_grip)
    seq = np.vstack([q[None, :]] + pre + list(targets[1:]))  # seq[0] = 지금 자세(보내지 않음)
    bad = check_steps(seq, step_body, step_grip)
    if bad is not None:
        res = {**base, "ok": False, "reason": "step_limit", "detail": {"tick": bad}}
        emit({"ev": "end", **res})
        return res
    seq = seq[1:]
    shift = len(pre) - 1  # targets[i] 가 seq 의 몇 번째인가 = i + shift
    markmap = {i + shift: label for i, label in marks}
    emit({"ev": "start", "ticks": len(seq), "pre_ticks": len(pre), "duration_s": round(len(seq) * period, 1)})

    sent: list[np.ndarray] = []
    max_err = np.zeros(5)
    over = 0
    last_ts, last_change = obs.timestamp, clock()
    t_next = clock()
    t0 = t_next
    q_meas = q
    try:
        for i, cmd in enumerate(seq):
            robot.send_action(cmd, goal_time=period)
            sent.append(cmd)
            if i in markmap:
                emit({"ev": "step", "i": i, "label": markmap[i], "t": round(clock() - t0, 2),
                      "measured": q_meas.round(1).tolist()})
            t_next += period
            dt = t_next - clock()
            if dt > 0:
                sleep(dt)
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
                j = int(np.argmax(err / track_max))
                raise Abort("tracking", {"tick": i, "joint": j, "err": err.round(2).tolist()})
        # 마지막 목표에서 잠깐 기다리며 한 번 더 본다
        end = clock() + settle_s
        while clock() < end:
            sleep(period)
            q_meas = np.asarray(robot.get_observation().joint_positions, float)
            err = np.abs(q_meas[:5] - sent[-1][:5])
            max_err = np.maximum(max_err, err)
            over = over + 1 if (err > track_max).any() else 0
            if over >= track_ticks:
                raise Abort("tracking", {"tick": len(seq), "joint": int(np.argmax(err / track_max)),
                                         "err": err.round(2).tolist()})
        res = {**base, "ok": True, "reason": "done", "sent": len(sent), "max_err": max_err.round(2).tolist(),
               "elapsed_s": round(clock() - t0, 1)}
    except Abort as e:
        # 추종 오차 = 팔이 막힌 것 → 측정 자세로(더 밀지 않게). 상태 끊김 = 측정값이 오래됨 → 마지막 명령 그대로(튀지 않게)
        body = q_meas if e.reason == "tracking" or not sent else sent[-1]
        hold(robot, body, sent[-1] if sent else q_meas, period)
        res = {**base, "ok": False, "reason": e.reason, "detail": e.detail, "sent": len(sent),
               "max_err": max_err.round(2).tolist(), "elapsed_s": round(clock() - t0, 1)}
    except KeyboardInterrupt:  # stop(home=True) 의 SIGINT
        hold(robot, q_meas, sent[-1] if sent else q_meas, period)
        res = {**base, "ok": False, "reason": "interrupted", "sent": len(sent), "max_err": max_err.round(2).tolist(),
               "elapsed_s": round(clock() - t0, 1)}
    emit({"ev": "end", **res})
    return res


def hold(robot, q_meas: np.ndarray, last_cmd: np.ndarray, period: float) -> None:
    """몸통 = 측정 자세, 그리퍼 = 마지막 명령(쥔 블록을 놓지 않게)."""
    try:
        robot.send_action(np.r_[np.asarray(q_meas)[:5], np.asarray(last_cmd)[5]], goal_time=period)
    except Exception as e:  # noqa: BLE001 — 멈추는 중 오류는 기록만
        emit_json({"ev": "hold_failed", "error": f"{type(e).__name__}: {e}"})


def container_return(payload: dict, robot_factory=None, *, clock=None, sleep=None) -> int:
    """지금 자세에서 payload["target"] 까지 관절 공간 직선으로 천천히 간다 — 정책이 끝난 뒤 시작 자세로(run_policy 2.3,
    끝 자세가 멀 때만), executor home(go_home). 이어 붙인 궤적을 stream() 의 안전장치(틱 한도·추종 오차·상태 끊김)로 보낸다.
    이미 already_tol 안이면 아무것도 안 보낸다. max_off(정규화)보다 멀면 거부(too_far). keep_gripper 면 그리퍼는 지금 값.
    종료 코드는 container_main 과 같다."""
    emit_json({"ev": "boot", "run_id": payload.get("run_id"), "mode": "return"})
    if robot_factory is not None:
        robot = robot_factory()
    else:
        from physicalai.robot import SharedRobot
        robot = SharedRobot.from_config(payload["driver"], name=payload["robot_name"])
    try:
        robot.connect()
    except Exception as e:  # noqa: BLE001
        emit_json({"ev": "end", "ok": False, "reason": "connect_refused", "error": f"{type(e).__name__}: {e}"})
        return 3
    try:
        lim = payload["limits"]
        q = np.asarray(robot.get_observation().joint_positions, float)
        target = np.array(payload["target"], float)
        if payload.get("keep_gripper"):  # 그리퍼는 지금 값 그대로(executor home 기본)
            target[5] = q[5]
        off = float(np.max(np.abs(q[:5] - target[:5])))
        emit_json({"ev": "measured", "q": [round(float(v), 2) for v in q], "off": round(off, 2)})
        if off <= float(payload.get("already_tol", 0.0)):
            emit_json({"ev": "end", "ok": True, "reason": "already_home", "off": round(off, 2)})
            return 0
        if payload.get("max_off") is not None and off > float(payload["max_off"]):  # 너무 멀면 안 보낸다
            emit_json({"ev": "end", "ok": False, "reason": "too_far", "off": round(off, 2),
                       "max_off": payload["max_off"], "sent": 0})
            return 2
        seq = np.vstack([q[None, :]] + ramp(q, target, np.broadcast_to(np.asarray(lim["step_body"], float), (5,)),
                                            float(lim["step_grip"])))
        kw = {k: v for k, v in (("clock", clock), ("sleep", sleep)) if v is not None}  # 시험용 가짜 시계
        res = stream(robot, seq, [(0, "복귀 시작"), (len(seq) - 1, "복귀 끝")], **lim, **kw)
    except Exception as e:  # noqa: BLE001
        emit_json({"ev": "end", "ok": False, "reason": "error", "error": f"{type(e).__name__}: {e}"})
        return 4
    finally:
        robot.disconnect()
    return 0 if res["ok"] else 2


def container_main(payload: dict, robot_factory=None) -> int:
    """컨테이너 안 진입점. 종료 코드 0 = done, 2 = 안전장치·멈춤으로 중단, 3 = 연결 거부, 4 = 오류.
    robot_factory: 호스트 시험용 가짜 로봇(executor 가 fake.py 소스를 함께 넣을 때). 없으면 SharedRobot."""
    emit_json({"ev": "boot", "run_id": payload.get("run_id"), "python": sys.version.split()[0]})
    if robot_factory is not None:
        robot = robot_factory()
    else:
        from physicalai.robot import SharedRobot
        robot = SharedRobot.from_config(payload["driver"], name=payload["robot_name"])
    try:
        robot.connect()
    except Exception as e:  # noqa: BLE001 — RobotDeviceAlreadyOwned 등: 움직이기 전에 실패
        emit_json({"ev": "end", "ok": False, "reason": "connect_refused", "error": f"{type(e).__name__}: {e}"})
        return 3
    emit_json({"ev": "connected", "robot": payload.get("robot_name")})
    try:
        res = stream(robot, payload["targets"], [tuple(m) for m in payload["marks"]], **payload["limits"])
    except Exception as e:  # noqa: BLE001
        emit_json({"ev": "end", "ok": False, "reason": "error", "error": f"{type(e).__name__}: {e}"})
        return 4
    finally:
        robot.disconnect()  # 소유 프로세스는 마지막 목표를 유지하고, 10초 뒤 토크를 켠 채 끝난다
    return 0 if res["ok"] else 2
