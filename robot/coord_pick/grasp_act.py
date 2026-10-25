"""하이브리드 집기(10-23): YOLO 로 찾고 IK 로 블록 위(시작 자세)까지 → ACT 집기 정책(policies.json grasp)이 내려가 쥐고 들기
→ IK 로 그릇에 나르기·놓기·홈. 실행기(_sort_wrist)가 config grasp.mode = "act" 일 때 부른다. 에이전트 인터페이스는 그대로.

공중 넘겨주기(IK 송신 → Studio 런타임 → 다시 IK 송신):
- 둘 다 같은 이름(robot_name)의 SharedRobot 이라 같은 소유 프로세스(_owner_worker)에 붙는다. 소유 프로세스는 /state 구독자가
  하나도 없으면 idle_timeout(기본 10초) 뒤 스스로 끝난다(끝날 때 토크는 켠 채 자세 유지). 그 뒤 새로 붙으면 SO101.connect() 가
  레지스터를 쓰려고 **모든 관절 토크를 잠깐 끈다**(_configure_servos) → 블록 위 공중에서 팔이 처지고, 다시 켜질 때 마지막 목표로 튄다.
- 그래서 넘겨주는 동안 컨테이너 안에 '유지 구독자'(OwnerKeeper)를 띄운다: SharedRobot.attach(이름) 으로 /state 만 구독하고
  동작은 보내지 않는다. 소유 프로세스가 끝나지 않으니 정책·IK 송신이 재연결 없이 그대로 붙는다. attach 만 하므로 소유 프로세스를
  새로 만들지 않고(없으면 실패 → 정책을 돌리지 않고 IK 집기로), 프로세스 이름에 coord_pick 이 없어 run_policy 의 '실행 중'
  판정(wait_previous·preflight)에 걸리지 않는다. 입력(stdin)이 닫히거나 max_s 가 지나면 스스로 끝난다.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time

import numpy as np

from robot import run_policy as rp

GRASP_DEFAULTS = {
    "mode": "ik",            # "ik" = 지금 IK 집기(손목 확인), "act" = 블록 위까지 IK → 집기 정책 → IK 나르기
    "policy": "grasp",       # policies.json 정책 이름(집기 정책 — 10-23 녹화·학습 후 --link)
    "timeout_s": 10,         # 정책 제어 루프 상한(에피소드 약 5초 + 여유). 끝 감지(닫고·들고·0.5초 정지)가 먼저 끝낸다
    "start_z_mm": 90,        # 정책 시작 자세 집게 가운데 높이(끝면으로 선 블록 꼭대기 54mm(10-23 실측) + 집게 끝 6~25mm 여유)
    "start_back_mm": 0.0,    # 시작 자세 집게 가운데를 블록에서 로봇 쪽으로(+) 이만큼(녹화 사람 중앙 33mm, 10-23 grasp 39개)
    "start_side_mm": 0.0,    # 시작 자세 옆(로봇에서 블록 쪽을 볼 때 왼쪽 +) 이만큼(녹화 중앙 −17mm)
    "start_tilt_deg": None,  # 시작 자세에서 먼저 해 볼 기울기(녹화 중앙 22°). None = 계획 집기 기울기
    # 시작 장면 맞춤(10-23 12:59 실물 뒤 — 정책을 켜기 전에 손목 화면을 녹화 시작 프레임처럼): "none" | "image".
    # image = ① 손목 화면 블록 긴 축이 start_axis_deg(0 = 가로)에서 start_axis_tol_deg 넘게 벗어나면 손목 롤을 돌리고(실제로
    # 돌려 본 배율로 한 번 더) ② 블록 가운데를 start_px 로 맞춘다(start_align_iters 번, ±start_tol_px). start_px 기본 =
    # grasp 녹화 39개 중 35개 첫 프레임 덩어리 중앙(198, 231 — 기준 사진 노란 원 202, 250 근처, 사분위 u 167~216 · v 194~261,
    # 연한 초록은 Lab 마스크로). 긴 축은 사람도 중앙 8° · 사분위 −22~30° 로 흩어졌지만 '가로'로 맞춘다(±10°).
    # 사람 시작 관절값과 가장 큰 관절 차 중앙: 이 맞춤(근사) 3.2 · 비킴 33/−17·22° 5.8(정규화, 1 = 1.8°).
    # 시작 사진에 블록이 없으면 정책을 켜지 않고 IK 집기로
    "start_align": "none",
    "start_px": [198.0, 231.0],
    "start_axis_deg": 0.0,
    "start_axis_tol_deg": 10.0,
    "start_align_iters": 3,
    "start_tol_px": 15.0,
    "act_align_iters": 1,    # 시작 자세에서 손목 사진으로 대강 맞추는 횟수(정책이 내려가며 마저 맞춘다)
    "approach_safe_mm": None,  # IK 집기도 블록 위 이 높이로 먼저 온 뒤 수직으로 hover 로(옆에서 쓸고 들어오지 않게). None = 예전
    "keeper_max_s": 120,     # 유지 구독자 수명 상한
    "fallback_ik": True,     # 정책이 못 쥐면(끝 감지 없음·손목 사진 기준 밖) 벌리고 올라가 IK 집기로 한 번
}

KEEPER_CODE = r"""
import json, sys, threading, time
from physicalai.robot import SharedRobot
name, max_s = sys.argv[1], float(sys.argv[2])
r = SharedRobot.attach(name, connect_timeout=5.0)
try:
    r.connect()
except Exception as e:
    print(json.dumps({"ev": "keeper_error", "error": f"{type(e).__name__}: {e}"}), flush=True)
    sys.exit(2)
obs = r.get_observation()
q = getattr(obs, "joint_positions", None)
q = getattr(obs, "state", None) if q is None else q
print(json.dumps({"ev": "keeper_attached", "q": None if q is None else [round(float(v), 2) for v in q]}), flush=True)
done = threading.Event()
threading.Thread(target=lambda: (sys.stdin.read(), done.set()), daemon=True).start()
t0 = time.monotonic()
while not done.is_set() and time.monotonic() - t0 < max_s:
    r.get_observation()
    time.sleep(0.2)
r.disconnect()
print(json.dumps({"ev": "keeper_end", "s": round(time.monotonic() - t0, 1)}), flush=True)
"""


class OwnerKeeper:
    """넘겨주기 동안 소유 프로세스를 붙잡는 읽기 전용 구독자(컨테이너 안). start() 는 붙었으면 True."""

    def __init__(self, log=None, *, max_s: float = 120.0, runner=None):
        self.log = log or (lambda *a, **k: None)
        self.max_s = max_s
        self.proc: subprocess.Popen | None = None
        self.lines: list[dict] = []
        self.runner = runner      # 시험용: argv → Popen 비슷한 것

    def start(self, timeout: float = 8.0) -> bool:
        rcfg = rp.load_config()
        argv = rp._docker(rcfg) + ["exec", "-i", "-u", "appuser", "-w", "/tmp", rcfg["container"], rp.CPY, "-I", "-c",
                                   KEEPER_CODE, rcfg["robot_name"], str(self.max_s)]
        self.proc = (self.runner or (lambda a: subprocess.Popen(a, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                                 stderr=subprocess.DEVNULL, text=True)))(argv)
        got = threading.Event()

        def reader():
            for line in self.proc.stdout:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                self.lines.append(ev)
                self.log("keeper", **ev)
                if ev.get("ev") in ("keeper_attached", "keeper_error"):
                    got.set()
            got.set()

        threading.Thread(target=reader, daemon=True).start()
        got.wait(timeout)
        ok = any(e.get("ev") == "keeper_attached" for e in self.lines)
        if not ok:
            self.stop()
        return ok

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None


def run_grasp_policy(policy: str, timeout_s: float, log, runner=None) -> dict:
    """집기 정책 한 번(복귀 없음). 반환 {ok, reason, end_q(정규화 6), early_stop, log_path, error, time_s}."""
    t0 = time.monotonic()
    r = (runner or rp.run_policy)(policy, timeout_s=timeout_s, return_home=False)
    es = r.get("early_stop") or {}
    out = {"ok": r.get("reason") == "grasp_done", "reason": r.get("reason"), "end_q": r.get("end_q"),
           "early_stop": es, "log_path": r.get("log_path"), "error": r.get("error"),
           "time_s": round(time.monotonic() - t0, 1)}
    log("act_policy", **out)
    return out


def standing_hint(blob: dict | None, ratio_max: float | None) -> bool | None:
    """손목 사진 덩어리의 가로세로비로 '끝면으로 선 블록' 의심. ratio_max 가 없으면 판정 안 함(None).
    10-23 실측(16×30×54)으로 전제('끝면은 네모에 가깝다')가 틀렸다: 끝면 30×16 = 1.9, 눕힌 윗면 54×30 = 1.8 —
    끈 채 둔다(GRASP_ACT.md 7절 ②). 집기 녹화 시작 프레임으로 다른 기준이 갈리면 그때 바꾼다."""
    if ratio_max is None or blob is None:
        return None
    return float(blob.get("aspect", 9)) <= ratio_max


def block_time_estimate(ik_hover_to_lift_s: float = 15.0, policy_load_s: float = 4.1, policy_motion_s: float = 4.5,
                        policy_end_s: float = 1.1, start_extra_s: float = 1.5) -> dict:
    """블록 하나 집기 구간(블록 위 도착 → 들어 올림 끝) 시간 비교. 기본값 = 10-22 실측:
    IK(손목 맞춤·near 맞춤·내려감·판정·집고 들기 5구간) 18:00 실물 15.0초, 정책 적재(첫 틱까지) ACT GPU 4.05~4.33초,
    끝난 뒤 카메라 놓기·결과 1.1초, 정책 동작 = 녹화 사람 내려감 1.0 + 닫기 0.2 + 들기 0.9 + 정지 0.5 + 시작 대기 약 1.5,
    start_extra = 시작 높이 90mm 에서 더 내려가는 시간."""
    act = policy_load_s + policy_motion_s + policy_end_s + start_extra_s
    return {"ik_s": ik_hover_to_lift_s, "act_s": round(act, 1), "act_breakdown": {
        "load_s": policy_load_s, "motion_s": policy_motion_s, "end_s": policy_end_s, "start_extra_s": start_extra_s}}


def end_pose(s, res: dict, q_fallback):
    """정책 끝 자세(정규화 6) → (q 라디안, 그리퍼). 없으면 (q_fallback, None)."""
    if res.get("end_q") is None:
        return q_fallback, None
    n = np.asarray(res["end_q"], float)
    return s.n_to_q(n), float(n[5])
