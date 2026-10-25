"""가짜 vision·robot. BALL_MOCK=1 이거나 실제 모듈 import 가 실패하면 backends.py 가 이것을 쓴다.

작업대 상태(구역별 블록 개수·구역 밖에 떨어진 블록·손)를 메모리에서 흉내 낸다. 함수 모양은 계약서 3절과 같다.
블록 색: red_* = 분홍, blue_* = 초록(계약서 머리말, 코드 식별자는 그대로). 흰 무광 식탁보 위 파스텔 블록으로 그린다.
환경변수:
  BALL_MOCK_SCENE       처음 장면: demo4(기본, 발표 실연 — 투입 분홍 2·초록 1·불량 1) | full(시연 전체 흐름 11개)
  BALL_MOCK_POLICY_S    정책 1회 걸리는 시간(초, 기본 0.3)
  BALL_MOCK_FAIL_AT     블록을 놓치는 시도 번호(쉼표 구분, 기본 "3"). 빈 값이면 실패 없음. 놓치면 작업대가 그대로다
  BALL_MOCK_DROP_AT     블록을 집었다가 구역 밖(outside)에 떨어뜨리는 시도 번호(쉼표 구분, 기본 없음)
  BALL_MOCK_ARRANGE_SCALE  좌표 집기(숫자 만들기·그릇 옮기기) 가짜 실행 시간 = 예상 시간 × 이 값(기본 0.02, 3분 → 3.6초)
  BALL_MOCK_COORD       그릇 옮기기(sort_blocks) 계획: 비면 흉내(블록당 BALL_MOCK_COORD_BLOCK_S 초, 기본 40),
                        robot = 로봇 담당 sort_blocks(dry_run=True) 로 계획(가짜 사진 자리를 작업대 투입 구역으로 옮겨)
  BALL_MOCK_COORD_FAIL  unreachable·violation(계획 거부) | tracking(실행 중 끊김) | miss(끝까지 돌았는데 마지막 블록이 그대로)
                        그릇에서 꺼내기(unsort_blocks, 10-23)도 블록당 BALL_MOCK_COORD_BLOCK_S 초 × BALL_MOCK_ARRANGE_SCALE
  BALL_MOCK_HOME        홈 맞춤(go_home) 흉내: 비면 팔이 홈 밖이면(멈춤으로 끊긴 뒤) 옮김, too_far = 60° 넘게 멀어 거부
  BALL_MOCK_HOME_S      홈으로 옮기는 데 걸리는 시간(초, 기본 0.05. 실물 5.3~6.6초)
  BALL_MOCK_STUDIO      처음에 Studio 세션 흉내: 비면 없음 | teleop(원격조작, 멈출 수 있음) | recording(녹화 중) | unknown(상태를 모름)
                        | lock(세션 없이 장치 잠금만 BALL_MOCK_STUDIO_LOCK_S 초, 기본 2). 세션을 멈추면 BALL_MOCK_STUDIO_RELEASE_S
                        (기본 0.5)초 뒤 잠금이 풀린다. 그동안 run_policy·go_home·좌표 집기 실행은 studio_session·robot_busy 로 거부
  BALL_MOCK_HAND        auto(기본) | always | never
                        auto = 로봇이 쉬는 동안 연속으로 HAND_DELAY 초 넘게 지켜보면 손이 보인다
  BALL_MOCK_HAND_DELAY  손이 나타나기까지 초(기본 2)
  BALL_MOCK_SNAP        1(기본)이면 runs/<날짜>/mock/ 에 작업대 그림을 저장
  BALL_MOCK_REFUSE      값이 있으면 run_policy 가 reason=refused 로 거부(예: studio_session)
  BALL_MOCK_END         정상 종료 reason: home_reached(기본, 홈 복귀 감지로 일찍 끝남 — 10-22 로봇 조기 종료) | duration_elapsed
  BALL_MOCK_DETECT_S    detect 1회 걸리는 시간(초, 기본 0.03)
"""
import os
import random
import threading
import time
from datetime import datetime

from .config import RUNS_DIR
from .events import now_iso

KEYS = ("red_ok", "blue_ok", "red_defect", "blue_defect")
ZONES = ("input", "bowl_red", "bowl_blue", "bowl_defect")
# 계약서 2절 "발표 실연"(10-21 밤 확정): 투입 분홍 2·초록 1·불량 1. 불량 색은 정해지지 않아 분홍으로 둔다.
# full = 위 표의 전체 흐름(블록 11개, 미리 찍은 성공 영상과 같은 구성) — selftest·eval 의 기대 개수는 full 기준
SCENES = {"demo4": {"red_ok": 2, "blue_ok": 1, "red_defect": 1, "blue_defect": 0},
          "full": {"red_ok": 4, "blue_ok": 4, "red_defect": 2, "blue_defect": 1}}


def scene_input() -> dict:
    return dict(SCENES.get(os.getenv("BALL_MOCK_SCENE", "demo4")) or SCENES["demo4"])

# 좌표 집기 그릇 옮기기(robot/coord_pick sort_blocks): 색 → (집는 클래스, 놓는 그릇). 이름은 robot plan.py 와 같다
SORT_TO = {"red": (("red_ok",), "bowl_red"), "blue": (("blue_ok",), "bowl_blue"),
           "defect": (("red_defect", "blue_defect"), "bowl_defect"),
           "red_defect": (("red_defect",), "bowl_defect"), "blue_defect": (("blue_defect",), "bowl_defect")}   # 불량 색(10-23)
SORT_DST = {k: b for ks, b in SORT_TO.values() for k in ks}
COLOR_KO = {"red_ok": "분홍", "blue_ok": "초록", "red_defect": "분홍 불량", "blue_defect": "초록 불량"}
BOWL_KO = {"bowl_red": "분홍 그릇", "bowl_blue": "초록 그릇", "bowl_defect": "불량 그릇"}

# 정책 이름 → (집는 구역, 집는 클래스, 놓는 구역). 놓는 구역 None = 사람 손
MOVES = {
    "pick_defect": ("input", ("red_defect", "blue_defect"), "bowl_defect"),
    "pick_red": ("input", ("red_ok",), "bowl_red"),
    "pick_blue": ("input", ("blue_ok",), "bowl_blue"),
    "handoff": ("bowl_red", ("red_ok",), None),
}


def policy_timeout(name: str) -> float:
    """robot/policies.json 의 정책별 timeout_s(없으면 defaults, 파일이 없으면 21초 — 10-22 15초 녹화 기준)."""
    import json

    from .config import ROOT
    try:
        d = json.loads((ROOT / "robot" / "policies.json").read_text(encoding="utf-8"))
        return float((d.get("policies", {}).get(name) or {}).get("timeout_s") or d["defaults"]["timeout_s"])
    except Exception:
        return 21.0


def _attempts_env(name: str, default: str = "") -> set[int]:
    raw = os.getenv(name, default)
    return {int(x) for x in raw.split(",") if x.strip()}


# 가짜 사진 속 구역 박스(픽셀). outside = 구역 밖에 떨어진 블록을 그리는 자리(박스는 안 그림)
BOX = {"input": (20, 20, 300, 220), "bowl_red": (330, 20, 480, 160), "bowl_blue": (490, 20, 630, 160),
       "bowl_defect": (330, 180, 480, 340), "outside": (20, 236, 180, 340)}
DIGIT_ORIGIN, DIGIT_STEP = (212, 244), 20     # 좌표 집기로 만든 숫자를 그리는 자리(5행×3열, 칸 20px)
HAND_BOX = (490, 180, 630, 340)
_COLS, _ROWS = ("왼쪽", "가운데", "오른쪽"), ("위", "가운데", "아래")
_CLS_OFF = {"red_ok": 0, "blue_ok": 10, "red_defect": 20, "blue_defect": 27}


def cell_name(zone: str, x: float, y: float) -> str | None:
    """구역 박스를 3×3 으로 나눈 칸(계약서 3절): "왼쪽 위" … "가운데" … "오른쪽 아래". 구역 밖이면 None."""
    if zone == "outside":
        return None
    x1, y1, x2, y2 = BOX[zone]
    c = min(2, max(0, int((x - x1) * 3 / (x2 - x1))))
    r = min(2, max(0, int((y - y1) * 3 / (y2 - y1))))
    return "가운데" if (c, r) == (1, 1) else f"{_COLS[c]} {_ROWS[r]}"


def _slots(zone: str) -> list[tuple[int, int]]:
    """구역 안에 블록을 놓을 자리들. 구역마다 정해진 순서로 섞어(늘 같음) 흩어져 놓인 것처럼 보이게 한다."""
    x1, y1, x2, y2 = BOX[zone]
    pts = [(x, y) for y in range(y1 + 36, y2 - 14, 34) for x in range(x1 + 20, x2 - 14, 34)]
    random.Random(zone).shuffle(pts)
    return pts


def layout(zones: dict, outside: dict) -> list[dict]:
    """구역별 개수 → 블록마다 {cls, zone, cell, px, conf}(계약서 3절 objects). 같은 클래스의 k번째 블록은 늘 같은 자리."""
    objs = []
    for z in (*ZONES, "outside"):
        counts = outside if z == "outside" else zones[z]
        slots = _slots(z)
        for k in KEYS:
            for j in range(counts.get(k, 0)):
                i = (_CLS_OFF[k] + j) % len(slots)
                x, y = slots[i]
                objs.append({"cls": k, "zone": z, "cell": cell_name(z, x, y), "px": [x, y],
                             "conf": round(0.85 + (i * 7 % 13) / 100, 2)})
    return objs


class Workbench:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.reset()

    def reset(self, input_counts: dict | None = None) -> None:
        with self._lock:
            self.zones = {z: dict.fromkeys(KEYS, 0) for z in ZONES}
            self.zones["input"].update(input_counts or scene_input())
            self.outside = dict.fromkeys(KEYS, 0)      # 구역 밖에 떨어진 블록
            self.handed = 0
            self.attempts = 0
            self.arranged = []                          # 좌표 집기로 놓은 블록 [{cell:[행,열], cls}]
            self.at_home = True                         # 팔이 홈 자세인가(멈춤으로 끊기면 False, go_home 이 되돌림)
            self.homed = 0                              # go_home 이 실제로 옮긴 횟수
            self.gripper_open = True                    # 정책·좌표 집기 뒤엔 닫혔다고 본다 → go_home(open_gripper=True) 가 연다
            self.opened = 0                             # 홈 복귀하며 집게를 연 횟수
            kind = os.getenv("BALL_MOCK_STUDIO", "")
            act = {"connected": True, "follower_source": "teleop", "is_recording": kind == "recording",
                   "dataset_loaded": kind == "recording"}
            self.studio = [] if kind in ("", "lock") else [{
                "session_name": "rt-00000000-0000-0000-0000-000000000024", "status": "starting" if kind == "unknown" else "running",
                "follower_name": "follower_24", "activity": None if kind == "unknown" else act}]
            self.lock_until = time.monotonic() + float(os.getenv("BALL_MOCK_STUDIO_LOCK_S", "2")) if kind == "lock" else 0.0
            self.studio_stops = []                      # 멈춘 세션 이름
            self.fail_at = _attempts_env("BALL_MOCK_FAIL_AT", "3")
            self.drop_at = _attempts_env("BALL_MOCK_DROP_AT")
            self._watch_since = None   # 로봇이 쉬는 동안 지켜보기 시작한 시각

    # ---- vision ----
    def _hand_present(self) -> bool:
        mode = os.getenv("BALL_MOCK_HAND", "auto")
        if mode in ("always", "never"):
            return mode == "always"
        delay = float(os.getenv("BALL_MOCK_HAND_DELAY", "2"))
        return time.monotonic() - self._watch_since >= delay

    def detect_objects(self, image_path: str | None = None) -> dict:
        t0 = time.perf_counter()
        time.sleep(float(os.getenv("BALL_MOCK_DETECT_S", "0.03")))
        with self._lock:
            if self._watch_since is None:
                self._watch_since = time.monotonic()
            hand = self._hand_present()
            zones = {z: dict(c) for z, c in self.zones.items()}
            objects = layout(zones, self.outside)
            arranged = list(self.arranged)
        snap = _draw_snapshot(objects, hand, arranged) if os.getenv("BALL_MOCK_SNAP", "1") == "1" else None
        return {
            "ts": now_iso(),
            "zones": zones,
            "hand": {"present": hand, "conf": 0.93 if hand else 0.0},
            "objects": objects,
            "image_path": snap,
            "device": "MOCK",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    # ---- Studio 흉내(10-22 r22) ----
    def studio_list(self) -> list[dict]:
        with self._lock:
            return [dict(x) for x in self.studio]

    def studio_stop(self, name: str) -> int:
        with self._lock:
            before = len(self.studio)
            self.studio = [x for x in self.studio if x["session_name"] != name]
            if len(self.studio) == before:
                return 404
            self.studio_stops.append(name)
            self.lock_until = time.monotonic() + float(os.getenv("BALL_MOCK_STUDIO_RELEASE_S", "0.5"))
            return 204

    def _studio_block(self) -> str | None:
        """Studio 가 팔을 쥐고 있으면 로봇 쪽 사전 점검 코드(robot run_policy.robot_preflight 와 같은 이름)."""
        with self._lock:
            if self.studio:
                return "studio_session"
            return "robot_busy" if time.monotonic() < self.lock_until else None

    def _refused(self, code: str) -> dict:
        return {"ok": False, "reason": "refused", "executed": False, "mock": True,
                "error": f"{code}: 가짜 — Studio 가 로봇을 쥐고 있다(follower_24)"}

    # ---- robot ----
    def run_policy(self, name: str, timeout_s: float | None = None) -> dict:
        """계약서 3절 robot 서명. timeout_s=None 이면 robot/policies.json 정책별 값(실제 로봇이 도는 시간, 기본 21초).
        가짜는 BALL_MOCK_POLICY_S(기본 0.3초)만 기다리고, 실제 로봇이면 돌 시간을 결과 policy_s 에 남긴다."""
        if name not in MOVES:
            raise ValueError(f"알 수 없는 정책: {name}")
        policy_s = timeout_s if timeout_s is not None else policy_timeout(name)
        started, t0 = now_iso(), time.monotonic()
        refuse = os.getenv("BALL_MOCK_REFUSE", "") or self._studio_block() or ""
        if refuse:                             # 시작 전 점검 거부 흉내(예: studio_session)
            return {"name": name, "ok": False, "started": started, "ended": now_iso(), "duration_s": 0.0,
                    "returncode": None, "log_path": None, "reason": "refused",
                    "error": f"{refuse}: 가짜 거부", "camera_released": True, "mock": True}
        self._stop.clear()
        dur = float(os.getenv("BALL_MOCK_POLICY_S", "0.3"))
        with self._lock:
            self.attempts += 1
            n = self.attempts
        self.gripper_open = False
        interrupted = self._stop.wait(min(dur, policy_s))
        if interrupted:
            self.at_home = False            # stop() = 그 자리 정지(실물 SIGKILL) → 팔이 홈 밖에 남는다
            rc = -15
        elif dur > policy_s:
            rc = 124
        else:
            rc = 0
            if n in self.drop_at:       # 집었다가 구역 밖에 떨어뜨림(정책은 끝까지 돎)
                self._move_one(name, drop=True)
            elif n not in self.fail_at:   # 실패 시도는 블록을 놓쳐 작업대가 그대로다
                self._move_one(name)
        with self._lock:
            self._watch_since = None
        end_ok = os.getenv("BALL_MOCK_END", "home_reached")
        return {  # robot/README 1절의 반환값과 같은 키
            "name": name, "ok": rc == 0, "started": started, "ended": now_iso(),
            "duration_s": round(time.monotonic() - t0, 2), "returncode": rc, "log_path": None,
            "reason": {0: end_ok, -15: "stopped", 124: "timeout"}[rc],
            "early_stop": {"triggered": rc == 0 and end_ok == "home_reached"},
            "camera_released": True, "mock": True, "policy_s": policy_s,
        }

    def _move_one(self, name: str, drop: bool = False) -> None:
        src, keys, dst = MOVES[name]
        with self._lock:
            have = [k for k in keys if self.zones[src][k] > 0]
            if not have:
                return
            k = max(have, key=lambda k: self.zones[src][k])
            self.zones[src][k] -= 1
            if drop:
                self.outside[k] += 1
            elif dst is None:
                self.handed += 1
            else:
                self.zones[dst][k] += 1

    def stop(self) -> None:
        self._stop.set()

    # ---- 좌표 집기(robot/COORD_PICK.md 9절 arrange_digit 과 같은 모양) ----
    def arrange_digit(self, digit, color: str = "any", dry_run: bool = True, *, detection: dict | None = None,
                      **kw) -> dict:
        """계획은 로봇 담당 arrange_digit(dry_run=True) 그대로(계획·시간·경고 + fake.py 가짜 로봇 모의 실행 sim).
        실행(dry_run=False)은 예상 시간을 BALL_MOCK_ARRANGE_SCALE 배로 줄여 기다리며 흉내 낸다 — 멈춤(stop)을 받고,
        놓은 블록은 투입 구역에서 빠져 사진의 숫자 자리에 그려진다. 실제 로봇·카메라는 안 쓴다."""
        import sys

        from .config import ROOT
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from robot.coord_pick.executor import arrange_digit as plan_digit

        det = detection or self.detect_objects()
        kw.setdefault("log_dir", RUNS_DIR / (datetime.now().strftime("%Y%m%d") + "_mock") / "coord_pick")
        r = plan_digit(digit, color, True, detection=robot_detection(det), **kw)
        if dry_run or not r.get("ok"):
            return r
        if self._studio_block():
            return dict(r, **self._refused(self._studio_block()), mode="real")
        self._stop.clear()
        self.gripper_open = False
        n = len(r["plan"])
        dur = float(r.get("time_s") or 0) * float(os.getenv("BALL_MOCK_ARRANGE_SCALE", "0.02"))
        t0 = time.monotonic()
        stopped = self._stop.wait(dur)
        done = n if not stopped else min(n, int(n * (time.monotonic() - t0) / max(dur, 1e-6)))
        with self._lock:
            for st in r["plan"][:done]:
                k = "red_ok" if st["color"] == "분홍" else "blue_ok"
                for z in ("input", "bowl_red", "bowl_blue"):     # 집은 곳(가짜): 투입 구역부터
                    if self.zones[z][k] > 0:
                        self.zones[z][k] -= 1
                        break
                self.arranged.append({"cell": list(st["to_cell"]), "cls": k})
            self._watch_since = None
        return dict(r, mode="real", executed=True, ok=not stopped, reason="stopped" if stopped else "done",
                    run={"steps_done": done, "stopped": stopped, "mock": True}, mock=True)

    # ---- 홈 맞춤(robot/coord_pick executor.go_home 과 같은 모양, 10-22 r20) ----
    def go_home(self, dry_run: bool = True, open_gripper: bool = False, **kw) -> dict:
        """측정 자세가 홈이면(open_gripper 면 집게도 열려 있으면) already_home(안 보냄), 아니면 BALL_MOCK_HOME_S 동안 옮김(done),
        BALL_MOCK_HOME=too_far 면 거부. Studio 가 쥐고 있으면 실제처럼 사전 점검에서 거부."""
        out = {"ok": False, "mode": "dry_run" if dry_run else "real", "executed": False, "reason": "planned",
               "summary": "홈 자세로(가짜)", "measured": None, "error": None, "open_gripper": open_gripper, "mock": True}
        if dry_run:
            return dict(out, ok=True)
        if self._studio_block():
            return dict(out, **self._refused(self._studio_block()), open_gripper=open_gripper)
        # too_far: 팔이 홈에서 BALL_MOCK_HOME_FAR_DEG(90°) 떨어져 있다 — max_deg(기본 60°)가 그보다 작으면 거부(홈 명령은 150°)
        far = float(os.getenv("BALL_MOCK_HOME_FAR_DEG", "90"))
        if os.getenv("BALL_MOCK_HOME") == "too_far" and float(kw.get("max_deg") or 60) < far:
            return dict(out, reason="too_far", error="홈에서 45.0(정규화) 떨어짐 > 33.33(가짜)")
        if self.at_home and (self.gripper_open or not open_gripper):
            return dict(out, ok=True, reason="already_home")
        self._stop.clear()
        if self._stop.wait(float(os.getenv("BALL_MOCK_HOME_S", "0.05"))):
            return dict(out, executed=True, reason="stopped")
        if not self.at_home:
            self.homed += 1
        self.at_home = True
        if open_gripper:
            self.gripper_open = True
            self.opened += 1
        return dict(out, ok=True, executed=True, reason="done")

    # ---- 좌표 집기 그릇 옮기기(robot/COORD_PICK.md 9절 sort_blocks 와 같은 모양, 10-22 r16) ----
    def sort_blocks(self, colors=("red", "blue"), dry_run: bool = True, *, detection: dict | None = None,
                    max_blocks: int | None = None, which: str | dict | None = None, **kw) -> dict:
        """계획은 흉내(_sort_plan) 또는 BALL_MOCK_COORD=robot 이면 로봇 담당 sort_blocks(dry_run=True).
        which: "left"|"right"|"middle" — 같은 색 후보를 가짜 사진 u(가로) 순서로 고른다(10-23, 로봇 함수와 같은 뜻).
        "top"|"bottom"(v, top = 로봇에서 먼 쪽)·"near"|"far"(가짜 사진 아래쪽 = 로봇 쪽) · {"next_to": 색} = 그 색 블록에 가장
        가까운 것(사진 거리)부터(10-23 13:30). 옆에 둘 블록이 없으면 거부.
        실행(dry_run=False)은 arrange_digit 흉내처럼 예상 시간 × BALL_MOCK_ARRANGE_SCALE 만큼 기다리며 멈춤(stop)을 받고,
        옮긴 블록은 투입 구역에서 제 그릇으로 간다. 실제 로봇·카메라는 안 쓴다."""
        colors = (colors,) if isinstance(colors, str) else tuple(colors)
        det = detection or self.detect_objects()
        fail = os.getenv("BALL_MOCK_COORD_FAIL", "")
        if os.getenv("BALL_MOCK_COORD") == "robot":
            r = self._sort_plan_robot(colors, det, max_blocks, **kw, **({"which": which} if which else {}))
        else:
            r = self._sort_plan(colors, det, max_blocks, fail, which)
        if dry_run or not r.get("ok"):
            return r
        if self._studio_block():
            return dict(r, **self._refused(self._studio_block()), mode="real")
        self._stop.clear()
        self.gripper_open = False
        n = len(r["plan"])
        dur = float(r.get("time_s") or 0) * float(os.getenv("BALL_MOCK_ARRANGE_SCALE", "0.02"))
        t0 = time.monotonic()
        stopped = self._stop.wait(dur / 2 if fail == "tracking" else dur)
        cut = stopped or fail == "tracking"
        if cut:
            self.at_home = False
        done = min(n, int(n * (time.monotonic() - t0) / max(dur, 1e-6))) if cut else n
        moved = done - 1 if fail == "miss" and done else done
        with self._lock:
            for st in r["plan"][:moved]:
                k = st.get("cls") or {v: c for c, v in COLOR_KO.items()}[st["color"]]
                if self.zones["input"][k] > 0:
                    self.zones["input"][k] -= 1
                    self.zones[SORT_DST[k]][k] += 1
            self._watch_since = None
        reason = "stopped" if stopped else "tracking" if fail == "tracking" else "done"
        return dict(r, mode="real", executed=True, ok=reason == "done", reason=reason, run_id="mock",
                    run={"returncode": 0 if reason == "done" else -9, "stopped": stopped, "timed_out": False,
                         "steps_done": done, "mock": True}, mock=True)

    def _sort_plan(self, colors: tuple, det: dict, max_blocks: int | None, fail: str,
                   which: str | dict | None = None) -> dict:
        """투입 구역의 그 색 블록을 하나씩(가짜 사진 아래쪽 = 로봇 쪽부터). 블록당 시간은 로봇 dry-run 실측(10-22 38~48초).
        which 가 있으면 u 순서: left = 작은 u 부터, right = 큰 u 부터, middle = u 순서 가운데부터. top = 작은 v 부터(로봇에서
        먼 쪽), bottom·near = 큰 v 부터(로봇 쪽), far = 작은 v 부터. {"next_to": 색} = 그 색 블록에 가장 가까운 것부터."""
        want = {k for c in colors for k in SORT_TO[c][0]}
        todo = sorted((o for o in det.get("objects") or [] if o.get("zone") == "input" and o["cls"] in want),
                      key=lambda o: -o["px"][1])
        if isinstance(which, dict):
            ref_cls = {"red": "red_ok", "blue": "blue_ok"}.get(which["next_to"], which["next_to"])
            refs = [o for o in det.get("objects") or [] if o.get("zone") == "input" and o["cls"] == ref_cls]
            if not refs:
                return {"ok": False, "mode": "dry_run", "executed": False, "reason": "refused", "colors": list(colors),
                        "blocks": 0, "plan": [], "time_s": 0, "which": which, "warnings": [], "violations": [],
                        "failures": [f"옆에 둘 {which['next_to']} 블록이 투입 구역에 없음(가짜)"], "mock": True}
            todo = sorted(todo, key=lambda o: min([(o["px"][0] - r["px"][0]) ** 2 + (o["px"][1] - r["px"][1]) ** 2
                                                   for r in refs if r is not o] or [float("inf")]))
        elif which:
            by_u = sorted(todo, key=lambda o: o["px"][0])
            by_v = sorted(todo, key=lambda o: o["px"][1])
            mid = (len(by_u) - 1) / 2
            todo = {"left": by_u, "right": by_u[::-1],
                    "middle": sorted(by_u, key=lambda o: abs(by_u.index(o) - mid)),
                    "top": by_v, "far": by_v, "bottom": by_v[::-1], "near": by_v[::-1]}[which]
        if max_blocks is not None:
            todo = todo[:max_blocks]
        plan = [{"order": i + 1, "color": COLOR_KO[o["cls"]], "cls": o["cls"], "from_px": o["px"], "from_mm": None,
                 "to": BOWL_KO[SORT_DST[o["cls"]]], "to_mm": None} for i, o in enumerate(todo)]
        failures, viol = [], []
        if fail == "unreachable" and plan:
            failures = [f"{plan[-1]['color']} 블록(작업대 [300.0, 140.0] mm)에 닿지 않는다 — 팔이 안 닿음(가짜)",
                        f"옮길 블록 {len(plan)}개 중 {len(plan) - 1}개만 계획됨"]
            plan = plan[:-1]
        elif fail == "violation":
            viol = [{"kind": "forbid_zone", "count": 12, "first_tick": 300, "detail": "다른 그릇 위 금지 높이 아래(가짜)"}]
        time_s = round(float(os.getenv("BALL_MOCK_COORD_BLOCK_S", "40")) * len(plan) + 1, 1)
        ok = not failures and not viol
        m, sec = divmod(int(round(time_s)), 60)
        return {"ok": ok, "mode": "dry_run", "executed": False, "reason": "planned" if ok else "refused",
                "colors": list(colors), "lift_mm": 0.0, "speed_deg_s": 15.0, "log_path": None, "error": None,
                "blocks": len(plan), "source": "detection", "plan": plan, "time_s": time_s, "which": which,
                "plan_time_s": round(time_s / 3, 1), "warnings": [], "failures": failures, "violations": viol,
                "sim": {"ok": True, "reason": "done", "sent": 0, "total": 0, "max_err": []},
                "summary": f"그릇 옮기기({'·'.join(colors)}): 블록 {len(plan)}개, 약 {m}분 {sec}초(관절 15°/s). "
                           f"실패 {len(failures)} · 공간 위반 {len(viol)} · 경고 0", "mock": True}

    def _sort_plan_robot(self, colors: tuple, det: dict, max_blocks: int | None, **kw) -> dict:
        """로봇 담당 sort_blocks(dry_run=True) 로 계획한다(가짜 사진 자리는 robot_detection 으로 옮겨 넘긴다)."""
        import sys

        from .config import ROOT
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from robot.coord_pick.executor import sort_blocks as plan_sort

        kw.setdefault("log_dir", RUNS_DIR / (datetime.now().strftime("%Y%m%d") + "_mock") / "coord_pick")
        return plan_sort(colors, True, detection=robot_detection(det), max_blocks=max_blocks, **kw)

    # ---- 그릇에서 꺼내기(robot/coord_pick unsort_blocks 와 같은 모양, 10-23 리허설 초기화) ----
    def unsort_blocks(self, dry_run: bool = True, colors=None, max_blocks: int | None = None, **kw) -> dict:
        """그릇(colors: red·blue·defect, None = 세 그릇 전부)에 있는 블록을 투입 구역으로 하나씩 되돌린다. 계획 시간·실행 흉내·
        멈춤은 sort_blocks 와 같다(블록당 BALL_MOCK_COORD_BLOCK_S 초, 실행은 × BALL_MOCK_ARRANGE_SCALE). 실제 로봇·카메라는 안 쓴다."""
        colors = (colors,) if isinstance(colors, str) else tuple(colors or ("red", "blue", "defect"))
        bowls = [SORT_TO[c][1] for c in colors]
        with self._lock:
            todo = [(b, k) for b in bowls for k in KEYS for _ in range(self.zones[b][k])]
        if max_blocks is not None:
            todo = todo[:max_blocks]
        plan = [{"order": i + 1, "color": COLOR_KO[k], "cls": k, "from": BOWL_KO[b], "to": "투입 구역"}
                for i, (b, k) in enumerate(todo)]
        time_s = round(float(os.getenv("BALL_MOCK_COORD_BLOCK_S", "40")) * len(plan) + 1, 1)
        m, sec = divmod(int(round(time_s)), 60)
        r = {"ok": True, "mode": "dry_run", "executed": False, "reason": "planned", "colors": list(colors),
             "log_path": None, "error": None, "blocks": len(plan), "plan": plan, "time_s": time_s,
             "warnings": [], "failures": [], "violations": [],
             "summary": f"그릇에서 꺼내기({'·'.join(colors)}): 블록 {len(plan)}개, 약 {m}분 {sec}초(가짜)", "mock": True}
        if dry_run or not plan:
            return r
        if self._studio_block():
            return dict(r, **self._refused(self._studio_block()), mode="real")
        self._stop.clear()
        self.gripper_open = False
        dur = time_s * float(os.getenv("BALL_MOCK_ARRANGE_SCALE", "0.02"))
        t0 = time.monotonic()
        stopped = self._stop.wait(dur)
        if stopped:
            self.at_home = False
        done = min(len(plan), int(len(plan) * (time.monotonic() - t0) / max(dur, 1e-6))) if stopped else len(plan)
        with self._lock:
            for b, k in todo[:done]:
                if self.zones[b][k] > 0:
                    self.zones[b][k] -= 1
                    self.zones["input"][k] += 1
            self._watch_since = None
        reason = "stopped" if stopped else "done"
        return dict(r, mode="real", executed=True, ok=not stopped, reason=reason, run_id="mock",
                    run={"returncode": -9 if stopped else 0, "stopped": stopped, "timed_out": False, "steps_done": done,
                         "mock": True})


def robot_detection(det: dict) -> dict:
    """가짜 사진 → 로봇 계획 함수(arrange_digit·sort_blocks dry-run)에 넘길 검출. 투입 구역 블록을 가짜 박스 안 비율 그대로
    작업대 투입 구역(robot/coord_pick/config.json input_area_mm) 자리의 위 카메라 픽셀로 옮긴다. 가짜 픽셀을 그대로 넘기면
    로봇 보정값에 따라 팔이 안 닿는 곳이 된다(10-22 r14 작업대 표시점을 바꾼 뒤 숫자 7 계획이 블록 2개로 줄었다)."""
    import sys

    import numpy as np

    from .config import ROOT
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from robot.coord_pick.plan import Setup, load_config

    s = Setup.load(load_config(), use_studio=False)
    (a0, b0), (a1, b1) = s.cfg["table"]["input_area_mm"]
    x1, y1, x2, y2 = BOX["input"]
    objs = []
    for o in det.get("objects") or []:
        if o.get("zone") == "input":
            fx, fy = (o["px"][0] - x1) / (x2 - x1), (o["px"][1] - y1) / (y2 - y1)
            q = s.w_to_px(np.array([[a0 + fx * (a1 - a0), b0 + fy * (b1 - b0)]]))[0]
            o = dict(o, px=[round(float(q[0]), 1), round(float(q[1]), 1)])
        objs.append(o)
    return dict(det, objects=objs)


def _block(img, center, side: int, angle: float, bgr, defect: bool) -> None:
    """위에서 본 스펀지 블록(약 3cm 정육면체 윗면) 하나. 불량은 면의 대각선을 따라 검은 X(기준서 3.3)."""
    import cv2
    import numpy as np

    pts = cv2.boxPoints((center, (side, side), angle)).astype(np.float32)
    cv2.fillPoly(img, [pts.astype(np.int32)], bgr, cv2.LINE_AA)
    cv2.polylines(img, [pts.astype(np.int32)], True, tuple(int(c * 0.55) for c in bgr), 1, cv2.LINE_AA)
    if defect:
        c = np.asarray(center, np.float32)
        p = c + (pts - c) * 0.72                               # 모서리에서 조금 안쪽까지
        for a, b in ((0, 2), (1, 3)):
            cv2.line(img, tuple(int(v) for v in p[a]), tuple(int(v) for v in p[b]), (0, 0, 0), 3, cv2.LINE_AA)


def _draw_snapshot(objects: list[dict], hand: bool, arranged: list | None = None) -> str | None:
    """UI 시험용 가짜 위 카메라 사진: 흰 무광 식탁보 위 파스텔 분홍·초록 블록. cv2 가 없으면 None."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    img = np.full((360, 640, 3), 250, np.uint8)
    # 사진에 보이는 이름이라 사람 말로(cv2 는 한글을 못 그려 영어). 코드 식별자는 zone 값 그대로
    names = {"input": "input", "bowl_red": "pink bowl", "bowl_blue": "green bowl", "bowl_defect": "defect bowl"}
    color = {"red": (194, 165, 248), "blue": (150, 212, 150)}   # BGR 파스텔 분홍·초록
    for z, name in names.items():
        x1, y1, x2, y2 = BOX[z]
        cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 2)
        cv2.putText(img, name, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    for o in objects:
        i = o["px"][0] * 7 + o["px"][1] * 3
        angle = (i % 31) - 15                           # 놓인 대로 약간씩 돌아간 모양(자리마다 늘 같음)
        _block(img, tuple(o["px"]), 24, angle, color[o["cls"].split("_")[0]], o["cls"].endswith("defect"))
    for a in arranged or []:                            # 좌표 집기로 만든 숫자(블록이 칸에 반듯이 놓임)
        r_, c_ = a["cell"]
        cx, cy = DIGIT_ORIGIN[0] + c_ * DIGIT_STEP + 8, DIGIT_ORIGIN[1] + r_ * DIGIT_STEP + 8
        _block(img, (cx, cy), 17, 0, color[a["cls"].split("_")[0]], False)
    if arranged:
        cv2.putText(img, "digit", (DIGIT_ORIGIN[0], DIGIT_ORIGIN[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    x1, y1, x2, y2 = HAND_BOX
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 160, 0) if hand else (150, 150, 150), 2)
    cv2.putText(img, "hand spot" + (" HAND" if hand else ""), (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
    now = datetime.now()
    d = RUNS_DIR / (now.strftime("%Y%m%d") + "_mock") / "mock"      # 가짜 기록은 runs/<날짜>_mock/ (runlog.day_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"snap_{now.strftime('%H%M%S_%f')}.jpg"
    cv2.imwrite(str(path), img)
    return str(path.relative_to(RUNS_DIR.parent))


workbench = Workbench()
detect_objects = workbench.detect_objects
run_policy = workbench.run_policy
stop = workbench.stop
