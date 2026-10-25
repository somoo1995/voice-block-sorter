"""vision·robot 연결. 실제 모듈(vision/detect.py, robot/run_policy.py)을 쓰고, 못 쓰면 mocks.py.

- BALL_MOCK=1      vision·robot 모두 가짜
- vision           import 되면 실제(카메라 한 장 찍고 닫음), 실패하면 가짜
- robot            BALL_ROBOT=1 일 때만 실제로 움직인다(계약서 5절: 실제 구동은 사용자 입회 때만).
                   BALL_ROBOT=dry 면 실제 모듈을 dry_run 으로 부른다(로봇 안 움직임, 설정만 검증).
                   그 밖에는 가짜. 실제 vision 을 못 쓰면 가짜 개수로 진짜 로봇을 움직이지 않게 robot 도 가짜.
- arrange          좌표 집기(robot/coord_pick arrange_digit, COORD_PICK.md 9절). robot 이 가짜면 mocks(계획은 로봇 함수
                   dry-run, 실행은 흉내), 아니면 로봇 함수 — 로봇 함수는 BALL_ROBOT=1·보정·금지 구역이 다 있어야 보낸다.
                   기록은 runs/<날짜>/coord_pick/ 에 남긴다(robot/ 에 쓰지 않음).
- home             정책·좌표 집기 직전 홈 맞춤(robot/coord_pick go_home, 10-22 r20). BALL_ROBOT=1 일 때만 실제, 가짜 로봇이면
                   mocks 흉내, BALL_ROBOT=dry·import 실패면 None(홈 맞춤 안 함).
- studio           Studio 런타임 세션 목록·멈춤(studio.py, 10-22 r22). 로봇이 실제(BALL_ROBOT=1)면 실제 API, 가짜 로봇이면 mocks
                   흉내, BALL_ROBOT=dry 면 None.
- sort             좌표 집기 그릇 옮기기(robot/coord_pick sort_blocks, 10-22 r16). 10-23 13:10 사용자 결정부터 defect·red·blue 는
                   늘 이것으로 한다(graph.coord_node, ACT 는 handoff 만). arrange 와 같은 규칙으로 가짜·실제를 고르고, 로봇 함수를
                   못 불러오면 None(info coord="off") → 그래프가 옮기기 명령을 거부한다(ACT 로 대신하지 않음).
- which            sort_blocks 가 which(왼쪽·오른쪽·가운데) 인자를 받는지(10-23, info["which"]). 로봇 함수의 인자 목록을 본다.
                   가짜 로봇은 받는다(BALL_MOCK_WHICH=0 이면 안 받는 것으로, BALL_MOCK_COORD=robot 이면 로봇 함수를 본다).
                   못 받으면 그래프가 위치 단서 명령을 거부한다(which 를 빼고 부르면 엉뚱한 블록을 집는다).
- which_values     로봇이 아는 which 값·next_to 기준(10-23 13:30, robot/coord_pick/plan.WHICH·NEXT_TO 를 시작 때 본다).
  next_to          가짜 로봇은 일곱 값(left·right·middle·top·bottom·near·far)과 네 기준을 다 흉내 낸다. 모르는 값이면 그래프가 거부.
- defect_color     sort_blocks 가 불량 색 이름(red_defect·blue_defect)을 아는지(10-23, info["defect_color"]). 로봇 SORT_CLASSES 를 본다.
                   가짜 로봇은 안다(BALL_MOCK_DEFECT_COLOR=0 이면 모르는 것으로, BALL_MOCK_COORD=robot 이면 로봇 것을 본다).
                   모르면 그래프가 "초록 불량 옮겨 줘" 같은 명령을 거부한다(색 없이 부르면 다른 색 불량까지 옮긴다).
- unsort           그릇에서 꺼내기(robot/coord_pick unsort_blocks, 10-23 리허설 초기화). sort 와 같은 규칙으로 가짜·실제를 고르고,
                   로봇 함수가 아직 없으면 None(info unsort="off") → 그래프가 "아직 그릇에서 꺼내기는 못 해요" 로 거부한다.
"""
import functools
import inspect
import os
import sys
import threading

from datetime import datetime

from .config import ROOT, RUNS_DIR

_lock = threading.Lock()
_loaded: dict | None = None


def _takes(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _robot_which() -> bool:
    try:
        from robot.coord_pick.executor import sort_blocks as plan_sort
        return _takes(plan_sort, "which")
    except Exception:
        return False


def _robot_targets() -> tuple[list, list]:
    """로봇이 아는 which 값·next_to 기준(robot/coord_pick/plan.WHICH·NEXT_TO, 10-23 13:30). 없으면 왼쪽·오른쪽·가운데만."""
    try:
        from robot.coord_pick import plan as rp
        return list(getattr(rp, "WHICH", ("left", "right", "middle"))), list(getattr(rp, "NEXT_TO", {}))
    except Exception:
        return ["left", "right", "middle"], []


def _robot_defect_color() -> bool:
    try:
        from robot.coord_pick.executor import SORT_CLASSES
        return {"red_defect", "blue_defect"} <= set(SORT_CLASSES)
    except Exception:
        return False


def _mock_caps() -> dict:
    """가짜 로봇이 which·불량 색을 받는가: 환경변수가 0 이면 못 받음, BALL_MOCK_COORD=robot(로봇 함수로 계획)이면 로봇 것을 본다."""
    robot = os.getenv("BALL_MOCK_COORD") == "robot"
    which = os.getenv("BALL_MOCK_WHICH") != "0" and (not robot or _robot_which())
    values, nxt = _robot_targets() if robot else (list(mocks_which()), ["red", "blue", "red_defect", "blue_defect"])
    return {"which": which, "which_values": values if which else [], "next_to": nxt if which else [],
            "defect_color": os.getenv("BALL_MOCK_DEFECT_COLOR") != "0" and (not robot or _robot_defect_color())}


def mocks_which() -> tuple:
    """가짜 로봇이 아는 which 값(mocks.Workbench._sort_plan 이 흉내 내는 것)."""
    return ("left", "right", "middle", "top", "bottom", "near", "far")


def _load() -> dict:
    from . import mocks

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    info = {"vision": "mock", "robot": "mock", "arrange": "mock", "coord": "mock", "unsort": "mock", "reason": {}}
    detect, run, stop = mocks.detect_objects, mocks.run_policy, mocks.stop
    arrange, sort, home = mocks.workbench.arrange_digit, mocks.workbench.sort_blocks, mocks.workbench.go_home
    unsort = mocks.workbench.unsort_blocks
    studio = {"list": mocks.workbench.studio_list, "stop": mocks.workbench.studio_stop}

    if os.getenv("BALL_MOCK") == "1":
        info["reason"] = {"vision": "BALL_MOCK=1", "robot": "BALL_MOCK=1", "arrange": "BALL_MOCK=1", "coord": "BALL_MOCK=1",
                          "unsort": "BALL_MOCK=1"}
        info["home"] = "mock"
        info.update(_mock_caps())
        return {"detect_objects": detect, "run_policy": run, "stop": stop, "arrange": arrange, "sort": sort,
                "unsort": unsort, "home": home, "studio": studio, "info": info}

    try:
        from vision.detect import detect_objects as detect
        info["vision"] = "real"
    except Exception as e:   # 모듈이 아직 없거나 import 중 오류
        info["reason"]["vision"] = f"{type(e).__name__}: {e}"

    arm = os.getenv("BALL_ROBOT", "")
    if info["vision"] != "real":
        info["reason"]["robot"] = "vision 이 가짜라 robot 도 가짜"
    elif arm not in ("1", "dry"):
        info["reason"]["robot"] = "BALL_ROBOT=1 이 아니라 가짜(실제 구동은 사용자 입회 때만)"
    else:
        try:
            from robot.run_policy import run_policy as real_run, stop as real_stop
            run, stop = real_run, real_stop
            if arm == "dry":
                run = functools.partial(real_run, dry_run=True)
            info["robot"] = "real" if arm == "1" else "dry_run"
        except Exception as e:
            info["reason"]["robot"] = f"{type(e).__name__}: {e}"
    if info["robot"] != "mock":
        try:
            from robot.coord_pick.executor import arrange_digit as real_arrange

            def arrange(digit, color="any", dry_run=True, **kw):
                kw.setdefault("log_dir", RUNS_DIR / datetime.now().strftime("%Y%m%d") / "coord_pick")
                return real_arrange(digit, color, dry_run, **kw)
            info["arrange"] = "real"
        except Exception as e:
            info["reason"]["arrange"] = f"{type(e).__name__}: {e}"
        try:
            from robot.coord_pick.executor import sort_blocks as real_sort

            def sort(colors, dry_run=True, **kw):
                kw.setdefault("log_dir", RUNS_DIR / datetime.now().strftime("%Y%m%d") / "coord_pick")
                return real_sort(colors, dry_run, **kw)
            info["coord"], info["which"], info["defect_color"] = "real", _takes(real_sort, "which"), _robot_defect_color()
            info["which_values"], info["next_to"] = _robot_targets() if info["which"] else ([], [])
        except Exception as e:     # 로봇 함수가 없으면 옮기기(불량·분홍·초록)는 거부(10-23 13:10, ACT 로 대신하지 않음).
            sort, info["coord"] = None, "off"   # 시작 기록(ui.log)에도 남긴다
            info["reason"]["coord"] = f"{type(e).__name__}: {e}"
            print(f"[agent] sort_blocks 를 못 불러와 옮기기 명령은 거부합니다(손 전달만 됨): {info['reason']['coord']}",
                  file=sys.stderr)
        try:
            from robot.coord_pick.executor import unsort_blocks as real_unsort
            unsort_logs = _takes(real_unsort, "log_dir")

            def unsort(dry_run=True, **kw):
                if unsort_logs:
                    kw.setdefault("log_dir", RUNS_DIR / datetime.now().strftime("%Y%m%d") / "coord_pick")
                return real_unsort(dry_run, **kw)
            info["unsort"] = "real"
        except Exception as e:     # 로봇 함수가 아직 없으면 그래프가 거부한다(graph.unsort_node)
            unsort, info["unsort"] = None, "off"
            info["reason"]["unsort"] = f"{type(e).__name__}: {e}"
        if info["robot"] == "real":
            from . import studio as real_studio          # 실제 로봇이면 Studio 정리도 실제 API(go_home 이 없어도)
            studio = {"list": real_studio.list_sessions, "stop": real_studio.stop_session}
            try:
                from robot.coord_pick.executor import go_home as real_home

                def home(dry_run=False, **kw):
                    kw.setdefault("log_dir", RUNS_DIR / datetime.now().strftime("%Y%m%d") / "coord_pick")
                    return real_home(dry_run, **kw)
                info["home"] = "real"
            except Exception as e:
                home, info["home"] = None, "off"
                info["reason"]["home"] = f"{type(e).__name__}: {e}"
                print(f"[agent] go_home 을 못 불러와 홈 맞춤 없이 정책을 돌립니다: {info['reason']['home']}", file=sys.stderr)
        else:                   # BALL_ROBOT=dry: 팔을 안 움직이는 시험이라 홈 맞춤·Studio 정리도 안 한다
            home, studio, info["home"] = None, None, "off"
            info["reason"]["home"] = "BALL_ROBOT=dry"
    else:
        info["reason"]["arrange"] = "robot 이 가짜라 좌표 집기도 가짜"
        info["reason"]["coord"] = "robot 이 가짜라 좌표 집기도 가짜"
        info["reason"]["unsort"] = "robot 이 가짜라 그릇에서 꺼내기도 가짜"
        info["home"] = "mock"
    if info["coord"] == "mock":
        info.update(_mock_caps())
    info.setdefault("which", False)
    info.setdefault("which_values", [])
    info.setdefault("next_to", [])
    info.setdefault("defect_color", False)
    return {"detect_objects": detect, "run_policy": run, "stop": stop, "arrange": arrange, "sort": sort, "unsort": unsort,
            "home": home, "studio": studio, "info": info}


def get() -> dict:
    global _loaded
    with _lock:
        if _loaded is None:
            _loaded = _load()
        return _loaded


def info() -> dict:
    return get()["info"]
