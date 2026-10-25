"""LangGraph StateGraph 와 run_command().

  classify ─┬─ defect/red/blue ─ plan ─ coord ─ finish   (늘 좌표 집기 sort_blocks, 10-23 13:10 사용자 결정)
            ├─ handoff ─ plan ─ hand_wait ─ act ─ verify ─┬─ hand_wait ─ ...     (학습한 동작(ACT)은 손 전달만)
            │                                              └─ finish
            ├─ plan(의도) ─ planner ─ [단계마다] plan ─ … ─ finish ─ next_step ─┬─ plan(다음 단계)
            │                                                                  └─ plan_report
            ├─ report ── report
            ├─ question ─ answer
            ├─ scene ─── scene
            ├─ plan(숫자·모양) ─ arrange ─ arrange_confirm(interrupt: "할까요?") ─ arrange_run   (좌표 집기)
            ├─ stop ──── halt
            ├─ home ──── home      (홈 자세 명령: go_home(open_gripper), 너무 멀면 HOME_CMD_MAX_DEG 로 한 번 더, 10-23)
            ├─ unsort ── unsort    (그릇에서 꺼내기: "다 꺼내 놔"·"초기화해 줘" 규칙 → unsort_blocks, 기록·보고에 안 넣음, 10-23)
            └─ none ──── none      (명령 아님: 잡담·마이크 시험·LLM 실패 + 자신 없음·낱말 근거 없음 → 안 움직임, 10-22 밤)
위치 단서(10-23): "왼쪽 초록 블록 옮겨 줘" → which=left, 개수가 없으면 1개, sort_blocks(which=) 로 같은 색 후보를 고른다.
13:30 부터 맨 위·맨 아래·가까운·먼(top·bottom·near·far)과 "분홍 옆"(next_to → which={"next_to": "red"})도. 규칙이 못 읽은
위치 낱말은 router.classify 가 LLM 에 묻는다. 못 읽었거나(LLM none·멈춤) 로봇이 그 값을 모르면 거부한다(위치를 버리고 부르면
엉뚱한 블록을 집는다).
불량 색(10-23): "초록 불량 옮겨 줘" → defect_cls=blue_defect, sort_blocks(["blue_defect"]) 로 그 색 불량만. 좌표 집기가 아니거나
로봇이 그 이름을 모르거나 여러 단계 계획이면 거부한다(색 없이 부르면 다른 색 불량까지 옮긴다).
"멈춰"는 그래프 앞(run_command)에서 규칙으로 바로 처리한다. 받아쓰기 환청("감사합니다." 등)도 그래프 앞에서 거른다(hearing.py).
집는 방식(10-23 13:10 사용자 결정): 학습한 동작(ACT)은 손 전달(handoff)에만 쓰고, 옮기기(defect·red·blue·plan 단계)는 늘
좌표 집기. pick_red·pick_blue·pick_defect 정책은 부르지 않는다(act_node·tools.run_policy 가 handoff 말고는 거부).
옮기기에 "ACT로·정책으로·학습한 동작으로" 단서가 붙으면 좌표로 하고 첫 말에 "학습한 동작은 손 전달에만 써요."를 붙인다.
좌표 집기를 못 불러왔으면 옮기기는 거부(ACT 로 대신하지 않음). set_method("act") 도 거부. 첫 말에 방식을 넣는다.
색: red_* = 분홍, blue_* = 초록(10-21 밤 변경, 코드 식별자는 그대로). 사람에게 하는 말만 분홍·초록.
"""
import operator
import queue
import re
import sqlite3
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Annotated, Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import backends, config, hearing, llm, report, runlog
from .events import emit, make, now_iso
from . import router as _rt
from .router import DEFECT_KO, GREEN as GREEN_RE, NEXT_KO, PINK as PINK_RE, WHICH_KO, WORD_INTENT, arrange_target, cant_do, \
    classify, colored_defect, defect_color, excluded, is_arrange, is_stop, looks_question, method_cue, parse_count, \
    parse_which
from .tools import detect_objects, run_policy, search_manual, stop_robot

STOP = threading.Event()        # "멈춰" 가 세우고, 다음 명령이 시작할 때 내린다

TASKS = {
    "defect": {"policy": "pick_defect", "src": "input", "keys": ("red_defect", "blue_defect"),
               "dst": "bowl_defect", "label": "불량", "verb": "분리했", "plan": "골라낼게요",
               "step": "불량 {n}개 골라내기", "doing": "불량을 골라내는 중이에요."},
    "red": {"policy": "pick_red", "src": "input", "keys": ("red_ok",), "dst": "bowl_red",
            "label": "분홍 블록", "verb": "담았", "plan": "옮길게요", "step": "분홍 블록 {n}개 분홍 그릇에 옮기기",
            "doing": "분홍 블록을 옮기는 중이에요."},
    "blue": {"policy": "pick_blue", "src": "input", "keys": ("blue_ok",), "dst": "bowl_blue",
             "label": "초록 블록", "verb": "담았", "plan": "옮길게요", "step": "초록 블록 {n}개 초록 그릇에 옮기기",
             "doing": "초록 블록을 옮기는 중이에요."},
    "handoff": {"policy": "handoff", "src": "bowl_red", "keys": ("red_ok",), "dst": None,
                "label": "분홍 블록", "verb": "전달했", "plan": "건네 드릴게요", "step": "분홍 블록 {n}개 손에 건네기",
                "doing": "분홍 블록을 건네 드릴게요."},
}
EMPTY_MSG = {"defect": "투입 구역에 불량품이 없습니다.", "red": "투입 구역에 분홍 블록이 없습니다.",
             "blue": "투입 구역에 초록 블록이 없습니다.", "handoff": "분홍 그릇에 분홍 블록이 없습니다."}


def _task(s) -> dict:
    """이 명령의 TASKS 항목. 불량 색이 있으면(10-23 defect_cls) 그 색 불량만 세고(keys) 말한다(label "초록 불량")."""
    t, c = TASKS[s["intent"]], s.get("defect_cls")
    return dict(t, keys=(c,), label=DEFECT_KO[c]) if s["intent"] == "defect" and c in DEFECT_KO else t


class State(TypedDict, total=False):
    text: str
    thread_id: str
    started: str              # 명령 시작 시각(ISO)
    intent: Optional[str]
    count: Optional[int]
    route: dict
    goal: int                 # 이번 명령에서 옮길 개수
    done: int                 # 옮긴 개수
    moved_by_class: dict
    attempts: int
    misses: int               # 지금 블록에서 연속으로 놓친 횟수
    dropped: int              # 이번 명령(단계)에서 구역 밖에 떨어뜨린 개수
    max_attempts: int
    obs: dict                 # 마지막 detect_objects 결과
    last_policy: dict
    status: str               # running | done | nothing | gave_up | limit | stopped | no_hand | error
    error_msg: str
    answer: str
    steps: list               # plan 의도: 검증한 단계 [{intent, count, expect}] (비면 단일 명령)
    step_i: int
    step_results: list
    plan_started: str
    arrange: dict             # 좌표 집기: {digit, color, det, dry, go}
    coord: dict               # 좌표 집기 그릇 옮기기 요약: {blocks, time_s, summary, gates, reason, moved, run_id}
    cmd: str                  # 방식 단서를 뺀 말(판별·계획에 씀, 10-22 r21). text 는 원문 그대로(기록·질문 검색)
    robot_method: str         # 옮기기 방식 — 늘 coord(10-23 13:10, handoff 는 이 값과 상관없이 ACT)
    method_by: str            # cue(말에 단서) | default(기본값)
    method_fallback: bool     # 10-23 13:10 부터 늘 False(좌표 집기를 못 불러오면 ACT 로 대신하지 않고 거부)
    method_asked: Optional[str]   # 말 속 방식 단서 act | coord | None — 옮기기에 act 면 첫 말에 ACT_ONLY_MSG
    which: Optional[str]      # 위치 단서 left | right | middle | top | bottom | near | far | mixed(둘 이상, 거부) — defect·red·blue 만
    next_to: Optional[str]    # 관계 "분홍 옆" red | blue | red_defect | blue_defect | defect(색 없는 불량, 거부) (10-23 13:30)
    target_by: Optional[str]  # 위치를 읽은 길 rule | llm | llm_none·need_llm·ref_rel·llm_down(못 읽음 → 거부) | None
    defect_cls: Optional[str]  # 불량 색 red_defect | blue_defect | mixed(뺀 색만, 거부) — defect 만(10-23)
    history: Annotated[list, operator.add]   # 대화(thread) 단위로 쌓이는 명령 요약 — 체크포인터가 보존


FRESH_OBS_S = 3.0            # plan 단계 사이에 다시 찍지 않고 쓸 수 있는 감지 결과의 나이


def _age_s(obs: dict) -> float | None:
    try:
        return (datetime.now().astimezone() - datetime.fromisoformat(obs["ts"])).total_seconds()
    except Exception:
        return None


def _count(obs: dict, zone: str, keys) -> int:
    return sum(obs["zones"][zone].get(k, 0) for k in keys)


class VisionError(RuntimeError):
    pass


VISION_MSG = "카메라로 개수를 세지 못했습니다."


def _detect(node: str) -> dict:
    emit("tool", node, name="detect_objects", status="start")
    t0 = time.perf_counter()
    try:
        obs = detect_objects.invoke({})
    except Exception as e:                  # 카메라 사용 중·모델 오류 등
        emit("error", node, message=f"detect_objects: {type(e).__name__}: {e}")
        raise VisionError(str(e)) from e
    emit("tool", node, name="detect_objects", status="end", result=obs,
         latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return obs


def _breakdown(moved_by_class: dict) -> str:
    parts = [f"{'분홍' if k.startswith('red') else '초록'} {n}" for k, n in sorted(moved_by_class.items(),
             key=lambda kv: kv[0].startswith("blue")) if n]
    return ", ".join(parts)


# ------------------------------------------------------------------ 노드
# 질문 미리 검색(10-22 r10): 질문으로 보이는 말이면 판별(LLM 이면 약 0.85초)과 동시에 search_manual 을 시작한다.
# 판별이 question 이 아니면 결과를 버린다(돌고 있는 검색은 끝까지 가지만 로봇·말에는 안 쓰임)
_PREFETCH: dict[str, Future] = {}
_PREFETCH_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qa-prefetch")
def _prefetch(text: str) -> bool:
    if not config.QA_PREFETCH or not _rt.looks_question(text):
        return False
    while len(_PREFETCH) >= 8:              # 꺼내지 않은 것(판별 오류 등)이 쌓이지 않게
        _PREFETCH.pop(next(iter(_PREFETCH)))
    _PREFETCH[text] = _PREFETCH_POOL.submit(search_manual.invoke, {"query": text})
    return True


# 집는 방식(10-22 r21 → 10-23 13:10 사용자 결정): 옮기기는 늘 좌표 집기, 학습한 동작(ACT)은 손 전달만.
# BALL_COORD 는 더 이상 방식을 정하지 않는다. 말 속 ACT 단서는 첫 말로 알리고 좌표로 한다
METHODS = ("act", "coord")
HOW = {"act": "학습한 동작으로", "coord": "좌표로"}          # 첫 말에 넣는 방식(handoff 는 늘 "학습한 동작으로")
METHOD = {"default": "coord"}
ACT_ONLY_MSG = "학습한 동작은 손 전달에만 써요."
COORD_OFF_MSG = "좌표 집기를 불러오지 못해 블록을 옮길 수 없어요."


def coord_available() -> bool:
    return backends.get().get("sort") is not None


def which_available() -> bool:
    """좌표 집기 로봇 함수가 which(왼쪽·오른쪽·가운데)를 받는가(backends info["which"], 10-23)."""
    return coord_available() and bool(backends.info().get("which"))


def which_values() -> list:
    """로봇이 아는 which 값(backends info["which_values"], 시작 때 robot/coord_pick/plan.WHICH 를 본다, 10-23 13:30)."""
    return list(backends.info().get("which_values") or (["left", "right", "middle"] if which_available() else []))


def next_to_values() -> list:
    """로봇이 아는 next_to 기준(backends info["next_to"], robot/coord_pick/plan.NEXT_TO)."""
    return list(backends.info().get("next_to") or [])


def defect_color_available() -> bool:
    """좌표 집기 로봇 함수가 불량 색 이름(red_defect·blue_defect)을 아는가(backends info["defect_color"], 10-23)."""
    return coord_available() and bool(backends.info().get("defect_color"))


def get_method() -> dict:
    """지금 기본 방식과 쓸 수 있는지(대시보드 표시용). coord_backend: real(BALL_ROBOT=1, 로봇 함수 — 보정·금지 구역 문은
    로봇 함수가 다시 봄) · mock(가짜 로봇) · off(못 불러옴)."""
    info = backends.info()
    return {"default": METHOD["default"], "coord_available": coord_available(), "coord_backend": info.get("coord"),
            "robot": info.get("robot"), "home": info.get("home"), "which": which_available(),
            "which_values": which_values() if which_available() else [], "next_to": next_to_values() if which_available() else [],
            "defect_color": defect_color_available(), "unsort": info.get("unsort"), "labels": dict(HOW),
            "act_for": ["handoff"]}


def set_method(method: str) -> dict:
    """기본 방식을 바꾼다(다음 명령부터). act 는 ValueError(10-23 13:10, 학습한 동작은 손 전달만).
    좌표 집기를 못 불러왔으면 coord 도 ValueError."""
    if method not in METHODS:
        raise ValueError(f"방식은 {METHODS} 중 하나: {method!r}")
    if method == "act":
        raise ValueError(ACT_ONLY_MSG + " 옮기기는 늘 좌표로 해요.")
    if method == "coord" and not coord_available():
        raise ValueError("좌표 집기(sort_blocks)를 불러오지 못했습니다: " + str(backends.info()["reason"].get("coord")))
    METHOD["default"] = method
    return get_method()


def classify_node(s: State) -> dict:
    emit("node", "classify", text=s["text"])
    pre = _prefetch(s["text"])
    asked, cue, clean = method_cue(s["text"])
    cmd = clean if asked and not looks_question(s["text"]) else s["text"]     # 질문꼴은 원문으로 판별
    r = classify(cmd)
    # 위치·관계(10-23, 13:30 top·bottom·near·far·next_to). router.classify 가 규칙 → 필요하면 LLM 으로 읽은 target.
    # 판별을 갈아 끼운 경우(시험 정답표)엔 규칙만
    tg = (r.get("target") if "target" in r else _rt.rule_target(cmd) if r["intent"] in COORD_INTENTS else None) or {}
    which, next_to, target_by = tg.get("which"), tg.get("next_to"), tg.get("by")
    dcls = defect_color(_rt.strip_ref(cmd)) if r["intent"] == "defect" else None   # 불량 색(10-23): 그 색 불량만
    count = r["count"] or (1 if (which and which != "mixed") or next_to else None)   # 위치가 있으면 1개
    if which:
        r = dict(r, which=which)
    if next_to:
        r = dict(r, next_to=next_to)
    if dcls:
        r = dict(r, defect_cls=dcls)
    if pre:
        r["prefetch"] = True
        if r["intent"] != "question":
            _PREFETCH.pop(s["text"], None)
    emit("tool", "classify", name="intent_router", status="end", result=r)
    if r.get("llm_down"):      # 화면에 띄운다(UI 는 "llm:" 으로 시작하는 오류를 로그로만 남긴다)
        t = f"{config.LLM_CLASSIFY_TIMEOUT_S:g}초"
        emit("error", "classify", message=f"LLM 서버 응답 없음({t} 안에 답 없음). " + (
            "판별이 불확실해 로봇을 움직이지 않았어요." if r["intent"] == "none"
            else f"임베딩 1위({r['intent']})로 처리했어요."), llm_down=True)
    # 옮기기는 늘 좌표 집기(10-23 13:10). 말 속 단서(asked)는 첫 말을 고르는 데만 쓴다(act → ACT_ONLY_MSG, handoff 에 coord →
    # "손에 건네기는 학습한 동작으로만 해요.")
    pick = {"method": "coord", "by": "cue" if asked else "default", "cue": cue, "asked": asked,
            "default": METHOD["default"], "fallback": False, "cmd": cmd if cmd != s["text"] else None}
    emit("tool", "classify", name="method_pick", status="end", result=pick)
    return {"intent": r["intent"], "count": count, "route": r, "cmd": cmd, "robot_method": "coord",
            "method_by": pick["by"], "method_fallback": False, "method_asked": asked, "which": which,
            "next_to": next_to, "target_by": target_by, "defect_cls": dcls}


WHICH_MSG = "아직 위치로 고르기는 못 해요."
POS_MSG = "그 위치로는 아직 고르지 못해요."       # 10-23 13:30: 읽지 못한 위치·로봇이 모르는 위치 — 위치 말을 버리지 않는다
POS_DEFECT_MSG = POS_MSG + " 불량 옆이면 '초록 불량 옆 분홍'처럼 불량 색을 말해 주세요."
POS_DOWN_MSG = "위치를 알아듣지 못해 옮기지 않았어요. 다시 말해 주세요."


def _which_refusal(s: State) -> str | None:
    """위치 단서 명령을 할 수 없는 이유. 로봇 함수가 which 를 받을 때만 한다(좌표 집기를 못 불러온 것은 plan_node 가 먼저 거부).
    LLM 이 멈췄거나·못 읽었거나(none)·다른 블록 기준 위·아래이거나·로봇이 그 값을 모르면 거부(10-23 13:30)."""
    which, nt, by = s.get("which"), s.get("next_to"), s.get("target_by")
    if not (which or nt or by):
        return None
    if by == "llm_down":
        return POS_DOWN_MSG
    if which == "mixed":
        return "위치는 한 번에 하나만 말해 주세요. 예를 들어 '왼쪽 초록 블록 옮겨 줘'처럼요."
    if nt == "defect":
        return POS_DEFECT_MSG
    if by in ("llm_none", "need_llm", "ref_rel", "unsupported"):
        return POS_MSG
    if not which_available():
        return WHICH_MSG
    if which and which not in which_values() or nt and nt not in next_to_values():
        return POS_MSG
    return None


DCOLOR_MSG = "아직 불량을 색으로 고르지는 못 해요."


def _defect_color_refusal(s: State) -> str | None:
    """불량 색 명령을 할 수 없는 이유(10-23). 좌표 집기 + 로봇이 red_defect·blue_defect 를 알 때만(pick_defect 정책은 색을 못 고른다)."""
    c = s.get("defect_cls")
    if not c:
        return None
    if c == "mixed":
        return "불량 색은 하나만 말해 주세요. 예를 들어 '초록 불량 옮겨 줘'처럼요."
    return None if defect_color_available() else DCOLOR_MSG


def plan_node(s: State) -> dict:
    intent, count, which, dcls = s["intent"], s.get("count"), s.get("which"), s.get("defect_cls")
    nt = s.get("next_to")
    t = _task(s)
    emit("node", "plan", intent=intent, policy=t["policy"], count=count, **({"which": which} if which else {}),
         **({"next_to": nt} if nt else {}), **({"defect_cls": dcls} if dcls else {}))
    # 좌표 집기를 못 불러왔으면 옮기기는 거부(10-23 13:10, ACT 정책으로 대신하지 않는다)
    why = COORD_OFF_MSG if intent in COORD_INTENTS and not coord_available() else None
    why = why or (_which_refusal(s) if intent in COORD_INTENTS else None) or _defect_color_refusal(s)
    if why:                       # 카메라·로봇을 거치지 않고 바로 거부(which·불량 색을 빼고 부르면 엉뚱한 블록을 집는다)
        emit("tool", "plan", name="sort_blocks", status="unavailable", which=which, next_to=nt,
             target_by=s.get("target_by"), defect_cls=dcls, method=s.get("robot_method"), error=why)
        return {"goal": 0, "status": "refused", "error_msg": why}
    obs = s.get("obs") or {}
    age = _age_s(obs)
    if s.get("steps") and age is not None and age <= FRESH_OBS_S:
        # plan 단계 사이: 앞 단계 verify(또는 planner)가 방금 찍은 결과를 그대로 쓴다(실제 카메라 한 장 0.75초, vision README)
        emit("tool", "plan", name="detect_objects", status="reuse", age_s=round(age, 2))
    else:
        try:
            obs = _detect("plan")
        except VisionError:
            return {"goal": 0, "status": "error", "error_msg": VISION_MSG}
    avail = _count(obs, t["src"], t["keys"])
    goal = min(count or 1, avail) if intent == "handoff" else (min(count, avail) if count else avail)
    if avail == 0:
        return {"obs": obs, "goal": 0, "status": "nothing"}
    how = HOW["coord" if coord_on(s) else "act"]      # 첫 말에 방식(10-22 r21). handoff 는 늘 학습한 동작
    where = WHICH_KO.get(which) if which else f"{NEXT_KO[nt]} 옆" if nt else None
    label = f"{where} {t['label']}" if where else t["label"]     # "맨 위 초록 블록 1개…" · "분홍 옆 초록 블록 1개를 좌표로 옮길게요."
    if intent == "defect" and not count and not dcls:
        cls = {k: obs["zones"]["input"][k] for k in t["keys"]}
        msg = f"불량품 {goal}개({_breakdown(cls)})를 {how} {t['plan']}."
    elif count and count > avail:
        msg = f"{t['label']}이 {avail}개뿐이라 {avail}개만 {how} {t['plan']}."
    else:
        msg = f"{label} {goal}개를 {how} {t['plan']}."
    if intent == "handoff" and s.get("method_asked") == "coord":
        msg = "손에 건네기는 학습한 동작으로만 해요. " + msg
    if intent in COORD_INTENTS and s.get("method_asked") == "act":     # 10-23 13:10: 옮기기에 ACT 단서 → 좌표로 하고 알림
        msg = f"{ACT_ONLY_MSG} {msg}"
    if s.get("steps"):           # plan 단계 안: 계획은 이미 알렸으니 한 문장만(실연 3분 예산, 10-22)
        msg = t["doing"]
        if intent == "handoff":
            msg = "분홍 블록은 학습한 동작으로 건네 드릴게요."
    emit("say", "plan", text=msg)
    return {"obs": obs, "goal": goal, "status": "running",
            "max_attempts": min(config.MAX_ATTEMPTS, goal * (1 + config.RETRY_PER_ITEM))}


def hand_wait_node(s: State) -> dict:
    emit("node", "hand_wait", timeout_s=config.HAND_WAIT_S)
    obs, asked = s["obs"], False
    deadline = time.monotonic() + config.HAND_WAIT_S
    while not obs["hand"]["present"]:
        if STOP.is_set():
            return {"status": "stopped", "obs": obs}
        if not asked:
            emit("say", "hand_wait", text="손을 내밀어 주세요.")
            asked = True
        if time.monotonic() >= deadline:
            return {"status": "no_hand", "obs": obs}
        time.sleep(config.HAND_POLL_S)
        try:
            obs = _detect("hand_wait")
        except VisionError:
            return {"status": "error", "error_msg": VISION_MSG}
    emit("node", "hand_wait", hand=obs["hand"])
    return {"obs": obs}


# 홈 맞춤(10-22 r20): 정책·좌표 집기 직전에 팔이 홈(config home_n)이 아니면 go_home 으로 천천히 되돌린다. 16:53 실물 handoff 는
# 앞 명령이 '멈춤'으로 끝나 팔이 홈 밖에서 시작했고, 조기 종료(시작 자세로 돌아와 멈춤)가 진짜 홈 정지를 못 알아채 남은 시간에
# 다음 블록을 집으러 갔다. 측정은 go_home 이 컨테이너에서 연결해 읽는다(이미 홈이면 already_home, 60° 넘게 멀면 too_far)
# ok    = HOME_ALIGN=after_stop 용: 마지막 로봇 동작이 정상으로 끝났나(멈춤·오류·첫 동작이면 False)
# fresh = 동작 뒤 홈 복귀(r22)가 된 시각(monotonic). 그 뒤 로봇이 안 움직였으면 다음 홈 맞춤은 다시 재지 않는다(이미 홈 4초)
# moved = 이번 명령(plan 단계)에서 팔이 움직였나 → 끝나면 집게를 열고 홈으로
_HOME = {"ok": False, "fresh": None, "moved": False}
HOME_FAR_MSG = "팔이 홈 자세에서 너무 멀어요. 팔을 홈 근처로 옮겨 주세요."


def _moving() -> None:
    """정책·좌표 집기·숫자 배치가 팔을 움직이기 시작했다(거부가 아니면 부른다)."""
    _HOME.update(moved=True, fresh=None)


def _go_home(node: str, *, open_gripper: bool = False, studio: bool = True, max_deg: float | None = None) -> dict:
    """go_home(dry_run=False) 한 번(tool 이벤트 go_home). studio=True 면 Studio 세션 거부를 정리하고 다시(_robot_call).
    max_deg: 홈에서 이 각도보다 멀면 거부(없으면 로봇 기본 safety.home_max_deg)."""
    home = backends.get()["home"]
    kw = {"max_deg": max_deg} if max_deg else {}

    def call() -> dict:
        try:
            return _clean(home(False, open_gripper=open_gripper, **kw))
        except Exception as e:
            return {"ok": False, "reason": "error", "error": f"{type(e).__name__}: {e}"}
    emit("tool", node, name="go_home", status="start", open_gripper=open_gripper, **kw)
    t0 = time.perf_counter()
    r = _robot_call(node, call) if studio else call()
    emit("tool", node, name="go_home", status="end", open_gripper=open_gripper, result=r, **kw,
         latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    return r


def _home_align(node: str) -> str | None:
    """홈 맞춤. 못 하면 말할 이유(그 명령은 거기서 멈춘다), 했거나 안 해도 되면 None."""
    if backends.get().get("home") is None or config.HOME_ALIGN == "off" \
            or (config.HOME_ALIGN == "after_stop" and _HOME["ok"]):
        return None
    fresh = _HOME.get("fresh")
    if _HOME["ok"] and fresh is not None and time.monotonic() - fresh < config.HOME_FRESH_S:
        emit("tool", node, name="go_home", status="skip", why="방금 동작 뒤 홈 복귀를 했고 그 뒤 안 움직임",
             age_s=round(time.monotonic() - fresh, 1))
        return None
    r = _go_home(node)
    _HOME["ok"] = bool(r.get("ok"))
    if r.get("ok") or r.get("reason") == "stopped":     # done·already_home / 멈춤은 부른 쪽이 STOP 을 본다
        return None
    emit("error", node, message=f"go_home: {r.get('reason')}: {r.get('error') or ''}")
    if r.get("reason") == "too_far":
        return HOME_FAR_MSG
    return r.get("studio_msg") or _REFUSED.get(_refuse_code(r) or "", "팔을 홈 자세로 맞추지 못했습니다.")


def _home_after(node: str, status: str | None) -> None:
    """동작이 끝나면 집게를 최대로 열고 홈으로(10-22 r22, go_home(open_gripper=True), BALL_ROBOT=1 일 때만 실제).
    '멈춤'이면 그 자리 유지 — 다음 명령 전에 홈 맞춤(r20)이 돈다. 실패해도 말하지 않는다(다음 홈 맞춤이 다시 본다)."""
    moved, _HOME["moved"] = _HOME["moved"], False
    if not moved or status == "stopped" or STOP.is_set() or config.HOME_AFTER == "off" \
            or backends.get().get("home") is None:
        return
    r = _go_home(node, open_gripper=True, studio=False)
    if r.get("ok"):
        _HOME.update(ok=True, fresh=time.monotonic())
    else:
        _HOME.update(ok=False, fresh=None)
        emit("error", node, message=f"go_home(open_gripper): {r.get('reason')}: {r.get('error') or ''}")


# Studio 세션 정리(10-22 r22, 사용자 허락 17:17): Studio 원격조작 탭을 닫거나 옮겨도 45초 idle + 10초 뒤에야 팔을 놓아,
# 그동안 정책·좌표 집기·홈 맞춤이 studio_session·robot_busy 로 거부됐다. 녹화 세션(is_recording·dataset_loaded)은 멈추지 않는다
STUDIO_CODES = ("studio_session", "robot_busy")
STUDIO_CLEAR_MSG = "Studio 연결을 정리하고 바로 할게요."
STUDIO_WAIT_MSG = "로봇이 풀리기를 잠깐 기다릴게요."
STUDIO_REC_MSG = "Studio 에서 녹화 중이라 그대로 뒀어요. 녹화를 끝낸 뒤 다시 말해 주세요."
STUDIO_OPEN_MSG = "Studio 화면이 열려 있어요. ←로 나와 주세요."


def _refuse_code(r: dict) -> str | None:
    return (r.get("error") or "").split(":", 1)[0].strip() if r.get("reason") == "refused" else None


def _session_brief(x: dict) -> dict:
    """세션 하나 → {session, status, source, recording}. recording: True(녹화·녹화 화면) · False · None(상태를 모름)."""
    a = x.get("activity") if isinstance(x.get("activity"), dict) else None
    if x.get("status") == "stopped":
        rec = False                       # 이미 끝나는 중
    elif a is None:
        rec = None
    else:
        rec = bool(a.get("is_recording") or a.get("dataset_loaded"))
    return {"session": x.get("session_name"), "status": x.get("status"), "follower": x.get("follower_name"),
            "source": (a or {}).get("follower_source"), "recording": rec}


def _robot_call(node: str, fn) -> dict:
    """로봇 함수 fn() 을 부르고, Studio 가 팔을 쥐고 있어 거부되면 정리하고 다시 부른다.
    a) 세션이 있으면 "Studio 연결을 정리하고 바로 할게요." → POST …/stop → 세션이 사라지고 잠금이 풀릴 때까지 다시
    b) 세션 없이 잠금만이면 기다리며 다시. 둘 다 STUDIO_WAIT_S(15초)까지. 녹화·상태 모름이면 멈추지 않고 studio_msg 를 붙여 돌려준다.
    c) 그래도 안 되면 마지막 거부 결과 그대로(부른 쪽이 지금처럼 거부 문구)."""
    r = fn()
    studio = backends.get().get("studio")
    if _refuse_code(r) not in STUDIO_CODES or studio is None or STOP.is_set():
        return r
    try:
        sessions = studio["list"]()
    except Exception as e:
        emit("error", node, message=f"studio sessions: {type(e).__name__}: {e}")
        return r
    brief = [_session_brief(x) for x in sessions]
    emit("tool", node, name="studio_sessions", status="end", result=brief, refused=_refuse_code(r))
    if any(b["recording"] for b in brief):
        return dict(r, studio_msg=STUDIO_REC_MSG)
    if any(b["recording"] is None for b in brief):
        return dict(r, studio_msg=STUDIO_OPEN_MSG)
    if sessions:
        emit("say", node, text=STUDIO_CLEAR_MSG)
        for b in brief:
            try:
                code = studio["stop"](b["session"])
            except Exception as e:
                code = f"{type(e).__name__}: {e}"
            emit("tool", node, name="studio_stop", status="end", session=b["session"], result=code)
    else:
        emit("say", node, text=STUDIO_WAIT_MSG)
    deadline = time.monotonic() + config.STUDIO_WAIT_S
    while not STOP.is_set() and time.monotonic() < deadline:
        time.sleep(config.STUDIO_POLL_S)
        if sessions:                       # 세션이 사라질 때까지는 목록만 본다(로봇 사전 점검보다 가볍다)
            try:
                sessions = studio["list"]()
            except Exception:
                sessions = []
            if sessions:
                continue
        r = fn()
        if _refuse_code(r) not in STUDIO_CODES:
            break
    emit("tool", node, name="studio_wait", status="end", ok=_refuse_code(r) not in STUDIO_CODES,
         reason=r.get("reason"), error=r.get("error"))
    return r


def act_node(s: State) -> dict:
    if STOP.is_set():
        return {"status": "stopped"}
    if s["intent"] != "handoff":     # 10-23 13:10: 학습한 동작은 손 전달만. pick_red·pick_blue·pick_defect 는 부르지 않는다
        emit("error", "act", message=f"act_only_handoff: {s['intent']}")
        return {"status": "error", "error_msg": f"{ACT_ONLY_MSG} 옮기기는 좌표로 해요."}
    t = TASKS[s["intent"]]
    n = s.get("attempts", 0) + 1
    emit("node", "act", policy=t["policy"], attempt=n, done=s.get("done", 0), goal=s["goal"])
    why = _home_align("act")
    if STOP.is_set():
        return {"status": "stopped"}
    if why:
        return {"status": "error", "error_msg": why}
    args = {"name": t["policy"]}            # timeout_s 는 넘기지 않는다(policies.json 정책별 값)
    emit("tool", "act", name="run_policy", status="start", args=args)
    def call() -> dict:
        try:
            return run_policy.invoke(args)
        except Exception as e:
            return {"name": t["policy"], "ok": False, "returncode": None, "reason": "error",
                    "error": f"{type(e).__name__}: {e}"}
    res = _robot_call("act", call)
    emit("tool", "act", name="run_policy", status="end", args=args, result=res)
    _HOME["ok"] = res.get("reason") in POLICY_OK_REASONS
    if res.get("reason") != "refused":
        _moving()
    # refused = 시작 전 점검에서 거부(Studio 세션·장치 잠금·모델 미연결), 팔은 안 움직였다 → 재시도하지 않는다.
    # error·timeout 은 실패한 시도로 보고 verify 에서 다시 세어 1회 재시도한다.
    if res.get("reason") == "refused":
        emit("error", "act", message=res.get("error") or "refused")
        return {"attempts": n, "last_policy": res, "status": "error", "error_msg": _refused_msg(t, res)}
    return {"attempts": n, "last_policy": res}


_REFUSED = {  # robot/README 1절 error 코드 → 말
    "studio_session": "Studio 화면이 로봇을 쓰고 있어 실행하지 않았습니다. Studio 화면에서 나온 뒤 다시 말해 주세요.",
    "robot_busy": "로봇을 다른 곳에서 쓰고 있어 실행하지 않았습니다.",
    "already_running": "다른 정책이 이미 실행 중이라 실행하지 않았습니다.",
}


def _refused_msg(t: dict, res: dict) -> str:
    if res.get("studio_msg"):
        return res["studio_msg"]
    code = (res.get("error") or "").split(":", 1)[0].strip()
    if code == "no_model":
        return f"{t['policy']} 정책에 연결된 모델이 없어 실행하지 않았습니다."
    return _REFUSED.get(code, "로봇이 실행을 거부했습니다.")


# run_policy 가 정상으로 끝난 방식(robot/README 1절). home_reached = 홈 복귀 정지 감지로 일찍 끝남(10-22 로봇 조기 종료)
POLICY_OK_REASONS = ("home_reached", "duration_elapsed")


def verify_node(s: State) -> dict:
    t = TASKS[s["intent"]]
    emit("node", "verify", attempt=s["attempts"])
    before = s["obs"]
    pol = s.get("last_policy") or {}
    if pol.get("camera_released") is False:   # 정책의 카메라 publisher 가 아직 안 끝났으면 잠깐 기다린다
        time.sleep(config.CAMERA_WAIT_S)
    try:
        after = _detect("verify")
    except VisionError:
        try:                                  # 카메라가 늦게 풀리는 경우 한 번 더
            time.sleep(config.CAMERA_WAIT_S)
            after = _detect("verify")
        except VisionError:
            return {"status": "error", "error_msg": VISION_MSG}
    dec = {k: max(0, before["zones"][t["src"]][k] - after["zones"][t["src"]][k]) for k in t["keys"]}
    n_dec = sum(dec.values())
    if t["dst"]:
        n_inc = _count(after, t["dst"], t["keys"]) - _count(before, t["dst"], t["keys"])
        moved = min(n_dec, n_inc) if n_dec >= 1 and n_inc >= 1 else 0
    else:                                   # 건네기: 분홍 그릇에서 줄었으면 옮긴 것
        moved = n_dec
    # 성공 = 정책이 정상 종료(robot ok: reason=home_reached 홈 복귀 감지로 일찍 끝남 · duration_elapsed 시간 경과)
    # + 다시 세어 정확히 1개 옮겨짐. 기준서 5.2·5.4: 두 개를 함께 옮겼으면 개수는 세되 시도는 실패. 개수는 카메라 기준
    policy_ok = bool(pol.get("ok")) or pol.get("reason") in POLICY_OK_REASONS
    ok = moved == 1 and policy_ok
    # 구역 밖(outside)에 늘어난 블록 = 옮기다 떨어뜨림(기준서 5.4 실패). objects 가 없는 카메라면 0
    dropped = max(0, sum(report.outside_counts(after).values()) - sum(report.outside_counts(before).values()))
    if ok:
        reason = None
    elif dropped and moved == 0:
        reason = "구역 밖으로 떨어짐"
    elif moved > 1:
        reason = "두 개 이상을 함께 집음"
    elif not policy_ok:
        reason = f"정책 {pol.get('reason') or 'error'}" + (f" ({pol['error']})" if pol.get("error") else "")
    else:
        reason = "옮겨지지 않음"
    this, left = {}, moved                  # 이번 시도에서 옮긴 개수를 클래스별로
    for k, n in dec.items():
        if min(n, left):
            this[k] = min(n, left)
            left -= this[k]
    by_class = dict(s.get("moved_by_class") or {})
    for k, n in this.items():
        by_class[k] = by_class.get(k, 0) + n
    runlog.append({
        "ts": now_iso(), "thread_id": s["thread_id"], "text": s["text"], "intent": s["intent"],
        "policy": t["policy"], "attempt": s["attempts"], "before": before["zones"], "after": after["zones"],
        "ok": ok, "moved": moved, "moved_by_class": this, "fail_reason": reason, "dropped": dropped,
        "policy_ok": policy_ok, "policy_reason": pol.get("reason"), "early_stop": pol.get("early_stop"),
        "returncode": pol.get("returncode"), "duration_s": pol.get("duration_s"),
        "backend": backends.info()["robot"],
    })
    done = s.get("done", 0) + moved
    misses = 0 if ok else s.get("misses", 0) + 1
    remaining = _count(after, t["src"], t["keys"])
    until_empty = s["intent"] != "handoff" and not s.get("count")
    if STOP.is_set():
        status = "stopped"
    elif remaining == 0 or (not until_empty and done >= s["goal"]):
        status = "done"
    elif misses > config.RETRY_PER_ITEM:
        status = "gave_up"
    elif s["attempts"] >= s["max_attempts"]:
        status = "limit"
    else:
        status = "running"
        if not ok:
            emit("say", "verify", text="두 개를 함께 집었어요. 다음은 하나씩 해 볼게요." if moved > 1
                 else "블록을 구역 밖에 떨어뜨렸어요. 한 번 더 해 볼게요." if dropped
                 else "놓쳤어요. 한 번 더 해 볼게요." if policy_ok
                 else "동작이 중간에 끊겼어요. 한 번 더 해 볼게요.")
    emit("node", "verify", ok=ok, moved=moved, done=done, remaining=remaining, status=status,
         fail_reason=reason)
    return {"obs": after, "done": done, "misses": misses, "moved_by_class": by_class, "status": status,
            "dropped": s.get("dropped", 0) + dropped}


def finish_node(s: State) -> dict:
    intent, st, done = s["intent"], s.get("status"), s.get("done", 0)
    t, colored = _task(s), s.get("defect_cls") in DEFECT_KO
    emit("node", "finish", status=st)
    if intent == "defect" and done and not colored:
        moved = f"불량 {done}개({_breakdown(s.get('moved_by_class') or {})})를 {t['verb']}습니다."
    else:
        moved = f"{t['label']} {done}개를 {t['verb']}습니다."
    if st == "nothing":
        msg = f"투입 구역에 {t['label']}이 없습니다." if colored else EMPTY_MSG[intent]
    elif st == "done":
        msg = moved
    elif st == "no_hand":
        msg = "손이 보이지 않아 건네지 않았습니다." if not done else f"{moved} 그다음은 손이 보이지 않아 멈췄습니다."
    elif st == "stopped":
        msg = "멈췄습니다." + (f" {moved}" if done else "")
    elif st == "refused":               # 위치 단서를 할 수 없음(10-23) — 로봇·카메라를 안 썼다
        msg = s.get("error_msg") or WHICH_MSG
    elif st == "error":
        why = s.get("error_msg") or "로봇 정책을 실행하지 못했습니다."
        msg = f"{moved + ' ' if done else ''}{why}" + ("" if why.endswith("주세요.") else " 확인해 주세요.")
    elif st == "gave_up":               # 기준서 5.5: 무엇이 왜 안 됐는지 말하고 사람 확인 요청
        tries = _KO_TIMES.get(1 + config.RETRY_PER_ITEM, str(1 + config.RETRY_PER_ITEM))
        msg = (f"{moved + ' 그다음 ' if done else ''}{t['label']}을 {tries} 번 옮기지 못했습니다. 확인해 주세요.")
    else:   # limit
        msg = f"시도 한도 {s.get('max_attempts')}회에 걸려 멈췄습니다. {moved + ' ' if done else ''}확인해 주세요."
    if s.get("dropped"):     # 떨어진 블록은 사람이 주워야 한다(기준서 5.5)
        msg += f" {s['dropped']}개는 구역 밖에 떨어져 있습니다." + ("" if msg.endswith("확인해 주세요.") else " 확인해 주세요.")
    if s.get("steps") and st in ("done", "nothing") and not s.get("dropped"):
        emit("node", "finish", status=st, text=msg, quiet=True)    # plan 단계: 문제 없으면 말하지 않음(plan_report 가 말함)
    else:
        emit("say", "finish", text=msg)
    _home_after("finish", st)                # 말이 나간 뒤 집게를 열고 홈으로(멈춤이면 그 자리)
    return {"answer": msg, "history": [{
        "ts": now_iso(), "started": s.get("started"), "elapsed_s": _elapsed(s), "text": s["text"],
        "intent": intent, "status": st, "done": done, "goal": s.get("goal", 0), "attempts": s.get("attempts", 0),
        "moved_by_class": s.get("moved_by_class") or {}, "say": msg, "dropped": s.get("dropped", 0),
        "route": {k: (s.get("route") or {}).get(k) for k in ("method", "latency_ms")},
        "by": "coord" if coord_on(s) else "act",
    }]}


_KO_TIMES = {2: "두", 3: "세", 4: "네"}


def _elapsed(s: State) -> float | None:
    try:
        return round((datetime.now().astimezone() - datetime.fromisoformat(s["started"])).total_seconds(), 2)
    except Exception:
        return None


def _hits_text(hits: list[dict]) -> str:
    return "\n\n".join(f"[근거 {i + 1}] {h['text']}" for i, h in enumerate(hits))


def report_node(s: State) -> dict:
    emit("node", "report")
    try:
        obs = _detect("report")
    except VisionError:
        msg = f"{VISION_MSG} 확인해 주세요."
        emit("say", "report", text=msg)
        return {"answer": msg, "status": "error"}
    f = report.facts(s["thread_id"], obs, s.get("history") or [])
    # 근거는 실시간 RAG 대신 상황별로 미리 골라 둔 조항(rag.CITATION_QUERIES, 기준서가 바뀌면 다시 계산)
    keys = ["report"] + (["mismatch"] if not f["match"] else []) + (["check"] if f["checks"] else []) \
        + list(dict.fromkeys(c["intent"] for c in f["commands"] if c["intent"] in TASKS))
    t0 = time.perf_counter()
    try:
        from .rag import quick_citations
        cites, how = quick_citations(keys)
    except Exception as e:
        cites, how = {}, "error"
        emit("error", "report", message=f"citations: {type(e).__name__}: {e}")
    hits, seen = [], set()
    for key in keys:
        for h in cites.get(key, []):
            if h["section"] not in seen:
                seen.add(h["section"])
                hits.append(dict(h, why=key))
    emit("tool", "report", name="search_manual", status="end", args={"citations": keys, "source": how},
         result=[{k: h.get(k) for k in ("section", "why")} for h in hits],
         latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    # 말은 짧은 보고를 바로 한다(10-22 r11, 5초 안팎). 아래 LLM 문장은 화면(report 이벤트·HTML)용이라 말을 기다리게 하지 않는다
    spoken = report.spoken_sentence(f)
    emit("say", "report", text=spoken)
    base = report.template_sentence(f)
    sentence, method = base, "template"
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        trouble = not f["match"] or f["checks"] or f["input_left"] or f.get("outside")
        sys_msg = ("너는 스펀지 블록 선별 로봇의 작업 보고 담당이다. [사실]만 써서 음성으로 읽을 결과 보고를 "
                   "한국어 2문장 안팎으로 쓴다. [예시 문장]의 말투와 순서를 따른다. 규칙: 사실에 있는 숫자를 하나도 "
                   "빠뜨리거나 바꾸지 않는다. 사실에 없는 평가·추측·권고를 만들지 않는다. 손에 건넨 개수는 그릇 개수와 "
                   "따로 말한다. 마크다운·목록·따옴표 없이 문장만 쓴다."
                   + (" 문제(기록 불일치·사람 확인 요청·남은 블록)가 있으니, 마지막에 기준서 근거의 조항 번호를 들어 "
                      "사람이 할 일을 한 문장 덧붙인다." if trouble else ""))
        user = f"[사실]\n{report.fact_lines(f)}\n\n[예시 문장]\n{base}"
        if trouble:   # 문제가 있을 때만 그 문제의 조항(mismatch·check)을 근거로 준다
            user += f"\n\n[기준서 근거]\n{_hits_text([h for h in hits if h['why'] in ('mismatch', 'check')] or hits)}"
        out = llm.chat(max_tokens=200).invoke([SystemMessage(sys_msg), HumanMessage(user)]).content.strip()
        if out and report.sentence_ok(out, f):
            sentence, method = out, "llm"
        else:
            method = "template(llm 문장 검사 탈락)"
            emit("node", "report", llm_rejected=out)
    except Exception as e:
        emit("error", "report", message=f"llm: {type(e).__name__}: {e}")
    html_doc = report.html_report(f, sentence, obs, hits, s["thread_id"])
    path = runlog.day_dir() / f"report_{datetime.now().strftime('%H%M%S')}.html"
    path.write_text(html_doc, encoding="utf-8")
    emit("report", "report", text=sentence, spoken=spoken, method=method, html=html_doc,
         html_path=str(path.relative_to(config.ROOT)), facts=f, image_path=obs.get("image_path"),
         sources=[{k: h.get(k) for k in ("section", "rerank", "text")} for h in hits])
    return {"obs": obs, "answer": spoken, "status": "done",
            "history": [{"ts": now_iso(), "started": s.get("started"), "elapsed_s": _elapsed(s),
                         "text": s["text"], "intent": "report", "status": "done", "say": spoken}]}


# 답 = "[<문서 이름> <조항 번호>]에 따르면 " + LLM 이 쓴 한 문장. 문서·번호는 코드가 붙인다:
# 4B 가 제목까지 쓰면 답이 2~7초(약 20자/초)였고 "5. 처리 규칙 > 5.4" 를 보고 5.2 로 적기도 했다(10-21).
# 10-22 세 문서: 조항 번호가 문서마다 따로라(4.3 이 셋 다 있음) LLM 은 조항 번호 대신 근거 번호(1·2·3)를 고른다
_ANSWER_SYS = ("너는 스펀지 블록 선별 로봇의 안내 담당이다. [자료 근거]와 [이번 대화의 작업 기록]만 보고 질문에 답한다.\n"
               "첫 줄: 답에 쓴 근거 번호만(예: 2). 자료가 아니라 작업 기록으로 답하거나 근거에 답이 없으면 0.\n"
               "둘째 줄: 질문에 바로 답하는 한국어 한 문장(40자 안팎, '~입니다/~합니다'체). 문서 이름·조항 번호는 쓰지 않는다. "
               "근거의 '~하지 않는다'·'~일 때만' 같은 금지·조건은 근거 낱말 그대로 써서 뜻을 뒤집지 않는다. "
               "근거에 없는 내용을 지어내지 않는다. 근거에도 기록에도 답이 없으면 "
               "'자료에는 그 내용이 없습니다.'라고 쓴다. 음성으로 읽으므로 마크다운 없이 문장만 쓴다.")
NO_ANSWER = "자료(작업 기준서·로봇 운용 설명서·시스템 안내서)에는 그 내용이 없습니다. 작업이나 로봇·시스템에 대해 물어봐 주세요."
# 나(안내 담당) 자신을 묻는 말은 검색 전에 거절한다. 세 문서로 넓히자 "너 이름이 뭐야?" 의 리랭크가 0.94(기준서 5.2)로
# 문서 밖 문턱을 넘었고, 4B 가 프롬프트 첫 줄을 보고 "내 이름은 스펀지 블록 선별 로봇입니다" 를 지어냈다(10-22).
# 프롬프트에 "자신·잡담은 0" 줄을 넣으면 문서 안 질문까지 거절해 세 문서 인용이 40 → 26/42 로 떨어져 코드로 막는다
_SELF_Q = _rt.SELF_Q


def answer_chain():
    """LangChain 답 체인: ChatPromptTemplate → ChatOpenAI(OVMS qwen3-4b). answer_node 가 뒤에 cited_answer_parser 를 잇는다."""
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", _ANSWER_SYS),
        ("human", "질문: {question}\n\n[자료 근거]\n{context}\n\n[이번 대화의 작업 기록]\n{history}"),
    ])
    return prompt | llm.chat(max_tokens=config.ANSWER_MAX_TOKENS)


SAFETY_CHAPTER = "안전"      # 장 제목에 이 말이 있는 조항(기준서 6. 안전 수칙)은 답을 지어 말하지 않고 원문을 읽는다


META_CHAPTER = "목적"        # 1장(목적과 적용 범위): 기준서 자체를 설명하는 조항
_META_CUES = ("목적", "범위", "적용", "기준서", "설명서", "안내서")


def _chapter(h: dict) -> str:
    return h["section"].split(" > ")[0]


def _drop_meta(q: str, hits: list[dict]) -> list[dict]:
    """1장(목적)은 질문이 목적·범위를 묻지 않으면 근거에서 뺀다. 리랭커가 모든 주제를 한 줄씩 담은 1.1 을 자주 1위로
    올리고, 블록 기준서 1.1 에 "어떤 순서로 처리하고" 가 들어가자 "작업 순서?" 에 4B 가 5.1 대신 1.1 을 인용했다(10-21 밤)."""
    if any(w in q for w in _META_CUES):
        return hits
    # 장 밑이 아닌 조각(" > " 없음) = 문서 머리말(문서 번호·적용 대상). "너 이름이 뭐야?" 가 여기서 답을 지어냈다
    return [h for h in hits if META_CHAPTER not in _chapter(h) and " > " in h["section"]] or hits


ANCHOR_RATIO = 0.7


def _safety_first(hits: list[dict], ratio: float = 0.8) -> list[dict]:
    """리랭크 점수가 1위의 ratio 이상인 안전 수칙 조항은 맨 앞에 둔다. 4B 는 대개 첫 근거를 인용한다.
    "로봇이 일할 때 손은 어디에?" 에서 5.6 손 전달(0.74)이 6.1 손 넣지 않기(0.66)보다 앞이라 5.6 으로 답했다."""
    top = max((h["rerank"] for h in hits if h.get("rerank") is not None), default=None)
    if top is None:
        return hits
    first = [h for h in hits if SAFETY_CHAPTER in _chapter(h) and (h.get("rerank") or 0) >= ratio * top]
    # 합친 순위 1위(anchor = 임베딩+BM25)도 리랭크가 높으면 맨 앞: 안전 조항이 둘(6.1·6.4)일 때 "로봇이 안 멈추면" 이
    # 6.1 을 인용했다(10-21 밤 색 변경 뒤). anchor 가 6.4
    # anchor 는 0.7 배까지: 10-22 0시 "로봇이 안 멈추면" 의 6.4(anchor) 0.54 / 6.1 0.70 = 0.77 이라 0.8 에 걸려 6.1 을 인용했다
    anchor = [h for h in hits if h.get("anchor") and (h.get("rerank") or 0) >= ANCHOR_RATIO * top]
    first = anchor + [h for h in first if h not in anchor]
    return first + [h for h in hits if h not in first]


def _clause(h: dict) -> tuple[str, str]:
    """('5.4 실패와 재시도', 본문). 조각 앞의 '[문서 > 장 > 조항]' 줄은 떼고 조항 제목만 붙인다."""
    body = h["text"].split("\n", 1)[-1] if h["text"].startswith("[") else h["text"]
    return h["section"].split(" > ")[-1], body


def cite(h: dict) -> str:
    """'[로봇 운용 설명서 4.3]' — 답과 화면에 쓰는 인용(문서 이름까지, 10-22)."""
    no = h.get("clause") or _clause(h)[0].split(" ")[0]
    return f"[{h.get('doc_name') or config.DOC_NAMES['standard']} {no}]"


def _clause_text(hits: list[dict]) -> str:
    return "\n\n".join(f"[근거 {i + 1}] {h.get('doc_name', '')} {t}\n{b}"
                         for i, (h, (t, b)) in enumerate(zip(hits, map(_clause, hits))))


# 4B 가 자료에 없다고 다르게 말한 것("서울에서 부산까지의 시간은 자료에 없습니다.") → NO_ANSWER 로 맞춘다
_NO_ANS = re.compile(r"자료에\s*(?:는\s*)?(?:그\s*|이\s*|관련\s*)?(?:내용이\s*|정보가\s*)?없|자료에\s*나와\s*있지\s*않")
_FOREIGN = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")      # 4B 가 섞는 한자·가나("자动 패키지", 10-22)


def _bigrams(t: str) -> set[str]:
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _overlap(body: str, h: dict) -> float:
    """답 문장의 글자 2-gram 중 근거(조항 제목 + 본문)에 있는 비율."""
    b = _bigrams(body)
    title, text = _clause(h)
    return len(b & _bigrams(title + " " + text)) / len(b) if b else 0.0


_SAFETY_Q = re.compile(r"손|안전|위험|다치|다칠|끼이|끼어|부딪")
CITE_SWITCH = 0.1       # LLM 이 고른 근거보다 답과의 겹침이 이만큼 큰 근거가 있으면 그쪽을 인용


def _pick_hit(q: str, no: str | None, body: str, hits: list[dict], has_history: bool) -> dict | None:
    """인용할 근거. 4B 의 첫 줄 번호는 반쯤만 믿는다(10-22 세 문서 51문항: 답을 근거로 하고도 0 이 23건, 다른 번호 4건).
    번호(0 이면 첫 근거)에서 시작해 답 문장과 겹침이 CITE_SWITCH 이상 큰 근거가 있으면 바꾼다 → 인용 40 → 48/51."""
    if not hits:
        return None
    if _SAFETY_Q.search(q) and SAFETY_CHAPTER in hits[0]["section"].split(" > ")[0]:
        return hits[0]                      # 안전 질문에 안전 수칙이 맨 앞이면 그 조항(_safety_first 가 앞에 둔다)
    ov = [_overlap(body, h) for h in hits]
    bi = max(range(len(hits)), key=lambda i: (ov[i], -i))
    if no and "." in no:                    # 조항 번호를 직접 썼으면 그 번호인 근거
        hit = next((h for h in hits if _clause(h)[0].split(" ")[0] == no), None)
        if hit:
            return hit
    ci = int(no) - 1 if no and "." not in no and 1 <= int(no) <= len(hits) else 0
    if no == "0" and has_history and ov[bi] < 0.2:
        return None                         # 작업 기록으로 답했다 — 인용 없음
    return hits[bi] if ov[bi] >= ov[ci] + CITE_SWITCH else hits[ci]


# 4B 가 반말 동사 줄기에 '입니다'를 붙인다("CPU에서 돌아입니다", "트레이너에서 해입니다", 10-22)
_ENDING_FIX = [(re.compile(r"돌아입니다"), "돌아갑니다"), (re.compile(r"(?<![가-힣])해입니다"), "합니다"),
               (re.compile(r"돼입니다"), "됩니다"), (re.compile(r"(?<=[가-힣])져입니다"), "집니다")]


def _first_sentence(text: str) -> str:
    t = " ".join(text.split())
    m = re.search(r"^.*?[다요]\.", t)
    return m.group(0) if m else t[:120]


def _compose_answer(raw: str, hits: list[dict], finish: str | None, has_history: bool = False, q: str = "") -> str:
    raw = (raw or "").strip()
    # 첫 줄 = 근거 번호(1·2·3, 0 = 없음). 4B 가 "근거 2" 나 조항 번호("4.3 …")를 쓰기도 해서 짧은 첫 줄이면 숫자만 뽑는다
    m = re.match(r"^\D{0,6}?(\d+(?:\.\d+)?)(?![\d.])[^\n]{0,30}\n+(.*)$", raw, re.S)
    no, body = (m.group(1), m.group(2).strip()) if m else (None, raw)
    body = " ".join(re.sub(r"^(?:둘째 줄|답)\s*[:：]\s*", "", body).split())   # 음성용 한 줄
    if finish == "length" and not re.search(r"[.?!]$", body):   # 80토큰에 잘렸으면 마지막 문장 끝까지만
        cut = max(body.rfind(". "), body.rfind("다."), body.rfind("요."))
        if cut > 0:
            body = body[:cut + 2].strip()
        elif ", " in body:      # 한 문장 나열이 잘림(블록 기준서 7.2 항목 8개) → 마지막 쉼표까지 + "등"
            body = body[:body.rfind(", ")] + " 등이 있습니다."
    if not body or "그 내용이 없습니다" in body or _NO_ANS.search(body):
        return NO_ANSWER
    # LLM 이 인용을 직접 붙였으면 떼고 코드가 붙인다("기준서 6.4에 따르면 …")
    body = re.sub(r"^[^,.]{0,25}?\d+\.\d+\S*\s*(?:에\s*따르면|에서는|에는)\s*,?\s*", "", body)
    for pat, rep_ in _ENDING_FIX:
        body = pat.sub(rep_, body)
    hit = _pick_hit(q, no, body, hits, has_history)
    if not hit:
        return _FOREIGN.sub("", body)
    if _FOREIGN.search(body):               # 한자가 섞였으면 근거 첫 문장을 그대로 읽는다
        body = _first_sentence(_clause(hit)[1])
    _, text = _clause(hit)
    if SAFETY_CHAPTER in hit["section"].split(" > ")[0]:
        # 안전 수칙은 기준서 문장을 그대로 읽는다. 4B 가 6.1 "손 자리 밖으로 작업대에 손을 넣지 않는다" 를
        # "손은 작업대에 두어야 합니다" 로 뒤집어 말했다(10-21, 프롬프트로 안 고쳐짐)
        body = " ".join(text.split())
    return f"{cite(hit)}에 따르면 {body}"


def cited_answer_parser(hits: list[dict], question: str = "", has_history: bool = False, info: dict | None = None):
    """LangChain 출력 파서: LLM 두 줄(근거 번호 + 한 문장) → "[로봇 운용 설명서 4.3]에 따르면 …".
    답은 구조화 출력(json_schema)을 쓰지 않는다 — 4B 가 문자열 칸에서 max_tokens 까지 같은 말을 되풀이했다(10-22 0시,
    planner 의 reason 칸). 의도 판별(IntentOut)·계획(PlanOut)은 구조화 출력이다."""
    from langchain_core.output_parsers import BaseOutputParser

    class CitedAnswerParser(BaseOutputParser[str]):
        def parse_result(self, result, *, partial: bool = False) -> str:
            g = result[0]
            msg = getattr(g, "message", None)
            finish = (getattr(msg, "response_metadata", None) or {}).get("finish_reason")
            if info is not None:
                info["finish"] = finish
            return _compose_answer(g.text, hits, finish, has_history=has_history, q=question)

        def parse(self, text: str) -> str:
            return _compose_answer(text, hits, None, has_history=has_history, q=question)

        @property
        def _type(self) -> str:
            return "cited_answer"

    return CitedAnswerParser()


def select_evidence(q: str, hits: list[dict]) -> tuple[list[dict], bool]:
    """검색 결과 → (LLM 에 줄 근거, 문서 밖 질문인가)."""
    scored = sorted((h for h in hits if h.get("rerank") is not None), key=lambda h: -h["rerank"])
    top_cos = hits[0].get("top_cos", 1.0) if hits else 0.0
    # 문서 밖 질문: 리랭크했으면 최고점, 건너뛰었으면 임베딩 1위 코사인으로 본다
    if bool(hits) and ((scored and scored[0]["rerank"] < config.RAG_MIN_SCORE) or top_cos < config.RAG_OOD_COS):
        return [], True
    hits = _drop_meta(q, hits)          # 점수로 거르기 전에 뺀다(1.1 이 최고점이면 60% 기준에 5.1 이 걸러졌다)
    scored = sorted((h for h in hits if h.get("rerank") is not None), key=lambda h: -h["rerank"])
    if scored:   # 근거는 최고점의 60% 이상 + 합친 순위 1위(anchor)만. 4B 가 낮은 조각까지 섞어 인용했다(10-21)
        keep = max(config.RAG_KEEP_SCORE, 0.6 * scored[0]["rerank"])
        hits = [h for h in hits if h.get("rerank") is None or h.get("anchor") or h["rerank"] >= keep]
    else:        # 리랭크 생략: anchor + 남은 1위 코사인에서 RAG_KEEP_COS_GAP 안쪽
        tc = max((h.get("cos", 0) for h in hits), default=0)
        hits = [h for h in hits if h.get("anchor") or h.get("cos", 0) >= tc - config.RAG_KEEP_COS_GAP]
    return _safety_first(hits), False


ASK_ACK = "자료를 찾아볼게요."


def answer_node(s: State) -> dict:
    emit("node", "answer")
    q = s["text"]
    self_q = bool(_SELF_Q.search(q))
    if not self_q:
        # 질문 첫마디(10-22 r10): 답이 중앙 3.1초라 말 끝 → 받아쓰기 → 그동안 조용했다. 판별이 question 으로 끝난 뒤에만
        # 말한다(명령을 질문으로 잘못 알아들었을 때 "찾아볼게요"는 더 나쁨). 작업 기록 질문은 report 로 가서 여기 안 옴
        emit("say", "answer", text=ASK_ACK, ack=True)
    fut = _PREFETCH.pop(q, None)
    emit("tool", "answer", name="search_manual", status="start", args={"query": q}, prefetched=bool(fut))
    try:
        hits = [] if self_q else fut.result(timeout=60) if fut else search_manual.invoke({"query": q})
    except Exception as e:
        hits = []
        emit("error", "answer", message=f"search_manual: {type(e).__name__}: {e}")
    emit("tool", "answer", name="search_manual", status="end", args={"query": q},
         result=[{"doc": h.get("doc"), "cite": cite(h), "section": h.get("section"), "rerank": h.get("rerank")}
                 for h in hits])
    hist = "\n".join(f"- {h['ts'][11:19]} \"{h['text']}\" → {h['intent']} {h.get('status')}, "
                     f"{h.get('done', 0)}개, 시도 {h.get('attempts', 0)}회"
                     for h in (s.get("history") or [])[-8:]) or "없음"
    hits, ood = ([], True) if self_q else select_evidence(q, hits)
    if ood and hist == "없음":
        # 작업 기록도 없음 → LLM 없이 바로 답한다(관계없는 조항을 끌어다 답하지 않게)
        answer = NO_ANSWER
        emit("say", "answer", text=answer, sources=[])
        return {"answer": answer, "status": "done",
                "history": [{"ts": now_iso(), "started": s.get("started"), "elapsed_s": _elapsed(s), "text": q,
                             "intent": "question", "status": "done", "say": answer}]}
    info: dict = {}
    try:
        chain = answer_chain() | cited_answer_parser(hits, q, has_history=hist != "없음", info=info)
        answer = chain.invoke({"question": q, "context": _clause_text(hits) or "없음", "history": hist})
    except Exception as e:
        emit("error", "answer", message=f"llm: {type(e).__name__}: {e}")
        answer = (f"{cite(hits[0])}에서 찾은 내용입니다. " + hits[0]["text"].split("\n", 1)[-1][:200]) if hits \
            else "지금은 답을 만들 수 없습니다."
    emit("say", "answer", text=answer, finish=info.get("finish"),
         sources=[{"doc": h.get("doc"), "cite": cite(h), "section": h.get("section"), "rerank": h.get("rerank")}
                  for h in hits])
    return {"answer": answer, "status": "done",
            "history": [{"ts": now_iso(), "started": s.get("started"), "elapsed_s": _elapsed(s), "text": q,
                         "intent": "question", "status": "done", "say": answer}]}


# ------------------------------------------------------------------ scene: 지금 작업대 질문(로봇 안 움직임)
ZONE_KO = {"input": "투입 구역", "bowl_red": "분홍 그릇", "bowl_blue": "초록 그릇", "bowl_defect": "불량 그릇",
           "outside": "구역 밖"}
_KEYS = ("red_ok", "blue_ok", "red_defect", "blue_defect")
CELLS = ("왼쪽 위", "가운데 위", "오른쪽 위", "왼쪽 가운데", "가운데", "오른쪽 가운데", "왼쪽 아래", "가운데 아래", "오른쪽 아래")
_KO_NUM = re.compile(r"(?<![가-힣])(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*(?:개|칸)")
# 질문 낱말 → 구역 / 종류(scene 대체 답). "남았어"는 투입 구역
_Q_ZONE = [("bowl_red", r"(?:분홍|핑크)\S*\s*그릇"), ("bowl_blue", r"(?:초록|녹색|연두)\S*\s*그릇"),
           ("bowl_defect", r"불량\S*\s*그릇"), ("outside", r"밖"), ("input", r"투입|남았|남아|남은")]
_Q_CLASS = [(("red_defect", "blue_defect"), r"불량|엑스"), (("red_ok", "red_defect"), r"분홍|핑크"),
            (("blue_ok", "blue_defect"), r"초록|녹색|연두")]


def _zone_counts(obs: dict, z: str) -> dict:
    return report.outside_counts(obs) if z == "outside" else obs["zones"][z]


def _line(counts: dict) -> str:
    parts = [f"{report.CLASS_KO[k]} {counts.get(k, 0)}개" for k in _KEYS if counts.get(k, 0)]
    if not parts:
        return "없음(합계 0개)"
    bad = counts.get("red_defect", 0) + counts.get("blue_defect", 0)
    return ", ".join(parts) + f"(합계 {sum(counts.get(k, 0) for k in _KEYS)}개" + (f", 불량 {bad}개)" if bad else ")")


def scene_facts(obs: dict) -> dict:
    """detect_objects 결과 → LLM 에 줄 한국어 JSON(이것만 근거). 구역마다 한 줄로 평평하게 준다: 처음엔 합계를 겹겹이
    담은 JSON 을 줬더니 4B 가 엉뚱한 구역의 숫자를 골랐다(10-21 밤 scene 4/8). 칸 이름은 detect 의 cell 그대로."""
    facts = {ZONE_KO[z]: _line(_zone_counts(obs, z)) for z in (*ZONE_KO,)}
    facts["손 자리"] = "손 보임" if obs["hand"]["present"] else "손 없음"
    groups: dict = {}
    for o in obs.get("objects") or []:
        key = (o["zone"], o.get("cell") or "", o["cls"])
        groups[key] = groups.get(key, 0) + 1
    facts["블록 위치"] = [f"{ZONE_KO.get(z, z)}{' ' + cell if cell else ''}: {report.CLASS_KO[c]} {n}개"
                       for (z, cell, c), n in sorted(groups.items(), key=lambda kv: (list(ZONE_KO).index(kv[0][0]), kv[0]))]
    return facts


def scene_answer_ok(ans: str, facts: dict) -> tuple[bool, str]:
    """지어내기 검사: 답이 말한 구역의 줄(위치 줄 포함)에 있는 숫자·칸만 쓸 수 있다. 구역을 말하지 않으면 숫자도 못 쓴다."""
    zones = [z for z in ZONE_KO.values() if z in ans]
    src = [facts[z] for z in zones] + [p for p in facts["블록 위치"] if any(p.startswith(z) for z in zones)]
    allowed = set(re.findall(r"\d+", " ".join(src))) | {"0"}
    bad = sorted(set(re.findall(r"\d+", ans)) - allowed)
    if bad:
        return False, f"말한 구역({'·'.join(zones) or '없음'})에 없는 숫자 {bad}"
    if _KO_NUM.search(ans):
        return False, "숫자를 말로 씀"
    t = ans
    for c in sorted(CELLS, key=len, reverse=True):
        if c in t:
            if not any(c in p for p in src):
                return False, f"말한 구역에 없는 칸 '{c}'"
            t = t.replace(c, "")
    return True, ""


def scene_summary(obs: dict) -> str:
    """요약 질문("지금 배치 어때?")의 말(10-22 r11: 발표 실연 ② 가 빈 그릇까지 읽어 한 문장 10.8초). 5초 안팎을 목표로
    중요한 것 먼저 — 손 → 구역 밖 → 그릇에 섞인 블록 → 투입 구역 → 그릇. 빈 구역은 말하지 않는다(자세한 건 say 의 facts)."""
    z, parts = obs["zones"], []
    if obs["hand"]["present"]:
        parts.append("손 자리에 손이 보여요.")
    out = report.outside_counts(obs)
    if out:
        parts.append(f"구역 밖에 {sum(out.values())}개가 떨어져 있어요.")
    for b, (keys, name) in report.BOWLS.items():
        mixed = sum(n for k, n in z[b].items() if k in _KEYS and k not in keys)
        if mixed:
            parts.append(f"{name}에 다른 블록 {mixed}개가 섞여 있어요.")
    inp = z["input"]
    items = [f"{lab} {n}개" for lab, n in (("분홍", inp.get("red_ok", 0)), ("초록", inp.get("blue_ok", 0)),
                                         ("불량", inp.get("red_defect", 0) + inp.get("blue_defect", 0))) if n]
    bowls = [f"{report.BOWLS[b][1]}에 {t}개" for b in ("bowl_red", "bowl_blue", "bowl_defect")
             if (t := sum(z[b].get(k, 0) for k in _KEYS))]
    if items and bowls:
        parts.append(f"투입 구역에 {', '.join(items)}가 있고, {', '.join(bowls)}가 있어요.")
    elif items:
        parts.append(f"투입 구역에 {', '.join(items)}가 있어요. 그릇은 비어 있어요.")
    elif bowls:
        parts.append(f"투입 구역은 비었고, {', '.join(bowls)}가 있어요.")
    elif not out:
        parts.append("작업대에 블록이 하나도 없어요.")
    return " ".join(parts)


def _is_summary(q: str) -> bool:
    """구역·종류·위치·손을 묻지 않는 요약 질문인가(scene_template 이 전체를 읽는 경우)."""
    return not ("손" in q or "어디" in q or any(re.search(p, q) for _, p in _Q_ZONE)
                or any(re.search(p, q) for _, p in _Q_CLASS))


def scene_template(q: str, obs: dict) -> str:
    """LLM 답이 검사에서 떨어지면 쓰는 답. 질문 낱말로 구역·종류를 고른다(없으면 전체 요약)."""
    if "손" in q:
        return "손 자리에 손이 보입니다." if obs["hand"]["present"] else "손 자리에 손이 보이지 않습니다."
    zone = next((z for z, pat in _Q_ZONE if re.search(pat, q)), None)
    cls = next((ks for ks, pat in _Q_CLASS if re.search(pat, q.split("그릇")[-1] if zone and zone.startswith("bowl") else q)), None)
    if "어디" in q and cls:
        where = [f"{ZONE_KO[o['zone']]}{' ' + o['cell'] if o.get('cell') else ''}" for o in obs.get("objects") or []
                 if o["cls"] in cls]
        if not where:
            return "지금 카메라에는 그 블록이 보이지 않습니다."
        cnt: dict = {}
        for w in where:
            cnt[w] = cnt.get(w, 0) + 1
        return ", ".join(f"{w}에 {n}개" for w, n in cnt.items()) + " 있습니다."
    zs = [zone] if zone else ["input", "bowl_red", "bowl_blue", "bowl_defect"]
    parts = []
    for z in zs:
        c = _zone_counts(obs, z)
        if cls:
            c = {k: v for k, v in c.items() if k in cls}
        items = [f"{report.CLASS_KO[k]} {c.get(k, 0)}개" for k in _KEYS if c.get(k, 0)]
        parts.append(f"{ZONE_KO[z]}에 " + (", ".join(items) if items else "0개"))
    out = report.outside_counts(obs)
    tail = f" 구역 밖에 {sum(out.values())}개가 떨어져 있습니다." if out and zone is None else ""
    return "지금 " + ", ".join(parts) + "가 있습니다." + tail


# 4B 가 JSON 에서 바로 답을 쓰면 엉뚱한 구역 숫자·깨진 어미("손 보여 없습니다", "있어입니다")가 나왔다(10-21 밤 scene 5/8).
# 보고 문장처럼 코드가 질문에 맞는 사실로 [예시 답]을 만들고, LLM 은 그걸 자연스럽게 다듬기만 한다
_SCENE_SYS = ("너는 스펀지 블록 선별 로봇의 작업대 안내 담당이다. [카메라 JSON]은 위 카메라가 방금 센 결과이고, [예시 답]은 그중 "
              "질문에 맞는 사실을 코드가 골라 쓴 문장이다. 질문에 바로 답하는 자연스러운 한국어 한 문장('~입니다/~습니다'체)으로 "
              "[예시 답]을 다듬는다. 규칙: [예시 답]의 숫자·구역 이름·칸 이름을 하나도 빼거나 바꾸지 않는다. 새 숫자를 만들지 않는다. "
              "분홍=핑크, 초록=녹색=연두. 마크다운 없이 문장만 쓴다.")
_BAD_END = re.compile(r"[어아]입니다|보여 없|있어입|\d+입니다\.?$")


def scene_node(s: State) -> dict:
    import json as _json
    emit("node", "scene")
    try:
        obs = _detect("scene")
    except VisionError:
        msg = f"{VISION_MSG} 확인해 주세요."
        emit("say", "scene", text=msg)
        return {"answer": msg, "status": "error"}
    facts = scene_facts(obs)
    method, why = "template", ""
    if _is_summary(s["text"]):
        # 요약은 짧은 코드 문장을 바로 말한다(LLM 다듬기 약 2초를 건너뜀, 10-22 r11). 숫자·칸 검사는 똑같이 지킨다
        answer, method = scene_summary(obs), "summary"
        if not scene_answer_ok(answer, facts)[0]:       # 만든 문장이 검사에 걸리면(있어선 안 됨) 예전 전체 답
            answer, method = scene_template(s["text"], obs), "template"
        emit("say", "scene", text=answer, method=method, facts=facts, image_path=obs.get("image_path"))
        return {"obs": obs, "answer": answer, "status": "done",
                "history": [{"ts": now_iso(), "started": s.get("started"), "elapsed_s": _elapsed(s), "text": s["text"],
                             "intent": "scene", "status": "done", "say": answer}]}
    answer = scene_template(s["text"], obs)
    # LLM 다듬기는 AGENT_SCENE_POLISH=1 일 때만(10-23 기본 끔: 09:35 GPU HANG 이 다듬기 중에 났다). 끄면 템플릿을 바로 말한다
    if config.SCENE_POLISH:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            user = (f"질문: {s['text']}\n\n[카메라 JSON]\n{_json.dumps(facts, ensure_ascii=False, indent=0)}"
                    f"\n\n[예시 답]\n{answer}")
            out = llm.chat(max_tokens=80).invoke([SystemMessage(_SCENE_SYS), HumanMessage(user)]).content.strip()
            out = " ".join(out.split())
            ok, why = scene_answer_ok(out, facts)
            missing = set(re.findall(r"\d+", answer)) - set(re.findall(r"\d+", out))
            if ok and missing:
                ok, why = False, f"예시 답의 숫자 {sorted(missing)} 를 뺌"
            if ok and _BAD_END.search(out):
                ok, why = False, "어미가 깨짐"
            if out and ok:
                answer, method = out, "llm"
            else:
                emit("node", "scene", llm_rejected=out, reason=why)
        except Exception as e:
            emit("error", "scene", message=f"llm: {type(e).__name__}: {e}")
    emit("say", "scene", text=answer, method=method, facts=facts, image_path=obs.get("image_path"))
    return {"obs": obs, "answer": answer, "status": "done",
            "history": [{"ts": now_iso(), "started": s.get("started"), "elapsed_s": _elapsed(s), "text": s["text"],
                         "intent": "scene", "status": "done", "say": answer}]}


# ------------------------------------------------------------------ plan: 배운 정책 4개를 엮는 명령
def _plan_models():
    from typing import Literal

    from pydantic import BaseModel, Field

    class PlanStep(BaseModel):
        intent: Literal["defect", "red", "blue", "handoff"] = Field(description="배운 동작 하나")
        count: Optional[int] = Field(default=None, description="말한 개수. 말하지 않았으면 null(그 종류 전부, handoff 는 1)")

    # 문자열 필드(reason)는 두지 않는다: json_schema 문법 제한 출력에서 4B 가 그 칸을 끝내지 못하고 max_tokens 까지
    # 갔다(10-22 0시 LengthFinishReasonError, 120·300토큰 모두). 빼면 49~52토큰·약 1.9초. 거절 문구는 고정(REFUSE_MSG)
    class PlanOut(BaseModel):
        feasible: bool = Field(description="배운 동작 4가지로 할 수 있으면 true")
        steps: list[PlanStep] = Field(default_factory=list, description="실행할 순서대로")

    return PlanOut


# 계획 LLM 은 규칙으로 못 읽은 말에만 쓴다(rule_plan). 뺀 동작·나머지·개수·손 전달 순서는 코드가 검증하므로
# 프롬프트는 동작 이름과 예 두 개만 둔다(10-21 밤: 긴 프롬프트 + 카메라 줄로 계획 알림이 3.7초)
_PLAN_SYS = ("너는 블록 선별 로봇의 계획 담당이다. 배운 동작은 4가지다: defect(투입 구역의 불량을 불량 그릇으로), "
             "red(정상 분홍·핑크를 분홍 그릇으로), blue(정상 초록·녹색·연두를 초록 그릇으로), handoff(분홍 그릇의 분홍을 사람 손에). "
             "말을 말한 순서대로 steps 로 바꾼다. count 는 말한 개수, 없으면 null. 4가지로 못 하는 일이면 feasible=false, steps=[].\n"
             "예) '분홍 두 개만 옮기고 불량 골라내 줘' → [red 2, defect null]\n"
             "예) '블록으로 하트 모양 만들어 줘' → feasible=false")
_PLAN_MAX_STEPS = 6
ACCEPT_MSG = "네, 계획을 세울게요."
ACCEPT_MSGS = {"coord": "네, 좌표로 할 계획을 세울게요."}   # 10-22 r21 방식. 10-23 13:10 부터 옮기기는 늘 좌표
ACCEPT_ACT_CUE = ACT_ONLY_MSG + " 좌표로 할 계획을 세울게요."   # 계획에 ACT 단서가 붙었을 때
REFUSE_MSG = ("그건 제가 배운 동작으로는 할 수 없어요. 제가 할 수 있는 건 불량 골라내기, 분홍·초록 블록을 같은 색 그릇에 옮기기, "
              "분홍 블록을 손에 건네기예요. 예를 들어 '불량 골라내고 분홍 블록 옮겨 줘'처럼 말해 주세요.")


def make_plan(text: str, obs: dict):
    """LLM 구조화 출력으로 단계 목록. 시험(selftest --offline)에서 바꿔 끼울 수 있게 함수로 둔다."""
    from langchain_core.messages import HumanMessage, SystemMessage
    model = llm.chat(max_tokens=120).with_structured_output(_plan_models(), method=config.LLM_SO_METHOD)
    return model.invoke([SystemMessage(_PLAN_SYS), HumanMessage(f"말: {text}")])


# 흔한 꼴은 규칙으로 계획한다(LLM 3초 → 0초). 구절(…고 / , / 다음에)마다 대상 하나(불량·분홍·초록, 분홍+손 = 손 전달)와
# 개수를 읽는다. 뺀 동작 구절은 지우고(check_plan 이 처리), 대상 없는 "나머지·다 정리" 는 채우기 표지로 둔다
_SPLIT = re.compile(r"(?<![말됐])고\s|,|그리고|다음에?|한\s*뒤에?")
_AND = re.compile(r"이랑|랑\s|와\s|과\s|하고|및|둘\s*다")
_TARGET = [("defect", r"불량|엑스"), ("pink", PINK_RE), ("blue", GREEN_RE)]


def rule_plan(text: str) -> list[dict] | None:
    """[{intent, count}] 또는 None(애매 → LLM). [] 는 '나머지·다 정리' 만 있어 check_plan 이 채운다는 뜻.
    "불량 빼고 …" 는 "불량을 빼내고"로 읽는다("불량 빼 줘" = 골라내기, 기준서 5.1 불량 먼저). 발표 실연 ③
    "불량 빼고 나머지 다 정리해 줘" 가 이 꼴이다(계약서 2절) — 불량이 이미 없으면 resolve_plan 이 그 단계를 건너뛴다."""

    def cut(m):                                          # 뺀 동작 구절은 지운다. "불량 빼고" 만 동작으로 남긴다
        return m.group(0) if m.group(1) in ("불량", "엑스") and m.group(0).endswith("빼고") else ","
    t = re.sub(r"(불량|엑스|" + PINK_RE + "|" + GREEN_RE + r")\S*\s*(?:블[록럭]\S*\s*|거\s*|것\s*)?"
               r"(?:그대로\s*두고|놔\s*두고|남겨\s*두고|두고|말고|제외하고|빼고)", cut, text)
    steps, rest = [], False
    for c in _SPLIT.split(t):
        c = c.strip()
        if not c:
            continue
        obj = re.sub(r"(?:" + PINK_RE + "|" + GREEN_RE + r")색?\s*그릇", "", c)   # "분홍 그릇에" 의 분홍은 대상이 아님(“분홍은 그릇에”는 대상)
        hit = sorted((m.start(), name) for name, pat in _TARGET for m in [re.search(pat, obj)] if m)
        names = [n for _, n in hit]
        if not names:
            if _REST.search(c) or _ALL_TO.search(c):
                rest = True
                continue
            return None                                      # 읽지 못한 구절
        if "defect" in names and len(names) > 1 and not _AND.search(obj):
            names = ["defect"]                               # "분홍 불량 빼 줘" 의 분홍은 불량의 색
        if "씩" in obj or len(names) > 1 and (not _AND.search(obj) or parse_count(obj) is not None):
            return None                                      # "두 개씩"·여러 대상에 개수가 붙으면 애매
        for n in names:
            it = ("handoff" if "손" in obj else "red") if n == "pink" else n
            steps.append({"intent": it, "count": parse_count(obj) if len(names) == 1 else None})
    return steps if steps or rest else None



# 코드 검증 ② 말에서 뺀 동작(…는 그대로 두고·말고·제외하고·놔두고)이 계획에 있으면 지운다. 4B 가
# "초록은 그대로 두고 나머지 정리해 줘" 를 [불량, 초록] 으로 거꾸로 세웠다(10-21 밤). "불량 빼고" 는
# "불량을 빼내고"로도 읽혀 강제하지 않는다
# 코드 검증 ③ "나머지 정리"(나머지 뒤에 색이 붙지 않은 것)·"다/전부 정리" = 빠진 동작을 불량→분홍→초록 순서로 채운다
_REST = re.compile(r"나머지(?!\s*(?:분홍|핑크|초록|녹색|연두|불량))|(?:다|전부|모두|싹)\s*(?:정리|치워)")
_REST_OF = re.compile(r"나머지\s*(분홍|핑크|초록|녹색|연두|불량)")
_ALL_TO = re.compile(r"(?:다|전부|모두|싹)\s*(?:그릇|옮겨|담아|넣어)")    # "불량 말고 전부 그릇에 담아 줘"(색을 말하면 그 색 전부)


def check_plan(text: str, steps: list[dict]) -> tuple[list[dict], list[str]]:
    excl = excluded(text, strict=True)
    unsure = excluded(text) - excl           # "불량 빼고": 뺄지 말지 LLM 이 읽은 대로 둔다(채우기에서도 건드리지 않음)
    notes = []
    kept = [st for st in steps if st["intent"] not in excl]
    if len(kept) < len(steps):
        notes.append("말씀하신 대로 " + "·".join(TASKS[it]["label"] for it in sorted(excl)) + "은 그대로 둡니다")
    any_color = re.search(r"분홍|핑크|초록|녹색|연두", text)     # "분홍이랑 초록 다 옮겨 줘" 의 '다'는 그 두 색
    if _REST.search(text) or _ALL_TO.search(text) and not any_color:
        have = {st["intent"] for st in kept}
        add = [it for it in ("defect", "red", "blue") if it not in excl | unsure and it not in have]
        kept += [{"intent": it, "count": None} for it in add]
    # 코드 검증 ④ 대상을 말했고 나머지·전부 정리 표지가 없으면, 말하지 않은 대상의 단계는 뺀다.
    # 4B 가 "분홍이랑 초록 다 옮겨 줘" 에 불량 골라내기를 덧붙였다(10-21 밤)
    named = {it for it, pat in WORD_INTENT if re.search(pat, text)}
    if named and not _REST.search(text) and not (_ALL_TO.search(text) and not any_color):
        ok_it = named | ({"handoff"} if "red" in named else set())
        drop = [st for st in kept if st["intent"] not in ok_it]
        if drop:
            kept = [st for st in kept if st["intent"] in ok_it]
            notes.append("말씀하지 않은 " + "·".join(sorted({TASKS[st['intent']]['label'] for st in drop})) + "은 하지 않습니다")
    # "나머지 분홍은 그릇에" = 그 색 전부 단계. 4B 가 [분홍 1 → 손 1] 만 세우고 나머지를 빠뜨렸다(10-21 밤)
    for m in _REST_OF.finditer(text):
        it = next(i for i, pat in WORD_INTENT if re.fullmatch(pat, m.group(1)))
        if not any(st["intent"] == it and st.get("count") is None for st in kept):
            kept.append({"intent": it, "count": None})
    return kept, notes


_EMPTY_NOTE = {"defect": "투입 구역에 불량품이 없어서", "red": "투입 구역에 분홍 블록이 없어서",
               "blue": "투입 구역에 초록 블록이 없어서", "handoff": "분홍 그릇에 분홍 블록이 없어서"}


def resolve_plan(steps: list[dict], obs: dict) -> tuple[list[dict], list[str]]:
    """LLM 단계 검증: 허용 intent 만, 개수 ≤ 그때 감지 개수(앞 단계를 개수로 흉내 내며), 없는 건 건너뜀.
    handoff 는 분홍 그릇에서 건네므로 그릇에 모자라면 투입 구역의 분홍을 먼저 옮기는 단계를 넣는다."""
    z = obs["zones"]
    inp = {k: z["input"].get(k, 0) for k in _KEYS}
    bowl_red = z["bowl_red"].get("red_ok", 0)
    out, notes = [], []

    def avail(it):
        return {"defect": inp["red_defect"] + inp["blue_defect"], "red": inp["red_ok"], "blue": inp["blue_ok"],
                "handoff": bowl_red}[it]

    def take(it, n):
        nonlocal bowl_red
        if it == "defect":
            for k in ("red_defect", "blue_defect"):
                m = min(n, inp[k])
                inp[k] -= m
                n -= m
        elif it == "red":
            inp["red_ok"] -= n
            bowl_red += n
        elif it == "blue":
            inp["blue_ok"] -= n
        else:
            bowl_red -= n

    for st in steps[:_PLAN_MAX_STEPS]:
        it, want = st["intent"], st.get("count")
        if it not in TASKS:
            notes.append(f"알 수 없는 동작 {it} 는 뺐습니다")
            continue
        want = want if want and want > 0 else None
        need = (want or 1) if it == "handoff" else want
        if it == "handoff" and bowl_red < need and inp["red_ok"] > 0:
            k = min(need - bowl_red, inp["red_ok"])
            out.append({"intent": "red", "count": k, "expect": k, "auto": True})
            take("red", k)
            notes.append(f"손에 건네려면 분홍 그릇에 있어야 해서 분홍 블록 {k}개를 먼저 옮깁니다")
        a = avail(it)
        if a == 0:
            notes.append(_EMPTY_NOTE[it] + " 그 단계는 건너뜁니다")
            continue
        n = a if need is None else min(need, a)
        if need is not None and need > a:
            notes.append(f"{TASKS[it]['label']}이 {a}개뿐이라 {a}개만 합니다")
        take(it, n)
        # 개수를 말하지 않은 단계는 count=None(0개 될 때까지) 그대로 실행하고, 알릴 때만 지금 개수를 쓴다
        out.append({"intent": it, "count": n if (need is not None or it == "handoff") else None, "expect": n})
    return out, notes


def _steps_text(steps: list[dict]) -> str:
    return ", ".join(f"{i + 1}) {TASKS[st['intent']]['step'].format(n=st['expect'])}" for i, st in enumerate(steps))


def _reset_step(st: dict) -> dict:
    return {"intent": st["intent"], "count": st["count"], "goal": 0, "done": 0, "moved_by_class": {}, "attempts": 0,
            "misses": 0, "max_attempts": 0, "status": "", "error_msg": "", "dropped": 0, "started": now_iso(),
            "coord": {}}


def planner_node(s: State) -> dict:
    emit("node", "planner", text=s["text"])
    text = s.get("cmd") or s["text"]          # 방식 단서를 뺀 말("좌표로"가 구절 읽기를 흐리지 않게)
    how = HOW["coord"]                         # 옮기는 단계는 늘 좌표(손 전달 단계는 그 단계 첫 말이 "학습한 동작으로")
    act_cue = s.get("method_asked") == "act"   # 10-23 13:10: ACT 단서 → 좌표로 하고 첫 말에 알림
    why = cant_do(text)
    if not why and _rt.rule_target(text)["by"]:   # 위치 단서는 한 가지 명령에서만(10-23). 단계로 나누면 위치가 빠져 엉뚱한 블록을 집는다
        emit("tool", "planner", name="make_plan", status="end", result={"feasible": False, "reason": "which", "by": "rule"})
        return {"steps": []} | _say_end(s, "위치로 고르기는 한 가지 명령에서만 해요. 한 가지씩 말해 주세요.", "refused")
    if not why and colored_defect(text):      # 불량 색도 한 가지 명령에서만(10-23). 단계는 색을 못 넘겨 다른 색 불량까지 옮긴다
        emit("tool", "planner", name="make_plan", status="end",
             result={"feasible": False, "reason": "defect_color", "by": "rule"})
        return {"steps": []} | _say_end(s, "불량을 색으로 고르기는 한 가지 명령에서만 해요. 한 가지씩 말해 주세요.", "refused")
    if why:                                   # LLM·카메라를 거치지 않고 바로 거절
        emit("tool", "planner", name="make_plan", status="end", result={"feasible": False, "reason": why, "by": "rule"})
        return {"steps": []} | _say_end(s, REFUSE_MSG, "refused")
    ruled = rule_plan(text)
    if ruled is None:                         # LLM 을 부르는 동안 먼저 받았다고 말한다(첫 말 1초 안)
        emit("say", "planner", text=ACCEPT_ACT_CUE if act_cue else ACCEPT_MSGS["coord"])
    try:
        obs = _detect("planner")
    except VisionError:
        return {"steps": []} | _say_end(s, f"{VISION_MSG} 확인해 주세요.", "error")
    t0 = time.perf_counter()
    if ruled is not None:
        raw = {"feasible": True, "steps": ruled, "reason": "", "by": "rule"}
    else:
        try:
            out = make_plan(text, obs)
            raw = {"feasible": out.feasible, "steps": [st.model_dump() for st in out.steps], "reason": "", "by": "llm"}
        except Exception as e:
            emit("error", "planner", message=f"llm: {type(e).__name__}: {e}")
            return {"steps": []} | _say_end(s, "계획을 세우지 못했어요. 한 가지씩 말해 주세요.", "error")
    fixed, notes = check_plan(text, raw["steps"]) if raw["feasible"] else ([], [])
    steps, more = resolve_plan(fixed, obs) if fixed else ([], [])
    notes += more
    emit("tool", "planner", name="make_plan", status="end", result=raw, steps=steps, notes=notes,
         latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    if not raw["feasible"] or (raw["by"] == "llm" and not raw["steps"]):
        return {"steps": []} | _say_end(s, REFUSE_MSG, "refused")
    if not steps:
        return {"steps": []} | _say_end(s, "지금 작업대에는 그 계획으로 할 일이 없습니다. " + ". ".join(notes) + ".",
                                        "nothing")
    if not coord_available() and any(st["intent"] in COORD_INTENTS for st in steps):    # ACT 로 대신하지 않는다(10-23 13:10)
        return {"steps": []} | _say_end(s, COORD_OFF_MSG, "refused")
    msg = f"{how} 순서대로 할게요. {_steps_text(steps)}." + (f" {'. '.join(notes)}." if notes else "")
    if act_cue and ruled is not None:         # LLM 경로는 수락 멘트에 이미 붙였다
        msg = f"{ACT_ONLY_MSG} {msg}"
    emit("say", "planner", text=msg, steps=steps)
    return {"steps": steps, "step_i": 0, "step_results": [], "plan_started": s.get("started"), "obs": obs,
            **_reset_step(steps[0])}


def _say_end(s: State, msg: str, status: str) -> dict:
    emit("say", "planner", text=msg)
    return {"answer": msg, "status": status,
            "history": [{"ts": now_iso(), "started": s.get("started"), "elapsed_s": _elapsed(s), "text": s["text"],
                         "intent": "plan", "status": status, "say": msg}]}


_PLAN_HALT = ("stopped", "gave_up", "limit", "error", "no_hand")


def next_step_node(s: State) -> dict:
    steps, i = s["steps"], s.get("step_i", 0)
    res = {"intent": s["intent"], "status": s.get("status"), "done": s.get("done", 0), "expect": steps[i]["expect"],
           "attempts": s.get("attempts", 0), "dropped": s.get("dropped", 0), "auto": steps[i].get("auto", False)}
    results = (s.get("step_results") or []) + [res]
    emit("node", "next_step", step=i + 1, of=len(steps), result=res)
    if res["status"] in _PLAN_HALT or STOP.is_set() or i + 1 >= len(steps):
        return {"step_results": results, "step_i": i + 1}
    nxt = steps[i + 1]
    return {"step_results": results, "step_i": i + 1, **_reset_step(nxt)}


_STATUS_KO = {"stopped": "멈춰서 중단", "gave_up": "두 번 실패해 중단", "limit": "시도 한도로 중단",
              "error": "실행할 수 없어 중단", "no_hand": "손이 보이지 않아 중단", "nothing": "할 블록이 없어 건너뜀"}


def plan_report_node(s: State) -> dict:
    """계획 대비 결과(코드가 만드는 문장, 숫자는 단계별 verify 의 카메라 개수)."""
    steps, results = s.get("steps") or [], s.get("step_results") or []
    emit("node", "plan_report", steps=len(steps), done_steps=len(results))
    parts = []
    for st, r in zip(steps, results):
        what = {"defect": "불량", "red": "분홍 블록", "blue": "초록 블록", "handoff": "손에 건넨 분홍 블록"}[st["intent"]]
        p = f"{what} {r['done']}개(계획 {r['expect']}개)"
        if r["status"] in _STATUS_KO:
            p += f" — {_STATUS_KO[r['status']]}"
        parts.append(p)
    finished = sum(r["status"] in ("done", "nothing") for r in results)
    halted = next((r for r in results if r["status"] in _PLAN_HALT), None)
    if halted is None and len(results) == len(steps):
        head = f"계획 {len(steps)}단계를 모두 마쳤습니다"
    else:
        head = f"계획 {len(steps)}단계 중 {finished}단계까지 했습니다"
    msg = f"{head}: {', '.join(parts)}."
    if halted and halted["status"] != "stopped":
        msg += " 남은 단계는 하지 않았습니다. 확인해 주세요."
    emit("say", "plan_report", text=msg, steps=steps, results=results)
    started = s.get("plan_started") or s.get("started")
    try:
        el = round((datetime.now().astimezone() - datetime.fromisoformat(started)).total_seconds(), 2)
    except Exception:
        el = None
    return {"intent": "plan", "answer": msg, "status": "done" if halted is None else halted["status"],
            "history": [{"ts": now_iso(), "started": started, "elapsed_s": el, "text": s["text"], "intent": "plan",
                         "status": "done" if halted is None else halted["status"], "say": msg,
                         "steps": steps, "step_results": results}]}


# ------------------------------------------------------------------ arrange: 블록으로 숫자 만들기(좌표 집기, plan 의 한 갈래)
# robot/COORD_PICK.md 9절 arrange_digit: detect → dry-run(계획·시간·경고) → 말하고 "할까요?"(interrupt) → 같은 detection 으로 실행
ACCEPT_ARRANGE = "네, 숫자 {d} 배치를 계획해 볼게요."
# 수락은 짧은 대답 전체가 맞을 때만("어디 있어?" 같은 말을 '어'로 읽어 실행하지 않게)
_YES_FULL = re.compile(r"(?:응|어|네|예|그래|좋아|오케이|ok|ㅇㅇ|해|시작|진행|그렇게)(?:\s*(?:해|해\s*줘|해\s*주세요|해요|"
                       r"할게|하자|요|줘|주세요))*")
_YES_ANY = re.compile(r"(?:그렇게|그대로)\s*해|시작해\s*(?:줘|주세요)|진행해\s*(?:줘|주세요)")
_NO = re.compile(r"아니|취소|하지\s*마|안\s*해|됐어|싫어|그만|ㄴㄴ|^노$")
_EUN = {"2", "4", "5", "9"}                     # 숫자 읽는 소리가 모음으로 끝남(이·사·오·구) → 는/를


def is_yes(text: str) -> bool:
    t = text.strip(" .!~,").lower()
    return bool(_YES_FULL.fullmatch(t) or _YES_ANY.search(t)) and not _NO.search(t)


def is_no(text: str) -> bool:
    return bool(_NO.search(text))


def _josa(d: str, a: str, b: str) -> str:
    return b if d in _EUN else a


def _clean(o):
    """체크포인터에 넣을 수 있게(numpy 값 등) JSON 으로 한 번 거른다."""
    import json as _json
    return _json.loads(_json.dumps(o, ensure_ascii=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


def arrange_layout(r: dict) -> dict:
    """대시보드가 그릴 칸 배치(README '좌표 집기 이벤트'). 행 0 = 위, 열 0 = 왼쪽(사람이 숫자를 읽는 방향)."""
    pat = list(r.get("pattern") or [])
    return {"rows": len(pat), "cols": len(pat[0]) if pat else 3, "pattern": pat,
            "cells": [{"order": p["order"], "row": p["to_cell"][0], "col": p["to_cell"][1], "color": p["color"],
                       "from_mm": p.get("from_mm"), "to_mm": p.get("to_mm")} for p in r.get("plan") or []]}


def arrange_gates(r: dict, which: str = "arrange") -> list[str]:
    """실제로 보낼 수 있나(robot arrange_digit·sort_blocks 의 송신 문과 같은 조건). 가짜면 문은 늘 열림(가짜로 끝까지).
    which: backends.info() 의 arrange(숫자 배치) | coord(그릇 옮기기)."""
    import json as _json
    import os
    if backends.info().get(which) == "mock":
        return []
    gates = []
    try:
        calibrated = _json.loads((config.ROOT / "robot" / "coord_pick" / "config.json").read_text(encoding="utf-8")) \
            .get("calibrated", False)
    except Exception:
        calibrated = False
    if not calibrated:
        gates.append("보정 전")
    if any("금지 구역 자료 없음" in w for w in r.get("warnings") or []):
        gates.append("금지 구역 자료(vision/zones.json) 없음")
    if os.getenv("BALL_ROBOT") != "1":
        gates.append("실제 구동 꺼짐(BALL_ROBOT)")
    return gates


def _arrange_refusal(d: str, color: str, r: dict) -> str:
    """dry-run 이 안 되는 이유를 쉬운 말로(블록 부족·작업 반경 밖·공간 위반·오류)."""
    fails = sorted(r.get("failures") or [], key=lambda f: "필요한데" not in f)     # 블록 부족을 먼저 말한다
    parts = []
    for f in fails:
        m = re.search(r"블록 (\d+)개가 필요한데 (\d+)개만", f)
        if m:
            what = {"red": "분홍 ", "blue": "초록 "}.get(color, "")
            parts.append(f"숫자 {d}{_josa(d, '은', '는')} 블록 {m.group(1)}개가 필요한데 지금 {what}블록으로는 "
                         f"{m.group(2)}개만 놓을 수 있어요")
        elif "닿지 않" in f or "IK" in f or "반경" in f:
            parts.append("팔이 닿지 않는 자리(작업 반경 밖)에 블록이 있어요")
    if r.get("violations"):
        parts.append("작업 공간 검사에 걸렸어요")
    if r.get("error") and not parts:
        parts.append("좌표 집기 계획을 세우지 못했어요")
    if not parts:
        parts.append("계획을 끝까지 세울 수 없었어요")
    return ". ".join(dict.fromkeys(parts)) + ". 지금은 7·1처럼 블록이 적은 숫자가 현실적이에요."


def _arrange_say_plan(d: str, r: dict) -> str:
    m, sec = divmod(int(round(float(r.get("time_s") or 0))), 60)
    return f"숫자 {d}: 블록 {r.get('blocks', len(r.get('plan') or []))}개, 약 {m}분 {sec}초 걸립니다."


def arrange_node(s: State) -> dict:
    emit("node", "arrange", text=s["text"])
    d, color, why = arrange_target(s["text"])
    if why:
        return {"intent": "arrange", "arrange": {}} | _arrange_end(s, why, "refused")
    emit("say", "arrange", text=ACCEPT_ARRANGE.format(d=d))          # 첫 말(1초 안)
    try:
        det = _detect("arrange")
    except VisionError:
        return {"intent": "arrange"} | _arrange_end(s, f"{VISION_MSG} 확인해 주세요.", "error")
    t0 = time.perf_counter()
    try:
        r = _clean(backends.get()["arrange"](d, color, True, detection=det))
    except Exception as e:
        emit("error", "arrange", message=f"arrange_digit: {type(e).__name__}: {e}")
        return {"intent": "arrange"} | _arrange_end(s, "좌표 집기 계획을 세우지 못했어요.", "error")
    gates = arrange_gates(r) if r.get("ok") else []
    emit("tool", "arrange", name="arrange_digit", status="end", mode="dry_run", digit=d, color=color, result=r,
         layout=arrange_layout(r), gates=gates, plan_only=bool(gates), latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    st = {"digit": d, "color": color, "det": _clean(det), "dry": r}
    if not r.get("ok"):
        return {"intent": "arrange", "arrange": st} | _arrange_end(s, _arrange_refusal(d, color, r), "refused")
    if gates:          # 보정 전·금지 구역 없음·BALL_ROBOT 없음 → 실행하지 않고 계획(칸 배치)만
        head = "보정 전이라" if "보정 전" in gates else f"{gates[0]}이라"
        msg = f"{head} 계획만 보여 드릴게요. {_arrange_say_plan(d, r)} 칸 배치는 화면에 띄웠어요."
        return {"intent": "arrange", "arrange": st} | _arrange_end(s, msg, "plan_only")
    w = len(r.get("warnings") or [])
    msg = _arrange_say_plan(d, r) + (f" 경고 {w}건은 화면에 띄웠어요." if w else "") + " 할까요?"
    emit("say", "arrange", text=msg)
    return {"intent": "arrange", "arrange": st, "status": "waiting_confirm", "answer": msg}


def _arrange_end(s: State, msg: str, status: str) -> dict:
    emit("say", "arrange", text=msg)
    return {"answer": msg, "status": status,
            "history": [{"ts": now_iso(), "started": s.get("started"), "elapsed_s": _elapsed(s), "text": s["text"],
                         "intent": "arrange", "status": status, "say": msg}]}


def arrange_confirm_node(s: State) -> dict:
    """사용자의 다음 말을 기다린다(LangGraph interrupt). run_command 가 같은 thread 의 다음 말로 이어 준다."""
    a = s.get("arrange") or {}
    ans = str(interrupt({"ask": "할까요?", "digit": a.get("digit")}) or "")
    if is_yes(ans) and not STOP.is_set():
        return {"arrange": {**a, "go": True}}
    return {"arrange": {**a, "go": False}} | _arrange_end(s, "알겠어요, 숫자 배치는 하지 않을게요.", "cancelled")


def arrange_run_node(s: State) -> dict:
    a = s["arrange"]
    d = a["digit"]
    emit("node", "arrange_run", digit=d)
    emit("say", "arrange_run", text="시작할게요. 멈추려면 '멈춰'라고 말해 주세요.")
    why = _home_align("arrange_run")      # arrange_digit 도 홈에서 시작해야 한다(not_at_home)
    if why or STOP.is_set():
        msg, status = (why, "error") if why else ("멈췄습니다.", "stopped")
        emit("say", "arrange_run", text=msg)
        return {"answer": msg, "status": status, "arrange": a,
                "history": [{"ts": now_iso(), "started": s.get("started"), "elapsed_s": _elapsed(s), "text": s["text"],
                             "intent": "arrange", "status": status, "say": msg}]}
    t0 = time.perf_counter()
    def call() -> dict:   # 몇 분 걸린다. "멈춰"는 다른 스레드의 run_command 가 robot.stop() 으로(정책과 같은 길)
        try:
            return _clean(backends.get()["arrange"](d, a["color"], False, detection=a["det"]))
        except Exception as e:
            return {"ok": False, "reason": "error", "error": f"{type(e).__name__}: {e}"}
    r = _robot_call("arrange_run", call)
    if r.get("reason") != "refused":
        _moving()
    emit("tool", "arrange_run", name="arrange_digit", status="end", mode=r.get("mode", "real"), digit=d, result=r,
         layout=arrange_layout(r) if r.get("plan") else None, latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    n = len(r.get("plan") or [])
    done = (r.get("run") or {}).get("steps_done")
    if r.get("reason") == "done":
        msg = f"숫자 {d}{_josa(d, '을', '를')} 다 놓았습니다(블록 {n}개)."
        status = "done"
    elif r.get("reason") == "stopped" or STOP.is_set():
        msg = "멈췄습니다." + (f" 블록 {done}개까지 놓았어요." if done is not None else "")
        status = "stopped"
    elif r.get("reason") == "refused":
        msg = r.get("studio_msg") or f"실행하지 않았어요. {r.get('error') or ''}".strip()
        status = "refused"
    else:
        msg = f"좌표 집기가 중간에 끝났어요({r.get('reason')}). 확인해 주세요."
        status = "error"
    try:
        obs = _detect("arrange_run")              # 끝나면 다시 보고(COORD_PICK.md 9절 5)
    except VisionError:
        obs = {}
    emit("say", "arrange_run", text=msg)
    _home_after("arrange_run", status)
    return {"obs": obs, "answer": msg, "status": status, "arrange": {**a, "result": r},
            "history": [{"ts": now_iso(), "started": s.get("started"), "elapsed_s": _elapsed(s), "text": s["text"],
                         "intent": "arrange", "status": status, "say": msg}]}


# ------------------------------------------------------------------ coord: 좌표 집기 그릇 옮기기(BALL_COORD=1, 10-22 r16)
# robot/COORD_PICK.md 9절 sort_blocks: plan 이 찍은 detection 으로 dry-run(계획·시간) → 말하고 → 같은 detection 으로 바로 실행
# (숫자 배치처럼 "할까요?" 를 묻지 않는다 — 시연 흐름) → 다시 찍어 옮겨진 개수를 센다. 실행은 그래프 스레드에서 몇 분 걸리고,
# "멈춰"는 다른 스레드의 run_command 가 robot.run_policy.stop() 으로(정책과 같은 길)
COORD_INTENTS = ("defect", "red", "blue")
_KO_CLS = {"분홍": "red_ok", "초록": "blue_ok", "분홍 불량": "red_defect", "초록 불량": "blue_defect"}  # robot plan.COLOR_KO
_COORD_END = {"tracking": "팔이 목표를 따라가지 못해", "stale_state": "로봇 상태가 끊겨", "not_at_home": "시작 자세가 아니라",
              "step_limit": "한 번에 움직이는 양이 한도를 넘어", "timeout": "시간이 넘어", "connect_refused": "로봇 연결이 거부돼",
              "interrupted": "중간에 끊겨", "error": "오류로"}


def coord_on(s: State) -> bool:
    """의도가 defect·red·blue 면 늘 좌표 집기(10-23 13:10, handoff 는 늘 ACT). 못 불러왔으면 plan_node 가 거부한다."""
    return s.get("intent") in COORD_INTENTS and coord_available()


def _secs(t) -> str:
    m, sec = divmod(int(round(float(t or 0))), 60)
    return f"{m}분 {sec}초" if m else f"{sec}초"


def _coord_refusal(r: dict, what: str = "좌표 집기") -> str:
    """dry-run 이 안 되는 이유를 쉬운 말로. 자세한 failures·violations·warnings 는 tool 이벤트에 있다."""
    if any("닿지 않" in f or "IK" in f for f in r.get("failures") or []):
        return f"팔이 닿지 않는 자리에 블록이 있어 {what} 계획을 세우지 못했어요."
    if r.get("violations"):
        return f"작업 공간 검사에 걸려 {what}를 하지 않았어요."
    return f"{what} 계획을 세우지 못했어요."


def coord_node(s: State) -> dict:
    if STOP.is_set():
        return {"status": "stopped"}
    intent = s["intent"]
    t = _task(s)
    before, sort = s["obs"], backends.get()["sort"]
    what = [s["defect_cls"]] if s.get("defect_cls") in DEFECT_KO else intent    # "초록 불량" → ["blue_defect"](10-23)
    kw = {"max_blocks": s["goal"]} if s.get("count") else {}        # "분홍 하나만" → 로봇에 가까운 것부터 그 개수
    if s.get("which"):                    # "왼쪽 초록" → 같은 색 후보를 위 카메라 사진 u 순서로(로봇 함수, 10-23)
        kw["which"] = s["which"]
    elif s.get("next_to"):                # "분홍 옆 초록" → 분홍 블록에 가장 가까운 초록(로봇 함수 which={"next_to": …}, 13:30)
        kw["which"] = {"next_to": s["next_to"]}
    n0 = s.get("attempts", 0)
    # 실제 로봇 함수면 detection 을 넘기지 않는다(10-22 17:50 실물 첫 성공): sort_blocks 가 위 카메라 원본을 직접 찍어 블록 긴 축
    # 방향(yaw)까지 재고 집게를 긴 축에 수직으로 맞춘다. 비전 detect_objects 의 image_path 는 구역·상자를 그린 사진이라 원본
    # (front_image)으로 못 쓴다. dry-run 도 같이 찍게 한다 — 방향 없는 계획은 "집게 두 끝 높이 차"로 거부되는 블록이 있었다.
    # 가짜 로봇은 plan 의 detection 을 그대로 넘긴다(같은 가짜 장면)
    live = backends.info().get("coord") == "real"
    dkw = {} if live else {"detection": before}
    emit("node", "coord", intent=intent, goal=s["goal"], detect="live" if live else "plan", which=s.get("which"),
         next_to=s.get("next_to"), defect_cls=s.get("defect_cls"))
    why = _home_align("coord") if not STOP.is_set() else None    # sort_blocks 는 홈에서 시작해야 하고(not_at_home), 위 카메라
    if STOP.is_set():                                            # 사진에 팔이 안 걸리게 찍기 전에 맞춘다
        return {"status": "stopped"}
    if why:
        return {"status": "error", "error_msg": why, "coord": {"reason": "home"}}
    t0 = time.perf_counter()
    try:
        dry = _clean(sort(what, True, **dkw, **kw))
    except Exception as e:
        dry = {"ok": False, "reason": "error", "error": f"{type(e).__name__}: {e}"}
    gates = arrange_gates(dry, "coord") if dry.get("ok") else []
    emit("tool", "coord", name="sort_blocks", status="end", mode="dry_run", color=intent, result=dry, gates=gates,
         detect="live" if live else "plan", which=s.get("which"), next_to=s.get("next_to"), defect_cls=s.get("defect_cls"),
         latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    n = int(dry.get("blocks") or len(dry.get("plan") or []))
    brief = {"blocks": n, "time_s": dry.get("time_s"), "summary": dry.get("summary"), "gates": gates,
             "block_yaws": dry.get("block_yaws")}
    if not dry.get("ok"):
        return {"status": "error", "error_msg": _coord_refusal(dry), "coord": brief | {"reason": dry.get("reason")}}
    if gates:          # 보정 전·금지 구역 없음·BALL_ROBOT 없음 → 보내지 않고 계획만(숫자 배치와 같은 문)
        head = "보정 전이라" if "보정 전" in gates else f"{gates[0]}이라"
        return {"status": "error", "coord": brief | {"reason": "plan_only"},
                "error_msg": f"{head} 좌표 집기는 계획만 세웠어요(블록 {n}개, 약 {_secs(dry.get('time_s'))}). "
                             "계획은 화면에 띄웠어요."}
    w = len(dry.get("warnings") or [])
    emit("say", "coord", text=f"블록 {n}개, 약 {_secs(dry.get('time_s'))} 걸려요."
                              + (f" 경고 {w}건은 화면에 띄웠어요." if w else ""), summary=dry.get("summary"))
    if STOP.is_set():
        return {"status": "stopped", "coord": brief | {"reason": "stopped"}}
    emit("tool", "coord", name="sort_blocks", status="start", mode="real", color=intent, blocks=n,
         detect="live" if live else "plan", which=s.get("which"), next_to=s.get("next_to"))
    t0 = time.perf_counter()

    def call() -> dict:
        try:
            return _clean(sort(what, False, **dkw, **kw))
        except Exception as e:
            return {"ok": False, "reason": "error", "error": f"{type(e).__name__}: {e}"}
    r = _robot_call("coord", call)
    if r.get("reason") != "refused":
        _moving()
    run_s = round(time.perf_counter() - t0, 2)
    emit("tool", "coord", name="sort_blocks", status="end", mode=r.get("mode", "real"), color=intent, result=r,
         latency_ms=round(run_s * 1000, 1))
    rr = r.get("reason")
    _HOME["ok"] = rr == "done"
    brief |= {"reason": rr, "run_id": r.get("run_id"), "log_path": r.get("log_path")}
    if rr == "refused":                        # 송신 전에 로봇이 거부(잠금·문) → 팔은 안 움직였다
        emit("error", "coord", message=r.get("error") or "refused")
        why = r.get("studio_msg") or ("다른 동작이 이미 실행 중이라 좌표 집기를 하지 않았습니다."
                                      if "already_running" in (r.get("error") or "") else
                                      _REFUSED.get(_refuse_code(r) or "", "로봇이 좌표 집기를 거부했습니다."))
        return {"status": "error", "error_msg": why, "coord": brief}
    try:
        after = _detect("coord")
    except VisionError:
        try:                                  # 카메라가 늦게 풀리는 경우 한 번 더
            time.sleep(config.CAMERA_WAIT_S)
            after = _detect("coord")
        except VisionError:
            return {"status": "error", "error_msg": VISION_MSG, "attempts": n0 + n, "coord": brief}
    dec = {k: max(0, before["zones"][t["src"]][k] - after["zones"][t["src"]][k]) for k in t["keys"]}
    n_inc = _count(after, t["dst"], t["keys"]) - _count(before, t["dst"], t["keys"])
    moved = max(0, min(sum(dec.values()), n_inc))
    dropped = max(0, sum(report.outside_counts(after).values()) - sum(report.outside_counts(before).values()))
    this, left = {}, moved                    # 카메라로 옮겨진 개수를 클래스별로
    for k, c in dec.items():
        if min(c, left):
            this[k] = min(c, left)
            left -= this[k]
    by_class = dict(s.get("moved_by_class") or {})
    for k, c in this.items():
        by_class[k] = by_class.get(k, 0) + c
    if STOP.is_set() or rr == "stopped":
        status, why = "stopped", ""
    elif rr == "done" and moved >= n:
        status, why = "done", ""
    elif rr == "done":
        status, why = "error", f"좌표 집기로 {n}개 중 {moved}개만 옮겨졌어요."
    else:
        status, why = "error", f"좌표 집기가 {_COORD_END.get(rr, '중간에')} 멈췄어요."
    # 작업 기록: 블록마다 한 줄(시도 = 블록 1개, ACT 와 같은 단위). 계획 순서대로 카메라가 센 개수만큼 성공으로 친다
    budget = dict(this)
    for i, st in enumerate(r.get("plan") or dry.get("plan") or []):
        k = st.get("cls") or _KO_CLS.get(st.get("color"))
        ok_i = budget.get(k, 0) > 0
        if ok_i:
            budget[k] -= 1
        runlog.append({
            "ts": now_iso(), "thread_id": s["thread_id"], "text": s["text"], "intent": intent,
            "policy": f"coord_{intent}", "attempt": n0 + i + 1, "before": before["zones"], "after": after["zones"],
            "ok": ok_i, "moved": int(ok_i), "moved_by_class": {k: 1} if ok_i else {},
            "fail_reason": None if ok_i else (why or "옮겨지지 않음"), "dropped": dropped if i == 0 else 0,
            "policy_ok": rr == "done", "policy_reason": rr, "duration_s": run_s, "method": "coord", "which": s.get("which"),
            "next_to": s.get("next_to"), "defect_cls": s.get("defect_cls"),
            "coord": {"run_id": r.get("run_id"), "blocks": n, "time_s": dry.get("time_s"), "log_path": r.get("log_path")},
            "backend": backends.info().get("coord"),
        })
    remaining = _count(after, t["src"], t["keys"])
    emit("node", "coord", ok=status == "done", moved=moved, done=s.get("done", 0) + moved, remaining=remaining,
         status=status, fail_reason=why or None)
    return {"obs": after, "done": s.get("done", 0) + moved, "moved_by_class": by_class, "status": status,
            "error_msg": why, "attempts": n0 + n, "dropped": s.get("dropped", 0) + dropped,
            "coord": brief | {"moved": moved}}


# 명령 아님(10-22 밤): 마이크 시험·잘 들려? → 짧은 답, 인사·잡담·청중에게 하는 말 → 말 없이 넘어감,
# LLM 실패 + 자신 없음·낱말 근거 없음 → 다시 말해 달라고 함. 로봇·카메라는 건드리지 않고 기록(history)에도 남기지 않는다
UNSURE_MSG = "잘 못 알아들었어요. 다시 말해 주세요."
NONE_MSGS = {"hear": "네, 잘 들려요.", "chat": "", "unsure": UNSURE_MSG, "vague": UNSURE_MSG}


def none_node(s: State) -> dict:
    r = s.get("route") or {}
    kind = r.get("none_kind") or "chat"
    emit("node", "none", kind=kind, why=r.get("none_check"), was=r.get("none_from"))
    msg = NONE_MSGS.get(kind, "")
    if msg:
        emit("say", "none", text=msg)
    return {"answer": msg, "status": "ignored"}


# 홈 자세 명령(10-23): "홈으로"·"홈 자세로 돌려 줘"·"처음 자세로"·"기본 자세로 되돌려"·"제자리로"(router.is_home 규칙).
# go_home(open_gripper=True) — 홈 맞춤과 같은 길(Studio 세션 정리 포함). 홈에서 너무 멀면(too_far) 한 번 더 HOME_CMD_MAX_DEG(150°)로.
# 작업 기록(history)에는 남기지 않는다(선별 작업이 아니라 결과 보고에 안 섞이게)
HOME_CMD_MSG = "팔을 홈 자세로 돌릴게요."
HOME_SLOW_MSG = "홈에서 너무 멀어 천천히 돌릴게요."
HOME_DONE_MSG = {"done": "홈 자세로 돌렸어요.", "already_home": "이미 홈이에요."}


def home_node(s: State) -> dict:
    emit("node", "home")
    if backends.get().get("home") is None:          # BALL_ROBOT=dry 거나 go_home 을 못 불러옴
        msg = "지금은 팔을 홈으로 돌릴 수 없어요(홈 이동을 불러오지 못했어요)."
        emit("error", "home", message=f"go_home: {backends.info()['reason'].get('home')}")
        emit("say", "home", text=msg)
        return {"answer": msg, "status": "error"}
    emit("say", "home", text=HOME_CMD_MSG)
    r = _go_home("home", open_gripper=True)
    if r.get("reason") == "too_far" and not STOP.is_set():
        emit("say", "home", text=HOME_SLOW_MSG)
        r = _go_home("home", open_gripper=True, max_deg=config.HOME_CMD_MAX_DEG)
    rr = r.get("reason")
    if r.get("ok"):
        _HOME.update(ok=True, fresh=time.monotonic(), moved=False)
    else:
        _HOME.update(ok=False, fresh=None, moved=False)
    if STOP.is_set() or rr == "stopped":            # "멈춰"는 run_command 가 이미 "멈췄습니다." 라고 했다
        return {"answer": "", "status": "stopped"}
    if r.get("ok"):
        msg, status = HOME_DONE_MSG.get(rr, HOME_DONE_MSG["done"]), "done"
    else:
        emit("error", "home", message=f"go_home: {rr}: {r.get('error') or ''}")
        msg, status = (HOME_FAR_MSG if rr == "too_far" else r.get("studio_msg")
                       or _REFUSED.get(_refuse_code(r) or "", "팔을 홈 자세로 돌리지 못했어요.")), "error"
    emit("say", "home", text=msg)
    return {"answer": msg, "status": status}


# 그릇에서 꺼내기(10-23 intent unsort, 리허설 초기화): "그릇에 있는 거 다시 투입 구역으로 옮겨 줘"·"다 꺼내 놔"·"초기화해 줘"
# (router.is_unsort 규칙, 색·불량을 말하면 그 그릇만, 개수를 말하면 그만큼). robot/coord_pick unsort_blocks(dry_run, colors,
# max_blocks) — sort_blocks 와 같은 꼴의 dict. 좌표 집기처럼 홈 맞춤 → dry-run(계획·송신 문) → 실행(Studio 정리 포함) → 집게 열고
# 홈. 시연 기록을 흐리지 않게 작업 기록(runlog)·대화 기록(history)·결과 보고에는 넣지 않는다
UNSORT_MSG = "그릇에 있는 블록을 투입 구역으로 꺼낼게요."
UNSORT_NO_MSG = "아직 그릇에서 꺼내기는 못 해요."
UNSORT_EMPTY_MSG = "그릇에 꺼낼 블록이 없어요."


def _unsort_moved(r: dict, planned: int) -> int:
    """꺼낸 개수: 손목 확인 실행이면 블록마다 ok, 아니면 끝까지 갔을 때(done) 계획한 개수."""
    if r.get("wrist"):
        return sum(1 for w in r["wrist"] if w.get("ok"))
    return planned if r.get("reason") == "done" else 0


def unsort_node(s: State) -> dict:
    colors, count = (s.get("route") or {}).get("colors"), s.get("count")
    unsort = backends.get().get("unsort")
    emit("node", "unsort", colors=colors, count=count)

    def end(msg: str, status: str, n: int = 0) -> dict:
        if msg:
            emit("say", "unsort", text=msg)
        _home_after("unsort", status)              # 팔이 움직였으면 집게를 열고 홈으로(멈춤이면 그 자리)
        return {"answer": msg, "status": status, "done": n}

    if unsort is None:                              # 로봇 함수가 아직 없다 — 로봇·카메라 0
        emit("tool", "unsort", name="unsort_blocks", status="unavailable", error=backends.info()["reason"].get("unsort"))
        return end(UNSORT_NO_MSG, "refused")
    emit("say", "unsort", text=UNSORT_MSG)
    why = _home_align("unsort")                     # 좌표 집기처럼 홈에서 시작한다(위 카메라 사진에 팔이 안 걸리게)
    if STOP.is_set():
        return end("", "stopped")
    if why:
        return end(why, "error")
    kw = {"colors": colors, "max_blocks": count}
    t0 = time.perf_counter()
    try:
        dry = _clean(unsort(True, **kw))
    except Exception as e:
        dry = {"ok": False, "reason": "error", "error": f"{type(e).__name__}: {e}"}
    gates = arrange_gates(dry, "unsort") if dry.get("ok") else []
    n = int(dry.get("blocks") or len(dry.get("plan") or []))
    emit("tool", "unsort", name="unsort_blocks", status="end", mode="dry_run", result=dry, gates=gates, **kw,
         latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    if not dry.get("ok"):
        emit("error", "unsort", message=f"unsort_blocks: {dry.get('reason')}: {dry.get('error') or ''}")
        return end(_coord_refusal(dry, "그릇에서 꺼내기"), "error")
    if n == 0:
        return end(UNSORT_EMPTY_MSG, "nothing")
    if gates:          # 보정 전·금지 구역 없음·BALL_ROBOT 없음 → 보내지 않고 계획만(좌표 집기와 같은 문)
        head = "보정 전이라" if "보정 전" in gates else f"{gates[0]}이라"
        return end(f"{head} 그릇에서 꺼내기는 계획만 세웠어요(블록 {n}개, 약 {_secs(dry.get('time_s'))}).", "error")
    if STOP.is_set():
        return end("", "stopped")
    emit("tool", "unsort", name="unsort_blocks", status="start", mode="real", blocks=n, **kw)
    t0 = time.perf_counter()

    def call() -> dict:
        try:
            return _clean(unsort(False, **kw))
        except Exception as e:
            return {"ok": False, "reason": "error", "error": f"{type(e).__name__}: {e}"}
    r = _robot_call("unsort", call)
    if r.get("reason") != "refused":
        _moving()
    emit("tool", "unsort", name="unsort_blocks", status="end", mode=r.get("mode", "real"), result=r, **kw,
         latency_ms=round((time.perf_counter() - t0) * 1000, 1))
    rr, moved = r.get("reason"), _unsort_moved(r, n)
    _HOME["ok"] = rr == "done"
    if STOP.is_set() or rr == "stopped":            # "멈춰"는 run_command 가 이미 "멈췄습니다." 라고 했다
        return end("", "stopped", moved)
    if rr == "refused":                             # 송신 전에 로봇이 거부(잠금·문) → 팔은 안 움직였다
        emit("error", "unsort", message=r.get("error") or "refused")
        return end(r.get("studio_msg") or _REFUSED.get(_refuse_code(r) or "", "로봇이 그릇에서 꺼내기를 거부했어요."), "error")
    if rr == "done" and moved >= n:
        return end(f"{moved}개를 꺼냈어요.", "done", moved)
    emit("error", "unsort", message=f"unsort_blocks: {rr}: {r.get('error') or ''}")
    tail = f"{n - moved}개는 못 꺼냈어요." if rr == "done" else f"그릇에서 꺼내기가 {_COORD_END.get(rr, '중간에')} 멈췄어요."
    return end((f"{moved}개를 꺼냈어요. " if moved else "") + tail, "error", moved)


def halt_node(s: State) -> dict:
    emit("node", "halt")
    STOP.set()
    _HOME.update(ok=False, fresh=None)
    stop_robot()
    emit("say", "halt", text="멈췄습니다.")
    return {"answer": "멈췄습니다.", "status": "stopped"}


# ------------------------------------------------------------------ 분기
def after_classify(s: State) -> str:
    if s["intent"] == "plan" and is_arrange(s.get("cmd") or s["text"]):
        return "arrange"
    return {"report": "report", "question": "answer", "stop": "halt", "scene": "scene",
            "plan": "planner", "none": "none", "home": "home", "unsort": "unsort"}.get(s["intent"], "plan")


def after_arrange(s: State) -> str:
    return "arrange_confirm" if s.get("status") == "waiting_confirm" else END


def after_arrange_confirm(s: State) -> str:
    return "arrange_run" if (s.get("arrange") or {}).get("go") else END


def after_planner(s: State) -> str:
    return "plan" if s.get("steps") else END


def after_finish(s: State) -> str:
    return "next_step" if s.get("steps") else END


def after_next_step(s: State) -> str:
    last = (s.get("step_results") or [{}])[-1]
    if s.get("step_i", 0) < len(s["steps"]) and last.get("status") not in _PLAN_HALT and not STOP.is_set():
        return "plan"
    return "plan_report"


def after_plan(s: State) -> str:
    if s["status"] != "running":
        return "finish"
    if s["intent"] == "handoff":
        return "hand_wait"
    return "coord" if coord_on(s) else "finish"     # 옮기기는 ACT 로 가지 않는다(10-23 13:10)


def after_hand_wait(s: State) -> str:
    return "finish" if s.get("status") in ("stopped", "no_hand", "error") else "act"


def after_act(s: State) -> str:
    return "finish" if s.get("status") in ("stopped", "error") else "verify"


def after_verify(s: State) -> str:
    if s["status"] != "running":
        return "finish"
    return "hand_wait" if s["intent"] == "handoff" else "act"


def build(checkpointer=None):
    g = StateGraph(State)
    for name, fn in [("classify", classify_node), ("plan", plan_node), ("hand_wait", hand_wait_node),
                     ("act", act_node), ("verify", verify_node), ("finish", finish_node),
                     ("report", report_node), ("answer", answer_node), ("halt", halt_node),
                     ("scene", scene_node), ("planner", planner_node), ("next_step", next_step_node),
                     ("plan_report", plan_report_node), ("arrange", arrange_node),
                     ("arrange_confirm", arrange_confirm_node), ("arrange_run", arrange_run_node),
                     ("coord", coord_node), ("none", none_node), ("home", home_node), ("unsort", unsort_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", after_classify,
                            ["plan", "report", "answer", "halt", "scene", "planner", "arrange", "none", "home", "unsort"])
    g.add_conditional_edges("arrange", after_arrange, ["arrange_confirm", END])
    g.add_conditional_edges("arrange_confirm", after_arrange_confirm, ["arrange_run", END])
    g.add_conditional_edges("planner", after_planner, ["plan", END])
    g.add_conditional_edges("finish", after_finish, ["next_step", END])
    g.add_conditional_edges("next_step", after_next_step, ["plan", "plan_report"])
    g.add_conditional_edges("plan", after_plan, ["act", "hand_wait", "coord", "finish"])
    g.add_edge("coord", "finish")
    g.add_conditional_edges("hand_wait", after_hand_wait, ["act", "finish"])
    g.add_conditional_edges("act", after_act, ["verify", "finish"])
    g.add_conditional_edges("verify", after_verify, ["act", "hand_wait", "finish"])
    for name in ("report", "answer", "halt", "scene", "plan_report", "arrange_run", "none", "home", "unsort"):
        g.add_edge(name, END)
    return g.compile(checkpointer=checkpointer)


_graph = None
_graph_lock = threading.Lock()
_busy: dict[str, threading.Lock] = {}
_warm = threading.Event()


def warmup() -> dict:
    """라우터 예문 임베딩·Chroma·보고 근거 캐시를 준비한다. UI 가 시작할 때 불러 두면 첫 명령이 빠르다."""
    from .rag import warmup as rag_warmup
    from .router import get_router

    t0 = time.perf_counter()
    get_router()
    out = rag_warmup()
    out["total_s"] = round(time.perf_counter() - t0, 2)
    return out


def _start_warmup() -> None:
    """첫 run_command 때 한 번, 백그라운드로 warmup. 실패해도 명령은 그대로 돈다(그때그때 만든다)."""
    if _warm.is_set():
        return
    _warm.set()

    def go():
        try:
            warmup()
        except Exception:
            pass
    threading.Thread(target=go, name="agent-warmup", daemon=True).start()


def get_graph():
    global _graph
    with _graph_lock:
        if _graph is None:
            config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(config.checkpoint_db()), check_same_thread=False)
            _graph = build(SqliteSaver(conn))
        return _graph


# ------------------------------------------------------------------ 진입점
def _detached(gen_fn) -> Iterator[dict]:
    """그래프를 뒤 스레드에서 돌리고 이벤트는 큐로만 넘긴다. LangGraph stream 은 받는 쪽이 당겨 갈 때만 진행해서,
    받는 쪽이 말(TTS)을 끝까지 하고 다음을 가져가면 그동안 팔이 멈춰 있었다(10-22 측정: say 마다 1.5초 멈추는 소비자 →
    다음 run_policy 시작도 1.5초 늦음). 이제 받는 쪽이 늦어도 그래프는 제 속도로 간다. 받는 쪽이 중간에 그만 읽어도
    명령은 끝까지 돈다(멈추려면 "멈춰")."""
    q: queue.Queue = queue.Queue()
    done = object()

    def work():
        try:
            for ev in gen_fn():
                q.put(ev)
        except Exception as e:
            q.put(make("error", "graph", {"message": f"{type(e).__name__}: {e}"}))
        finally:
            q.put(done)
    threading.Thread(target=work, name="agent-graph", daemon=True).start()
    while (ev := q.get()) is not done:
        yield ev


def run_command(text: str, thread_id: str = "demo", stt: dict | None = None) -> Iterator[dict]:
    """말 한 문장을 처리하며 이벤트를 흘려보낸다(계약서 3절). 끝에 node="end" 이벤트 1개.
    stt: 받아쓰기 녹음 정보(선택) {"speech_s": 말소리 초} 또는 {"wav_path": 녹음 파일} — 있으면 말소리 길이로도 환청을
    거른다(hearing.py). 없으면 글 기준만."""
    text = (text or "").strip()
    t0 = time.perf_counter()
    mi = llm.model_info()
    yield make("node", "start", {"text": text, "thread_id": thread_id, "backend": backends.info(),
                                 "robot_method_default": METHOD["default"],
                                 "llm": mi["name"], "llm_fallback": mi["fallback"]})
    _start_warmup()
    if not text:
        yield make("error", "start", {"message": "빈 명령입니다."})
        return
    if is_stop(text):                   # LLM·그래프를 거치지 않고 바로 멈춘다
        STOP.set()
        _HOME.update(ok=False, fresh=None)   # 그 자리 정지 → 다음 동작 전에 홈 맞춤(HOME_ALIGN=after_stop 이어도)
        try:
            stop_robot()
            yield make("tool", "stop", {"name": "robot.stop", "status": "end"})
        except Exception as e:
            yield make("error", "stop", {"message": f"{type(e).__name__}: {e}"})
        yield make("say", "stop", {"text": "멈췄습니다."})
        yield from _cancel_pending(thread_id)          # 숫자 배치 "할까요?" 를 기다리던 중이면 그것도 취소
        yield make("node", "end", {"intent": "stop", "method": "rule", "status": "stopped",
                                   "elapsed_s": round(time.perf_counter() - t0, 3)})
        return
    why = hearing.check(text, stt)
    if why:      # 받아쓰기 환청: 그래프·"할까요?" 기다림·실행 중 명령을 건드리지 않고 다시 말해 달라고만 한다
        yield make("tool", "unheard", {"name": "stt_guard", "status": "end", "result": {"ignored": True, "why": why}})
        yield make("say", "unheard", {"text": hearing.UNHEARD_MSG, "unheard": True})
        yield make("node", "end", {"intent": "unheard", "method": "rule", "status": "ignored",
                                   "answer": hearing.UNHEARD_MSG, "elapsed_s": round(time.perf_counter() - t0, 3)})
        return

    with _graph_lock:
        busy = _busy.setdefault(thread_id, threading.Lock())
    if not busy.acquire(blocking=False):
        yield make("error", "start", {"message": "이전 명령이 아직 실행 중입니다. '멈춰'로 멈춘 뒤 다시 말해 주세요."})
        return
    yield from _detached(lambda: _run_graph(text, thread_id, t0, busy))


def _run_graph(text: str, thread_id: str, t0: float, busy: threading.Lock) -> Iterator[dict]:
    try:
        STOP.clear()
        graph = get_graph()
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 500}   # plan 은 단계 × 시도만큼 노드가 돈다
        if _pending(graph, cfg):
            if is_yes(text) or is_no(text):          # "할까요?" 에 대한 답 → 멈춘 자리에서 이어 간다
                yield from _stream(graph, Command(resume=text), cfg)
                yield _end_event(graph, cfg, t0)
                return
            yield from _stream(graph, Command(resume="취소"), cfg)   # 다른 말 → 기다림을 끝내고 새 명령으로
        init: State = {"text": text, "thread_id": thread_id, "started": now_iso(), "intent": None,
                       "count": None, "route": {},
                       "goal": 0, "done": 0, "moved_by_class": {}, "attempts": 0, "misses": 0,
                       "max_attempts": 0, "obs": {}, "last_policy": {}, "status": "", "error_msg": "",
                       "answer": "", "dropped": 0, "steps": [], "step_i": 0, "step_results": [], "plan_started": "",
                       "arrange": {}, "coord": {}, "cmd": text, "robot_method": "", "method_by": "",
                       "method_fallback": False, "method_asked": None, "which": None, "next_to": None, "target_by": None}
        yield from _stream(graph, init, cfg)
        yield _end_event(graph, cfg, t0)
    finally:
        busy.release()


def _stream(graph, inp, cfg) -> Iterator[dict]:
    try:
        yield from graph.stream(inp, cfg, stream_mode="custom")
    except Exception as e:
        yield make("error", "graph", {"message": f"{type(e).__name__}: {e}"})


def _end_event(graph, cfg, t0: float) -> dict:
    v = graph.get_state(cfg).values
    return make("node", "end", {"intent": v.get("intent"), "count": v.get("count"),
                                "method": (v.get("route") or {}).get("method"), "status": v.get("status"),
                                "robot_method": v.get("robot_method"), "which": v.get("which"), "next_to": v.get("next_to"),
                                "target_by": v.get("target_by"),
                                "defect_cls": v.get("defect_cls"),
                                "answer": v.get("answer"), "attempts": v.get("attempts"), "done": v.get("done"),
                                "elapsed_s": round(time.perf_counter() - t0, 3)})


def _pending(graph, cfg) -> bool:
    """이 thread 가 "할까요?"(interrupt) 답을 기다리는 중인가."""
    try:
        st = graph.get_state(cfg)
        return bool(st.next) and any(getattr(t, "interrupts", ()) for t in st.tasks)
    except Exception:
        return False


def _cancel_pending(thread_id: str) -> Iterator[dict]:
    with _graph_lock:
        busy = _busy.setdefault(thread_id, threading.Lock())
    if not busy.acquire(blocking=False):      # 실행 중이면 robot.stop() 이 이미 멈췄다
        return
    try:
        graph = get_graph()
        cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 500}
        if _pending(graph, cfg):
            yield from _stream(graph, Command(resume="취소"), cfg)
    finally:
        busy.release()
