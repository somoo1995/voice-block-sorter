"""발표용 대시보드 — 말로 시키는 스펀지 블록 선별 로봇.

실행(프로젝트 루트에서):  .venv/bin/python -m ui.app      → http://127.0.0.1:7870
버튼 누르고 말하기 → STT(OVMS) → run_command → 이벤트 스트리밍(노드 진행·로그·카메라·개수·디바이스) → 음성 응답.

환경변수
  UI_PORT       7870 (7860 은 Physical AI Studio 가 쓴다)
  UI_AGENT      auto | agent | fake   auto = agent/ 를 불러 보고 없으면 가짜 스트림(ui/fake_agent.py)
  UI_AGENT_FN   run_command 위치를 직접 지정, 예) agent.graph:run_command
  UI_TTS_OUT    speaker | browser | silent | timed | off   speaker = 노트북 스피커로 차례대로(기본, pw-play 있을 때)
                silent = 합성만 하고 소리는 안 냄(밤 리허설·시험, 말하기 시간은 잰다)
                timed = 소리는 안 내고 음성 길이만큼 기다림(실연 시간 흉내 — 다음 말이 실제처럼 밀린다)
  UI_NET_GUARD  block | log | off   block(기본) = 이 프로세스의 인터넷 연결을 막는다(ui/netguard.py)
  UI_VIEW       0 | 1   1 = 보기 전용 화면을 함께 연다(ui/view.py, 0.0.0.0:UI_VIEW_PORT=7880, 주소는 시작 로그·머리줄).
                입력·버튼·마이크 없이 읽기만 — 와이파이의 노트북·폰이 본다. 이 대시보드(Gradio)는 그대로 127.0.0.1
  LLM_MODEL·STT_MODEL·LLM_BASE_URL 등은 SETUP.md / .env 를 따른다
"""
from __future__ import annotations

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # 시연은 외부 통신 0건
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import copy
import html
import importlib
import inspect
import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import types
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

from ui import netguard  # gradio·에이전트를 불러오기 전에 건다

netguard.install()

import gradio as gr
import requests

from ui import fake_agent, micdev, stt, tts
from ui.report import render_report

PORT = int(os.getenv("UI_PORT", "7870"))
BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8100/v3").rstrip("/")
REPORT_DIR = ROOT / "ui" / "reports"
TTS_OUT = os.getenv("UI_TTS_OUT") or ("speaker" if tts.player() else "browser")
# 받은 녹음을 남긴다(10-22 밤: 실물 받아쓰기가 나빴는데 녹음이 /tmp 에만 있어 재부팅으로 사라짐) → python -m ui.stt_real_check --voice-log
VOICE_DIR = ROOT / "ui" / "voice_log"
VOICE_LOG = os.getenv("UI_VOICE_LOG", "1") != "0"
VOICE_KEEP = int(os.getenv("UI_VOICE_KEEP", "400"))  # 이보다 많으면 오래된 녹음부터 지운다(약 0.1MB/개)

# 빠른 명령 = 발표 실연 순서(계약서 2절 끝 "발표 실연", 블록 4개). 멈춤은 옆의 ■ 멈춰 버튼
QUICK = [("①", "불량품 골라내 줘"), ("②", "지금 배치 어때?"), ("③", "불량 빼고 나머지 다 정리해 줘"),
         ("④", "분홍 블록 하나 내 손에 줘"), ("⑤", "결과 알려 줘")]
# 접어 둔 다른 명령(리허설·질문 시연용)
EXTRA = [("", "분홍 블록 옮겨 줘"), ("", "초록 블록 옮겨 줘"), ("▶", "분홍 두 개만 옮기고 불량 골라내 줘"),
         ("?", "초록 불량은 왜 불량이야?"), ("?", "그리퍼가 뜨거우면 어떻게 해?"), ("?", "인터넷 없이 돌아가?"),
         ("#", "분홍으로 숫자 1 만들어 줘")]

# LangGraph 노드 → 화면 이름. 에이전트가 agent.NODE_LABELS / agent.NODE_ORDER 를 내보내면 그것을 쓴다.
# 이름은 agent/graph.py 의 노드(classify·plan·hand_wait·act·verify·finish·report·answer·halt)와 같다.
# scene·make_plan·step 은 10-21 밤 intent scene·plan 용(ui/fake_agent.py 이름). 에이전트 이름이 다르면 처음 지날 때 뒤에 붙는다.
NODE_LABELS = {"classify": "의도 파악", "plan": "개수 세기", "hand_wait": "손 대기", "act": "정책 실행",
               "verify": "다시 세기", "finish": "마무리", "answer": "기준서 답변", "report": "결과 보고", "halt": "정지",
               "scene": "배치 보기", "make_plan": "계획 세우기", "step": "단계",
               "planner": "계획 세우기", "next_step": "다음 단계", "plan_report": "계획 보고",
               "arrange": "숫자 계획", "arrange_confirm": "확인 대기", "arrange_run": "숫자 놓기", "home": "홈 자세",
               "unsort": "그릇에서 꺼내기"}
NODE_ORDER = ["classify", "plan", "hand_wait", "act", "verify", "finish", "answer", "report"]
# home = "홈으로"·"처음 자세로" 명령(10-23, agent RESULT 15절) — go_home 으로 팔을 움직이는 칸
# unsort = "다 꺼내 놔"·"초기화해 줘"(10-23, RESULT 16절, 리허설 초기화) — unsort_blocks 로 그릇 블록을 투입 구역에 되돌리는 칸
NODE_LAYER = {"classify": "판단", "report": "판단", "answer": "검색", "plan": "보기", "verify": "보기",
              "hand_wait": "보기", "act": "행동", "scene": "보기", "make_plan": "판단", "planner": "판단",
              "arrange": "판단", "arrange_run": "행동", "none": "판단", "unheard": "판단", "coord": "행동", "home": "행동",
              "unsort": "행동"}
BOOKKEEPING = ("start", "end")  # run_command 의 시작·끝 표시(그래프 노드 아님)
TOOL_LAYER = {"detect_objects": "보기", "run_policy": "행동", "search_manual": "검색", "intent_router": "판단", "sort_blocks": "행동",
              "go_home": "행동", "unsort_blocks": "행동"}
DEVICES = [("듣기", "Whisper large-v3-turbo INT8", os.getenv("STT_DEVICE", "NPU")),
           ("보기", "YOLO n 5클래스 INT8", "NPU"),
           ("판단", os.getenv("LLM_MODEL", "Qwen3 INT4"), os.getenv("LLM_DEVICE", "GPU")),
           ("검색", "Qwen3-Embedding 0.6B + Reranker", "CPU"),
           ("행동", "ACT 정책 4개", "GPU"),
           ("말하기", "Supertonic 3 한국어", "CPU"),
           ("흐름", "LangGraph 상태 그래프", "CPU")]
STOP_WORDS = ("멈춰", "스톱", "정지", "중지", "그만")
CLASS_KO = {"red_ok": "정상 분홍", "blue_ok": "정상 초록", "red_defect": "분홍 불량", "blue_defect": "초록 불량"}  # 코드 red_* = 분홍
STATUS_KO = {"done": "완료", "nothing": "옮길 것 없음", "stopped": "멈춤", "gave_up": "두 번 실패", "no_hand": "손 없음",
             "limit": "시도 한도", "error": "오류", "skipped": "건너뜀", "refused": "거부", "waiting_confirm": "확인 기다림",
             "plan_only": "계획만", "cancelled": "취소", "ignored": "안 움직임"}  # agent/graph.py State.status
# status ignored 는 의도로 나눠 보인다: none = 명령 아님(잡담·마이크 시험·LLM 실패 때 자신 없음), unheard = 받아쓰기 환청
IGNORED_KO = {"none": "명령 아님", "unheard": "못 알아들음"}
NOTHING_KO = {"unsort": "꺼낼 것 없음"}  # status nothing 을 의도로 나눠 보임(그릇에서 꺼내기는 그릇이 빔)
UNSORT_KO = {"red": "분홍", "blue": "초록", "defect": "불량"}  # unsort_blocks colors(꺼낼 그릇). 없으면 그릇 전부
NONE_KO = {"hear": "마이크 시험", "chat": "잡담", "unsure": "LLM 실패·자신 없음", "vague": "움직일 낱말 없음"}  # intent_router none_kind
# 거부(refused: 할 수 없어 하지 않음 — 위치로 고르기 못 함·배운 적 없는 동작·로봇이 송신 전에 막음)는 실패(✗ 빨강)와 달리 ⊘ 주황
REFUSE_ICON = "⊘"
# 위치 단서(10-23, 13:30 맨 위·맨 아래·가까운·먼, RESULT 19절): end.which·node plan·coord·tool sort_blocks which. 문구는 에이전트
# router.WHICH_KO 와 같다. top = 위 카메라 v 작은 쪽(로봇에서 먼 쪽) → 작업대 그림 아래쪽이 로봇 쪽. mixed = 위치를 둘 이상 말함(거부)
WHICH_KO = {"left": "왼쪽", "right": "오른쪽", "middle": "가운데", "top": "맨 위", "bottom": "맨 아래",
            "near": "로봇에 가장 가까운", "far": "로봇에서 가장 먼", "mixed": "위치 둘"}
# 관계(10-23 13:30): end.next_to·tool sort_blocks next_to = 그 블록에 가장 가까운 것. defect = 색 없는 불량(로봇 이름이 없어 거부)
NEXT_KO = {"red": "분홍", "blue": "초록", "red_defect": "분홍 불량", "blue_defect": "초록 불량", "defect": "불량"}
NEXT_CLS = {"red": ("red_ok",), "blue": ("blue_ok",), "red_defect": ("red_defect",), "blue_defect": ("blue_defect",),
            "defect": ("red_defect", "blue_defect")}  # 작업대 그림에서 테두리를 두를 기준 블록(로봇 plan.NEXT_TO 와 같음)
# 위치를 읽은 길(target_by). 앞 둘은 읽음, 나머지는 못 읽음 → 에이전트가 거부("그 위치로는 아직 고르지 못해요." 등)
BY_KO = {"rule": "규칙", "llm": "LLM", "llm_none": "LLM 도 못 읽음", "need_llm": "LLM 안 물음",
         "ref_rel": "다른 블록 기준 위·아래", "llm_down": "LLM 응답 없음"}
BY_FAIL = ("llm_none", "need_llm", "ref_rel", "llm_down")
TARGET_KO = {"red": "분홍", "blue": "초록", "defect": "불량"}  # 위치가 붙은 결과 문구의 대상 색("맨 위 초록 1개 · 완료")


def _where(which=None, next_to=None, by=None) -> str | None:
    """위치·관계 → 화면 문구: "맨 위"·"분홍 옆"·"위치 못 읽음". 없으면 None."""
    if which in WHICH_KO:
        return WHICH_KO[which]
    if next_to in NEXT_KO:
        return f"{NEXT_KO[next_to]} 옆"
    return "위치 못 읽음" if by in BY_FAIL else None

# 불량 색(10-23, RESULT 16절): end.defect_cls·node plan·coord·tool sort_blocks. mixed = "초록 말고 불량"처럼 뺀 색만 말함
# (에이전트가 "불량 색은 하나만 말해 주세요"로 거부). 두 색 다 말하면 에이전트가 불량 전부로 보고 defect_cls 를 안 보낸다
DEFECT_KO = {"red_defect": "분홍 불량", "blue_defect": "초록 불량", "mixed": "불량 색 불분명"}
# 계획(intent plan) 단계 이름: 에이전트가 label 을 안 주면 intent·count 로 만든다
STEP_KO = {"defect": ("불량", "골라내기"), "red": ("분홍", "옮기기"), "blue": ("초록", "옮기기"), "handoff": ("분홍", "손에 건네기"),
           "scene": ("배치", "보기"), "report": ("결과", "보고"), "question": ("기준서", "답하기")}
STEP_STATE = {"done": "done", "ok": "done", "end": "done", "nothing": "skip", "skipped": "skip", "stopped": "stop",
              "gave_up": "fail", "limit": "fail", "no_hand": "fail", "error": "fail", "failed": "fail", "refused": "refuse"}
# agent/graph.py 의 report·answer 노드는 LLM·근거 조회가 실패해도 템플릿·검색 결과로 답을 낸다 → 대화창에 ⚠ 를 띄우지 않음
FALLBACK_ERR = re.compile(r"^(llm|citations):")


def _load_agent():
    mode = os.getenv("UI_AGENT", "auto")
    notes = []
    if mode != "fake":
        specs = [os.getenv("UI_AGENT_FN")] if os.getenv("UI_AGENT_FN") else \
            ["agent:run_command", "agent.graph:run_command", "agent.app:run_command"]
        for spec in specs:
            mod, fn = spec.split(":")
            try:
                m = importlib.import_module(mod)
                NODE_LABELS.update(getattr(m, "NODE_LABELS", {}))
                if getattr(m, "NODE_ORDER", None):
                    NODE_ORDER[:] = list(m.NODE_ORDER)
                return getattr(m, fn), spec, notes
            except Exception as e:
                notes.append(f"{spec} 불러오기 실패: {e.__class__.__name__}: {e}")
        if mode == "agent":
            raise RuntimeError("; ".join(notes))
    return fake_agent.run_command, "fake", notes


AGENT_FN, AGENT_NAME, AGENT_NOTES = _load_agent()
# 에이전트 run_command(text, thread_id, stt=…) 가 녹음 정보를 받으면 소리 기준 환청 거르기(agent/hearing.py)에 넘긴다
AGENT_STT = os.getenv("UI_STT_SOUND_CHECK", "1") != "0" and "stt" in inspect.signature(AGENT_FN).parameters


def _short_ts(ts: str | None) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%H:%M:%S") if ts else datetime.now().strftime("%H:%M:%S")
    except ValueError:
        return str(ts)[-8:]


def _resolve(p: str | None) -> str | None:
    if not p:
        return None
    path = Path(p) if Path(p).is_absolute() else ROOT / p
    return str(path) if path.exists() else None


def _ms(ms) -> str:
    if ms is None:
        return "-"
    return f"{ms / 1000:.1f}초" if ms >= 1000 else f"{ms:.1f} ms" if ms < 100 else f"{ms:.0f} ms"


def _cite_line(sources, text: str) -> str:
    """질문 답 say 의 sources[{doc, cite: "[로봇 운용 설명서 4.3]"}] → 말풍선 아래 "근거: 로봇 운용 설명서 4.3" 줄(말로는 안 읽음).
    답이 인용한 조항을 먼저, 함께 찾은 조항은 2개까지. sources 가 없으면(문서 밖·다른 노드의 say) 빈 글."""
    cites = []
    for s in sources if isinstance(sources, list) else []:
        c = str((s or {}).get("cite") or "").strip() if isinstance(s, dict) else ""
        if c and c not in cites:
            cites.append(c)
    if not cites:
        return ""
    cites.sort(key=lambda c: c not in text)  # 답 첫머리 "[문서 조항]에 따르면" 의 조항이 앞
    main, rest = cites[0].strip("[]"), [c.strip("[]") for c in cites[1:3]]
    return (f'\n\n<span class="cite">근거: {html.escape(main)}'
            + (f' · 함께 찾음 {html.escape(", ".join(rest))}' if rest else "") + "</span>")


def _refuse_html(text: str) -> str:
    """거부 말풍선: 에이전트·로봇 문구는 그대로 두고 앞 아이콘(⚠ 대신 ⊘)과 왼쪽 띠 색만 바꾼다."""
    return f'<span class="refuse"><b>{REFUSE_ICON}</b> {html.escape(text)}</span>'


class Board:
    """화면 상태 한 벌(발표용 1인 화면이라 전역). 구역마다 버전을 올려 바뀐 곳만 다시 그린다."""

    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.thread_id = "demo" if not hasattr(self, "thread_id") else f"demo-{datetime.now():%H%M%S}"
            self.chat: list[dict] = []
            self.logs: deque = deque(maxlen=300)
            self.nodes = {n: {"state": "idle", "visits": 0} for n in NODE_ORDER}
            self.order = list(NODE_ORDER)
            self.current = None
            self.current_note = ""
            self.snapshot = None
            self.snapcap = "카메라 대기 중 — 개수를 세면 위 카메라 사진이 여기에 뜹니다"
            self.zones = None
            self.objects = None  # detect_objects 의 objects(블록마다 zone·cell). 없으면(옛 형식) 개수만 그린다
            self.hand = False
            self.det_ts = None
            self.plan = None  # {"text", "steps": [{intent, count, label, state, done}], "cur", "end"}
            self.arrange = None  # 좌표 집기(숫자 배치) 계획: agent/README "좌표 집기(arrange)" 의 tool arrange_digit 이벤트
            # 이번 명령의 위치·관계 {"which", "next_to", "by", "intent", "refused"} → 작업대 그림 표·대상 그릇 카드 표
            self.target = None
            self.route_pos: list = []  # intent_router target.pos — 규칙이 위치로 못 읽은 낱말("사이" 등, 못 읽음 표시에 씀)
            self.cmd_intent = None  # 이번 명령의 의도(node plan·end) — 위치 표를 붙일 그릇 카드
            self.defect = None  # 이번 명령의 불량 색 {"cls": red_defect|blue_defect|mixed, "refused": bool} → 불량 카드 표
            self.refused_tool = None  # 바로 앞 tool 결과가 reason refused(로봇이 송신 전에 막음) → 뒤따르는 error 를 거부로 보임
            self.prev_cards: dict = {}
            self.handed = 0
            self.dev = {layer: {"model": model, "device": dev, "last": None, "calls": 0, "sum": 0.0, "max": 0.0, "active": 0}
                        for layer, model, dev in DEVICES}
            self.report_html = None
            self.report_link = None
            self.report_alt = None  # 에이전트가 따로 만든 보고서(html_path)
            self.report_open = False
            self.first_snap = None  # (경로, 시각) 이 대화의 첫 위 카메라 사진 = 보고서 "작업 전"
            self.pol_time: dict = {}  # 정책 이름 → 실행 시간 합(초)
            self.cmds: list[dict] = []  # 명령 기록(보고서 "명령별 소요 시간")
            self.running = None  # 실행 중인 명령 글
            self.t_cmd = None  # 말이 끝난 시각(음성) 또는 보낸 시각(글)
            self.departed = False
            self.audio = None
            self.voice_t0 = None
            self.backend = getattr(self, "backend", None)  # 에이전트 start 이벤트의 {"vision": real|mock, "robot": ...}
            self.ver = {k: 1 for k in ("hdr", "snap", "layout", "nodes", "cards", "dev", "chat", "log", "report", "audio")}

    def bump(self, *keys):
        for k in keys:
            self.ver[k] += 1

    # ── 기록 ──
    def log(self, kind: str, node: str, text: str, ts: str | None = None):
        self.logs.append((_short_ts(ts), kind, node, text))
        self.bump("log")

    def add_chat(self, role: str, text: str):
        self.chat.append({"role": role, "content": text})
        self.bump("chat")

    def dev_hit(self, layer: str | None, ms, device: str | None = None):
        if layer not in self.dev or ms is None:
            return
        d = self.dev[layer]
        d["last"], d["calls"], d["sum"], d["max"] = float(ms), d["calls"] + 1, d["sum"] + float(ms), max(d["max"], float(ms))
        if device:
            d["device"] = device
        self.bump("dev")

    def set_active(self, layer: str | None, on: bool):
        if layer in self.dev:
            self.dev[layer]["active"] = max(0, self.dev[layer]["active"] + (1 if on else -1))
            self.bump("dev")

    def clear_active(self, keep=("말하기",)):
        for layer, d in self.dev.items():
            if layer not in keep:
                d["active"] = 0
        self.bump("dev")

    # ── 이벤트 적용 (계약서 3절 형식) ──
    def apply(self, ev: dict, quiet: bool = False) -> str | None:
        """반환: 말할 문장(say) 이면 그 글, 아니면 None. quiet = 멈춘 뒤라 대화창에 더 쓰지 않음."""
        with self.lock:
            kind, node, data, ts = ev.get("type"), ev.get("node") or "", ev.get("data") or {}, ev.get("ts")
            refused_tool, self.refused_tool = self.refused_tool, None  # 바로 다음 이벤트에만 쓴다
            if isinstance(data, dict):
                self._plan_track(kind, node, data)
                if data.get("intent"):
                    self.cmd_intent = data["intent"]  # 대상 그릇 카드를 고르는 데 쓴다(node plan·end)
                if _where(data.get("which"), data.get("next_to"), data.get("target_by")):  # node plan·coord·end, tool sort_blocks
                    self._set_target(data, data.get("status") in ("refused", "unavailable"))
                if data.get("defect_cls") in DEFECT_KO:  # node plan·coord·end, tool sort_blocks
                    self._set_defect(data["defect_cls"], data.get("status") in ("refused", "unavailable"))
            if kind == "node" and node in BOOKKEEPING:
                if node == "end" and self.arrange is not None:
                    self.arrange["waiting"] = data.get("status") == "waiting_confirm"
                    self.arrange["status"] = data.get("status")
                    self.bump("layout")
                if node == "end":
                    if data.get("elapsed_s") is not None:
                        self.dev_hit("흐름", float(data["elapsed_s"]) * 1000)
                    self.log("refuse" if data.get("status") == "refused" else "info", "end",
                             ", ".join(f"{k}={data[k]}" for k in ("intent", "count", "which", "next_to", "target_by", "defect_cls",
                                                                  "status", "elapsed_s")
                                       if data.get(k) is not None), ts)
                else:
                    if isinstance(data.get("backend"), dict):
                        self.backend = data["backend"]
                        self.bump("hdr")
                    self.log("info", "start", ", ".join(f"{k}={v}" for k, v in data.items() if k in ("backend", "llm") and v), ts)
            elif kind == "node":
                self._enter(node, data)
                note = ", ".join(f"{k}={v}" for k, v in data.items() if k in ("intent", "count", "policy", "attempt", "status") and v is not None)
                where = _where(data.get("which"), data.get("next_to"))
                self.log("node", node, note + (f" · {where}" if where else ""), ts)
                if data.get("latency_ms") is not None:
                    self.dev_hit(NODE_LAYER.get(node), data["latency_ms"], data.get("device"))
            elif kind == "tool":
                self._tool(node, data, ts)
            elif kind == "say":
                text = str(data.get("text") or "").strip()
                if text and not quiet:
                    self.add_chat("assistant", text + _cite_line(data.get("sources"), text))
                    self.log("say", node, text, ts)
                    return text
            elif kind == "report":
                self._report(data)
                self.log("report", node, data.get("summary") or data.get("text") or "보고서", ts)
            elif kind == "error":
                msg = data.get("message") or data.get("error") or str(data)
                if FALLBACK_ERR.match(msg):  # 에이전트가 대체 경로로 이어 가는 오류(답은 그대로 나옴) → 로그만
                    self.log("warn", node, f"대체 경로: {msg}", ts)
                elif refused_tool:  # run_policy·sort_blocks 실행이 reason refused(로봇이 막음) → 문구 그대로, 아이콘·색만 거부
                    self.add_chat("assistant", _refuse_html(msg))
                    self.log("refuse", node, f"{REFUSE_ICON} {refused_tool} 거부: {msg}", ts)
                else:
                    self.add_chat("assistant", f"⚠ {msg}")
                    self.log("error", node, msg, ts)
            else:
                self.log(str(kind), node, str(data)[:200], ts)
        return None

    def _arrange(self, data: dict):
        """tool arrange_digit(dry_run 계획 / arrange_run 실행) → 칸 배치 그림."""
        lay = data.get("layout") if isinstance(data.get("layout"), dict) else {}
        res = data.get("result") if isinstance(data.get("result"), dict) else {}
        prev = self.arrange or {}
        self.arrange = {
            "digit": data.get("digit"), "color": data.get("color"), "mode": data.get("mode"),
            "plan_only": bool(data.get("plan_only")), "gates": list(data.get("gates") or []),
            "layout": lay or prev.get("layout") or {}, "ok": res.get("ok"), "reason": res.get("reason"),
            "blocks": res.get("blocks"), "time_s": res.get("time_s"), "warnings": list(res.get("warnings") or []),
            "failures": [str(x) for x in res.get("failures") or []],
            "placed": ((res.get("run") or {}).get("steps_done") if isinstance(res.get("run"), dict) else None),
            "waiting": False, "status": None,  # 실행(real)은 "그렇게 해" 대답으로 오므로 제목은 계획 때 명령 그대로
            "text": (prev.get("text") if data.get("mode") == "real" and prev.get("text") else self.running or prev.get("text", "")),
        }
        self.bump("layout")

    def _set_target(self, data: dict, refused: bool):
        """위치·관계 → 작업대 그림의 표·대상 그릇 카드의 표. 한 번 거부되면 그 명령 동안 거부로 둔다(end 가 다시 보내도).
        못 읽은 위치(target_by llm_none 등)는 which·next_to 없이 와서 늘 거부."""
        prev = self.target or {}
        which, nt = data.get("which"), data.get("next_to")
        same = (prev.get("which"), prev.get("next_to")) == (which, nt)
        by = data.get("target_by") or (prev.get("by") if same else None)  # node plan·coord 는 target_by 를 안 보낸다
        # 위치 둘·색 없는 불량 옆·못 읽은 위치는 에이전트가 늘 거부 → node plan 때부터 거부로(주황 표가 잠깐 파랗게 뜨지 않게)
        refused = refused or by in BY_FAIL or which == "mixed" or nt == "defect" or (same and prev.get("refused", False))
        cur = {"which": which, "next_to": nt, "by": by, "intent": data.get("color") or self.cmd_intent, "refused": refused}
        if prev != cur:
            self.target = cur
            self.bump("layout", "cards")

    def _pos_note(self) -> str:
        """못 읽은 위치 낱말(intent_router target.pos) → ": 사이·끝". 없으면 빈 글."""
        return f": {'·'.join(self.route_pos)}" if self.route_pos else ""

    def _set_defect(self, cls: str, refused: bool):
        """불량 색 → 불량 카드의 표. 위치 단서와 같이 한 번 거부되면 그 명령 동안 거부로 둔다."""
        prev = self.defect or {}
        refused = refused or (prev.get("cls") == cls and prev.get("refused", False))
        if prev != {"cls": cls, "refused": refused}:
            self.defect = {"cls": cls, "refused": refused}
            self.bump("cards")

    def _plan_track(self, kind: str, node: str, data: dict):
        """intent plan 의 단계 목록·진행. 읽는 필드(ui/README.md "plan 이벤트"):
        data.steps = [{intent, count, expect?, label?}] → 목록(같은 목록이 다시 오면 진행은 그대로),
        data.results = [{status, done}] → 단계별 결과, data.step(1부터)·step_index(0부터) + status 또는 result{status, done}
        → 그 단계의 진행. 번호가 없으면 node 의 intent 로 다음 단계를 짐작한다."""
        steps = data.get("steps")
        if isinstance(steps, list) and steps and all(isinstance(s, dict) for s in steps):
            sig = [(s.get("intent"), s.get("count")) for s in steps]
            if not self.plan or self.plan["sig"] != sig:  # 에이전트는 세울 때·말할 때·끝 보고 때 같은 steps 를 다시 보낸다
                self.plan = {"text": self.running or "", "sig": sig, "steps": [dict(s) for s in steps], "cur": None, "end": None}
                for s in self.plan["steps"]:
                    s["state"] = STEP_STATE.get(s.get("status"), "wait")
                self.bump("layout")
        p = self.plan
        if not p:
            return
        if isinstance(data.get("results"), list):  # 끝 보고: 단계별 결과 한꺼번에
            for s, r in zip(p["steps"], data["results"]):
                if isinstance(r, dict):
                    s["state"] = STEP_STATE.get(r.get("status"), s["state"])
                    s["done"] = r.get("done", s.get("done"))
            self.bump("layout")
        if p["end"]:
            return
        if kind == "node" and node == "end":
            for s in p["steps"]:
                if s["state"] == "active":
                    s["state"] = STEP_STATE.get(data.get("status"), "done")
            p["end"] = data.get("status") or "done"
            self.bump("layout")
            return
        idx, explicit = None, True
        if isinstance(data.get("step"), int):
            idx = data["step"] - 1
        elif isinstance(data.get("step_index"), int):
            idx = data["step_index"]
        elif kind == "node" and data.get("intent") in STEP_KO:
            explicit = False
            start = p["cur"] or 0
            idx = next((i for i in range(start, len(p["steps"])) if p["steps"][i].get("intent") == data["intent"]
                        and p["steps"][i]["state"] in ("wait", "active")), None)
        if idx is None or not 0 <= idx < len(p["steps"]):
            return
        for s in p["steps"][:idx]:
            if s["state"] in ("wait", "active"):
                s["state"] = "done"
        cur = p["steps"][idx]
        res = data.get("result") if isinstance(data.get("result"), dict) else {}
        status = res.get("status", data.get("status")) if explicit else None
        cur["state"] = STEP_STATE.get(status) or "active"
        if res.get("done", data.get("done")) is not None:
            cur["done"] = res.get("done", data.get("done"))
        p["cur"] = idx
        self.bump("layout")

    def _enter(self, node: str, data: dict):
        if node not in self.nodes:
            self.nodes[node] = {"state": "idle", "visits": 0}
            self.order.append(node)
        if self.current and self.current != node and self.nodes[self.current]["state"] == "active":
            self.nodes[self.current]["state"] = "done"
            self.set_active(NODE_LAYER.get(self.current), False)
        if self.current != node or data.get("status") != "end":
            if self.current != node:
                self.nodes[node]["visits"] += 1
                self.set_active(NODE_LAYER.get(node), True)
            self.nodes[node]["state"] = "active"
        self.current = node
        bits = [str(data[k]) for k in ("policy", "intent") if data.get(k)]
        if data.get("attempt", 1) and data.get("attempt", 1) > 1:
            bits.append(f"{data['attempt']}번째 시도")
        if data.get("count") is not None:
            bits.append(f"{data['count']}개")
        self.current_note = " · ".join(bits) if bits else self.current_note if data.get("status") == "end" else ""
        self.bump("nodes")

    def _tool(self, node: str, data: dict, ts):
        name = data.get("name") or data.get("tool") or "?"
        result = data.get("result", data.get("output"))
        layer = TOOL_LAYER.get(name)
        if data.get("status") == "reuse":  # plan 단계 사이: 앞 단계가 3초 안에 찍은 감지를 다시 씀(카메라 안 엶)
            self.log("tool", node, f"↺ {name} 앞 결과 다시 씀({data.get('age_s', '-')}초 전)", ts)
            return
        if data.get("status") == "skip":  # go_home: 방금 동작 뒤 홈 복귀를 했고 그 뒤 안 움직임 → 팔 안 움직임
            self.log("tool", node, f"↷ {name} 건너뜀 · {data.get('why') or ''}", ts)
            return
        if data.get("status") == "unavailable":  # 위치 단서·불량 색·꺼내기를 할 수 없어 부르지 않음(로봇·카메라 0, 10-23)
            by = data.get("target_by")
            why = [_where(data.get("which"), data.get("next_to"), by) or data.get("which"),
                   f"{BY_KO[by]}{self._pos_note()}" if by in BY_FAIL else None,
                   DEFECT_KO.get(data.get("defect_cls"), data.get("defect_cls")), data.get("error")]
            self.log("refuse", node, " · ".join([f"{REFUSE_ICON} {name} 안 부름"] + [str(w) for w in why if w]), ts)
            return
        if isinstance(result, dict) and result.get("reason") == "refused" and data.get("mode") != "dry_run":
            self.refused_tool = name  # 에이전트가 곧바로 error(로봇의 거부 문구)를 보낸다(agent/graph.py act·coord)
        if result is None and data.get("status") != "end":
            self.set_active(layer, True)
            args = data.get("args") or {k: data[k] for k in ("mode", "color", "blocks", "which", "next_to", "open_gripper", "max_deg",
                                                             "colors", "max_blocks") if data.get(k) is not None}
            self.log("tool", node, f"▶ {name}({', '.join(f'{k}={v}' for k, v in args.items())})", ts)
            if name == "run_policy" and not self.departed and self.t_cmd:
                self.departed = True
                self.log("info", "", f"말 → 로봇 출발 {time.time() - self.t_cmd:.1f}초", ts)
            return
        self.set_active(layer, False)
        if isinstance(result, dict) and "zones" in result:
            self._detect(result)
            self.log("tool", node, f"◀ {name} 투입 {sum(result['zones'].get('input', {}).values())}개 · "
                                   f"손 {'있음' if (result.get('hand') or {}).get('present') else '없음'} · "
                                   f"{result.get('device', '')} {_ms(result.get('latency_ms'))}", ts)
        elif name == "run_policy" and isinstance(result, dict):
            ok = bool(result.get("ok"))
            self.dev_hit("행동", (result.get("duration_s") or 0) * 1000, data.get("device"))
            pname = result.get("name") or "?"
            self.pol_time[pname] = self.pol_time.get(pname, 0.0) + float(result.get("duration_s") or 0)
            if ok and result.get("name") == "handoff":
                self.handed += 1
                self.bump("cards")
            self.log("ok" if ok else "error", node, f"{'✓' if ok else '✗'} {result.get('name')} {result.get('duration_s', '-')}초", ts)
        elif name == "search_manual":
            hits = result if isinstance(result, list) else result.get("hits", []) if isinstance(result, dict) else []
            ids = ", ".join((h.get("cite") or f"{h.get('id') or h.get('section')} {h.get('title', '')}").strip()
                            for h in hits[:3] if isinstance(h, dict))
            self.log("tool", node, f"◀ {name} 근거: {ids or str(result)[:120]}", ts)
            self.dev_hit("검색", data.get("latency_ms"), data.get("device"))
        elif name == "sort_blocks" and isinstance(result, dict):
            r = result
            self.log("ok" if r.get("ok") else "refuse" if r.get("reason") == "refused" else "error", node,
                     f"◀ {name} {data.get('mode') or r.get('mode') or ''} · 블록 {r.get('blocks', len(r.get('plan') or []))} · "
                     f"{r.get('reason', '')}" + (f" · {_where(data.get('which'), data.get('next_to')) or data.get('which')}"
                                                 if data.get("which") or data.get("next_to") else "")
                     + (f" · {DEFECT_KO.get(data['defect_cls'], data['defect_cls'])}" if data.get("defect_cls") else "")
                     + (f" · {r['error']}" if r.get("error") else ""), ts)
        elif name == "unsort_blocks" and isinstance(result, dict):
            r = result
            bowls = "·".join(UNSORT_KO.get(c, str(c)) for c in data.get("colors") or []) or "전부"
            self.log("ok" if r.get("ok") else "refuse" if r.get("reason") == "refused" else "error", node,
                     f"◀ {name} {data.get('mode') or r.get('mode') or ''} · 그릇 {bowls} · 블록 {r.get('blocks', len(r.get('plan') or []))}"
                     + (f" · 최대 {data['max_blocks']}개" if data.get("max_blocks") else "") + f" · {r.get('reason', '')}"
                     + (f" · 막힘 {', '.join(data['gates'])}" if data.get("gates") else "")
                     + (f" · {r['error']}" if r.get("error") else ""), ts)
        elif name == "go_home" and isinstance(result, dict):
            r = result
            # too_far = 팔이 홈에서 멀어 로봇이 안 움직임(홈 명령은 한도 150° 로 한 번 더) → 주황, 끝내 안 되면 에이전트 error 가 빨강
            self.log("ok" if r.get("ok") else "refuse" if r.get("reason") == "refused" else
                     "warn" if r.get("reason") == "too_far" else "error", node,
                     f"◀ {name} {r.get('reason', '')}" + (" · 집게 열기" if data.get("open_gripper") else "")
                     + (f" · 한도 {data['max_deg']:g}°" if isinstance(data.get("max_deg"), (int, float)) else "")
                     + f" · {_ms(data.get('latency_ms'))}" + (f" · {r['error']}" if r.get("error") else ""), ts)
        elif name == "arrange_digit":
            self._arrange(data)
            r = result if isinstance(result, dict) else {}
            self.log("tool", node, f"◀ {name} 숫자 {data.get('digit')} · {data.get('mode')} · 블록 {r.get('blocks', '-')} · "
                                   f"{r.get('reason', '')}" + (f" · 막힘 {', '.join(data.get('gates') or [])}" if data.get("gates") else ""), ts)
        elif name == "intent_router" and isinstance(result, dict):
            self.dev_hit("판단", result.get("latency_ms"), data.get("device"))
            extra = ""
            if result.get("intent") == "none":
                extra = f" · 명령 아님({NONE_KO.get(result.get('none_kind'), result.get('none_kind'))})" + (
                    f" · 막기 전 {result['none_from']}" if result.get("none_from") else "")
            if result.get("llm_down"):
                extra += " · ⚠ LLM 응답 없음"
            tg = result.get("target") if isinstance(result.get("target"), dict) else {}
            self.route_pos = [str(p) for p in tg.get("pos") or []]
            w, by = tg.get("which") or result.get("which"), tg.get("by")
            where = _where(w, tg.get("next_to") or result.get("next_to"), by)
            if where:  # 10-23 13:30: "위치 맨 위(규칙)" · "분홍 옆(LLM)" · "위치 못 읽음(LLM 도 못 읽음: 사이)"
                extra += (f" · {'위치 ' if w in WHICH_KO and w != 'mixed' else ''}{where}"
                          + (f"({BY_KO.get(by, by)}{self._pos_note() if by in BY_FAIL else ''})" if by else ""))
            self.log("tool", node, "◀ 의도 " + ", ".join(f"{k}={result[k]}" for k in ("intent", "count", "method", "error")
                                                       if result.get(k) is not None) + extra, ts)
        else:
            self.log("tool", node, f"◀ {name} {'' if result is None else str(result)[:160]}", ts)

    def _detect(self, res: dict):
        self.zones = res.get("zones") or {}
        objs = res.get("objects")
        self.objects = [o for o in objs if isinstance(o, dict)] if isinstance(objs, list) else None
        self.hand = bool((res.get("hand") or {}).get("present"))
        self.det_ts = _short_ts(res.get("ts"))
        self.bump("layout")
        img = _resolve(res.get("image_path"))
        if img:
            self.snapshot = img
            self.first_snap = self.first_snap or (img, _short_ts(res.get("ts")))
            self.bump("snap")
        hand = (res.get("hand") or {}).get("present")
        n_in = sum((self.zones.get("input") or {}).values())
        self.snapcap = (f"{_short_ts(res.get('ts'))} · 투입 구역 {n_in}개 · 손 자리 {'손 있음' if hand else '비어 있음'}"
                        f" · {res.get('device', '?')} {_ms(res.get('latency_ms'))}")
        self.dev_hit("보기", res.get("latency_ms"), res.get("device"))
        self.bump("snap", "cards")

    def _facts_report(self, data: dict) -> dict:
        """에이전트 report 이벤트의 facts(agent/report.py) + 화면이 모은 사진·정책 시간·장치 시간 → render_report 데이터."""
        f = data["facts"]
        exp, det, wrong, left = (f.get(k) or {} for k in ("expected", "detected", "wrong", "input_left"))

        def mixed(bowl):
            return " · ".join(f"{CLASS_KO.get(k, k)} {n}개 섞임" for k, n in (wrong.get(bowl) or {}).items())

        counts = [
            {"key": "defect", "label": "불량 그릇", "record": exp.get("bowl_defect"), "camera": det.get("bowl_defect"),
             "detail": " · ".join(x for x in (f"분홍 {f.get('defect_red', 0)} · 초록 {f.get('defect_blue', 0)}",
                                              mixed("bowl_defect")) if x)},
            {"key": "red", "label": "분홍 그릇", "record": exp.get("bowl_red"), "camera": det.get("bowl_red"),
             "detail": mixed("bowl_red") or ("손에 건넨 것 빼고" if f.get("handed") else "")},
            {"key": "blue", "label": "초록 그릇", "record": exp.get("bowl_blue"), "camera": det.get("bowl_blue"),
             "detail": mixed("bowl_blue")},
            {"key": "hand", "label": "손 전달", "record": f.get("handed"), "camera": None, "detail": "카메라로 셀 수 없음(기록만)"},
            {"key": "input", "label": "투입 구역 남음", "record": None, "camera": sum(left.values()),
             "detail": ", ".join(f"{CLASS_KO.get(k, k)} {n}개" for k, n in left.items())},
        ]
        policies = [{"name": p, "attempts": v.get("attempts", 0), "success": v.get("ok", 0),
                     "total_s": round(self.pol_time.get(p, 0.0), 1)} for p, v in (f.get("policies") or {}).items()]
        devices = [{"layer": layer, "model": d["model"], "device": d["device"], "calls": d["calls"],
                    "avg_ms": d["sum"] / d["calls"], "max_ms": d["max"]} for layer, d in self.dev.items() if d["calls"]]
        commands = [{"ts": c["ts"], "text": c["text"], "intent": c.get("intent"), "duration_s": c.get("duration_s"),
                     "result": c.get("result")} for c in self.cmds
                    # 지금 도는 "결과 알려 줘"와 그릇에서 꺼내기(리허설 초기화 — 에이전트도 작업 기록·보고에 안 넣음)는 빼고
                    if c.get("duration_s") is not None and c.get("intent") not in ("report", "unsort")]
        checks = [f"{c.get('text')} — {c.get('why')}" + (f": {c['say']}" if c.get("say") else "") for c in f.get("checks") or []]
        ng = netguard.status()
        now = datetime.now()
        started = self.cmds[0]["started"] if self.cmds else now
        return {
            "job_id": self.thread_id, "started": f"{started:%Y-%m-%d %H:%M:%S}", "ended": f"{now:%H:%M:%S}",
            "duration_s": f.get("total_s"), "summary": data.get("text"),
            "offline": ng["mode"] != "off" and not ng["blocked"],
            "net_note": f"외부 접속 시도 {ng['blocked']}건(차단)" if ng["blocked"] else "외부 차단 꺼짐",
            "before_image": self.first_snap[0] if self.first_snap else None,
            "before_ts": self.first_snap[1] if self.first_snap else None,
            "after_image": _resolve(data.get("image_path")) or self.snapshot, "after_ts": f"{now:%H:%M:%S}",
            "counts": counts, "policies": policies, "commands": commands, "devices": devices, "human_checks": checks,
        }

    def _report(self, data: dict):
        html_text, src = None, None
        self.report_alt = _resolve(data.get("html_path"))
        if isinstance(data.get("facts"), dict) and not isinstance(data.get("data"), dict):
            data = dict(data, data=self._facts_report(data))
        if isinstance(data.get("data"), dict):  # 데이터가 있으면 UI 가 잰 듣기·말하기 줄을 더해 다시 그린다
            rep = dict(data["data"])
            have = {d.get("layer") for d in rep.get("devices") or []}
            extra = []
            for layer in ("듣기", "말하기", "흐름"):
                d = self.dev[layer]
                if layer not in have and d["calls"]:
                    extra.append({"layer": layer, "model": d["model"], "device": d["device"], "calls": d["calls"],
                                  "avg_ms": d["sum"] / d["calls"], "max_ms": d["max"]})
            rep["devices"] = (rep.get("devices") or []) + extra
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            src = REPORT_DIR / f"report_{datetime.now():%Y%m%d_%H%M%S}.html"
            html_text = render_report(rep, str(src))
        elif data.get("html"):
            html_text = data["html"]
        elif _resolve(data.get("html_path")):
            src = Path(_resolve(data["html_path"]))
            html_text = src.read_text(encoding="utf-8")
        if html_text:
            self.report_html, self.report_link, self.report_open = html_text, str(src) if src else None, True
            self.bump("report")


BOARD = Board()
STOP = threading.Event()
VIEW = None  # 보기 전용 화면(ui/view.py) — UI_VIEW=1 로 켰을 때만 그 모듈(아니면 불러오지도 스레드를 띄우지도 않는다)
_status_cache = {"t": 0.0}


def _status() -> dict:
    """OVMS·와이파이 상태(5초 캐시)."""
    if time.time() - _status_cache["t"] < 5:
        return _status_cache
    s = {"t": time.time()}
    try:
        s["models"] = [m["id"] for m in requests.get(f"{BASE_URL}/models", timeout=0.8).json().get("data", [])]
    except Exception:
        s["models"] = None
    try:
        radio = subprocess.run(["nmcli", "-t", "-f", "WIFI", "radio"], capture_output=True, text=True, timeout=2).stdout.strip()
        conn = subprocess.run(["nmcli", "-t", "-f", "TYPE,STATE", "dev"], capture_output=True, text=True, timeout=2).stdout
        s["wifi"] = "off" if radio == "disabled" else "connected" if "wifi:connected" in conn else "on"
    except Exception:
        s["wifi"] = None
    _status_cache.clear()
    _status_cache.update(s)
    return s


_net_seen = {"n": 0}


def _log_net(ng: dict):
    """새로 막은 외부 접속을 로그에 남긴다(머리줄을 그릴 때마다 확인)."""
    new = ng["blocked"] - _net_seen["n"]
    if new <= 0:
        return
    _net_seen["n"] = ng["blocked"]
    for ts, target, event in ng["recent"][-min(new, len(ng["recent"])):]:
        BOARD.log("warn", "net", f"외부 접속 {'차단' if ng['mode'] == 'block' else '감지'}: {target} ({event})", ts)


# ── 그리기 ────────────────────────────────────────────────────────────────
def _pill(text, cls=""):
    return f'<span class="pill {cls}">{html.escape(text)}</span>'


def html_header() -> str:
    s = _status()
    pills = []
    if AGENT_NAME == "fake":
        pills.append(_pill("가짜 스트림", "warn"))
    else:
        pills.append(_pill("에이전트", "ok"))
        b = BOARD.backend or {}
        for key, name in (("vision", "비전"), ("robot", "로봇")):
            if b.get(key):  # agent/backends.py: real | mock | dry_run(BALL_ROBOT=dry)
                pills.append(_pill(f"{name} {({'mock': '가짜', 'real': '실물', 'dry_run': '시험(안 움직임)'}).get(b[key], b[key])}",
                                   "ok" if b[key] == "real" else "warn"))
    wifi = s.get("wifi")
    pills.append(_pill({"off": "와이파이 꺼짐", "connected": "와이파이 연결됨", "on": "와이파이 켜짐"}.get(wifi, "와이파이 ?"),
                       "ok" if wifi == "off" else "warn"))
    ng = netguard.status()
    if ng["mode"] == "off":
        pills.append(_pill("외부 차단 꺼짐", "warn"))
    else:
        pills.append(_pill(f"외부 접속 {'차단' if ng['mode'] == 'block' else '감지'} {ng['blocked']}건" if ng["blocked"]
                           else "외부 접속 0건", "warn" if ng["blocked"] else "ok"))
        _log_net(ng)
    models = s.get("models")
    if models is None:
        pills.append(_pill("OVMS 꺼짐", "bad"))
    else:
        pills.append(_pill(f"OVMS {len(models)}개 모델", "ok"))
        pills.append(_pill("STT 준비" if stt.MODEL in models else "STT 없음", "ok" if stt.MODEL in models else "warn"))
    src = micdev.default_source()  # 젠북 기본 입력(10초 캐시). 이어폰 단자 마이크면 녹음이 완전 무음일 수 있다(10-22 밤)
    if src:
        pills.append(_pill(f"마이크 {'음소거' if src['muted'] else src.get('nick') or src.get('name')}",
                           "bad" if src["muted"] else "warn" if micdev.is_jack(src) else "ok"))
    if TTS_OUT == "off":
        pills.append(_pill("음성 출력 끔"))
    else:
        st = tts.status()
        pills.append(_pill({"ready": "TTS 준비", "missing": "TTS 없음 · 글자만", "loading": "TTS 로딩"}[st]
                           + {"silent": " · 무음", "timed": " · 무음(시간 맞춤)"}.get(TTS_OUT, ""),
                           {"ready": "ok", "missing": "warn", "loading": ""}[st]))
    if VIEW:  # 보기 전용 화면 주소(교육장 사람들이 폰으로 친다)·지금 보는 브라우저 수
        pills.append(_pill(f"보기 전용 {VIEW.URL} · {VIEW.viewers()}명", "view"))
    run = ""
    if BOARD.running:
        el = time.time() - (BOARD.t_cmd or time.time())
        run = f'<span class="run"><i class="dot"></i>실행 중 · “{html.escape(BOARD.running)}” · {el:.0f}초</span>'
    return (f'<div class="hdr"><div class="ttl"><b>말로 시키는 스펀지 블록 선별 로봇</b>'
            f'<small>OpenVINO 온디바이스 · NPU · GPU · CPU</small></div>{"".join(pills)}{run}</div>')


def html_nodes() -> str:
    chips = []
    for i, n in enumerate(BOARD.order):
        st = BOARD.nodes[n]
        visits = f'<em>×{st["visits"]}</em>' if st["visits"] > 1 else ""
        chips.append(f'<div class="node {st["state"]}">{html.escape(NODE_LABELS.get(n, n))}{visits}<small>{html.escape(n)}</small></div>')
        if i < len(BOARD.order) - 1:
            chips.append('<span class="arrow">›</span>')
    cur = BOARD.current
    now = (f'지금: <b>{html.escape(NODE_LABELS.get(cur, cur))}</b> {html.escape(BOARD.current_note)}' if cur and BOARD.running
           else "대기 중" if not cur else f'마지막: {html.escape(NODE_LABELS.get(cur, cur))}')
    return f'<div class="ptitle">LangGraph 노드 진행</div><div class="flow">{"".join(chips)}</div><div class="now">{now}</div>'


def html_cards() -> str:
    z = BOARD.zones or {}
    tot = lambda k: sum((z.get(k) or {}).values()) if k in z else None
    bd = z.get("bowl_defect") or {}
    vals = {
        "defect": (tot("bowl_defect"), "불량", f"분홍 {bd.get('red_defect', 0)} · 초록 {bd.get('blue_defect', 0)}" if bd else "카메라 기준"),
        "red": (tot("bowl_red"), "분홍", "카메라 기준"),
        "blue": (tot("bowl_blue"), "초록", "카메라 기준"),
        "hand": (BOARD.handed, "손 전달", "기록 기준"),
    }
    dc, tg = BOARD.defect, BOARD.target
    tags: dict = {}
    # 위치·관계(10-23 13:30): 옮길 블록이 갈 그릇 카드 오른쪽 위에 "맨 위"·"분홍 옆"(카드 색), 거부되면 점선 주황 "⊘ …"
    if tg and tg.get("intent") in vals:
        where = _where(tg["which"], tg["next_to"], tg["by"])
        tags[tg["intent"]] = [f'<i class="pos{" refuse" if tg["refused"] else ""}">'
                              f'{html.escape((REFUSE_ICON + " " if tg["refused"] else "") + where)}</i>']
    # 불량 색 명령(10-23): 불량 카드 오른쪽 위에 "초록 불량"(그 색 칠), 거부되면 점선 주황 "⊘ 초록 불량". 위치가 있으면 그 아래
    if dc:
        tags.setdefault("defect", []).append(f'<i class="dcls {"refuse" if dc["refused"] else dc["cls"]}">'
                                             f'{html.escape((REFUSE_ICON + " " if dc["refused"] else "") + DEFECT_KO[dc["cls"]])}</i>')
    out = []
    for key, (v, label, sub) in vals.items():
        bump = " bump" if BOARD.prev_cards.get(key) not in (None, v) else ""
        tag = f'<span class="tags">{"".join(tags[key])}</span>' if key in tags else ""
        out.append(f'<div class="card {key}{bump}">{tag}<span>{label}</span>'
                   f'<b>{"-" if v is None else v}</b><em>{html.escape(sub)}</em></div>')
        BOARD.prev_cards[key] = v
    return f'<div class="cards">{"".join(out)}</div>'


def html_devices() -> str:
    rows = []
    for layer, d in BOARD.dev.items():
        active = d["active"] > 0 or (layer == "흐름" and BOARD.running)
        avg = d["sum"] / d["calls"] if d["calls"] else None
        rows.append(f'<tr class="{"active" if active else ""}"><td>{"<i class=dot></i>" if active else ""}{layer}</td>'
                    f'<td>{html.escape(d["model"])}</td><td><span class="dev {d["device"]}">{d["device"]}</span></td>'
                    f'<td class="num">{_ms(d["last"])}</td><td class="num">{_ms(avg)}</td><td class="num">{d["calls"]}</td></tr>')
    return ('<div class="ptitle">디바이스 — 어디서 무엇이 도나</div><table class="devtbl"><thead><tr><th>계층</th><th>모델</th>'
            '<th>장치</th><th class="num">최근</th><th class="num">평균</th><th class="num">횟수</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def html_log() -> str:
    lines = [f'<div class="l t-{k}"><span class="ts">{ts}</span><span class="k">{html.escape(n or k)}</span>{html.escape(t)}</div>'
             for ts, k, n, t in reversed(BOARD.logs)]
    return f'<div class="ptitle">로그</div><div class="log">{"".join(lines) or "<div class=l>아직 기록이 없습니다</div>"}</div>'


def html_report() -> str:
    if not BOARD.report_html:
        return '<div class="empty">"결과 알려 줘" 라고 하면 결과 보고서가 여기에 뜹니다.</div>'
    link = (f'<a class="open" href="/gradio_api/file={html.escape(BOARD.report_link)}" target="_blank">새 창에서 열기 · 인쇄</a>'
            if BOARD.report_link else "")
    if BOARD.report_alt and BOARD.report_alt != BOARD.report_link:
        link += (f' <a class="open" href="/gradio_api/file={html.escape(BOARD.report_alt)}" target="_blank">'
                 f'에이전트 보고서(기준서 근거 포함)</a>')
    return f'{link}<iframe class="report-frame" srcdoc="{html.escape(BOARD.report_html, quote=True)}"></iframe>'


def html_snapcap() -> str:
    return f'<div class="snapcap">{html.escape(BOARD.snapcap)}</div>'


# ── 보기 전용 화면(ui/view.py) 조각 ──
def _view_chat() -> str:
    """대화는 글자만(말풍선 안 HTML 은 벗겨 다시 이스케이프 — 받아쓴 글이 태그로 해석되지 않게)."""
    msgs = [f'<div class="msg {m["role"]}">{html.escape(html.unescape(re.sub(r"<[^>]+>", " ", str(m["content"]))).strip())}</div>'
            for m in BOARD.chat[-30:]]
    return f'<div class="ptitle">대화</div><div class="chatv">{"".join(msgs) or "<div class=empty>아직 대화가 없습니다</div>"}</div>'


def _view_cmds() -> str:
    rows = [f'<li class="{"" if c.get("result") else "run"}"><span class="ts">{c.get("ts", "")}</span><b>{html.escape(c["text"])}</b>'
            f'<em>{html.escape(c.get("result") or "실행 중")}</em></li>' for c in reversed(BOARD.cmds[-12:])]
    return f'<div class="ptitle">명령 결과</div><ol class="cmds">{"".join(rows) or "<li>아직 명령이 없습니다</li>"}</ol>'


def _view_report() -> str:
    if not BOARD.report_html:
        return '<div class="empty">"결과 알려 줘" 라고 하면 결과 보고서가 여기에 뜹니다.</div>'
    return f'<iframe class="report-frame" sandbox srcdoc="{html.escape(BOARD.report_html, quote=True)}"></iframe>'  # 링크·스크립트 없음


def view_sections() -> tuple[dict, str | None]:
    """보기 전용 화면이 부른다(1초에 한 번). 대시보드와 같은 html_* 조각 + 위 카메라 사진 경로.
    BOARD 는 잠금 안에서 복사만 하고 바로 놓는다. 그리기는 잠금 밖에서 복사본으로 — 이 모듈의 함수들을 BOARD 만 복사본으로
    바꾼 이름표(globals)에 다시 묶어 부른다. 그래서 카드 강조(prev_cards)·로그 등 대시보드 상태를 하나도 건드리지 않는다."""
    with BOARD.lock:
        state = copy.deepcopy({k: v for k, v in vars(BOARD).items() if k != "lock"})
    board = Board.__new__(Board)
    board.__dict__.update(state, lock=threading.RLock())
    g = dict(globals(), BOARD=board, _log_net=lambda ng: None)  # 막은 외부 접속 로그는 대시보드가 남긴다
    for name, f in list(g.items()):
        if isinstance(f, types.FunctionType) and f.__globals__ is globals():
            g[name] = types.FunctionType(f.__code__, g, name, f.__defaults__, f.__closure__)
    out = {k: g[fn]() for k, fn in (("hdr", "html_header"), ("snapcap", "html_snapcap"), ("layout", "html_layout"),
                                    ("cards", "html_cards"), ("nodes", "html_nodes"), ("cmds", "_view_cmds"), ("chat", "_view_chat"),
                                    ("log", "html_log"), ("dev", "html_devices"), ("report", "_view_report"))}
    return out, _resolve(board.snapshot)


# ── 현재 배치(작업대 그림) ──
# 위 카메라에서 본 자리(투입 구역 왼쪽, 그릇 셋·손 자리 오른쪽)를 그린 틀. 블록은 objects 의 zone·cell(3×3 칸)에 점으로 찍는다.
# (x, y, 폭, 높이, 이름) — viewBox 0 0 400 262
LAYOUT = {"input": (6, 24, 196, 214, "투입 구역"), "bowl_red": (216, 24, 86, 86, "분홍 그릇"),
          "bowl_blue": (310, 24, 86, 86, "초록 그릇"), "bowl_defect": (216, 152, 86, 86, "불량 그릇"),
          "hand_spot": (310, 152, 86, 86, "손 자리")}
CELL_ROW, CELL_COL = {"위": 0, "아래": 2}, {"왼쪽": 0, "오른쪽": 2}


def _cell_rc(cell) -> tuple[int, int]:
    """'왼쪽 위'·'위 가운데'·'가운데' … → (행, 열). 순서가 바뀌어도, 모르는 이름이면 가운데."""
    words = str(cell or "가운데").split()
    r = next((CELL_ROW[w] for w in words if w in CELL_ROW), 1)
    c = next((CELL_COL[w] for w in words if w in CELL_COL), 1)
    return r, c


def _block_svg(cls: str, cx: float, cy: float, size: float) -> str:
    color = "pink" if cls.startswith("red") else "green"
    h = size / 2
    out = f'<rect class="blk {color}" x="{cx - h:.1f}" y="{cy - h:.1f}" width="{size:.1f}" height="{size:.1f}" rx="2.5"/>'
    if cls.endswith("defect"):
        k = h * 0.62
        out += (f'<path class="xmark" d="M{cx - k:.1f} {cy - k:.1f}L{cx + k:.1f} {cy + k:.1f}'
                f'M{cx - k:.1f} {cy + k:.1f}L{cx + k:.1f} {cy - k:.1f}"/>')
    return out


def _zone_text(counts: dict) -> str:
    red = counts.get("red_ok", 0)
    blue = counts.get("blue_ok", 0)
    bad = counts.get("red_defect", 0) + counts.get("blue_defect", 0)
    return " · ".join(t for t, n in ((f"분홍 {red}", red), (f"초록 {blue}", blue), (f"불량 {bad}", bad)) if n) or "비어 있음"


def _plan_html() -> str:
    p = BOARD.plan
    if not p:
        return ""
    icon = {"done": "✓", "active": "▶", "wait": "", "skip": "–", "stop": "■", "fail": "✗", "refuse": REFUSE_ICON}
    items = []
    for i, s in enumerate(p["steps"], 1):
        what, verb = STEP_KO.get(s.get("intent"), (s.get("intent") or "?", ""))
        label = s.get("label") or s.get("text") or (f"{what} {s['count']}개 {verb}" if s.get("count") else f"{what} {verb}")
        if s.get("auto"):
            label += " (자동으로 넣은 단계)"
        expect = s.get("expect") or s.get("count")
        got = (f'<em>{s["done"]}{f"/{expect}" if expect else ""}개</em>' if s.get("done") is not None
               else f'<em>{expect}개 예정</em>' if expect else "")
        items.append(f'<li class="st-{s["state"]}"><i>{icon.get(s["state"], "")}</i><span>{i}. {html.escape(str(label).strip())}</span>{got}</li>')
    head = "계획" + (f" — {STATUS_KO.get(p['end'], p['end'])}" if p["end"] else " — 진행 중")
    return (f'<div class="plan"><div class="ptitle">{head} · “{html.escape(p["text"])}”</div>'
            f'<ol>{"".join(items)}</ol></div>')


def html_arrange() -> str:
    """숫자 배치 계획: 왼쪽 = 작업대 위 블록(from_mm, 번호 = 놓는 순서), 오른쪽 = 숫자 칸(row 0 = 위), 화살표 = 어디서 어디로."""
    a = BOARD.arrange
    lay = a.get("layout") or {}
    cells = [c for c in lay.get("cells") or [] if isinstance(c, dict)]
    rows, cols = int(lay.get("rows") or 5), int(lay.get("cols") or 3)
    pattern = lay.get("pattern") or []
    parts = ['<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">'
             '<path d="M0 0L8 4L0 8z" class="ah"/></marker></defs>']
    bx, by, bw, bh = 6, 24, 190, 214
    cs = min(40.0, bh / rows, 140 / cols)
    gx, gy = 236 + (150 - cols * cs) / 2, by + (bh - rows * cs) / 2
    parts.append(f'<rect class="zone" x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="6"/>'
                 f'<text class="zlabel" x="{bx + 2}" y="{by - 7}">작업대 위 블록</text>'
                 f'<text class="zlabel" x="{gx}" y="{by - 7}">숫자 {html.escape(str(a.get("digit") or "?"))}</text>')
    for r in range(rows):
        for c in range(cols):
            if r < len(pattern) and c < len(pattern[r]) and pattern[r][c] == "#":
                parts.append(f'<rect class="slot" x="{gx + c * cs + 2:.1f}" y="{gy + r * cs + 2:.1f}" width="{cs - 4:.1f}" height="{cs - 4:.1f}" rx="4"/>')
    src = [c["from_mm"] for c in cells if isinstance(c.get("from_mm"), (list, tuple)) and len(c["from_mm"]) == 2]
    if src:
        xs, ys = [p[0] for p in src], [p[1] for p in src]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        k = min((bw - 40) / span, (bh - 40) / span)
        ox = bx + (bw - (max(xs) - min(xs)) * k) / 2
        oy = by + (bh - (max(ys) - min(ys)) * k) / 2
    placed = a.get("placed")
    for c in cells:
        color = "pink" if "분홍" in str(c.get("color")) or c.get("color") == "red" else "green" if "초록" in str(c.get("color")) or c.get("color") == "blue" else "gray"
        n = c.get("order")
        tx, ty = gx + (c.get("col", 0) + 0.5) * cs, gy + (c.get("row", 0) + 0.5) * cs
        done = a.get("mode") == "real" and placed is not None and n is not None and n <= placed
        h = cs * 0.36
        parts.append(f'<rect class="blk {color}{" ghost" if a.get("mode") != "real" or not done else ""}" x="{tx - h:.1f}" y="{ty - h:.1f}" '
                     f'width="{2 * h:.1f}" height="{2 * h:.1f}" rx="3"/><text class="ord" x="{tx:.1f}" y="{ty + 4.5:.1f}">{n}</text>')
        fm = c.get("from_mm")
        if src and isinstance(fm, (list, tuple)) and len(fm) == 2:
            sx, sy = ox + (fm[0] - min(xs)) * k, oy + (fm[1] - min(ys)) * k
            parts.append(f'<line class="arw {color}" x1="{sx + 10:.1f}" y1="{sy:.1f}" x2="{tx - h - 3:.1f}" y2="{ty:.1f}" marker-end="url(#ah)"/>'
                         f'<rect class="blk {color}" x="{sx - 9:.1f}" y="{sy - 9:.1f}" width="18" height="18" rx="3"/>'
                         f'<text class="ord" x="{sx:.1f}" y="{sy + 4:.1f}">{n}</text>')
    svg = f'<svg class="bench" viewBox="0 0 400 252" role="img" aria-label="숫자 배치 계획">{"".join(parts)}</svg>'
    t = a.get("time_s")
    bits = [f"블록 {a['blocks']}개" if a.get("blocks") is not None else "",
            f"약 {int(t // 60)}분 {int(t % 60)}초" if isinstance(t, (int, float)) else "",
            f"경고 {len(a['warnings'])}건" if a.get("warnings") else ""]
    head = {"real": "숫자 놓기", "dry_run": "숫자 배치 계획"}.get(a.get("mode"), "숫자 배치")
    status = STATUS_KO.get(a.get("status"), a.get("status") or "")
    cap = " · ".join(x for x in bits if x)
    extra = ""
    if a.get("ok") is False:  # dry-run 이 안 됨(블록 부족·안 닿음·공간 위반) → 에이전트가 거절
        extra += f'<div class="arr-note">할 수 없는 계획 — {html.escape((a.get("failures") or [a.get("reason") or "이유 없음"])[0])}</div>'
    if a.get("plan_only"):
        extra += f'<div class="arr-note">계획만 보여 주는 중 — {html.escape(", ".join(a["gates"]) or "실행 안 함")}</div>'
    if a.get("mode") == "real" and placed is not None:
        extra += f'<div class="arr-note">놓은 블록 {placed}개(진한 칸)</div>'
    for w in a.get("warnings")[:2]:
        extra += f'<div class="arr-warn">⚠ {html.escape(str(w))}</div>'
    if a.get("waiting"):
        extra += '<div class="arr-ask">할까요? <b>응</b> / <b>아니</b> — 말하거나 아래 버튼</div>'
    return (f'<div class="ptitle">{head} — “{html.escape(a.get("text") or "")}”{f" · {status}" if status else ""}</div>{svg}'
            f'<div class="bench-cap">{html.escape(cap)} · 번호 = 놓는 순서, 왼쪽은 로봇 좌표(mm)로 본 블록 자리</div>{extra}')


def html_layout() -> str:
    if BOARD.arrange:
        return html_arrange()
    z, objs = BOARD.zones, BOARD.objects
    total = sum(sum(v.values()) for v in (z or {}).values())
    placed = objs is not None and (objs or total == 0)  # objects 가 비었는데 개수는 있으면 옛 형식
    parts = []
    for key, (x, y, w, h, name) in LAYOUT.items():
        counts = (z or {}).get(key) or {}
        n = sum(counts.values())
        label = name if z is None or key == "hand_spot" else f"{name} {n}개"
        if key == "input":
            grid = "".join(f'<line class="grid" x1="{x + w * k / 3:.1f}" y1="{y}" x2="{x + w * k / 3:.1f}" y2="{y + h}"/>'
                           f'<line class="grid" x1="{x}" y1="{y + h * k / 3:.1f}" x2="{x + w}" y2="{y + h * k / 3:.1f}"/>'
                           for k in (1, 2))
            parts.append(f'<rect class="zone" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>{grid}')
        elif key == "hand_spot":
            parts.append(f'<rect class="zone hand{" on" if BOARD.hand else ""}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>'
                         f'<text class="hand-t" x="{x + w / 2}" y="{y + h / 2 + 5}">{"손 있음" if BOARD.hand else "손 없음"}</text>')
        else:
            parts.append(f'<circle class="zone bowl {key}" cx="{x + w / 2}" cy="{y + h / 2}" r="{w / 2 - 1}"/>')
        parts.append(f'<text class="zlabel" x="{x + 2}" y="{y - 7}">{html.escape(label)}</text>')
        if z is not None and not placed and key != "hand_spot":  # 옛 형식: 개수만
            parts.append(f'<text class="cnt" x="{x + w / 2}" y="{y + h / 2 + 2}">{n}</text>'
                         f'<text class="cnt-s" x="{x + w / 2}" y="{y + h / 2 + 20}">{html.escape(_zone_text(counts))}</text>')
    outside = 0
    tg = BOARD.target
    ref = NEXT_CLS.get(tg["next_to"], ()) if tg else ()  # "분홍 옆" → 투입 구역의 분홍 블록에 테두리(로봇은 투입 구역 안 기준만 본다)
    if placed:
        cells: dict = {}
        for o in objs:
            if o.get("zone") not in LAYOUT:
                outside += 1
                continue
            cells.setdefault((o["zone"], *_cell_rc(o.get("cell"))), []).append(str(o.get("cls") or ""))
        for (zone, r, c), clss in cells.items():
            x, y, w, h, _ = LAYOUT[zone]
            cw, ch = w / 3, h / 3
            cols = 1 if len(clss) == 1 else 2 if len(clss) <= 4 else 3
            rows = -(-len(clss) // cols)
            size = min(17.0, (min(cw / cols, ch / rows)) * 0.78)
            for i, cls in enumerate(sorted(clss)):
                cx = x + cw * c + cw * ((i % cols) + 0.5) / cols
                cy = y + ch * r + ch * ((i // cols) + 0.5) / rows
                parts.append(_block_svg(cls, cx, cy, size))
                if zone == "input" and cls in ref:
                    parts.append(f'<rect class="ref{" refuse" if tg["refused"] else ""}" x="{cx - size * 0.85:.1f}" '
                                 f'y="{cy - size * 0.85:.1f}" width="{size * 1.7:.1f}" height="{size * 1.7:.1f}" rx="5"/>')
    if tg:  # 위치·관계: 투입 구역 안 그 쪽에 짧은 표. 고르는 기준은 같은 색 후보의 위 카메라 u(가로)·v(세로) 순서·로봇까지 거리
        x, y, wd, h, _ = LAYOUT["input"]  # 그림 아래쪽(v 큰 쪽)이 로봇 쪽 — 맨 아래·가까운은 아래에, 맨 위·먼은 위에
        which = tg["which"]
        word = _where(which, tg["next_to"], tg["by"])
        label = (f"{REFUSE_ICON} {word}" if tg["refused"] else f"↔ {word}" if tg["next_to"] else
                 {"left": "◀ {}", "right": "{} ▶", "top": "▲ {}", "far": "▲ {}"}.get(which, "▼ {}").format(word))
        tw = 13 * len(label) + 6  # 글자 폭 대략(13px)
        tx = {"left": x + 6, "right": x + wd - 6 - tw}.get(which, x + (wd - tw) / 2)
        # 표는 구역 테두리에 걸쳐 놓는다(위: 구역 이름 줄 바로 아래, 아래: 범례 바로 위) — 맨 위·맨 아래 줄 블록을 가리지 않게
        ty = y + h - 12 if which in ("bottom", "near") else y - 5
        parts.append(f'<g class="which{" refuse" if tg["refused"] else ""}"><rect x="{tx:.1f}" y="{ty}" width="{tw}" height="20" rx="10"/>'
                     f'<text x="{tx + tw / 2:.1f}" y="{ty + 14.5}">{html.escape(label)}</text></g>')
    if z is None:
        status = "아직 세지 않았습니다 — 명령을 내리거나 “지금 작업대 어때?”"
    elif placed:
        status = f"카메라 {BOARD.det_ts} · 블록 {len(objs)}개를 칸 위치로" + (f" · 구역 밖 {outside}개" if outside else "")
    else:
        status = f"카메라 {BOARD.det_ts} · 개수만(이 결과에는 위치 정보가 없음)"
    if tg:
        which, by = tg["which"], tg["by"]
        status += (" · " + ("위치 " if which in WHICH_KO and which != "mixed" else "") + _where(which, tg["next_to"], by)
                   + ("(그림 아래쪽이 로봇 쪽)" if which in ("top", "bottom", "near", "far") else "")
                   + ("(테두리 = 기준 블록)" if ref and placed else "")
                   + (f"({BY_KO[by]}{BOARD._pos_note()})" if by in BY_FAIL else "")
                   + (" — 거부(이 위치로 고르지 못함)" if tg["refused"] else ""))
    legend = (f'{_block_svg("red_ok", 10, 254, 12)}<text class="lg" x="21" y="259">분홍</text>'
              f'{_block_svg("blue_ok", 70, 254, 12)}<text class="lg" x="81" y="259">초록</text>'
              f'{_block_svg("red_defect", 130, 254, 12)}{_block_svg("blue_defect", 145, 254, 12)}'
              f'<text class="lg" x="156" y="259">불량(X 표시)</text>')
    svg = f'<svg class="bench" viewBox="0 0 400 264" role="img" aria-label="작업대 현재 배치">{"".join(parts)}{legend}</svg>'
    return (f'<div class="ptitle">현재 배치 — 위에서 본 작업대</div>{svg}'
            f'<div class="bench-cap">{html.escape(status)}</div>{_plan_html()}')


SECTIONS = ["hdr", "snap", "snapcap", "layout", "confirm", "nodes", "cards", "dev", "chat", "log", "report", "acc", "audio"]
_VER_KEY = {"snapcap": "snap", "acc": "report", "confirm": "layout"}
_audio_sent = {"ver": 1}  # 새 음성은 어느 핸들러가 그리든 한 번만 재생
_timer_sent: dict = {}  # 타이머(2초)는 바뀐 구역만 다시 보낸다


def render(sent: dict | None = None) -> list:
    """sent 가 있으면 지난번과 같은 구역은 gr.skip() 으로(깜빡임·전송량 줄이기)."""
    with BOARD.lock:
        makers = {
            "hdr": html_header, "snap": lambda: BOARD.snapshot, "snapcap": html_snapcap, "layout": html_layout,
            "confirm": lambda: gr.Row(visible=bool(BOARD.arrange and BOARD.arrange.get("waiting"))), "nodes": html_nodes,
            "cards": html_cards, "dev": html_devices, "chat": lambda: list(BOARD.chat), "log": html_log,
            "report": html_report, "acc": lambda: gr.Accordion(open=BOARD.report_open), "audio": lambda: BOARD.audio,
        }
        out = []
        for key in SECTIONS:
            v = BOARD.ver[_VER_KEY.get(key, key)]
            if key == "audio":
                if TTS_OUT == "browser" and BOARD.audio and v > _audio_sent["ver"]:
                    _audio_sent["ver"] = v
                    out.append(BOARD.audio)
                else:
                    out.append(gr.skip())
            elif sent is not None and key != "hdr" and sent.get(key) == v:
                out.append(gr.skip())
            else:
                out.append(makers[key]())
                if sent is not None:
                    sent[key] = v
        return out


# ── 말하기 ────────────────────────────────────────────────────────────────
def _warm_tts():
    tts.available()


def say_out(text: str):
    if TTS_OUT == "off":
        return
    BOARD.set_active("말하기", True)

    def done(_text, path):
        with BOARD.lock:
            BOARD.set_active("말하기", False)
            BOARD.dev_hit("말하기", tts.last_info.get("latency_ms"))
            if not path and tts.last_info.get("error"):
                BOARD.log("warn", "tts", f"음성 없이 글자만: {tts.last_info['error']}")

    if TTS_OUT in ("speaker", "silent", "timed"):
        tts.enqueue(text, on_done=done, sound={"speaker": True, "silent": False, "timed": "timed"}[TTS_OUT])
    else:  # browser: 합성만 하고 화면의 오디오가 자동 재생
        path = tts.speak(text, trim=True)
        done(text, path)
        if path:
            BOARD.audio = path
            BOARD.bump("audio")


# ── 멈춤 ─────────────────────────────────────────────────────────────────
def is_stop_text(text: str) -> bool:
    s = re.sub(r"[\W_]+", "", text or "")
    return 0 < len(s) <= 10 and any(w in s for w in STOP_WORDS)


def do_stop(origin: str):
    """에이전트: run_command("멈춰") — 그래프 앞에서 규칙으로 바로 멈춘다(계약서 2절). 실패하면 robot.run_policy.stop().
    가짜 스트림: fake_agent.stop()."""
    STOP.set()
    tts.clear_queue()
    with BOARD.lock:
        BOARD.log("error", "stop", f"정지 요청({origin})")
    said = False
    if AGENT_NAME == "fake":
        fake_agent.stop()
    else:
        try:
            for ev in AGENT_FN("멈춰", thread_id=BOARD.thread_id):
                if ev.get("type") == "node" and ev.get("node") == "start":
                    continue
                text = BOARD.apply(ev)
                if text:
                    said = True
                    say_out(text)
        except Exception as e:
            with BOARD.lock:
                BOARD.log("error", "stop", f"에이전트 정지 실패 {e.__class__.__name__}: {e} → robot.run_policy.stop()")
            try:
                importlib.import_module("robot.run_policy").stop()
            except Exception as e2:
                BOARD.log("error", "stop", f"robot.run_policy.stop() 실패: {e2}")
    if not said:
        with BOARD.lock:
            BOARD.add_chat("assistant", "멈췄습니다.")
        say_out("멈췄습니다.")


# ── 실행 ─────────────────────────────────────────────────────────────────
def _pump(text: str, thread_id: str, q: queue.Queue, heard: dict | None = None):
    try:
        for ev in AGENT_FN(text, thread_id=thread_id, **({"stt": heard} if heard and AGENT_STT else {})):
            q.put(("ev", ev))
    except Exception as e:
        q.put(("err", f"{e.__class__.__name__}: {e}"))
    finally:
        q.put(("end", None))


def run_pending(pending):
    """pending = (글, 출처 'voice'|'text'|'quick', 말이 끝난 시각[, 녹음 정보 {"wav_path"}]). 이벤트마다 화면을 다시 그린다."""
    sent: dict = {}
    if not pending or not pending[0]:
        yield render(sent)
        return
    text, origin, t0, *rest = pending
    STOP.clear()
    with BOARD.lock:
        BOARD.running, BOARD.t_cmd, BOARD.departed = text, t0 or time.time(), False
        for st in BOARD.nodes.values():
            st.update(state="idle", visits=0)
        BOARD.current, BOARD.current_note = None, ""
        BOARD.plan = None  # 계획 명령이면 에이전트의 steps 이벤트로 다시 잡힌다
        BOARD.target, BOARD.route_pos, BOARD.cmd_intent, BOARD.defect, BOARD.refused_tool = None, [], None, None, None
        chat0 = len(BOARD.chat)  # 이번 명령의 말풍선은 여기부터
        if BOARD.arrange and not BOARD.arrange.get("waiting"):
            BOARD.arrange = None  # 숫자 배치 확인 대답("응"·"아니")이면 그림을 남겨 둔다
        elif BOARD.arrange:
            BOARD.arrange["waiting"] = False
        BOARD.bump("nodes", "hdr", "layout", "cards")
        BOARD.log("cmd", origin, text)
        cmd = {"ts": datetime.now().strftime("%H:%M:%S"), "started": datetime.now(), "text": text}
        BOARD.cmds.append(cmd)
    yield render(sent)

    q: queue.Queue = queue.Queue()
    threading.Thread(target=_pump, args=(text, BOARD.thread_id, q, rest[0] if rest else None), daemon=True).start()
    stop_deadline = None
    end: dict = {}
    said = 0
    while True:
        try:
            kind, item = q.get(timeout=0.5)
        except queue.Empty:
            if STOP.is_set():
                stop_deadline = stop_deadline or time.time() + 5
                if time.time() > stop_deadline:
                    BOARD.log("warn", "", "에이전트가 5초 안에 끝나지 않아 화면 갱신만 멈춥니다")
                    break
            yield render(sent)
            continue
        if kind == "end":
            break
        if kind == "err":
            with BOARD.lock:
                BOARD.add_chat("assistant", f"⚠ 에이전트 오류: {item}")
                BOARD.log("error", "agent", item)
            break
        if item.get("type") == "node" and item.get("node") == "end":
            end = item.get("data") or {}
        say = BOARD.apply(item, quiet=STOP.is_set())
        if say and not STOP.is_set():
            said += 1
            say_out(say)
        yield render(sent)

    with BOARD.lock:
        st = (IGNORED_KO.get(end.get("intent")) if end.get("status") == "ignored" else
              NOTHING_KO.get(end.get("intent")) if end.get("status") == "nothing" else None) \
            or STATUS_KO.get(end.get("status"), end.get("status") or "")
        if end.get("intent") == "none" and not said:  # 잡담(none_kind chat)은 에이전트가 말하지 않는다 → 소리 없이 한 줄만
            BOARD.add_chat("assistant", '<span class="cite">명령이 아니어서 그대로 두었어요(로봇 안 움직임)</span>')
        if end.get("status") == "refused":  # 거부 이유는 에이전트가 이미 말했다 → 그 말풍선에 아이콘·색만
            last = next((m for m in reversed(BOARD.chat[chat0:]) if m["role"] == "assistant"), None)
            if last and 'class="refuse"' not in last["content"]:
                last["content"] = _refuse_html(last["content"])
                BOARD.bump("chat")
        # 결과 문구: "왼쪽 초록 불량 1개 · 완료" · "맨 위 초록 1개 · 완료" · "분홍 옆 초록 1개 · 완료" · "초록 · 위치 못 읽음 · 거부"
        # · "분홍 · 위치 둘 · 거부"(위치를 고르지 못한 것은 대상 뒤에 따로)
        fail = "위치 못 읽음" if end.get("target_by") in BY_FAIL else WHICH_KO["mixed"] if end.get("which") == "mixed" else ""
        where = None if fail else _where(end.get("which"), end.get("next_to"))
        dcls = DEFECT_KO.get(end.get("defect_cls"))
        what = " ".join(x for x in (where, dcls or (TARGET_KO.get(end.get("intent")) if where or fail else None)) if x)
        n = f"{end['done']}개" if end.get("done") and end.get("intent") not in ("report", "question") else ""
        cmd.update(intent=end.get("intent"), duration_s=end.get("elapsed_s", round(time.time() - cmd["started"].timestamp(), 1)),
                   result=" · ".join(x for x in (" ".join(x for x in (what, n) if x), fail, st) if x))
        if BOARD.current in BOARD.nodes and BOARD.nodes[BOARD.current]["state"] == "active":
            BOARD.nodes[BOARD.current]["state"] = "done"
        BOARD.log("info", "", f"끝 · {time.time() - BOARD.t_cmd:.1f}초")
        BOARD.running = None
        BOARD.clear_active()
        BOARD.bump("nodes", "hdr")
    yield render(sent)


def take_text(text: str, origin: str = "text", heard: dict | None = None):
    """입력을 받아 pending 으로. '멈춰'는 기다리지 않고 바로 멈춘다. heard = 음성일 때 녹음 정보(에이전트 환청 거르기용)."""
    text = (text or "").strip()
    if not text:
        return "", None
    icon = {"voice": "🎙 ", "quick": ""}.get(origin, "")
    if is_stop_text(text):
        BOARD.add_chat("user", icon + text)
        do_stop({"voice": "말", "quick": "버튼"}.get(origin, "글"))
        return "", None
    BOARD.add_chat("user", icon + text)
    return "", (text, origin, BOARD.voice_t0 if origin == "voice" else time.time(), heard)


MIC_SOURCES = ["microphone", "upload"]  # 마이크가 기본, 아래 아이콘으로 녹음 파일 올리기(마이크가 안 될 때·시험용)
# 녹음은 브라우저가 16kHz 로 풀어 올린다(Whisper 입력과 같은 표본율 — 44.1kHz 를 서버에서 줄일 때의 접힘이 없다)
MIC_RATE = int(os.getenv("UI_MIC_RATE", "16000"))
# Gradio 는 녹음 버튼마다 getUserMedia({audio:true})로 마이크를 새로 연다(크롬 기본 = 에코 제거·잡음 제거·자동 음량 켬).
# UI_MIC_CONSTRAINTS: 그 위에 덮을 값(JSON, 예 {"noiseSuppression":false}). UI_MIC_WARM_S: 마지막 녹음 뒤 이만큼 마이크를 열어 두고
# 다음 녹음은 그 복제본으로 바로 시작(마이크 여는 동안 첫 말소리를 잃지 않게). 0 = 예전처럼 매번 연다
MIC_CONSTRAINTS = json.loads(os.getenv("UI_MIC_CONSTRAINTS", "{}") or "{}")
MIC_WARM_S = float(os.getenv("UI_MIC_WARM_S", "600"))
HEAD_MIC = """<script>
(() => {
  const md = navigator.mediaDevices;
  if (!md || !md.getUserMedia || md.__ballMic) return;
  md.__ballMic = true;
  const orig = md.getUserMedia.bind(md), CONS = %s, WARM_MS = %d;
  let master = null, key = '', timer = null;
  md.getUserMedia = async (c) => {
    if (!c || !c.audio || c.video) return orig(c);
    const want = {audio: {...(c.audio === true ? {} : c.audio), ...CONS}};
    if (!WARM_MS) return orig(want);
    const k = JSON.stringify(want);
    const live = master && master.getAudioTracks().some(t => t.readyState === 'live');
    if (!live || k !== key) { if (master) master.getTracks().forEach(t => t.stop()); master = await orig(want); key = k; }
    clearTimeout(timer);
    timer = setTimeout(() => { if (master) master.getTracks().forEach(t => t.stop()); master = null; }, WARM_MS);
    return master.clone();
  };
  // 마이크를 이미 허락한 브라우저면 화면을 열 때 미리 열어 둔다(첫 녹음도 바로 시작)
  if (WARM_MS && navigator.permissions) navigator.permissions.query({name: 'microphone'})
    .then(p => { if (p.state === 'granted') md.getUserMedia({audio: true}).then(s => s.getTracks().forEach(t => t.stop())); })
    .catch(() => {});
})();
</script>""" % (json.dumps(MIC_CONSTRAINTS), int(MIC_WARM_S * 1000))


# 파일을 올린 뒤 칸이 "파일 올리기" 모드로 남는다(서버 쪽 갱신으로는 안 바뀜) → 브라우저에서 마이크 아이콘을 눌러 되돌린다
JS_MIC_MODE = """() => { const b = document.querySelector('#mic button[aria-label="Record audio"]')
    || [...document.querySelectorAll('#mic button[aria-label]')].find(e => /record|녹음/i.test(e.getAttribute('aria-label')));
  if (b && !b.classList.contains('selected')) b.click(); }"""


def _mic_reset():
    """듣고 나면 칸을 비운다."""
    return gr.Audio(value=None)


def _keep_voice(audio_path: str, text: str, err: str | None) -> str | None:
    """받은 녹음을 ui/voice_log/<날짜>/<시각>.wav 로, 받아쓰기 결과·녹음 상태를 같은 폴더 voice.jsonl 에 남긴다."""
    if not VOICE_LOG:
        return None
    try:
        now = datetime.now()
        day = VOICE_DIR / now.strftime("%Y%m%d")
        day.mkdir(parents=True, exist_ok=True)
        dst = day / f"{now:%H%M%S}_{now.microsecond // 10000:02d}{Path(audio_path).suffix or '.wav'}"
        shutil.copy(audio_path, dst)
        rec = {"ts": now.isoformat(timespec="milliseconds"), "file": dst.name, "text": text, "err": err, **stt.last_info}
        with open(day / "voice.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        for old in sorted(VOICE_DIR.glob("*/*.wav"))[:-VOICE_KEEP]:
            old.unlink(missing_ok=True)
        return str(dst)
    except Exception as e:
        BOARD.log("warn", "stt", f"녹음 남기기 실패 {e.__class__.__name__}: {e}")
        return None


def _rec_note(m: dict) -> str:
    """녹음 상태 한 줄 + 문제면 짧은 조언(로그용)."""
    note = f"녹음 {m.get('seconds', 0):.1f}초 · 말소리 {m.get('speech_s', 0):.1f}초 · 최대 {m.get('peak', 0):.2f}"
    if m.get("clip", 0) > 0.001:
        note += " · ⚠ 소리가 너무 커 잘림(마이크에서 조금 떨어져서)"
    elif m.get("peak", 1) < 0.05:
        note += " · ⚠ 소리가 작음(마이크 가까이)"
    if m.get("lead_s") is not None and m["lead_s"] < 0.06:
        note += " · ⚠ 말이 녹음 시작에 붙음(버튼 누르고 반 박자 뒤에 말하기)"
    return note


def on_voice(audio_path):
    """녹음이 끝나면: 듣기(STT) → 글 상자에 보여 주고 pending 으로."""
    if not audio_path:
        return gr.skip(), "", None, *render()
    BOARD.voice_t0 = time.time()
    with BOARD.lock:
        BOARD.set_active("듣기", True)
    try:
        text = stt.transcribe(audio_path)
        err = None
    except Exception as e:
        text, err = "", str(e)
    kept = _keep_voice(audio_path, text, err)
    with BOARD.lock:
        BOARD.set_active("듣기", False)
        BOARD.dev_hit("듣기", stt.last_info.get("latency_ms"))
        if err:
            BOARD.log("error", "stt", err)
            BOARD.add_chat("assistant", "⚠ 음성 인식이 안 됩니다. 아래 글 상자에 입력해 주세요.")
        elif not text and stt.last_info.get("peak", 1) <= 0:  # 소리가 하나도 안 들어옴 = 마이크 문제(말을 못 알아들은 것과 다름)
            BOARD.log("error", "stt", f"마이크 입력이 0(완전 무음, {stt.last_info.get('seconds', 0):.1f}초) — 입력 장치 확인 · "
                                      f"지금 기본 입력: {micdev.label()} · 점검: python -m ui.mic_check --device")
            BOARD.add_chat("assistant", f"⚠ 마이크 입력이 0입니다(완전 무음). 입력 장치를 확인해 주세요 — 지금 기본 입력: "
                                        f"{micdev.label()}. 바꾼 뒤에는 화면을 새로고침하고, 급하면 아래 글 상자에 입력하세요.")
        elif not text:
            BOARD.log("warn", "stt", f"말소리를 못 찾았습니다 · {_rec_note(stt.last_info)}")
            BOARD.add_chat("assistant", "잘 못 들었어요. 버튼을 누르고 다시 말씀해 주세요.")
        else:
            raw = stt.last_info.get("raw")
            BOARD.log("stt", "stt", f"“{text}” · {_ms(stt.last_info.get('latency_ms'))}"
                                    + (f" · 바로잡음(원문 “{raw}”)" if raw and raw != text else "")
                                    + f" · {_rec_note(stt.last_info)}")
    # 소리 기준 환청 거르기(agent/hearing.py)에는 말소리 길이를 넘긴다. 녹음 파일(wav_path)을 넘기면 에이전트가 절대 문턱(0.02)으로
    # 재서 작게 녹음된 진짜 명령을 거른다(실물 흉내 -35dBFS 에서 936개 중 641개). ui/stt 는 잡음 바닥에 맞춘 상대 문턱
    heard = {"speech_s": stt.last_info.get("speech_s")} if stt.last_info.get("speech_s") is not None else None
    _, pending = take_text(text, "voice", heard) if text else ("", None)
    return _mic_reset(), "", pending, *render()


def on_stop_button():
    do_stop("버튼")
    return render()


def on_reset():
    if BOARD.running:
        do_stop("초기화")
    if AGENT_NAME == "fake":
        fake_agent.reset()
    elif os.getenv("BALL_MOCK") == "1":  # 에이전트의 가짜 작업대(agent/mocks.py)도 처음 상태로
        try:
            importlib.import_module("agent.mocks").workbench.reset()
        except Exception as e:
            BOARD.log("warn", "", f"가짜 작업대 초기화 실패 {e.__class__.__name__}: {e}")
    BOARD.reset()
    BOARD.log("info", "", f"초기화 · thread_id={BOARD.thread_id}")
    return render()


# ── 화면 ─────────────────────────────────────────────────────────────────
CSS = (Path(__file__).parent / "app.css").read_text(encoding="utf-8")
THEME = gr.themes.Default(primary_hue="indigo", neutral_hue="slate", radius_size="md",
                          font=["Noto Sans CJK KR", "NanumBarunGothic", "sans-serif"],
                          font_mono=["NanumGothicCoding", "DejaVu Sans Mono", "monospace"])


def build() -> gr.Blocks:
    with gr.Blocks(title="스펀지 블록 선별 로봇") as demo:
        pending = gr.State(None)
        hdr = gr.HTML(elem_id="hdr")
        # 조작 줄을 맨 위에: 발표 중 스크롤 없이 마이크·멈춤·빠른 명령을 누른다
        with gr.Row(equal_height=True, elem_id="controls"):
            mic = gr.Audio(sources=MIC_SOURCES, type="filepath", format="wav", show_label=False,
                           elem_id="mic", scale=3, min_width=240, waveform_options=gr.WaveformOptions(sample_rate=MIC_RATE))
            with gr.Column(scale=7, min_width=420):
                with gr.Row(equal_height=True):
                    txt = gr.Textbox(show_label=False, placeholder="말 대신 글로 — 예) 불량품 골라내 줘", lines=1, max_lines=1,
                                     scale=6, container=False)
                    send = gr.Button("보내기", variant="primary", scale=1, min_width=90)
                    stopb = gr.Button("■ 멈춰", variant="stop", scale=1, min_width=90, elem_id="stopbtn")
                with gr.Row(elem_id="quick"):
                    quick = [gr.Button(f"{n} {t}", size="sm", elem_classes="qbtn") for n, t in QUICK]
                    reset = gr.Button("초기화", size="sm", variant="secondary", elem_id="resetbtn")
                with gr.Accordion("다른 명령", open=False, elem_id="extra"):
                    with gr.Row():
                        extra = [gr.Button(f"{n} {t}".strip(), size="sm") for n, t in EXTRA]
        with gr.Row(equal_height=False):
            with gr.Column(scale=6, min_width=420):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        snap = gr.Image(type="filepath", show_label=False, interactive=False, height=360, elem_id="snap",
                                        buttons=["fullscreen"])
                        snapcap = gr.HTML()
                    with gr.Column(scale=1, min_width=300):
                        layout = gr.HTML(elem_id="layout", elem_classes="panel")
                        with gr.Row(visible=False, elem_id="confirm") as confirm:  # 숫자 배치 "할까요?" 대답
                            yes = gr.Button("응, 해 줘", variant="primary", size="sm")
                            no = gr.Button("아니", size="sm")
                cards = gr.HTML()
            with gr.Column(scale=5, min_width=380):
                nodes = gr.HTML(elem_classes="panel")
                devs = gr.HTML(elem_classes="panel")
        with gr.Row(equal_height=False):
            with gr.Column(scale=6, min_width=420):
                chat = gr.Chatbot(show_label=False, height=300, elem_id="chat", buttons=[],
                                  placeholder="마이크 버튼을 누르고 말하거나, 위 글 상자에 입력하세요")
            with gr.Column(scale=5, min_width=380):
                log = gr.HTML(elem_classes="panel")
        with gr.Accordion("결과 보고서", open=False) as acc:
            report = gr.HTML()
        audio_out = gr.Audio(autoplay=True, visible=TTS_OUT == "browser", show_label=False, interactive=False)

        outs = [hdr, snap, snapcap, layout, confirm, nodes, cards, devs, chat, log, report, acc, audio_out]
        run_kw = dict(concurrency_id="run", concurrency_limit=1, show_progress="hidden")

        demo.load(lambda: render(), None, outs, show_progress="hidden")
        gr.Timer(2.0).tick(lambda: render(_timer_sent), None, outs, show_progress="hidden")

        for trig in (send.click, txt.submit):
            trig(take_text, txt, [txt, pending], show_progress="hidden").then(run_pending, pending, outs, **run_kw)
        for b, (_, t) in zip(quick + extra + [yes, no], QUICK + EXTRA + [("", "응"), ("", "아니")]):
            b.click(lambda t=t: take_text(t, "quick"), None, [txt, pending], show_progress="hidden") \
                .then(run_pending, pending, outs, **run_kw)
        voice_kw = dict(fn=on_voice, inputs=mic, outputs=[mic, txt, pending, *outs], show_progress="hidden")
        mic.stop_recording(**voice_kw).then(run_pending, pending, outs, **run_kw)
        uploaded = mic.upload(**voice_kw)
        uploaded.then(run_pending, pending, outs, **run_kw)
        uploaded.then(None, js=JS_MIC_MODE)
        stopb.click(on_stop_button, None, outs, concurrency_limit=None, show_progress="hidden")
        reset.click(on_reset, None, outs, show_progress="hidden")
    return demo


def _warm_agent():
    """에이전트 warmup(라우터 임베딩·기준서 검색 준비)을 미리 — 첫 명령이 빨라진다(agent/__init__.py)."""
    try:
        mod = importlib.import_module(AGENT_NAME.split(":")[0])
        fn = getattr(mod, "warmup", None)
        if fn:
            out = fn()
            with BOARD.lock:
                BOARD.log("info", "", f"에이전트 준비 끝 {out.get('total_s', '?')}초")
        # 방식(r21)·위치로 고르기(10-23, which = 로봇 함수가 which 인자를 받나)·불량 색(defect_color = 로봇이 red_defect·
        # blue_defect 를 아나)·그릇에서 꺼내기(unsort = real·mock·off, off 면 로봇 함수 없음)
        gm = getattr(mod, "get_method", None)
        if gm:
            m = gm()
            unsort = {"real": "됨(실제)", "mock": "됨(가짜)", "off": "안 됨(로봇 함수 없음 — 꺼내기 명령은 거부)"}
            # 10-23 13:30: which_values(로봇 plan.WHICH)·next_to(plan.NEXT_TO) — 로봇이 모르는 값은 에이전트가 부르기 전에 거부
            vals = "·".join(WHICH_KO.get(w, str(w)) for w in m.get("which_values") or [])
            near = "·".join(NEXT_KO.get(c, str(c)) for c in m.get("next_to") or [])
            pos = " / ".join(x for x in (vals, f"옆: {near}" if near else "") if x)
            parts = [f"방식 기본 = {m.get('default')}", f"좌표 집기 {m.get('coord_backend')}",
                     "위치로 고르기 " + (f"됨({pos})" if m.get("which") and pos else "됨" if m.get("which")
                                    else "안 됨(위치 단서 명령은 거부)")]
            if "defect_color" in m:  # 10-23 이전 에이전트에는 두 키가 없다
                parts.append("불량 색 고르기 " + ("됨" if m["defect_color"] else "안 됨(색 붙은 불량 명령은 거부)"))
            if "unsort" in m:
                parts.append("그릇에서 꺼내기 " + unsort.get(m["unsort"], str(m["unsort"])))
            with BOARD.lock:
                BOARD.log("info", "", " · ".join(parts))
    except Exception as e:
        with BOARD.lock:
            BOARD.log("warn", "", f"에이전트 warmup 실패 {e.__class__.__name__}: {e}")


def _view_log(text: str):
    with BOARD.lock:
        BOARD.log("warn", "view", text)


def _start_view():
    """보기 전용 화면을 연다. 무엇이 실패해도(포트 사용 중 등) 로그 한 줄만 남기고 대시보드는 그대로 뜬다."""
    global VIEW
    try:
        from ui import view
        url = view.start(view_sections, CSS, _view_log)
        VIEW = view
        BOARD.log("info", "", f"보기 전용 화면 = {url} (와이파이 안 노트북·폰, 읽기만 · 1초마다 새로 그림)")
        print(f"보기 전용 화면: {url}", flush=True)
    except Exception as e:
        BOARD.log("warn", "", f"보기 전용 화면 못 엶 {e.__class__.__name__}: {e} — 대시보드는 그대로")


def main():
    threading.Thread(target=_warm_tts, daemon=True).start()
    if AGENT_NAME != "fake":
        threading.Thread(target=_warm_agent, daemon=True).start()
    BOARD.log("info", "", f"에이전트 = {AGENT_NAME} · 음성 출력 = {TTS_OUT} · STT = {stt.BASE_URL} ({stt.MODEL})"
                          f" · 외부 차단 = {netguard.status()['mode']}")
    for note in AGENT_NOTES:
        BOARD.log("warn", "", note)
    src = micdev.default_source(0)
    BOARD.log("warn" if not src or src["muted"] or micdev.is_jack(src) else "info", "",
              f"마이크 기본 입력 = {micdev.label(src)} · 녹음 남김 {'켬' if VOICE_LOG else '끔'}(ui/voice_log)"
              f" · 미리 열기 {MIC_WARM_S:.0f}초 · 크롬 설정 {json.dumps(MIC_CONSTRAINTS) if MIC_CONSTRAINTS else '기본'}")
    for d in (REPORT_DIR, fake_agent.OUT, tts.OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if os.getenv("UI_VIEW", "0") == "1":
        _start_view()
    build().queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=PORT, theme=THEME, css=CSS, head=HEAD_MIC, show_error=True, footer_links=[],
        allowed_paths=[str(ROOT / "runs"), str(fake_agent.OUT), str(REPORT_DIR), str(tts.OUT_DIR)])


if __name__ == "__main__":
    main()
