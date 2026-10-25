"""가짜 에이전트 — agent/ 가 아직 없을 때 대시보드를 움직이는 이벤트 스트림.

    run_command(text, thread_id="demo") -> Iterator[dict]   # 계약서 3절 이벤트 형식 그대로
    stop()    # 실행 중인 가짜 정책을 멈춘다
    reset()   # 작업대를 시연 시작 상태로: FAKE_SCENE=demo4(기본, 발표 실연 — 정상 분홍 2·초록 1·불량 1) | full(분홍 4·초록 4·불량 3)

작업대를 메모리에서 흉내 내고, 위 카메라 사진도 그려서(박스 포함) 진짜와 같은 모양의 결과를 낸다.
코드 식별자는 계약서대로 red_* = 분홍, blue_* = 초록. detect 결과에 objects(블록마다 zone·cell·px)도 넣는다.
scene(배치 질문)·plan(여러 단계 명령)은 UI 패널 시험용 흉내다. plan 이벤트 모양은 ui/README.md "plan 이벤트" 절.
환경변수: FAKE_SPEED(기본 0.15 = 정책 1회 약 2.5초), FAKE_FAIL(기본 0.1 = 정책 실패 확률)
"""
from __future__ import annotations

import os
import random
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ui" / "mock_out"
SPEED = float(os.getenv("FAKE_SPEED", "0.15"))
FAIL = float(os.getenv("FAKE_FAIL", "0.1"))
SCENES = {"demo4": (("red_ok", 2), ("blue_ok", 1), ("red_defect", 1)),  # 계약서 2절 "발표 실연"
          "full": (("red_ok", 4), ("blue_ok", 4), ("red_defect", 2), ("blue_defect", 1))}
SCENE = os.getenv("FAKE_SCENE", "demo4")
CLASSES = ("red_ok", "blue_ok", "red_defect", "blue_defect")

# 위 카메라 640x480 화면 배치(가짜). 진짜 구역은 vision/zones.json
INPUT_BOX = (40, 170, 340, 400)
BOWLS = {"bowl_red": (420, 105), "bowl_blue": (555, 105), "bowl_defect": (487, 250)}
BOWL_R = 58
HAND_BOX = (420, 345, 620, 465)
COLORS = {"red_ok": (242, 167, 197), "blue_ok": (160, 216, 168), "red_defect": (242, 167, 197), "blue_defect": (160, 216, 168)}
BOX_COLORS = {"red_ok": (214, 64, 159), "blue_ok": (70, 167, 88), "red_defect": (20, 20, 20), "blue_defect": (20, 20, 20),
              "hand": (226, 163, 54)}

POLICY = {"defect": "pick_defect", "red": "pick_red", "blue": "pick_blue", "handoff": "handoff"}
TARGET = {"defect": ("red_defect", "blue_defect"), "red": ("red_ok",), "blue": ("blue_ok",)}
DEST = {"defect": "bowl_defect", "red": "bowl_red", "blue": "bowl_blue"}
NAME = {"defect": "불량", "red": "분홍 블록", "blue": "초록 블록", "handoff": "분홍 블록"}
STEP_LABEL = {"defect": ("불량", "골라내기"), "red": ("분홍", "옮기기"), "blue": ("초록", "옮기기"), "handoff": ("분홍", "손에 건네기")}
RED_WORDS, BLUE_WORDS = ("분홍", "핑크"), ("초록", "녹색", "연두")
ROWS, COLS = ("위", "가운데", "아래"), ("왼쪽", "가운데", "오른쪽")


def _font(size: int, bold: bool = False):
    try:
        name = "NanumSquare:bold" if bold else "NanumSquare"
        path = subprocess.run(["fc-match", "-f", "%{file}", name], capture_output=True, text=True, timeout=3).stdout
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


_F_LABEL, _F_BOX, _F_HUD = _font(15, True), _font(11), _font(13, True)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _zone_box(zone: str) -> tuple[int, int, int, int]:
    if zone == "input":
        return INPUT_BOX
    if zone == "hand_spot":
        return HAND_BOX
    cx, cy = BOWLS[zone]
    return cx - BOWL_R, cy - BOWL_R, cx + BOWL_R, cy + BOWL_R


def cell_name(box, x: float, y: float) -> str:
    """구역 박스를 3×3 으로 나눈 칸 이름(계약서 3절): '왼쪽 위' … 가운데 칸은 '가운데'."""
    x0, y0, x1, y1 = box
    c = min(2, max(0, int(3 * (x - x0) / max(1, x1 - x0))))
    r = min(2, max(0, int(3 * (y - y0) / max(1, y1 - y0))))
    return "가운데" if (r, c) == (1, 1) else f"{COLS[c]} {ROWS[r]}"


class World:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.reset()

    def reset(self):
        rng = random.Random()
        self.input: list[list] = []  # [cls, x, y]
        for cls, n in SCENES.get(SCENE, SCENES["demo4"]):
            for _ in range(n):
                self.input.append([cls, *self._free_spot(rng)])
        self.bowls = {k: Counter() for k in BOWLS}
        self.handed = 0
        self.hand_present = False
        self.snap_no = 0
        self.first_snap: tuple[str, str] | None = None
        self.policies: dict[str, list] = {}  # name -> [attempts, success, total_s]
        self.commands: list[dict] = []
        self.dev: dict[str, list[float]] = {}
        self.human_checks: list[str] = []
        self.started = datetime.now()
        OUT.mkdir(parents=True, exist_ok=True)
        for f in OUT.glob("snap_*.jpg"):
            f.unlink(missing_ok=True)

    def _free_spot(self, rng):
        x0, y0, x1, y1 = INPUT_BOX
        for _ in range(200):
            x, y = rng.randint(x0 + 25, x1 - 25), rng.randint(y0 + 25, y1 - 25)
            if all((x - b[1]) ** 2 + (y - b[2]) ** 2 > 40**2 for b in getattr(self, "input", [])):
                return x, y
        return x, y

    def zones(self) -> dict:
        inp = Counter(b[0] for b in self.input)
        z = {"input": {c: inp.get(c, 0) for c in CLASSES}}
        for k, cnt in self.bowls.items():
            z[k] = {c: cnt.get(c, 0) for c in CLASSES}
        return z

    def placed(self) -> list[tuple[str, str, int, int]]:
        """(cls, zone, x, y) — 그릇 안 블록은 그릇 가운데 둘레에 놓는다."""
        out = [(b[0], "input", b[1], b[2]) for b in self.input]
        for k, (cx, cy) in BOWLS.items():
            i = 0
            for cls, n in sorted(self.bowls[k].items()):
                for _ in range(n):
                    dx, dy = (i % 3 - 1) * 30, (i // 3 - 1) * 30
                    out.append((cls, k, cx + dx, cy + dy))
                    i += 1
        return out

    def snapshot(self) -> str:
        img = Image.new("RGB", (640, 480), (246, 246, 243))  # 흰 무광 식탁보
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((14, 14, 120, 120), 14, fill=(205, 207, 212))
        d.text((67, 67), "로봇", font=_F_LABEL, fill=(90, 94, 100), anchor="mm")
        d.rectangle(INPUT_BOX, outline=(150, 150, 150), width=2)
        d.text((INPUT_BOX[0] + 6, INPUT_BOX[1] - 20), "투입 구역", font=_F_LABEL, fill=(90, 90, 90))
        for k, (cx, cy) in BOWLS.items():
            d.ellipse((cx - BOWL_R, cy - BOWL_R, cx + BOWL_R, cy + BOWL_R), fill=(252, 252, 250), outline=(160, 160, 160), width=3)
            label = {"bowl_red": "분홍", "bowl_blue": "초록", "bowl_defect": "불량"}[k]
            d.text((cx, cy + BOWL_R + 12), label, font=_F_LABEL, fill=(90, 90, 90), anchor="mm")
        d.rectangle(HAND_BOX, outline=(200, 160, 90), width=2)
        d.text((HAND_BOX[0] + 6, HAND_BOX[1] - 20), "손 자리", font=_F_LABEL, fill=(150, 110, 40))

        dets = [(cls, x, y) for cls, _, x, y in self.placed()]
        rng = random.Random(self.snap_no)
        for cls, x, y in dets:
            self._block(d, cls, x, y)
        for cls, x, y in dets:
            self._box(d, (x - 19, y - 19, x + 19, y + 19), cls, 0.86 + rng.random() * 0.12)
        if self.hand_present:
            hx, hy = (HAND_BOX[0] + HAND_BOX[2]) // 2, (HAND_BOX[1] + HAND_BOX[3]) // 2 + 6
            d.ellipse((hx - 42, hy - 30, hx + 42, hy + 30), fill=(232, 190, 160))
            for j in range(4):
                fx = hx - 33 + j * 22
                d.rounded_rectangle((fx, hy - 62, fx + 15, hy - 20), 7, fill=(232, 190, 160))
            if self.handed:
                self._block(d, "red_ok", hx, hy)
            self._box(d, (hx - 48, hy - 66, hx + 48, hy + 34), "hand", 0.91)
        self.snap_no += 1
        path = OUT / f"snap_{self.snap_no:04d}.jpg"
        img.save(path, quality=90)
        for old in sorted(OUT.glob("snap_*.jpg"))[:-60]:
            old.unlink(missing_ok=True)
        return str(path)

    @staticmethod
    def _block(d, cls, x, y, r=13):
        d.rounded_rectangle((x - r, y - r, x + r, y + r), 4, fill=COLORS[cls], outline=tuple(int(c * 0.7) for c in COLORS[cls]), width=2)
        if cls.endswith("defect"):
            d.line((x - 8, y - 8, x + 8, y + 8), fill=(15, 15, 15), width=4)
            d.line((x - 8, y + 8, x + 8, y - 8), fill=(15, 15, 15), width=4)

    @staticmethod
    def _box(d, box, cls, conf):
        col = BOX_COLORS[cls]
        d.rectangle(box, outline=col, width=2)
        label = f"{cls} {conf:.2f}"
        w = d.textlength(label, font=_F_BOX) + 6
        d.rectangle((box[0], box[1] - 15, box[0] + w, box[1]), fill=col)
        d.text((box[0] + 3, box[1] - 14), label, font=_F_BOX, fill=(255, 255, 255))


W = World()


def reset():
    with W.lock:
        W.reset()


def stop():
    W.stop_flag.set()


def _ev(kind, node, **data):
    return {"type": kind, "node": node, "data": data, "ts": _now()}


def _sleep(sec: float) -> bool:
    """멈춤 요청이 오면 바로 깨어나 False."""
    return not W.stop_flag.wait(sec)


def _stat(layer: str, ms: float):
    W.dev.setdefault(layer, []).append(ms)


def _detect():
    t0 = time.perf_counter()
    path = W.snapshot()
    ms = 10.5 + random.random() * 4
    _stat("보기", ms)
    objects = [{"cls": cls, "zone": zone, "cell": cell_name(_zone_box(zone), x, y), "px": [x, y],
                "conf": round(0.86 + random.random() * 0.12, 2)} for cls, zone, x, y in W.placed()]
    res = {"ts": _now(), "zones": W.zones(), "hand": {"present": W.hand_present, "conf": 0.91 if W.hand_present else 0.0},
           "objects": objects, "image_path": str(Path(path).relative_to(ROOT)), "device": "NPU", "latency_ms": round(ms, 1)}
    if W.first_snap is None:
        W.first_snap = (res["image_path"], datetime.now().strftime("%H:%M:%S"))
    time.sleep(max(0.0, 0.25 * SPEED / 0.15 - (time.perf_counter() - t0)))
    return res


def _count_words(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(개|만)", text)
    if m:
        return int(m.group(1))
    words = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5}
    m = re.search(r"(하나|다섯|한|두|둘|세|셋|네|넷)\s*(개|만)", text) or re.search(r"(하나)", text)
    return words[m.group(1)] if m else None


def _color(t: str) -> str | None:
    tail = re.split(r"말고|됐고|건됐|은됐", t)[-1]
    hits = [(tail.find(k), v) for words, v in ((RED_WORDS, "red"), (BLUE_WORDS, "blue")) for k in words if k in tail]
    return min(hits)[1] if hits else None


PLAN_SPLIT = re.compile(r"(?<=[가-힣])고\s+|다음에\s*|그다음\s*|그리고\s*")
ACTION = ("옮기", "옮겨", "담", "골라", "빼", "넣", "정리", "줘", "주세요", "건네", "치워")
KEEP = ("그대로", "두고", "말고")


def _parts(text: str) -> list[str]:
    return [re.sub(r"\s+", "", p) for p in PLAN_SPLIT.split(text.strip()) if p.strip()]


def _is_plan(text: str) -> bool:
    """동작이 둘 이상이거나, '나머지'·'그대로 두고' 같은 조건이 붙은 명령."""
    parts = _parts(text)
    acts = sum(any(a in p for a in ACTION) for p in parts)
    keeps = sum(any(k in p for k in KEEP) for p in parts)
    return acts >= 2 or (acts >= 1 and len(parts) >= 2 and keeps >= 1) or ("나머지" in "".join(parts) and acts >= 1)


def _single(text: str) -> tuple[str | None, int | None]:
    t = re.sub(r"\s+", "", text)
    n = _count_words(text)
    if any(k in t for k in ("손", "나한테", "건네")):
        return "handoff", n
    if any(k in t for k in ("불량", "X", "엑스")):
        return "defect", n
    color = _color(t)
    return (color, n) if color else (None, None)


def classify(text: str) -> tuple[str | None, int | None]:
    t = re.sub(r"\s+", "", text)
    if any(k in t for k in ("어때", "남았", "남아있", "배치", "위치", "어디있", "어디에있")):
        return "scene", None
    if any(k in t for k in ("왜", "뭐", "어떻게", "센티", "기준", "순서", "어디", "무엇")):
        return "question", None
    if any(k in t for k in ("결과", "보고", "몇개", "몇번", "성공", "옮겼", "들어있")):
        return "report", None
    if any(k in t for k in ("멈춰", "스톱", "정지", "중지", "그만")):
        return "stop", None
    if _is_plan(text):
        return "plan", None
    return _single(text)


def make_plan(text: str) -> list[dict]:
    """'분홍 두 개만 옮기고 불량 골라내 줘' → [{intent: red, count: 2}, {intent: defect, count: None}].
    '그대로 두고'·'말고' 가 붙은 색은 빼고, '나머지' 는 기준서 5.1 순서(불량 → 분홍 → 초록)로 채운다."""
    steps, keep = [], set()
    parts = _parts(text)
    for i, t in enumerate(parts):
        if t.endswith("빼") and i + 1 < len(parts) and "나머지" in parts[i + 1]:  # "불량 빼고 나머지 …" = 불량은 빼고
            keep.add("defect" if "불량" in t else _color(t))
            continue
        if any(k in t for k in KEEP):
            c = re.split(r"말고", t)[0] if "말고" in t else t
            c = _color(c.replace("말", ""))
            if c:
                keep.add(c)
            continue
        if "나머지" in t or ("전부" in t and not _color(t)):
            for intent in ("defect", "red", "blue"):
                if intent not in keep and all(s["intent"] != intent for s in steps):
                    steps.append({"intent": intent, "count": None})
            continue
        intent, n = _single(t)
        if intent in POLICY:
            steps.append({"intent": intent, "count": n})
    for s in steps:
        what, verb = STEP_LABEL[s["intent"]]
        s["label"] = f"{what} {s['count']}개 {verb}" if s["count"] else f"{what} {verb}"
    return steps


def _manual_chunks():
    text = (ROOT / "docs" / "standard.md").read_text(encoding="utf-8")
    out = []
    for block in re.split(r"\n(?=### )", text)[1:]:
        head, _, body = block.partition("\n")
        m = re.match(r"### (\d+\.\d+) (.+)", head)
        if m:
            body = re.split(r"\n## ", body)[0].strip()
            out.append({"id": m.group(1), "title": m.group(2).strip(), "text": body})
    return out


def _search(query: str, k: int = 3):
    def grams(s):
        s = re.sub(r"\s+", "", s)
        return {s[i:i + 2] for i in range(len(s) - 1)}

    q = grams(query)
    scored = []
    for c in _manual_chunks():
        score = len(q & grams(c["title"])) * 2 + len(q & grams(c["text"]))
        scored.append(dict(c, score=round(score / max(1, len(q)), 3)))
    return sorted(scored, key=lambda c: -c["score"])[:k]


def _run_policy(name: str, intent: str):
    """가짜 정책 1회. (ok, duration_s)"""
    dur = 14 + random.random() * 6
    if not _sleep(dur * SPEED):
        return False, dur * SPEED, True
    ok = random.random() >= FAIL
    with W.lock:
        if ok and intent in TARGET:
            cand = [b for b in W.input if b[0] in TARGET[intent]]
            if cand:
                block = random.choice(cand)
                W.input.remove(block)
                W.bowls[DEST[intent]][block[0]] += 1
        elif ok and intent == "handoff":
            W.bowls["bowl_red"]["red_ok"] -= 1
            W.handed += 1
    rec = W.policies.setdefault(name, [0, 0, 0.0])
    rec[0] += 1
    rec[1] += int(ok)
    rec[2] += dur
    _stat("행동", dur * 1000)
    return ok, dur, False


def _remaining(intent: str) -> int:
    if intent == "handoff":
        return W.bowls["bowl_red"]["red_ok"]
    return sum(1 for b in W.input if b[0] in TARGET[intent])


def _scene_answer(text: str, det: dict) -> str:
    z, objs = det["zones"], det.get("objects") or []
    inp = z["input"]
    t = re.sub(r"\s+", "", text)
    if "불량" in t:
        n = inp["red_defect"] + inp["blue_defect"]
        cells = [o["cell"] for o in objs if o["zone"] == "input" and o["cls"].endswith("defect")]
        if not n:
            return "투입 구역에 불량이 없습니다."
        return f"투입 구역에 불량이 {n}개 남았습니다(분홍 {inp['red_defect']}, 초록 {inp['blue_defect']}). 자리는 {', '.join(cells)}입니다."
    bowls = ", ".join(f"{name} 그릇 {sum(z[k].values())}개" for k, name in (("bowl_red", "분홍"), ("bowl_blue", "초록"), ("bowl_defect", "불량")))
    return (f"투입 구역에 분홍 {inp['red_ok']}개, 초록 {inp['blue_ok']}개, 불량 {inp['red_defect'] + inp['blue_defect']}개가 있고, "
            f"{bowls}입니다. 손 자리에는 {'손이 있습니다' if det['hand']['present'] else '손이 없습니다'}.")


def _work(intent: str, count: int | None, step: int | None = None):
    """한 intent(defect·red·blue·handoff) 실행. yield from 으로 부르면 (result, done) 을 돌려준다."""
    policy = POLICY[intent]
    extra = {"step": step} if step else {}
    yield _ev("node", "plan", intent=intent, policy=policy, count=count, **extra)
    yield _ev("tool", "plan", status="start", name="detect_objects", args={})
    det = _detect()
    yield _ev("tool", "plan", status="end", name="detect_objects", args={}, result=det)
    have = _remaining(intent)
    goal = 1 if intent == "handoff" and count is None else min(count or have, have)
    if have == 0:
        where = "분홍 그릇" if intent == "handoff" else "투입 구역"
        yield _ev("say", "plan", text=f"{where}에 {NAME[intent]}이 없습니다.")
        return "nothing", 0
    if intent != "handoff":
        yield _ev("say", "plan", text=f"{NAME[intent]} {have}개를 찾았습니다." if goal == have else f"{NAME[intent]} {goal}개만 옮길게요.")
    if intent == "handoff":
        yield _ev("node", "hand_wait")
        yield _ev("say", "hand_wait", text="손 자리에 손바닥을 펴 주세요.")
        for i in range(3):
            if not _sleep(0.8 * SPEED / 0.15):
                break
            W.hand_present = i >= 1
            det = _detect()
            yield _ev("tool", "hand_wait", status="end", name="detect_objects", args={}, result=det)
            if W.hand_present:
                break
    done = 0
    for _ in range(goal):
        ok = False
        for attempt in (1, 2):
            yield _ev("node", "act", policy=policy, attempt=attempt)
            started = _now()
            yield _ev("tool", "act", status="start", name="run_policy", args={"name": policy})
            ok, dur, stopped = _run_policy(policy, intent)
            yield _ev("tool", "act", status="end", name="run_policy", args={"name": policy},
                      result={"name": policy, "ok": ok, "started": started, "ended": _now(),
                              "duration_s": round(dur, 1), "returncode": 0 if ok else 1,
                              "log_path": f"runs/mock/{policy}.log", "stopped": stopped})
            if stopped:
                break
            yield _ev("node", "verify")
            det = _detect()
            yield _ev("tool", "verify", status="end", name="detect_objects", args={}, result=det)
            if ok:
                break
        if W.stop_flag.is_set():
            yield _ev("say", "act", text="멈췄습니다.")
            return "stopped", done
        if not ok:
            msg = f"{NAME[intent]}을 두 번 집지 못했습니다. 확인해 주세요."
            W.human_checks.append(f"{datetime.now():%H:%M} {msg}")
            yield _ev("say", "act", text=msg)
            return "gave_up", done
        done += 1
    W.hand_present = False
    if intent == "defect":
        z = W.bowls["bowl_defect"]
        msg = f"불량 {done}개(분홍 {z['red_defect']}, 초록 {z['blue_defect']})를 분리했습니다." if done == sum(z.values()) \
            else f"불량 {done}개를 분리했습니다."
    elif intent == "handoff":
        msg = f"분홍 블록 {done}개를 전달했습니다."
    else:
        msg = f"{NAME[intent]} {done}개를 담았습니다."
    yield _ev("node", "finish", status="done")
    yield _ev("say", "finish", text=msg)
    return "done", done


def run_command(text: str, thread_id: str = "demo"):
    W.stop_flag.clear()
    t_start = time.time()
    yield _ev("node", "start", text=text, thread_id=thread_id, backend="fake")
    yield _ev("node", "classify", status="start", text=text)
    _sleep(0.6 * SPEED / 0.15)
    intent, count = classify(text)
    llm_ms = 650 + random.random() * 500
    _stat("판단", llm_ms)
    yield _ev("node", "classify", status="end", intent=intent, count=count, latency_ms=round(llm_ms), device="GPU")
    result, status, done = "", None, None

    if intent is None:
        yield _ev("say", "classify", text="잘 못 알아들었어요. 다시 말씀해 주세요.")
        result = "의도 불명"
    elif intent == "stop":
        yield _ev("node", "halt")
        yield _ev("say", "halt", text="멈췄습니다.")
        result = "정지"
    elif intent == "scene":
        yield _ev("node", "scene")
        yield _ev("tool", "scene", status="start", name="detect_objects", args={})
        det = _detect()
        yield _ev("tool", "scene", status="end", name="detect_objects", args={}, result=det)
        yield _ev("say", "scene", text=_scene_answer(text, det))
        result, status = "배치 답", "done"
    elif intent == "question":
        yield _ev("node", "answer")
        yield _ev("tool", "answer", status="start", name="search_manual", args={"query": text})
        t0 = time.perf_counter()
        hits = _search(text)
        ms = (time.perf_counter() - t0) * 1000 + 35
        _stat("검색", ms)
        yield _ev("tool", "answer", status="end", name="search_manual", args={"query": text}, latency_ms=round(ms, 1), device="CPU",
                  result=[{k: h[k] for k in ("id", "title", "score")} | {"doc": "standard", "cite": f"[작업 기준서 {h['id']}]", "text": h["text"][:160]}
                          for h in hits])
        _sleep(0.8 * SPEED / 0.15)
        top = hits[0]
        first = " ".join(re.split(r"(?<=다\.)\s+", top["text"].replace("\n", " "))[:2])
        yield _ev("say", "answer", text=f"[작업 기준서 {top['id']}]에 따르면 {first}",  # 에이전트 10-22 형식(문서 이름 + 조항)
                  sources=[{"doc": "standard", "cite": f"[작업 기준서 {h['id']}]", "section": h["id"]} for h in hits])
        result = f"기준서 {top['id']}"
    elif intent == "report":
        yield _ev("node", "report")
        yield _ev("tool", "report", status="start", name="detect_objects", args={})
        det = _detect()
        yield _ev("tool", "report", status="end", name="detect_objects", args={}, result=det)
        _sleep(0.8 * SPEED / 0.15)
        data = build_report(det)
        from ui.report import render_report

        path = OUT / f"report_{datetime.now():%H%M%S}.html"
        render_report(data, str(path))
        yield _ev("report", "report", html_path=str(path.relative_to(ROOT)), summary=data["summary"], data=data)
        yield _ev("say", "report", text=data["summary"])
        result = "보고서"
    elif intent == "plan":
        steps = make_plan(text)
        yield _ev("node", "make_plan", intent="plan", steps=steps)
        if not steps:
            yield _ev("say", "make_plan", text="할 일을 정하지 못했어요. 다시 말씀해 주세요.")
            result, status = "계획 없음", "nothing"
        else:
            yield _ev("say", "make_plan", text=f"{len(steps)}단계로 할게요: " + ", ".join(s["label"] for s in steps) + ".")
            done = 0
            for i, s in enumerate(steps, 1):
                yield _ev("node", "step", step=i, intent=s["intent"], count=s["count"], status="start")
                status, n = yield from _work(s["intent"], s["count"], step=i)
                done += n
                yield _ev("node", "step", step=i, intent=s["intent"], status=status, done=n)
                if status in ("stopped", "gave_up"):
                    break
            result = f"{done}개"
    else:
        status, done = yield from _work(intent, count)
        result = {"nothing": "대상 없음", "stopped": "정지", "gave_up": "사람 확인"}.get(status, f"{done}개")
    W.commands.append({"ts": datetime.fromtimestamp(t_start).strftime("%H:%M:%S"), "text": text, "intent": intent,
                       "duration_s": round(time.time() - t_start, 1), "result": result})
    yield _ev("node", "end", intent=intent, count=count, status=status or result, done=done, elapsed_s=round(time.time() - t_start, 3))


def build_report(det: dict) -> dict:
    z = det["zones"]
    s = lambda d: sum(d.values())
    rec_defect = W.policies.get("pick_defect", [0, 0])[1]
    rec_red = W.policies.get("pick_red", [0, 0])[1] - W.handed
    rec_blue = W.policies.get("pick_blue", [0, 0])[1]
    att = sum(v[0] for v in W.policies.values())
    suc = sum(v[1] for v in W.policies.values())
    cam_red, cam_blue, cam_def = s(z["bowl_red"]), s(z["bowl_blue"]), s(z["bowl_defect"])
    ok = (cam_def, cam_red, cam_blue) == (rec_defect, rec_red, rec_blue)
    red_txt = f"분홍 {cam_red}개" + (f"({W.handed}개 전달)" if W.handed else "")
    summary = (f"불량 {cam_def}개, {red_txt}, 초록 {cam_blue}개입니다. 시도 {att}회 중 {suc}회 성공, "
               + ("기록과 카메라 개수가 일치합니다." if ok else "기록과 카메라 개수가 다릅니다. 확인해 주세요."))
    devs = [("보기", "YOLO n INT8 (가짜)", "NPU"), ("판단", os.getenv("LLM_MODEL", "Qwen3 INT4"), "GPU"),
            ("검색", "Qwen3-Embedding + Reranker", "CPU"), ("행동", "ACT 정책 4개", "GPU")]
    devices = []
    for layer, model, dev in devs:
        v = W.dev.get(layer, [])
        devices.append({"layer": layer, "model": model, "device": dev, "calls": len(v),
                        "avg_ms": sum(v) / len(v) if v else None, "max_ms": max(v) if v else None})
    return {
        "job_id": "mock-" + W.started.strftime("%H%M%S"),
        "started": W.started.strftime("%Y-%m-%d %H:%M:%S"), "ended": datetime.now().strftime("%H:%M:%S"),
        "duration_s": round((datetime.now() - W.started).total_seconds(), 1),
        "summary": summary, "offline": True,
        "before_image": W.first_snap[0] if W.first_snap else None, "before_ts": W.first_snap[1] if W.first_snap else None,
        "after_image": det["image_path"], "after_ts": datetime.now().strftime("%H:%M:%S"),
        "counts": [
            {"key": "defect", "label": "불량 그릇", "record": rec_defect, "camera": cam_def,
             "detail": f"분홍 {z['bowl_defect']['red_defect']} · 초록 {z['bowl_defect']['blue_defect']}"},
            {"key": "red", "label": "분홍 그릇", "record": rec_red, "camera": cam_red},
            {"key": "blue", "label": "초록 그릇", "record": rec_blue, "camera": cam_blue},
            {"key": "hand", "label": "손 전달", "record": W.handed, "camera": None, "detail": "카메라로 셀 수 없음(기록만)"},
            {"key": "input", "label": "투입 구역 남음", "record": None, "camera": s(z["input"])},
        ],
        "policies": [{"name": k, "attempts": v[0], "success": v[1], "total_s": round(v[2], 1)} for k, v in W.policies.items()],
        "commands": list(W.commands),
        "devices": devices,
        "human_checks": list(W.human_checks),
    }
