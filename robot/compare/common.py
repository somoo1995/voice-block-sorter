"""비교 공통: 관절값(정규화) → 집게 가운데 작업대 좌표, 그리퍼 닫힘 찾기, 장면의 블록·정답·'가장 가까운' 판정."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robot.coord_pick.plan import Setup, load_config

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "robot" / "logs" / "n2"
TARGETS = {"red": ("red_ok",), "green": ("blue_ok",), "defect": ("red_defect", "blue_defect")}
POLICY_TASK = {"act_red": "red", "act_green": "green", "act_defect": "defect", "smolvla_red": "red"}
TASK_OF_DS = {"pick_red": "red", "pick_green": "green", "pick_defect": "defect"}
HIT_MM = 35.0       # 닫은 자리에서 블록 중심까지 이보다 멀면 '허공'(블록 한 변 약 30mm)
KO = {"red_ok": "분홍", "blue_ok": "초록", "red_defect": "분홍 불량", "blue_defect": "초록 불량"}
_S: list[Setup] = []


def setup() -> Setup:
    """config.json 은 읽기만(10-22 오후 실물 보정: base·tcp_mm·호모그래피)."""
    if not _S:
        _S.append(Setup.load(load_config(), use_studio=False))
    return _S[0]


def tcp_w(n) -> np.ndarray:
    """관절값 6개(정규화, 그리퍼 포함 가능) → 집게 가운데의 작업대 좌표(x, y mm)·높이 z(mm)."""
    s = setup()
    p = s.tcp_pos(s.n_to_q(np.asarray(n, float)))
    return np.r_[s.frames.r2w(p[:2]), p[2]]


def closing(g: np.ndarray, rest: float | None = None) -> int | None:
    """그리퍼 값 열 → 닫기 시작 인덱스(녹화 분석 r14 ex2 와 같은 규칙: 쉬는 값 + 6 넘게 연 뒤, 연 최댓값에서 2 넘게 떨어진
    첫 자리). 열지 않았거나 닫지 않았으면 None."""
    g = np.asarray(g, float)
    rest = float(np.median(g[:10])) if rest is None else rest
    i = 0
    while i < len(g) and g[i] < rest + 6:
        i += 1
    if i == len(g):
        return None
    m = g[i]
    while i < len(g) and g[i] >= m - 2:
        m = max(m, g[i])
        i += 1
    return i if i < len(g) else None


def blocks(det_objects: list[dict]) -> list[dict]:
    """검출 objects → 투입 구역 블록 [{cls, px, w(작업대 mm), r(로봇 원점 거리 mm)}]."""
    s = setup()
    out = []
    for o in det_objects:
        if o.get("zone") != "input" or o["cls"] not in KO:
            continue
        w = s.px_to_w(o["px"])[0]
        out.append({"cls": o["cls"], "px": [float(v) for v in o["px"]], "w": w.round(1).tolist(),
                    "r": round(float(np.linalg.norm(s.frames.w2r(w))), 1), "conf": o.get("conf")})
    return sorted(out, key=lambda b: b["r"])


def scene_kind(bl: list[dict], task: str) -> str | None:
    """'쉬움' = 로봇에 가장 가까운 블록이 목표, '방해' = 목표보다 가까운 다른 블록이 있음, None = 목표 없음."""
    tg = [b for b in bl if b["cls"] in TARGETS[task]]
    if not tg:
        return None
    return "쉬움" if bl[0]["cls"] in TARGETS[task] else "방해"


def judge(point_w, bl: list[dict], task: str) -> dict:
    """닫은 자리(작업대 xy) → 가장 가까운 블록·거리·정답 여부·그 블록이 로봇에 가장 가까운 블록인지."""
    if point_w is None or not bl:
        return {"hit": None, "d_mm": None, "ok": False, "nearest_robot": None, "air": point_w is not None}
    p = np.asarray(point_w[:2], float)
    d = [float(np.linalg.norm(np.asarray(b["w"]) - p)) for b in bl]
    k = int(np.argmin(d))
    air = d[k] > HIT_MM
    return {"hit": None if air else bl[k]["cls"], "hit_index": k, "d_mm": round(d[k], 1), "air": air,
            "ok": (not air) and bl[k]["cls"] in TARGETS[task], "nearest_robot": (not air) and k == 0}


def load(name: str):
    return json.loads((OUT / name).read_text())
