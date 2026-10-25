"""위 카메라 사진 → YOLO 라벨 초안(HSV 규칙). 사람은 review.py 로 검수만 하고 바로 train.py.

  python vision/autolabel.py vision/data/raw/20261022 --name day1 [--marker black|white|yellow|H,S,V,H,S,V]
      → vision/labels_draft/day1/draft.json  (박스·자신 없는 표시·상태)
      → vision/datasets/auto_day1/            (train.py --dir 로 바로 학습. --zip 이면 Roboflow 업로드용 zip 도)
  python vision/review.py day1                         # 검수 화면 http://127.0.0.1:8791 (손 박스는 여기서 그린다)
      같은 이름으로 다시 돌리면 검수한 사진은 그대로 두고 새 사진·미검수 사진만 라벨한다(--fresh 면 전부)
  python vision/autolabel.py --export day1 [--zip]     # review.py 로 고친 뒤 다시 내보내기 (review.py 버튼과 같음)
  python vision/autolabel.py --calibrate day1          # 검수(통과)한 실물 사진으로 임계값 → vision/autolabel.json
  python vision/autolabel.py --eval vision/_test/w_black.zip [--marker black]   # 정답 라벨과 비교 (정밀도·재현율)

규칙:
  블록  = 분홍(red_*)/초록(blue_*) HSV 마스크 덩어리. 볼록 껍질로 박스. 정사각형에 가깝고 볼록하면 한 개, 아니면 맞닿은 블록으로 보고
          면적으로 개수를 어림해 나누고, 나눈 조각이 블록 모양·크기가 아니면 'split' 표시.
  불량  = 블록 안쪽(가장자리 15% 깎음)에서 마커 색 픽셀 비율 ≥ defect_ratio. 경계 구간이면 'defect?' 표시.
  손    = 사람이 review.py 에서 그린다(10-21 확정). 기본(hands=hint)은 피부색 덩어리를 'hand?' 제안 박스로만
          넣어 그 사진을 검수로 보낸다. --hands off 면 제안도 없음, auto 면 hand_spot 안 덩어리를 그대로 라벨.
  구역  = zones.json 이 있으면 5구역 어디에도 중심이 없는 블록 후보는 버리고 수를 경고한다(로봇 몸체·의자·접시).
          로봇·접시에 붙어 같이 버려질 블록은 구역 안 조각만 'zone' 표시로 남긴다. --keep-outside 로 끈다.
  자신 없는 박스가 하나라도 있는 사진은 status=review → 검수 전에는 데이터셋에서 빠진다.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

VISION = Path(__file__).resolve().parent
DRAFTS = VISION / "labels_draft"
DATASETS = VISION / "datasets"
CONFIG = VISION / "autolabel.json"
NAMES = ["blue_defect", "blue_ok", "hand", "red_defect", "red_ok"]  # Roboflow 와 같은 알파벳순
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# 이전 진한 빨강·파랑 블록/볼(합성 비교용): --colors strong
STRONG = {"red": {"h_lo": 10, "h_hi": 165, "s_min": 110, "v_min": 50},
          "blue": {"h_min": 95, "h_max": 130, "s_min": 90, "v_min": 40}}

DEFAULTS = {
    # OpenCV HSV (H 0~179). 키 red/blue 는 클래스 식별자: red_* = 분홍, blue_* = 초록 (10-21 밤 파스텔 스펀지).
    # 분홍은 179 를 넘어 0 으로 이어지는 두 구간(h ≤ h_lo 또는 h ≥ h_hi). 파스텔은 채도가 낮아(합성 59~123) s_min 이 낮다.
    # h_hi 125: 10-22 실물 분홍 스펀지(4600K)는 색상 중앙값 148·하위 5% 119 로 자홍 쪽이다(합성은 167~169). 예전 150 은
    # 블록 픽셀의 1/3 만 잡아 블록이 조각나거나 통째로 빠졌다. 푸른 식탁보는 색상 98~100 이라 안 걸린다.
    "red": {"h_lo": 4, "h_hi": 125, "s_min": 35, "v_min": 80},
    "blue": {"h_min": 30, "h_max": 90, "s_min": 35, "v_min": 80},
    "markers": {  # [[H,S,V 하한], [H,S,V 상한]]
        "black": [[0, 0, 0], [179, 255, 70]],
        "white": [[0, 0, 170], [179, 50, 255]],
        "yellow": [[18, 90, 120], [35, 255, 255]],
    },
    "marker": "black",
    "defect_ratio": 0.05,          # 이 비율 이상이면 불량
    "black_rel": 0.45,             # 검정 마커는 그 블록 밝기(중앙값)의 이 배보다 어두운 픽셀만 — 그늘진 옆면·기공 제외
    "defect_band": [0.025, 0.08],  # 이 구간이면 'defect?'
    "min_size_px": 14,             # 한 변이 이보다 작은 덩어리는 무시 (3cm 블록이 몇 px 인지는 calibrate 가 잼)
    "obj_area_px": None,           # 블록 하나가 보이는 면적(px², 옆면 포함). None 이면 사진마다 중앙값으로 어림
    # 블록 하나의 모양(덩어리 볼록도, 회전 사각형 가로세로비): min/max 밖이면 블록 하나로 보지 않고, odd 밖이면 'irregular'.
    # 기본값은 합성(위에서 본 블록) 기준. 10-22 실물 카메라는 비스듬해 윗면·앞면이 같이 보여 가로세로비 중앙값 1.52·
    # 볼록도 중앙값 0.91 이라 블록 대부분이 'irregular' 가 된다 → calibrate 가 검수한 블록으로 다시 잰다.
    "shape": {"solid_min": 0.88, "aspect_max": 1.6, "solid_odd": 0.92, "aspect_odd": 1.4},
    "skin_ycrcb": [[0, 135, 85], [255, 175, 135]],
    "skin_s_min": 40,              # HSV 채도 하한 — 따뜻한 조명의 흰 작업대가 피부색 범위에 드는 것을 막는다
    "skin_h": [[0, 25], [172, 179]],  # 피부 색상 구간. 분홍 스펀지(합성 H 167~169)도 YCrCb 로는 피부라 색상으로 가른다
    "hand_min_area": 1500,
    "hand_zone_frac": 0.5,         # 덩어리의 이 비율 이상이 hand_spot 안이어야 자신 있음 (hands=auto 일 때만)
    "hands": "hint",               # hint = 제안 박스('hand?')만, off = 없음, auto = 자신 있으면 라벨
    "bridge_px": 5,                # 마커 선이 블록 색 마스크를 끊어도 한 덩어리로 묶는 거리
    "close_px": 7,
    # zones.json 이 있으면 5구역 어디에도 중심이 없는 블록 후보는 버린다. 10-22 실물: 카메라를 더 못 숙여 화면 위
    # 25% 는 의자·바닥, 아래는 흰 로봇 몸체·접시가 들어오고, 흰 부품이 초록 마스크에 넓게 걸린다. --keep-outside 로 끈다.
    "zones_only": True,
    "zone_tol_px": 10,             # 중심이 구역 밖 이 거리 안이면 남긴다(반쯤 걸친 블록이 박스 중심 몇 px 차이로 빠지지 않게)
    "zone_margin_px": 40,          # 구역 둘레 이 거리 밖의 색 픽셀은 덩어리를 찾기 전에 지운다(경계에 걸친 블록은 온전히 남는다)
}
SHOWN = {"red": "분홍", "blue": "초록"}  # 사람이 보는 색 이름


# ---------------------------------------------------------------- 설정

def merge_config(cfg: dict, d: dict) -> dict:
    """설정 d 를 cfg 위에 덮은 사본(dict 값은 키별로 덮는다)."""
    out = copy.deepcopy(cfg)
    for k, v in d.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def load_config(path: Path = CONFIG, marker: str | None = None, colors: str = "pastel") -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if colors == "strong":
        cfg.update(copy.deepcopy(STRONG))
    if path and Path(path).exists():
        cfg = merge_config(cfg, json.loads(Path(path).read_text()))
    if marker:
        if marker in cfg["markers"]:
            cfg["marker"] = marker
        else:  # H,S,V,H,S,V
            v = [int(x) for x in marker.split(",")]
            if len(v) != 6:
                sys.exit("--marker 는 black|white|yellow 또는 H,S,V,H,S,V")
            cfg["markers"]["custom"] = [v[:3], v[3:]]
            cfg["marker"] = "custom"
    return cfg


def load_zone_polys(path) -> dict | None:
    if not path or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    return {"image_size": d.get("image_size"), "zones": {k: np.array(v, np.int32) for k, v in d["zones"].items()}}


def zone_mask(zones: dict | None, name: str, shape) -> np.ndarray | None:
    if not zones or name not in zones["zones"]:
        return None
    h, w = shape[:2]
    poly = zones["zones"][name].astype(np.float32)
    if zones["image_size"] and tuple(zones["image_size"]) != (w, h):
        poly = poly * (w / zones["image_size"][0], h / zones["image_size"][1])
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly.round().astype(np.int32)], 255)
    return m


def zone_keep(zones: dict | None, cfg, shape) -> dict | None:
    """구역 합집합 마스크 셋: exact(그대로), center(박스 중심 판정용, zone_tol_px 넓힘), near(덩어리 찾기용, zone_margin_px 넓힘).
    구역이 없거나 zones_only 가 꺼져 있으면 None."""
    if not zones or not zones["zones"] or not cfg.get("zones_only", True):
        return None
    keep = np.zeros(shape[:2], np.uint8)
    for name in zones["zones"]:
        keep |= zone_mask(zones, name, shape)
    return {"exact": keep, "center": cv2.dilate(keep, _k(2 * cfg["zone_tol_px"] + 1)),
            "near": cv2.dilate(keep, _k(2 * cfg["zone_margin_px"] + 1))}


def _center_in(keep, xyxy) -> bool:
    h, w = keep.shape
    x1, y1, x2, y2 = xyxy
    return keep[min(h - 1, (y1 + y2) // 2), min(w - 1, (x1 + x2) // 2)] > 0


# ---------------------------------------------------------------- 마스크

def _k(n: int):
    n = max(1, int(n))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def masks(img, cfg) -> dict:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    r, b = cfg["red"], cfg["blue"]
    red = ((h <= r["h_lo"]) | (h >= r["h_hi"])) & (s >= r["s_min"]) & (v >= r["v_min"])
    blue = (h >= b["h_min"]) & (h <= b["h_max"]) & (s >= b["s_min"]) & (v >= b["v_min"])
    lo, hi = cfg["markers"][cfg["marker"]]
    marker = cv2.inRange(hsv, np.array(lo), np.array(hi)) > 0
    ylo, yhi = cfg["skin_ycrcb"]
    skin = (cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb), np.array(ylo), np.array(yhi)) > 0) \
        & (s >= cfg["skin_s_min"]) & np.any([(h >= a) & (h <= b) for a, b in cfg["skin_h"]], axis=0)
    out = {}
    for name, m in (("red", red), ("blue", blue)):
        out[name] = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, _k(3))
    out["marker"] = marker.astype(np.uint8) * 255
    out["v"] = v
    out["skin"] = skin.astype(np.uint8) * 255
    return out


# ---------------------------------------------------------------- 블록

def _touches_edge(x1, y1, x2, y2, w, h) -> bool:
    return x1 <= 1 or y1 <= 1 or x2 >= w - 2 or y2 >= h - 2


def marker_ratio(inner, marker, v, v_obj: float, cfg) -> float:
    """안쪽 영역에서 마커 픽셀 비율. 검정은 블록 밝기 대비로 한 번 더 거른다(JPEG 때문에 채도로는 못 거름)."""
    n_in = int(np.count_nonzero(inner))
    if not n_in:
        return 0.0
    m = (marker > 0) & (inner > 0)
    if cfg["marker"] == "black":
        m &= v < cfg["black_rel"] * v_obj
    return float(np.count_nonzero(m)) / n_in


def _box_from_pixels(pts: np.ndarray, marker, v, cfg, shape, flags: list[str]) -> dict:
    """색 픽셀 좌표 → 볼록 껍질 → 박스, 불량 비율."""
    h, w = shape[:2]
    hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32))
    x, y, bw, bh = cv2.boundingRect(hull)
    region = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(region, hull, 255)
    area = max(1.0, float(cv2.contourArea(hull)))
    r_eq = (area / np.pi) ** 0.5
    inner = cv2.erode(region, _k(2 * max(1, int(0.15 * r_eq)) + 1))
    ratio = marker_ratio(inner, marker, v, float(np.median(v[pts[:, 1], pts[:, 0]])), cfg)
    flags = list(flags)
    lo, hi = cfg["defect_band"]
    if lo <= ratio <= hi:
        flags.append("defect?")
    if _touches_edge(x, y, x + bw, y + bh, w, h):
        flags.append("edge")
    if area ** 0.5 < 1.3 * cfg["min_size_px"]:
        flags.append("small")
    return {"xyxy": [int(x), int(y), int(x + bw), int(y + bh)], "ratio": round(ratio, 4),
            "defect": ratio >= cfg["defect_ratio"], "flags": flags, "area": area}


def _shape(points) -> tuple[float, float, float]:
    """(껍질 면적, 볼록도 = 면적/껍질 면적, 가로세로비 = 최소 회전 사각형의 긴 변/짧은 변)."""
    points = points.reshape(-1, 1, 2).astype(np.int32)
    hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(points))))
    (_, _), (rw, rh), _ = cv2.minAreaRect(points)
    return hull_area, float(cv2.contourArea(points)) / hull_area, max(rw, rh) / max(1.0, min(rw, rh))


# 블록 하나의 모양(cfg["shape"]): 볼록하고 정사각형에 가깝다. 화면 가장자리에서는 옆면이 보여 가로세로비가 1.5 쯤까지
# 늘어난다. 맞닿은 블록 2개는 나란하면 가로세로비 ~2, 비스듬하면 볼록도가 떨어진다. 면적이 블록 하나의 PAIR_AREA 배를
# 넘으면 모양과 상관없이 여러 개로 본다.
PAIR_AREA = 1.7


def find_components(color_mask, marker, cfg, shape, zk=None) -> list[dict]:
    """한 색 마스크의 덩어리(마커 선이 끊은 곳은 이어 붙임)에 모양·구역 판정을 붙여 돌려준다."""
    h, w = shape[:2]
    near = cv2.dilate(color_mask, _k(2 * cfg["bridge_px"] + 1))
    bridged = cv2.morphologyEx(color_mask | (marker & near), cv2.MORPH_CLOSE, _k(cfg["close_px"]))
    contours, _ = cv2.findContours(bridged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = cfg["min_size_px"] ** 2 * 0.5
    sh = cfg["shape"]
    found = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        area, solid, aspect = _shape(c)
        comp = np.zeros((h, w), np.uint8)
        cv2.drawContours(comp, [c], -1, 255, -1)
        ys, xs = np.nonzero(comp & color_mask)
        if len(xs) < min_area * 0.5:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        found.append({"area": area, "solid": solid, "aspect": aspect,
                      "shape_ok": solid >= sh["solid_min"] and aspect <= sh["aspect_max"],
                      "odd": solid < sh["solid_odd"] or aspect > sh["aspect_odd"], "pts": np.stack([xs, ys], 1), "mask": comp,
                      "inside": zk is None or _center_in(zk["center"], (x, y, x + bw, y + bh)),
                      "n_in": len(xs) if zk is None else int(np.count_nonzero(zk["exact"][ys, xs]))})
    return found


def block_area(found: list[dict], cfg) -> float | None:
    """블록 하나가 보이는 면적: 설정값(calibrate), 없으면 구역 안 블록 모양 덩어리의 면적 중앙값. 두 색을 합쳐 잰다 —
    블록 크기는 색과 상관없다. 10-22 실물은 분홍에 블록 모양 덩어리가 하나도 없는 사진이 많았고, 그때마다 분홍 덩어리를
    전부 둘로 잘라 블록 하나가 박스 2개가 됐다. 모르면 None(자르지 않고 'irregular' 로 검수에 보낸다)."""
    if cfg.get("obj_area_px"):
        return float(cfg["obj_area_px"])
    singles = [c["area"] for c in found if c["shape_ok"] and c["inside"]]
    return float(np.median(singles)) if singles else None


def find_objects(found, color_mask, marker, v, cfg, shape, ref, zk=None) -> tuple[list[dict], list[str], int]:
    """한 색의 블록들(found = find_components, ref = block_area). zk(zone_keep)를 주면 구역 밖 후보는 버린다.
    반환: (블록 목록, 사진 단위 경고, 버린 수). 블록 크기 덩어리는 박스 중심으로, 블록 2개 이상 크기는 픽셀이 절반 넘게
    구역 밖이면 버린다 — 로봇 몸체 띠와 접시가 이어진 큰 덩어리는 박스 중심이 구역 안에 떨어지기도 한다.
    버린 큰 덩어리의 구역 안 조각은 'zone' 표시로 남긴다."""
    min_area = cfg["min_size_px"] ** 2 * 0.5
    comps, n_out, warnings = [], 0, []
    for c in found:
        if zk is not None:
            big = c["area"] >= PAIR_AREA * (ref or 4 * min_area)
            if (c["n_in"] < 0.5 * len(c["pts"])) if big else not c["inside"]:
                n_out += 1
                if big:  # 로봇·접시에 블록이 붙어 한 덩어리가 된 경우: 구역 안 조각만 'zone' 표시로 남겨 검수로 보낸다
                    comps += _zone_parts(c, color_mask, zk["exact"], cfg, min_area)
                continue
        comps.append(c)
    objs = []
    for c in comps:
        single = c["shape_ok"] and (ref is None or c["area"] / ref < PAIR_AREA)
        # 모양이 어긋나도 면적이 블록 하나쯤이면 나누지 않고 'irregular' 로 검수에 보낸다(원근으로 길쭉해진 블록).
        # 블록 크기를 모르면(ref None) 몇 개인지 모르므로 자르지 않는다
        k = 1 if single else max(1, round(c["area"] / ref)) if ref else 1
        if k == 1 and not single:
            c["odd"] = True
        if k > 8:
            warnings.append(f"큰 색 영역 무시(블록 {k}개 크기) — 분홍/초록 그릇·물건?")
            continue
        zone = ["zone"] if c.get("zone") else []
        if k == 1:
            objs.append(_box_from_pixels(c["pts"], marker, v, cfg, shape, (["irregular"] if c["odd"] else []) + zone))
            continue
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, lab, _ = cv2.kmeans(c["pts"].astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        subs = [c["pts"][lab.ravel() == i] for i in range(k)]
        sure = ref is not None and all(len(sub) >= 10 and _piece_ok(sub, ref, c["mask"], cfg["shape"]["aspect_max"])
                                       for sub in subs)
        for sub in subs:
            if len(sub) >= 10:
                b = _box_from_pixels(sub, marker, v, cfg, shape, ([] if sure else ["split"]) + zone)
                if zk is not None and not _center_in(zk["center"], b["xyxy"]):
                    n_out += 1
                    continue
                objs.append(b)
    return objs, warnings, n_out


def _zone_parts(c: dict, color_mask, exact, cfg, min_area: float) -> list[dict]:
    """구역 밖 큰 덩어리에서 구역 안 조각만 떼어 블록 후보로(표시 'zone', 모양 판정 없이 검수로). 블록 최소 크기보다
    가는 조각은 구역 경계를 손으로 찍은 오차로 보고 버린다. 구역이 로봇·접시를 덮으면 큰 조각이 나와 '큰 색 영역' 경고가 된다."""
    contours, _ = cv2.findContours(c["mask"] & exact, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for p in contours:
        _, _, bw, bh = cv2.boundingRect(p)
        if cv2.contourArea(p) < min_area or min(bw, bh) < cfg["min_size_px"]:
            continue
        m = np.zeros_like(exact)
        cv2.drawContours(m, [p], -1, 255, -1)
        ys, xs = np.nonzero(m & color_mask)
        if len(xs) < min_area * 0.5:
            continue
        out.append({"area": _shape(p)[0], "shape_ok": False, "odd": True, "zone": True,
                    "pts": np.stack([xs, ys], 1), "mask": m})
    return out


def _piece_ok(pts: np.ndarray, ref: float, comp_mask, aspect_max: float) -> bool:
    """나눈 조각이 블록 하나처럼 생겼나: 정사각형에 가깝고(가로세로비 ≤ aspect_max) 면적이 기준의 0.6~1.8배이며,
    조각의 껍질이 덩어리 안에 꽉 차 있다(≥ 0.9). 두 블록에 걸쳐 잘못 나눈 조각은 껍질이 블록 사이 빈 곳까지 덮는다."""
    hull = cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32))
    area, _, aspect = _shape(hull)
    m = np.zeros_like(comp_mask)
    cv2.fillConvexPoly(m, hull, 255)
    dense = np.count_nonzero(m & comp_mask) / max(1, np.count_nonzero(m))
    return aspect <= aspect_max and 0.6 <= area / ref <= 1.8 and dense >= 0.9


# ---------------------------------------------------------------- 손

def find_hands(skin, obj_mask, zones, cfg, shape) -> list[dict]:
    if cfg["hands"] == "off":
        return []
    h, w = shape[:2]
    m = skin & ~cv2.dilate(obj_mask, _k(7))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, _k(5)), cv2.MORPH_CLOSE, _k(9))
    spot = zone_mask(zones, "hand_spot", shape)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hands = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < cfg["hand_min_area"] or area > 0.35 * w * h:  # 화면 1/3 넘는 피부색 = 작업대·조명 색
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        flags = []
        if spot is not None:
            comp = np.zeros((h, w), np.uint8)
            cv2.drawContours(comp, [c], -1, 255, -1)
            frac = np.count_nonzero(comp & spot) / max(1, np.count_nonzero(comp))
            if frac < 0.15:
                continue  # hand_spot 밖 피부색은 손 후보가 아님
            if frac < cfg["hand_zone_frac"]:
                flags.append("hand?")
        else:
            flags.append("hand?")  # 구역을 모르면 모두 검수
        if _touches_edge(x, y, x + bw, y + bh, w, h):
            flags.append("arm?")   # 화면 끝에 닿음 → 팔까지 박스에 들어갔을 수 있음
        if area > 0.2 * w * h or cfg["hands"] == "hint":
            flags.append("hand?")  # hint: 손은 사람이 확정하므로 전부 제안
        hands.append({"cls": "hand", "xyxy": [int(x), int(y), int(x + bw), int(y + bh)],
                      "flags": sorted(set(flags)), "ratio": None})
    return hands


# ---------------------------------------------------------------- 한 장

def label_image(img, cfg, zones=None, info: dict | None = None) -> tuple[list[dict], list[str]]:
    """info 를 주면 info["outside"] 에 구역 밖이라 버린 블록 후보 수를 넣는다(경고만, 사진을 검수로 보내지는 않는다)."""
    cv2.setRNGSeed(0)  # 맞닿은 블록 k-means 시작점 고정 → 같은 사진은 다시 돌려도 같은 박스
    mk = masks(img, cfg)
    zk = zone_keep(zones, cfg, img.shape)
    boxes, warnings, n_out = [], [], 0
    cms = {color: mk[color] if zk is None else mk[color] & zk["near"] for color in ("red", "blue")}
    found = {color: find_components(cm, mk["marker"], cfg, img.shape, zk) for color, cm in cms.items()}
    ref = block_area(found["red"] + found["blue"], cfg)
    for color, cm in cms.items():
        objs, warn, n = find_objects(found[color], cm, mk["marker"], mk["v"], cfg, img.shape, ref, zk)
        warnings += warn
        n_out += n
        for b in objs:
            boxes.append({"cls": f"{color}_{'defect' if b['defect'] else 'ok'}", "xyxy": b["xyxy"],
                          "flags": b["flags"], "ratio": b["ratio"]})
    boxes += find_hands(mk["skin"], mk["red"] | mk["blue"], zones, cfg, img.shape)
    if info is not None:
        info["outside"] = n_out
    return boxes, warnings


def draw_boxes(img, boxes):
    colors = {"red_ok": (203, 102, 255), "red_defect": (140, 0, 170), "blue_ok": (60, 200, 60),
              "blue_defect": (0, 110, 0), "hand": (0, 170, 255)}
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        c = colors[b["cls"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 1 if not b["flags"] else 2)
        txt = b["cls"] + (f" {b['ratio']:.2f}" if b.get("ratio") is not None else "")
        if b["flags"]:
            txt += " " + ",".join(b["flags"])
            cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 255), 1)
        cv2.putText(out, txt, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    return out


# ---------------------------------------------------------------- 초안·내보내기

def draft_path(name: str) -> Path:
    return DRAFTS / name / "draft.json"


def save_draft(draft: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(draft, ensure_ascii=False, indent=1))
    tmp.replace(path)


def run(src: Path, name: str, cfg: dict, zones_path, debug: bool, fresh: bool = False) -> dict:
    """사진 폴더 → 초안. 같은 이름의 초안이 있으면 사람이 손댄 사진(통과·제외·검수 화면에서 저장)은 그대로 두고
    나머지만 다시 라벨한다 — 녹화가 끝날 때마다 사진을 같은 폴더에 더하고 다시 돌리는 방법 B 용. fresh 면 전부 다시."""
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"사진 없음: {src}")
    path = draft_path(name)
    kept = {}
    if path.exists():
        old = json.loads(path.read_text())
        if any(im["status"] in ("pass", "exclude") or im.get("reviewed") for im in old["images"]):
            bak = path.with_name(f"draft.json.bak-{datetime.now():%Y%m%d_%H%M%S}")
            shutil.copy2(path, bak)
            print(f"검수한 초안을 {bak.name} 로 백업했다")
        if not fresh:
            kept = {im["stem"]: im for im in old["images"] if im["status"] in ("pass", "exclude") or im.get("reviewed")}
    zones = load_zone_polys(zones_path)
    if zones is None:
        print(f"⚠ {zones_path} 없음 — 손은 화면 전체에서 찾고 모두 'hand?' 로 표시, 구역 밖 블록 후보도 버리지 않는다")
    elif not cfg.get("zones_only", True):
        print("구역 밖 블록 후보도 남긴다(--keep-outside)")
    out_dir = DRAFTS / name
    if debug:
        (out_dir / "debug").mkdir(parents=True, exist_ok=True)
    images = []
    out_total = out_imgs = 0
    for p in files:
        if p.stem in kept:  # 사람이 손댄 사진은 다시 라벨하지 않는다
            images.append(kept[p.stem])
            continue
        img = cv2.imread(str(p))
        if img is None:
            print("못 읽음:", p)
            continue
        info = {}
        boxes, warnings = label_image(img, cfg, zones, info)
        reasons = sorted({f for b in boxes for f in b["flags"]}) + warnings
        if not any(b["cls"] != "hand" for b in boxes):
            reasons.append("블록 0개")
        images.append({"stem": p.stem, "path": str(p.resolve()), "w": img.shape[1], "h": img.shape[0],
                       "status": "review" if reasons else "auto", "reasons": reasons, "boxes": boxes,
                       "outside": info["outside"]})
        out_total += info["outside"]
        out_imgs += info["outside"] > 0
        if debug:
            cv2.imwrite(str(out_dir / "debug" / f"{p.stem}.jpg"), draw_boxes(img, boxes))
    draft = {"name": name, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
             "source": str(src.resolve()), "zones": str(zones_path) if zones else None,
             "marker": cfg["marker"], "config": cfg, "names": NAMES, "images": images}
    if kept:
        print(f"사람이 손댄 사진 {sum(im['stem'] in kept for im in images)}장은 그대로 두고 나머지를 라벨했다")
    if out_total:
        print(f"⚠ 구역 밖 블록 후보 {out_total}개를 버렸다(사진 {out_imgs}장) — 로봇 몸체·의자·접시 등. "
              "구역 밖에 진짜 블록이 있던 사진은 review.py 에서 직접 그린다. 끄려면 --keep-outside")
    save_draft(draft, path)
    return draft


def summarize(draft: dict) -> dict:
    st = {}
    for im in draft["images"]:
        st[im["status"]] = st.get(im["status"], 0) + 1
    boxes = {n: 0 for n in NAMES}
    flagged = 0
    for im in draft["images"]:
        for b in im["boxes"]:
            boxes[b["cls"]] += 1
            flagged += bool(b["flags"])
    return {"images": len(draft["images"]), "status": st, "boxes": boxes, "flagged_boxes": flagged,
            "outside_dropped": sum(im.get("outside", 0) for im in draft["images"])}


def yolo_txt(boxes, w, h) -> str:
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = (max(0, min(v, lim)) for v, lim in zip(b["xyxy"], (w, h, w, h)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{NAMES.index(b['cls'])} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def export(name: str, make_zip: bool = False, include_review: bool = False, val_pct: int = 20) -> dict:
    """draft.json → Roboflow 형식 폴더(train/valid/data.yaml). auto·pass 만 넣는다(review 는 검수 전이라 뺀다)."""
    draft = json.loads(draft_path(name).read_text())
    out = DATASETS / f"auto_{name}"
    if out.exists():
        shutil.rmtree(out)  # 이 도구가 만드는 폴더라 매번 새로 만든다
    keep = {"auto", "pass"} | ({"review"} if include_review else set())
    counts = {"train": 0, "valid": 0, "skipped": 0}
    boxes = {n: 0 for n in NAMES}
    for im in draft["images"]:
        if im["status"] not in keep:
            counts["skipped"] += 1
            continue
        # 사진 이름의 해시로 나눈다 → 다시 내보내도 같은 사진은 같은 쪽
        split = "valid" if int(hashlib.md5(im["stem"].encode()).hexdigest()[:8], 16) % 100 < val_pct else "train"
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        src = Path(im["path"])
        shutil.copy2(src, out / split / "images" / src.name)
        (out / split / "labels" / f"{src.stem}.txt").write_text(yolo_txt(im["boxes"], im["w"], im["h"]))
        counts[split] += 1
        for b in im["boxes"]:
            boxes[b["cls"]] += 1
    if not counts["valid"] and counts["train"]:
        print("⚠ valid 0장 — 사진이 너무 적다")
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\n\n"
                                   f"nc: {len(NAMES)}\nnames: {NAMES}\n\n"
                                   f"# vision/autolabel.py --export {name} ({datetime.now():%Y-%m-%d %H:%M})\n")
    res = {"dir": str(out), **counts, "boxes": boxes}
    if make_zip:
        zp = DATASETS / f"auto_{name}.zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out))
        res["zip"] = str(zp)
    return res


# ---------------------------------------------------------------- 평가

def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(pred: list[dict], gt: list[dict], thr: float = 0.5) -> dict:
    """클래스별 TP/FP/FN. 같은 클래스끼리 IoU 큰 순서로 짝짓는다."""
    res = {n: {"tp": 0, "fp": 0, "fn": 0} for n in NAMES}
    for n in NAMES:
        P = [p for p in pred if p["cls"] == n]
        G = [g for g in gt if g["cls"] == n]
        pairs = sorted(((iou(p["xyxy"], g["xyxy"]), i, j) for i, p in enumerate(P) for j, g in enumerate(G)),
                       reverse=True)
        used_p, used_g = set(), set()
        for v, i, j in pairs:
            if v < thr:
                break
            if i not in used_p and j not in used_g:
                used_p.add(i)
                used_g.add(j)
        res[n]["tp"] += len(used_p)
        res[n]["fp"] += len(P) - len(used_p)
        res[n]["fn"] += len(G) - len(used_g)
    return res


def read_yolo(txt: Path, w: int, h: int, names: list[str]) -> list[dict]:
    out = []
    if txt.exists():
        for ln in txt.read_text().splitlines():
            p = ln.split()
            if len(p) == 5:
                c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                out.append({"cls": names[c], "xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                                                      (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return out


OBJ_CLASSES = [n for n in NAMES if n != "hand"]


def score_table(total: dict) -> tuple[dict, str]:
    """클래스별 + '블록'(4클래스 합) + '전체'(손 포함 5클래스 합)."""
    rows, lines = {}, ["| 클래스 | 정답 | 맞음 | 틀린 박스 | 놓침 | 정밀도 | 재현율 |", "|---|---:|---:|---:|---:|---:|---:|"]
    groups = {"블록": OBJ_CLASSES, "전체": NAMES}
    for n in NAMES + list(groups):
        t = total[n] if n in total else {k: sum(total[m][k] for m in groups[n]) for k in ("tp", "fp", "fn")}
        p = t["tp"] / max(1, t["tp"] + t["fp"])
        r = t["tp"] / max(1, t["tp"] + t["fn"])
        rows[n] = {**t, "P": round(p, 3), "R": round(r, 3)}
        lines.append(f"| {n} | {t['tp'] + t['fn']} | {t['tp']} | {t['fp']} | {t['fn']} | {p:.3f} | {r:.3f} |")
    return rows, "\n".join(lines)


def evaluate(src: Path, cfg: dict, zones_path=None) -> dict:
    """정답 라벨이 있는 YOLO 데이터셋(zip 또는 폴더)으로 자동 라벨을 채점."""
    import yaml

    if src.suffix == ".zip":
        root = VISION / "_test" / f"eval_{src.stem}"
        if not root.exists():
            with zipfile.ZipFile(src) as z:
                z.extractall(root)
    else:
        root = src
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    zones = load_zone_polys(zones_path)
    total = {n: {"tp": 0, "fp": 0, "fn": 0} for n in NAMES}
    n_img = n_review = n_flag = n_box = 0
    flagged_wrong = wrong = 0
    by_flag: dict[str, dict[str, int]] = {}  # 표시별 맞은/틀린 박스 수 → 어느 표시가 검수를 늘리는지
    review_obj = hand_imgs = hand_caught = n_out = 0
    for img_path in sorted(root.glob("*/images/*")):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt = read_yolo(img_path.parent.parent / "labels" / f"{img_path.stem}.txt", w, h, names)
        info = {}
        pred, warnings = label_image(img, cfg, zones, info)
        n_out += info["outside"]
        m = match(pred, gt)
        for n in NAMES:
            for k in ("tp", "fp", "fn"):
                total[n][k] += m[n][k]
        n_img += 1
        n_box += len(pred)
        n_flag += sum(bool(p["flags"]) for p in pred)
        n_review += bool(warnings or any(p["flags"] for p in pred))
        review_obj += bool(warnings or any(p["flags"] for p in pred if p["cls"] != "hand"))
        if any(g["cls"] == "hand" for g in gt):
            hand_imgs += 1
            hand_caught += any(p["cls"] == "hand" for p in pred)
        # 틀린 박스 중 몇 개가 '자신 없음' 표시였나 (검수에서 잡히는 비율)
        for p in pred:
            ok = any(g["cls"] == p["cls"] and iou(p["xyxy"], g["xyxy"]) >= 0.5 for g in gt)
            if not ok:
                wrong += 1
                flagged_wrong += bool(p["flags"])
            for f in p["flags"]:
                by_flag.setdefault(f, {"ok": 0, "wrong": 0})["ok" if ok else "wrong"] += 1
    rows, md = score_table(total)
    return {"images": n_img, "review_images": n_review, "review_by_object": review_obj,
            "hand_images": hand_imgs, "hand_images_proposed": hand_caught, "boxes": n_box, "flagged_boxes": n_flag,
            "wrong_boxes": wrong, "wrong_flagged": flagged_wrong, "outside_dropped": n_out, "by_flag": by_flag,
            "rows": rows, "table": md}


# ---------------------------------------------------------------- 보정

def _table_color(ab, bgm) -> tuple[np.ndarray, float]:
    """사진 전체 배경(구역 둘레 안, 박스 밖)의 대표 색(a·b 중앙값 = 식탁보)과, 그 색으로 볼 거리(중앙값 + 6·MAD, 최소 8)."""
    px = ab[bgm]
    ref = np.median(px, axis=0)
    d = np.linalg.norm(px - ref, axis=1)
    med = float(np.median(d))
    return ref, max(8.0, med + 6 * 1.4826 * float(np.median(np.abs(d - med))))


def _object_pixels(ab, xyxy, bgm, table=None) -> np.ndarray:
    """박스 안에서 박스 둘레 배경과 색(Lab 의 a·b)이 다른 픽셀(bool). 채도로 가르면 안 된다: 10-22 실물(4600K)은
    식탁보가 푸르게 찍혀 채도 56 으로 파스텔 블록(분홍 45)보다 높다. 밝기는 빼서 그늘이 블록으로 잡히지 않게 한다.
    table(_table_color)을 주면 둘레 중 식탁보 색에 가까운 픽셀만 배경으로 쓴다 — 10-22 녹화 사진은 투입 구역 가장자리
    블록의 둘레 절반 이상이 주황 테이프라 기준이 테이프 색이 되고 문턱이 70 대로 뛰어 블록 픽셀을 못 찾았다.
    둘레에 식탁보가 거의 없으면(그릇 안) 둘레 전체를 쓴다."""
    h, w = bgm.shape
    x1, y1, x2, y2 = xyxy
    pad = 10
    ys, xs = slice(max(0, y1 - pad), min(h, y2 + pad + 1)), slice(max(0, x1 - pad), min(w, x2 + pad + 1))
    inside = np.zeros((h, w), bool)
    inside[y1:y2 + 1, x1:x2 + 1] = True
    ring = bgm[ys, xs] & ~inside[ys, xs]
    ref_px = ab[ys, xs][ring] if ring.sum() >= 30 else ab[bgm]  # 둘레가 다른 박스로 막히면 사진 전체 배경
    if table is not None:
        cloth = ref_px[np.linalg.norm(ref_px - table[0], axis=1) <= table[1]]
        if len(cloth) >= 30:
            ref_px = cloth
    ref = np.median(ref_px, axis=0)
    d = np.linalg.norm(ab[ys, xs] - ref, axis=2)
    thr = max(6.0, float(np.percentile(np.linalg.norm(ref_px - ref, axis=1), 99)) + 3)
    out = np.zeros((h, w), bool)
    out[ys, xs] = (d > thr) & inside[ys, xs]
    return out


BG_HIT_MAX = 0.005  # 구역 안 배경 중 새 색 규칙에 걸려도 되는 비율(사진 한 장이면 블록 2~3개 넓이)


def _dense_range(vals: np.ndarray, q: float) -> tuple[float, float]:
    """값의 q 가 들어가는 가장 좁은 구간(블록 색상은 한곳에 몰리고, 섞여 든 다른 픽셀은 흩어져 있다)."""
    s = np.sort(vals)
    n = len(s)
    k = max(1, int(np.ceil(q * n)))
    if k >= n:
        return float(s[0]), float(s[-1])
    i = int(np.argmin(s[k - 1:] - s[:n - k + 1]))
    return float(s[i]), float(s[i + k - 1])


def _hue_rule(color: str, p: np.ndarray, dense: float) -> tuple[dict, float]:
    """블록 픽셀(HSV)의 색상이 dense 비율만큼 가장 좁게 모인 구간 ± 5 → 색상 규칙. 분홍은 0/179 경계를 넘는다.
    두 번째 값은 분홍이 0 쪽으로 퍼진 끝(15 를 넘으면 잘랐다는 뜻)."""
    if color == "red":
        hh = (p[:, 0].astype(int) + 90) % 180  # 0/179 경계를 90 근처로 옮겨 구간 하나로
        lo_, hi_ = _dense_range(hh, dense)
        return {"h_lo": int(max(-1, min(15, hi_ - 90 + 5))), "h_hi": int(max(120, lo_ + 90 - 5))}, hi_ - 90 + 5
    lo_, hi_ = _dense_range(p[:, 0], dense)
    return {"h_min": int(lo_ - 5), "h_max": int(hi_ + 5)}, 0.0


def _in_hue(color: str, rule: dict, hsv_px: np.ndarray) -> np.ndarray:
    h = hsv_px[:, 0]
    if color == "red":
        return (h <= rule["h_lo"]) | (h >= rule["h_hi"])
    return (h >= rule["h_min"]) & (h <= rule["h_max"])


def calibrate(name: str, cfg: dict, out: Path = CONFIG) -> dict:
    """검수에서 '통과'한 사진의 박스(=정답)로 HSV 범위·불량 비율 기준·블록 크기·모양을 계산해 out(autolabel.json)에 쓴다.
    배경 통계는 구역(zones.json) 안에서만 잰다 — 구역 밖 로봇 몸체·접시는 어차피 버리므로 기준을 올리면 안 된다."""
    draft = json.loads(draft_path(name).read_text())
    ims = [im for im in draft["images"] if im["status"] == "pass"]
    if len(ims) < 3:
        sys.exit(f"통과한 사진이 {len(ims)}장 — review.py 에서 실물 사진 10장쯤 검수(통과)한 뒤 다시")
    zones = load_zone_polys(draft.get("zones") or VISION / "zones.json")
    px = {"red": [], "blue": []}
    ratios = {c: [] for c in OBJ_CLASSES}
    areas, hand_areas, bg, notes, skin_hsv = [], [], [], [], []
    for im in ims:
        img = cv2.imread(im["path"])
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        ab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 1:].astype(np.float32)
        mk = masks(img, cfg)
        occupied = np.zeros((h, w), np.uint8)
        for b in im["boxes"]:
            x1, y1, x2, y2 = map(int, b["xyxy"])
            cv2.rectangle(occupied, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), 255, -1)
        bgm = occupied == 0
        zk = zone_keep(zones, cfg, img.shape)
        if zk is not None:
            bgm &= zk["near"] > 0
        # 채도 하한을 정하는 배경은 구역 안만 — 둘레 40px 까지 넣으면 아래쪽 구역 옆 로봇 흰 부품(10-22 실물 H80 S49,
        # 초록 블록과 같은 색)이 걸려 하한이 89 로 올라갔다. 블록 둘레(_object_pixels)는 그대로 둘레까지 쓴다
        bg.append(hsv[bgm & (zk["exact"] > 0)][::7] if zk is not None else hsv[bgm][::7])
        table = _table_color(ab, bgm)
        for b in im["boxes"]:
            x1, y1, x2, y2 = (max(0, int(v)) for v in b["xyxy"])
            if b["cls"] == "hand":
                hand_areas.append((x2 - x1) * (y2 - y1))
                skin_hsv.append(hsv[y1:y2 + 1, x1:x2 + 1][mk["skin"][y1:y2 + 1, x1:x2 + 1] > 0])
                continue
            # 블록 영역 = 박스 안에서 배경과 색이 다른 픽셀의 볼록 껍질. 회전한 블록은 박스 모서리가 작업대라 박스를 그대로 쓰면 안 된다
            sat = _object_pixels(ab, (x1, y1, x2, y2), bgm, table)
            ys, xs = np.nonzero(sat)
            if len(xs) < 20:
                notes.append(f"{im['stem']}: {b['cls']} 박스 안에 색 픽셀이 거의 없음 — 박스 위치 확인")
                continue
            hull = cv2.convexHull(np.stack([xs, ys], 1).reshape(-1, 1, 2).astype(np.int32))
            region = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(region, hull, 255)
            area = max(1.0, float(cv2.contourArea(hull)))
            inner = cv2.erode(region, _k(2 * max(1, int(0.15 * (area / np.pi) ** 0.5)) + 1))
            body = (inner > 0) & sat & (mk["marker"] == 0)
            v_obj = float(np.median(hsv[..., 2][body])) if body.any() else 255.0
            ratios[b["cls"]].append(marker_ratio(inner, mk["marker"], hsv[..., 2], v_obj, cfg))
            px[b["cls"].split("_")[0]].append(hsv[body])
            areas.append(area)
    bgp = np.concatenate(bg)
    new = {}
    for color in ("red", "blue"):
        if not px[color]:
            notes.append(f"{SHOWN[color]} 블록 박스 없음 — 기존 값 유지")
            continue
        p = np.concatenate(px[color])
        # 색상은 블록 픽셀이 가장 좁게 모인 구간(98%) ± 5. 백분위 1~99 로 잡으면 섞여 든 테이프·이웃 블록·식탁보 픽셀
        # (10-22 실물 2~5%, 흩어져 있음) 쪽으로 범위가 늘어나 초록이 색상 21~161 이 됐다.
        # 채도 하한은 블록 기준(하위 2% - 15)으로 정한다. 같은 색상 범위의 배경이 그 하한에 넓게 걸리면 먼저 구간을
        # 95·90·85·80% 로 좁혀 보고(10-22 실물 초록 H70~90 바로 위에 푸른 식탁보 H98~100), 그래도 걸리면 하한을 올린다.
        # 처음부터 좁게 잡으면 블록 가장자리 픽셀이 빠져 'irregular' 검수가 는다(합성 4600K 색 입힘 8 → 22장).
        # 검수에서 박스를 안 친 작은 분홍 조각(10-22 실물 60px) 몇 개 때문에 하한을 올리면 블록을 다 놓친다.
        for dense in (0.98, 0.95, 0.9, 0.85, 0.8):
            rule, spread = _hue_rule(color, p, dense)
            in_rule, in_hue = _in_hue(color, rule, p), _in_hue(color, rule, bgp)
            q = p[in_rule] if np.count_nonzero(in_rule) >= 50 else p
            s_lo, v_lo = np.percentile(q[:, 1], 2), np.percentile(q[:, 2], 2)
            v_min = int(max(25, v_lo - 15))
            s_min = int(max(20, s_lo - 15))
            cand = bgp[in_hue & (bgp[:, 2] >= v_min)]
            hit = float(np.count_nonzero(cand[:, 1] >= s_min)) / max(1, len(bgp))
            if hit <= BG_HIT_MAX:
                break
        if dense < 0.98 and hit <= BG_HIT_MAX:
            notes.append(f"{SHOWN[color]}: 배경을 피하려고 색상 구간을 블록 픽셀 {dense:.0%} 로 좁혔다(블록 픽셀의 "
                         f"{np.count_nonzero(in_rule) / len(p):.0%} 가 범위 안)")
        if hit > BG_HIT_MAX:
            bg_s = float(np.percentile(cand[:, 1], 99.5))
            s_min = max(s_min, int(bg_s + 10))
            notes.append(f"{SHOWN[color]}: 구역 안 배경의 {hit:.1%} 가 이 색 범위에 걸려 채도 하한을 {s_min} 으로 올렸다"
                         + (f" — 블록 채도 하위 2% 가 {s_lo:.0f} 라 블록을 놓친다. 색온도·조명을 확인(배경 오검출 주의)"
                            if s_min > s_lo else ""))
        if color == "red" and spread > 15:
            notes.append(f"분홍 색상이 0 쪽(주황·살구)으로 {spread:.0f} 까지 퍼져 있다 — 15 에서 잘랐다. "
                         "손 피부(색상 0~25)와 겹치니 분홍이 연어색이면 손을 분홍 블록으로 볼 수 있다")
        new[color] = {**rule, "s_min": s_min, "v_min": v_min}
    ok = ratios["red_ok"] + ratios["blue_ok"]
    bad = ratios["red_defect"] + ratios["blue_defect"]
    if ok and bad:
        mx_ok, mn_bad = max(ok), min(bad)
        if mx_ok < mn_bad:
            thr = (mx_ok + mn_bad) / 2
            new["defect_ratio"] = round(thr, 4)
            new["defect_band"] = [round(max(mx_ok, thr * 0.6), 4), round(min(mn_bad, thr * 1.5), 4)]
        else:
            notes.append(f"정상 블록 최대 {mx_ok:.3f} ≥ 불량 블록 최소 {mn_bad:.3f} — 마커로 안 갈린다. "
                         "X 를 더 굵게 칠하거나 그 블록 사진을 다시 확인")
    else:
        notes.append("정상·불량 블록이 둘 다 있어야 불량 기준을 잰다 — 기존 값 유지")
    if areas:
        new["obj_area_px"] = round(float(np.median(areas)), 1)
        # 최소 크기는 면적 하위 5% 로 — 가장 작은 박스 하나(반쯤 가려진 블록)로 정하면 6px 까지 내려가 작은 얼룩이 다 후보가 된다
        new["min_size_px"] = int(max(6, 0.6 * float(np.percentile(areas, 5)) ** 0.5))
    if hand_areas:
        new["hand_min_area"] = int(max(300, 0.4 * min(hand_areas)))
    # 블록 모양 기준: 새 색 규칙으로 덩어리를 다시 찾아, 검수한 블록 박스 하나에만 맞는(IoU ≥ 0.5, 다른 박스와 안 겹침)
    # 덩어리의 볼록도·가로세로비로 잰다. 합성 기준값보다 엄하게는 바꾸지 않는다.
    cfg_new = merge_config(cfg, new)
    sol, asp = [], []
    for im in ims:
        img = cv2.imread(im["path"])
        gt = [b for b in im["boxes"] if b["cls"] != "hand"]
        mk = masks(img, cfg_new)
        zk = zone_keep(zones, cfg_new, img.shape)
        for color in ("red", "blue"):
            cm = mk[color] if zk is None else mk[color] & zk["near"]
            for c in find_components(cm, mk["marker"], cfg_new, img.shape, zk):
                bb = [*c["pts"].min(0).tolist(), *c["pts"].max(0).tolist()]
                hits = [g for g in gt if iou(bb, g["xyxy"]) > 0.1]
                if len(hits) == 1 and hits[0]["cls"].startswith(color) and iou(bb, hits[0]["xyxy"]) >= 0.5:
                    sol.append(c["solid"])
                    asp.append(c["aspect"])
    if len(sol) >= 8:
        d = DEFAULTS["shape"]
        new["shape"] = {
            "solid_min": round(float(np.clip(np.percentile(sol, 1) - 0.05, 0.55, d["solid_min"])), 3),
            "aspect_max": round(float(np.clip(np.percentile(asp, 99) + 0.15, d["aspect_max"], 2.5)), 2),
            "solid_odd": round(float(np.clip(np.percentile(sol, 5) - 0.02, 0.6, d["solid_odd"])), 3),
            "aspect_odd": round(float(np.clip(np.percentile(asp, 95) + 0.05, d["aspect_odd"], 2.2)), 2)}
    else:
        notes.append(f"블록 하나로 맞는 덩어리가 {len(sol)}개뿐 — 모양 기준은 기존 값 유지")
    if skin_hsv and "red" in new and len(sk := np.concatenate(skin_hsv)):
        r = new["red"]
        in_pink = ((sk[:, 0] <= r["h_lo"]) | (sk[:, 0] >= r["h_hi"])) & (sk[:, 1] >= r["s_min"]) & (sk[:, 2] >= r["v_min"])
        if in_pink.mean() > 0.2:
            notes.append(f"손 피부 픽셀의 {in_pink.mean():.0%} 가 새 분홍 범위에 든다 — 손이 분홍 블록으로 잡힐 수 있다. "
                         "검수 때 손 위의 분홍 박스를 지우고, 시연 때 손 자리 판정을 확인")
    new["marker"] = cfg["marker"]
    if cfg["marker"] == "custom":
        new["markers"] = {"custom": cfg["markers"]["custom"]}
    stats = {"images": len(ims), "ratio_ok_max": round(max(ok), 4) if ok else None,
             "ratio_defect_min": round(min(bad), 4) if bad else None, "obj_area_px": new.get("obj_area_px"),
             "shape_blocks": len(sol)}
    # 새 값으로 같은 사진을 다시 라벨해 검수 결과와 비교하고, 지금 값보다 못 맞추면 쓰지 않는다
    cfg2 = merge_config(load_config(None), new)
    cfg2["zones_only"] = cfg.get("zones_only", True)
    rows, md = _check(ims, cfg2, zones)
    rows_old, _ = _check(ims, cfg, zones)
    f1 = {k: 2 * r["블록"]["P"] * r["블록"]["R"] / max(1e-9, r["블록"]["P"] + r["블록"]["R"])
          for k, r in (("new", rows), ("old", rows_old))}
    out = Path(out)
    written = f1["new"] + 0.02 >= f1["old"]
    if written:
        if out.exists():
            shutil.copy2(out, out.with_name(f"{out.name}.bak-{datetime.now():%Y%m%d_%H%M%S}"))
        out.write_text(json.dumps(new, ensure_ascii=False, indent=1))
    else:
        rej = out.with_name(f"{out.stem}.rejected.json")
        rej.write_text(json.dumps(new, ensure_ascii=False, indent=1))
        notes.append(f"새 값이 검수한 사진을 지금 값보다 못 맞춘다(블록 정밀도/재현율 {rows_old['블록']['P']}/{rows_old['블록']['R']}"
                     f" → {rows['블록']['P']}/{rows['블록']['R']}) — {out.name} 은 그대로 두고 {rej.name} 에 남겼다. "
                     "검수 박스(블록마다 하나, 손은 손만)를 확인하고 다시")
    return {"config": new, "stats": stats, "notes": notes, "check": rows, "check_old": rows_old, "table": md,
            "written": written, "path": str(out if written else out.with_name(f"{out.stem}.rejected.json"))}


def _check(ims: list[dict], cfg: dict, zones) -> tuple[dict, str]:
    """검수한 사진(박스 = 정답)을 cfg 로 다시 라벨해 채점."""
    total = {n: {"tp": 0, "fp": 0, "fn": 0} for n in NAMES}
    for im in ims:
        pred, _ = label_image(cv2.imread(im["path"]), cfg, zones)
        m = match(pred, im["boxes"])
        for n in NAMES:
            for k in ("tp", "fp", "fn"):
                total[n][k] += m[n][k]
    return score_table(total)


# ---------------------------------------------------------------- 노출 점검

def exposure_stats(img, cfg, zones=None) -> dict:
    """식탁보 영역(input 구역, 없으면 화면 가운데 절반)에서 식탁보 밝기(채도 80 미만·밝기 120 초과 픽셀의 V 중앙값)와
    블록 채도(분홍·초록 마스크의 S 중앙값). 색별 가장 큰 덩어리(%)는 늘 화면 가운데 절반 대비 — 색온도는 식탁보 전체의 일이고
    input 구역은 작아서(10-22 zones.json 1만 px) 블록 두어 개로도 10% 를 넘는다. 포화 비율은 화면 전체(어느 채널이든 250 이상).
    10-22 실물: 4600K 에서 식탁보가 푸르게(채도 56) 찍혀, 예전 기준(화면 전체·채도 25 미만)은 식탁보 대신 의자·로봇을 쟀다."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    mk = masks(img, cfg)
    center = np.zeros((h, w), bool)
    center[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
    region = zone_mask(zones, "input", img.shape)
    region, name = (center, "center") if region is None else (region > 0, "input")
    obj = ((mk["red"] | mk["blue"]) > 0) & region
    table = region & (s < 80) & (v > 120) & ~cv2.dilate(obj.astype(np.uint8), _k(9)).astype(bool)
    blob = {}
    for color in ("red", "blue"):
        n, _, st, _ = cv2.connectedComponentsWithStats(((mk[color] > 0) & center).astype(np.uint8))
        blob[color] = round(float(st[1:, cv2.CC_STAT_AREA].max()) / center.sum() * 100, 1) if n > 1 else 0.0
    return {"region": name, "table_v": float(np.median(v[table])) if table.any() else None,
            "clip_pct": round(float((img.max(2) >= 250).mean() * 100), 2),
            "obj_s": float(np.median(s[obj])) if obj.any() else None, "obj_px": int(obj.sum()), "blob_pct": blob}


def exposure_verdict(st: dict) -> str:
    """식탁보 밝기·포화 기준은 10-21 밤 합성 파스텔 시험(README '노출'), 색온도·블록 채도 기준은 10-22 실물 실측."""
    if st["table_v"] is None:
        return "식탁보를 못 찾음(화면 가운데·input 구역에 흰 식탁보가 보이게)"
    if st["blob_pct"]["blue"] >= EXPO["blob_max"]:
        return (f"식탁보가 초록으로 잡힌다(가장 큰 초록 덩어리가 화면 가운데의 {st['blob_pct']['blue']}%): "
                "white_balance_temperature 를 300 내린다(10-22: 5000K 이상이면 식탁보 전체가 초록)")
    if st["blob_pct"]["red"] >= EXPO["blob_max"]:
        return (f"식탁보가 분홍으로 잡힌다(가장 큰 분홍 덩어리가 화면 가운데의 {st['blob_pct']['red']}%): "
                "조명 색과 white_balance_temperature 를 확인")
    if st["table_v"] >= EXPO["table_v_max"] or st["clip_pct"] >= EXPO["clip_max"]:
        return "밝다: exposure_time_absolute 를 20% 내린다"
    if st["table_v"] < EXPO["table_v_min"]:
        return "어둡다: exposure_time_absolute 를 20% 올린다(식탁보가 흰색이 아니면 이 판정은 무시)"
    if st["obj_px"] < EXPO["obj_px_min"]:
        return "식탁보 밝기는 좋음. 블록이 안 보여 블록 채도는 못 봤다(분홍·초록 블록 몇 개를 화면 가운데에 두고 다시)"
    if st["obj_s"] < EXPO["obj_s_min"]:
        return "블록 채도가 낮다: exposure 를 10% 내려 보고, 그래도 낮으면 조명·WB 를 본다"
    return "좋음"


# obj_s_min: 합성 파스텔은 80 안팎이었지만 실물 파스텔은 4600K·노출 200 에서 분홍 45 정도로 보인다(10-22) → 40.
# blob_max: 블록을 몇 개만 두고 찍을 때 한 덩어리가 화면 가운데 절반의 10% 를 넘으면 식탁보가 색 마스크에 걸린 것이다
# (10-22 4600K 3%, 5000K 28%). obj_px_min: 이보다 적으면 블록이 없는 것으로 본다(10-22 사진의 분홍 덩어리 하나 약 630px).
EXPO = {"table_v_min": 200, "table_v_max": 250, "clip_max": 5.0, "obj_s_min": 40, "obj_px_min": 300, "blob_max": 10.0}


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", type=Path, help="사진 폴더")
    ap.add_argument("--name", help="초안 이름 (기본: 사진 폴더 이름)")
    ap.add_argument("--marker", help="불량 표시 색: black|white|yellow 또는 H,S,V,H,S,V")
    ap.add_argument("--zones", default=str(VISION / "zones.json"))
    ap.add_argument("--keep-outside", action="store_true",
                    help="zones.json 이 있어도 5구역 밖에 중심이 있는 블록 후보를 버리지 않는다(기본은 버림)")
    ap.add_argument("--hands", choices=["hint", "off", "auto"], help="손 박스: hint(기본)=제안만, off, auto")
    ap.add_argument("--colors", choices=["pastel", "strong"], default="pastel",
                    help="기본 색 규칙: pastel = 분홍·초록(기본), strong = 이전 진한 빨강·파랑(합성 비교용)")
    ap.add_argument("--config", type=Path, default=CONFIG, help="임계값 파일(읽기, --calibrate 는 여기에 쓴다)")
    ap.add_argument("--zip", action="store_true", help="Roboflow 업로드용 zip 도 만든다")
    ap.add_argument("--debug", action="store_true", help="박스 그린 사진을 labels_draft/<name>/debug/ 에")
    ap.add_argument("--fresh", action="store_true", help="같은 이름 초안이 있어도 검수한 사진까지 전부 다시 라벨")
    ap.add_argument("--export", metavar="NAME", help="검수 뒤 다시 내보내기")
    ap.add_argument("--include-review", action="store_true", help="검수 안 한 'review' 사진도 넣기")
    ap.add_argument("--calibrate", metavar="NAME", help="검수한 사진으로 임계값 계산 → autolabel.json")
    ap.add_argument("--eval", type=Path, metavar="ZIP_OR_DIR", help="정답 라벨과 비교")
    ap.add_argument("--exposure", nargs="+", type=Path, metavar="JPG", help="노출 점검(식탁보 밝기·포화·블록 채도)")
    a = ap.parse_args()

    cfg = load_config(a.config, a.marker, a.colors)
    if a.hands:
        cfg["hands"] = a.hands
    if a.keep_outside:
        cfg["zones_only"] = False
    if a.exposure:
        zones = load_zone_polys(a.zones)
        for p in a.exposure:
            img = cv2.imread(str(p))
            if img is None:
                print(p, "못 읽음")
                continue
            st = exposure_stats(img, cfg, zones)
            print(p.name, json.dumps(st), "→", exposure_verdict(st))
    elif a.eval:
        res = evaluate(a.eval, cfg, a.zones if a.zones != str(VISION / "zones.json") else None)
        print(json.dumps({k: v for k, v in res.items() if k != "table"}, ensure_ascii=False))
        print(res["table"])
    elif a.calibrate:
        res = calibrate(a.calibrate, cfg, a.config)
        print(f"{res['path']}{'' if res['written'] else ' (쓰지 않음 — 아래 ⚠)'}:", json.dumps(res["config"], ensure_ascii=False))
        print("통계:", json.dumps(res["stats"], ensure_ascii=False))
        for n in res["notes"]:
            print("⚠", n)
        b, o = res["check"]["블록"], res["check_old"]["블록"]
        print(f"새 값으로 다시 라벨 vs 검수 결과 (블록 정밀도/재현율: 지금 값 {o['P']}/{o['R']} → 새 값 {b['P']}/{b['R']}):\n" + res["table"])
    elif a.export:
        print(json.dumps(export(a.export, a.zip, a.include_review), ensure_ascii=False))
    elif a.src:
        name = a.name or a.src.name
        draft = run(a.src, name, cfg, a.zones, a.debug, a.fresh)
        print("초안:", draft_path(name), json.dumps(summarize(draft), ensure_ascii=False))
        print("데이터셋:", json.dumps(export(name, a.zip), ensure_ascii=False))
        print(f"검수: python vision/review.py {name}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
