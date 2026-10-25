"""YOLO 모델이 아직 없을 때 쓰는 색 검출 어댑터 — vision/autolabel.py 의 label_image(HSV 규칙) 박스 → det.json(계약 objects).

    python -m robot.coord_pick.detect_hsv /tmp/cam_now.jpg -o det.json [--draw det.jpg]
    python -m robot.coord_pick sort --detect det.json         # 이 파일로 분홍·초록 옮기기 계획

블록 쪼개짐: HSV 규칙은 블록 하나를 박스 2~3개로 나눌 때가 있다. 같은 색 박스끼리 틈이 gap_px 이하이고 합친 크기가
블록 한 개 크기(max_span_px) 안이면 한 블록으로 합치고, 조각 박스 중심의 평균을 블록 중심으로 쓴다
(10-22 15:30 실측: 쪼갠 조각 중심 평균이 실제 중심과 3px 안). 손 박스('hand')는 뺀다.
사진은 읽기만 한다 — 카메라를 열지 않는다(사진 파일을 받는다).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
ZONES_JSON = ROOT / "vision" / "zones.json"
GAP_PX = 6          # 조각 박스 사이 틈(px) — 이보다 가까우면 같은 블록 후보
MAX_SPAN_PX = 34    # 합친 박스 한 변 상한(px). 투입 구역의 블록 한 개 = 가로 17~24·세로 18~28px(10-22 녹화 첫 장면)
ZONE_TOL_PX = 6     # 구역 판정 여유


def _gap(a, b) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return float(max(dx, dy))


def merge_boxes(boxes: list[dict], gap_px: float = GAP_PX, max_span_px: float = MAX_SPAN_PX) -> list[dict]:
    """[{cls, xyxy}] → [{cls, px, box, parts}]. 색(red/blue)이 같고 가까운 조각을 합친다. 조각 중 하나라도 불량이면 불량."""
    items = [b for b in boxes if b["cls"] != "hand"]
    groups: list[list[dict]] = []
    for b in sorted(items, key=lambda b: (b["xyxy"][0], b["xyxy"][1])):
        color = b["cls"].split("_")[0]
        best = None
        for g in groups:
            if g[0]["cls"].split("_")[0] != color:
                continue
            if min(_gap(b["xyxy"], o["xyxy"]) for o in g) > gap_px:
                continue
            u = np.array([o["xyxy"] for o in g + [b]], float)
            if max(u[:, 2].max() - u[:, 0].min(), u[:, 3].max() - u[:, 1].min()) <= max_span_px:
                best = g
                break
        if best is None:
            groups.append([b])
        else:
            best.append(b)
    out = []
    for g in groups:
        xy = np.array([o["xyxy"] for o in g], float)
        centers = np.c_[(xy[:, 0] + xy[:, 2]) / 2, (xy[:, 1] + xy[:, 3]) / 2]
        color = g[0]["cls"].split("_")[0]
        defect = any(o["cls"].endswith("defect") for o in g)
        flags = sorted({f for o in g for f in o.get("flags", [])})
        out.append({"cls": f"{color}_{'defect' if defect else 'ok'}", "px": centers.mean(0).round(1).tolist(),
                    "box": [float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 2].max()), float(xy[:, 3].max())],
                    "parts": len(g), "conf": 0.5 if flags else 0.8, "flags": flags})
    return out


def _inside(p, poly) -> bool:
    x, y, inside = p[0], p[1], False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _near(p, poly, tol) -> bool:
    if _inside(p, poly):
        return True
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab, ap = b - a, np.asarray(p, float) - a
        t = float(np.clip(ap @ ab / max(ab @ ab, 1e-9), 0, 1))
        if np.linalg.norm(ap - t * ab) <= tol:
            return True
    return False


def zone_of(px, zones: dict | None, tol: float = ZONE_TOL_PX) -> str | None:
    """블록 중심이 든 구역 이름(input·bowl_red·bowl_blue·bowl_defect·hand_spot), 없으면 None."""
    if not zones:
        return None
    for name, poly in zones.items():
        if _near(px, np.asarray(poly, float), tol):
            return name
    return None


def load_zones(path: str | Path = ZONES_JSON, size: tuple[int, int] = (640, 480)) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    w, h = d.get("image_size", size)
    return {k: (np.array(v, float) * (size[0] / w, size[1] / h)).tolist() for k, v in d["zones"].items()}


def measure_yaw(img, objects: list[dict], s, *, near_px: float = 25.0) -> list[dict]:
    """위 카메라 사진에서 블록마다 색 덩어리 긴 축(주성분) → 작업대 방위 yaw_w(rad, 긴 축)·가로세로비 aspect 를 objects 에 붙인다.
    s = plan.Setup(호모그래피). 덩어리 = 목표 색 마스크에서 블록 중심(px)에 가장 가까운 것(near_px 안). 10-22 녹화 102건:
    위 카메라 긴 축과 손목 화면 긴 축의 방향 차 중앙 10°(사분위 4~23)."""
    import math

    import cv2
    sys.path.insert(0, str(ROOT))
    from vision import autolabel

    from robot.coord_pick.wrist import long_axis
    mk = autolabel.masks(img, autolabel.load_config())
    out = []
    for o in objects:
        o = dict(o)
        color = o["cls"].split("_")[0]
        if color in ("red", "blue"):
            m = cv2.morphologyEx(mk[color], cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, st, cen = cv2.connectedComponentsWithStats(m)
            cand = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 60]
            if cand:
                i = min(cand, key=lambda j: float(np.hypot(*(cen[j] - np.asarray(o["px"], float)))))
                if float(np.hypot(*(cen[i] - np.asarray(o["px"], float)))) <= near_px:
                    ys, xs = np.nonzero(lab == i)
                    ax, asp = long_axis(np.stack([xs, ys], 1))
                    c = np.array([xs.mean(), ys.mean()])
                    w0, w1 = s.px_to_w(c)[0], s.px_to_w(c + 10 * np.array([math.cos(ax), math.sin(ax)]))[0]
                    o["yaw_w"] = round(math.atan2(w1[1] - w0[1], w1[0] - w0[0]), 4)
                    o["aspect"] = round(asp, 2)
        out.append(o)
    return out


def detect_hsv(img, zones_path: str | Path | None = ZONES_JSON, *, gap_px: float = GAP_PX,
               max_span_px: float = MAX_SPAN_PX) -> dict:
    """사진(BGR 배열) → {"source", "objects": [{cls, px, box, parts, conf, flags, zone}], "boxes", "warnings"}."""
    sys.path.insert(0, str(ROOT))
    from vision import autolabel  # 비전 담당 파일(읽기만)
    cfg = autolabel.load_config()
    zl = autolabel.load_zone_polys(zones_path) if zones_path else None
    boxes, warnings = autolabel.label_image(img, cfg, zl)
    objs = merge_boxes(boxes, gap_px, max_span_px)
    zones = load_zones(zones_path, (img.shape[1], img.shape[0])) if zones_path else None
    for o in objs:
        o["zone"] = zone_of(o["px"], zones)
    return {"source": "hsv(vision/autolabel.label_image) + 조각 합치기", "image_size": [img.shape[1], img.shape[0]],
            "objects": objs, "boxes": [{"cls": b["cls"], "xyxy": b["xyxy"], "flags": b.get("flags", [])} for b in boxes],
            "warnings": warnings}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.coord_pick.detect_hsv", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", help="det.json 경로(없으면 화면에만)")
    ap.add_argument("--zones", default=str(ZONES_JSON))
    ap.add_argument("--draw", help="박스·중심을 그린 사진 경로")
    a = ap.parse_args(argv)
    import cv2
    img = cv2.imread(a.image)
    if img is None:
        print(f"[안됨] 사진을 못 읽음: {a.image}")
        return 1
    det = detect_hsv(img, a.zones)
    det["image"] = str(a.image)
    ko = {"red_ok": "분홍", "blue_ok": "초록", "red_defect": "분홍 불량", "blue_defect": "초록 불량"}
    print(f"블록 {len(det['objects'])}개(조각 박스 {sum(b['cls'] != 'hand' for b in det['boxes'])}개를 합침)")
    for o in det["objects"]:
        print(f"  {ko.get(o['cls'], o['cls']):<6} 중심 ({o['px'][0]:.0f},{o['px'][1]:.0f}) 조각 {o['parts']} 구역 {o['zone']}"
              + (f" 표시 {','.join(o['flags'])}" if o["flags"] else ""))
    for w in det["warnings"]:
        print(f"  [주의] {w}")
    if a.out:
        Path(a.out).write_text(json.dumps(det, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"저장: {a.out}")
    if a.draw:
        col = {"red": (203, 102, 255), "blue": (60, 200, 60)}
        for b in det["boxes"]:
            if b["cls"] != "hand":
                x1, y1, x2, y2 = b["xyxy"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
        for o in det["objects"]:
            c = col[o["cls"].split("_")[0]]
            x1, y1, x2, y2 = (int(round(v)) for v in o["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 1)
            cv2.drawMarker(img, (int(round(o["px"][0])), int(round(o["px"][1]))), c, cv2.MARKER_CROSS, 8, 1)
        cv2.imwrite(a.draw, img)
        print(f"그림: {a.draw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
