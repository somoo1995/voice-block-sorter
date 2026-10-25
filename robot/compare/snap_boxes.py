"""실물 실행 직전 검출 사진(runs/20261022/snap_*.jpg)에 그려진 YOLO 박스를 색으로 되읽는다.

그 사진은 vision/detect.draw() 가 박스(두께 2)·이름표(박스 위 칠한 사각형)·구역을 그린 것이라 YOLO 를 다시 돌리면 거의 못
찾는다(10-22 밤 실측 19장 중 블록 0~2개). 그때의 실시간 검출 결과를 그대로 쓰려고 박스 색(BOX_COLORS)으로 테두리를 찾는다:
색 마스크의 연결 성분마다 위쪽의 칠한 이름표 줄을 떼고, 남은 세로 테두리의 좌우 끝·아래 끝으로 박스를 복원한다.

    .venv/bin/python -m robot.compare.snap_boxes      # → robot/logs/n2/truth_fix.json 의 real_* 항목, n2/snap_check.jpg
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from robot.compare.common import OUT, ROOT
from robot.coord_pick.detect_hsv import load_zones, zone_of

BOX_COLORS = {"red_ok": (203, 102, 255), "blue_ok": (60, 200, 60), "red_defect": (140, 0, 170),
              "blue_defect": (0, 110, 0)}  # vision/detect.py 와 같은 값(BGR)
TOL = 48


def boxes(img: np.ndarray) -> list[dict]:
    poly = np.array(load_zones()["input"])
    x0, y0 = poly.min(0).astype(int) - (8, 30)
    x1, y1 = poly.max(0).astype(int) + 8
    out = []
    for cls, c in BOX_COLORS.items():
        m = (np.abs(img.astype(int) - np.array(c)).max(2) < TOL).astype(np.uint8)
        m[:y0], m[y1:], m[:, :x0], m[:, x1:] = 0, 0, 0, 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for k in range(1, n):
            if st[k, cv2.CC_STAT_AREA] < 40 or st[k, cv2.CC_STAT_HEIGHT] < 14:
                continue
            ys, xs = np.nonzero(lab == k)
            top, bot = ys.min(), ys.max()
            # 이름표 = 위쪽의 가로로 꽉 찬 줄들. 박스 줄은 좌우 테두리(각 2px)만 칠해져 있다
            rows = [(y, xs[ys == y]) for y in range(top, bot + 1)]
            label_end = top
            for y, xr in rows:
                if len(xr) >= 0.5 * (xr.max() - xr.min() + 1) and len(xr) > 8:
                    label_end = y
                elif y > top + 4:
                    break
            body = [(y, xr) for y, xr in rows if label_end + 2 < y < bot - 2 and len(xr)]
            if len(body) < 6:
                continue
            # 세로 테두리 = 몸통 줄의 절반 넘게 칠해진 열. 같은 색 박스가 나란히 붙으면 한 성분이 되므로 테두리를 둘씩 짝짓는다
            cols = np.zeros(img.shape[1], int)
            for _, xr in body:
                cols[xr] += 1
            edge = np.nonzero(cols >= 0.45 * len(body))[0]
            runs = np.split(edge, np.nonzero(np.diff(edge) > 2)[0] + 1) if len(edge) else []
            xs_e = [float(r.mean()) for r in runs if len(r)]
            pairs = [(xs_e[i], xs_e[i + 1]) for i in range(0, len(xs_e) - 1, 2)]
            if not pairs:
                pairs = [(float(np.median([xr.min() for _, xr in body])), float(np.median([xr.max() for _, xr in body])))]
            for bx0, bx1 in pairs:
                if bx1 - bx0 < 8:
                    continue
                ys_b = [y for y, xr in body if np.any(np.abs(xr - bx0) <= 2) and np.any(np.abs(xr - bx1) <= 2)]
                yb0, yb1 = (min(ys_b), max(ys_b)) if ys_b else (label_end, bot)
                px = [(bx0 + bx1) / 2, (yb0 + yb1) / 2]
                out.append({"cls": cls, "px": [round(px[0], 1), round(float(px[1]), 1)],
                            "box": [int(bx0), int(yb0), int(bx1), int(yb1)]})
    zones = load_zones()
    for o in out:
        o["zone"] = zone_of(o["px"], zones)
    return out


def main() -> int:
    doc = json.loads((OUT / "scenes.json").read_text())
    fx = OUT / "truth_fix.json"
    fix = json.loads(fx.read_text()) if fx.exists() else {}
    tiles = []
    for r in doc["real"]:
        if not r["snap"]:
            continue
        img = cv2.imread(str(ROOT / r["snap"]))
        b = boxes(img)
        fix[r["id"]] = b
        vis = img.copy()
        for o in b:
            x, y = map(int, o["px"])
            cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
            cv2.putText(vis, o["cls"][:5] + ("X" if "defect" in o["cls"] else ""), (x - 18, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
        t = cv2.resize(vis[140:290, 110:350], None, fx=1.6, fy=1.6)
        cv2.putText(t, r["start"], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        tiles.append(t)
        print(r["start"], r["policy"], [(o["cls"], o["px"], o["zone"]) for o in b])
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(OUT / "snap_check.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    fx.write_text(json.dumps({"_doc": "정답 블록 고침: real_* = 실물 직전 사진의 그려진 박스(snap_boxes.py)", **{
        k: v for k, v in fix.items() if k != "_doc"}}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
