"""[시험용] 가짜 데이터 — 진짜 사진이 오기 전에 autolabel → review → train.py → detect.py 경로를 끝까지 돌려 보는 용도.

물체는 스펀지 블록(기본, 10-21 밤 결정): 약 3cm 정육면체를 위에서 본 회전한 사각형 + 원근으로 보이는 옆면 +
기공 질감, 불량은 굵은 검정 X(6면 전부 칠함 → 위로 온 면에 늘 X). --object ball 은 이전 부직포볼(시험 비교용).
색은 파스텔 분홍·초록(기본, 10-21 밤 결정. 코드의 red_* = 분홍, blue_* = 초록). --palette strong 은 이전 진한 빨강·파랑.
--jitter 는 사진마다 밝기 흔들림(카메라 노출 고정이면 0.06 쯤), --gain 1.2 는 노출이 20% 높아 색이 날아간 상태.

  python vision/synth_dataset.py --n 200 --table white --out vision/_test/b_white.zip      # autolabel 평가용 zip
  python vision/synth_dataset.py --raw vision/_test/b_raw --n 150 --table white          # capture.py 처럼 사진만 (+ _gt/ 정답)
  python vision/synth_dataset.py --scene vision/_test/scene_block.jpg --table white      # 구역 장면 + zones_test.json + 정답 개수

Roboflow 와 같게 만든다: train/valid/test 분할, 클래스 알파벳순, data.yaml 의 '../train/images', 라벨 끝 줄바꿈 없음.
진짜 데이터로 학습한 모델과 성능을 비교하는 용도가 아니다.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np

NAMES = ["blue_defect", "blue_ok", "hand", "red_defect", "red_ok"]  # Roboflow 는 알파벳순으로 번호를 매긴다
W, H = 640, 480
PALETTES = {  # BGR. 키 red/blue 는 클래스 식별자(red_* = 분홍, blue_* = 초록)
    "pastel": {"red": (192, 160, 245), "blue": (150, 228, 178)},  # 파스텔 분홍·연두 스펀지: 채도 낮고 밝다
    "strong": {"red": (40, 35, 200), "blue": (190, 90, 30)},      # 이전 진한 빨강·파랑
}
COLOR = dict(PALETTES["pastel"])
RENDER = {"palette": "pastel", "gain": 1.0, "jitter": 0.2}  # main 이 CLI 값으로 바꾼다
PX_PER_CM = (11, 18)   # 위 카메라 높이에 따른 3cm 물체 크기 33~54px (공 시절 지름 32~60px 와 같은 범위)
BLOCK_CM = (2.5, 3.5)  # 손으로 깍둑썬 블록 한 변
MARKER = {"black": (15, 15, 15), "white": (240, 240, 240), "yellow": (30, 215, 235)}  # 불량 X 색


def background(rng, table="gray"):
    lo, hi = (222, 246) if table == "white" else (150, 215)  # white = 흰 무광 작업대
    if table == "white" and RENDER["palette"] == "pastel":
        # 흰 무광 식탁보 + WB 고정: 거의 무채색(채널 차 ±4). 색이 끼면 파스텔 블록과 구별이 안 된다
        base = (rng.uniform(lo, hi) + rng.uniform(-4, 4, 3)).astype(np.float32)
    else:
        base = rng.integers(lo, hi, 3).astype(np.float32)
    img = np.empty((H, W, 3), np.float32)
    img[:] = base
    img += rng.normal(0, 6, img.shape)
    for _ in range(rng.integers(1, 4)):  # 그릇처럼 보이는 방해물
        c = (int(rng.integers(60, W - 60)), int(rng.integers(60, H - 60)))
        cv2.circle(img, c, int(rng.integers(45, 75)), tuple(float(v) for v in base * 0.75), 4)
    return img


def obj_color(color, rng) -> np.ndarray:
    """물체 하나의 색. 파스텔은 흰색 쪽으로 섞인 정도(채도)를 흔들고, 진한 색은 밝기만 흔든다."""
    base = np.array(COLOR[color], np.float32)
    if RENDER["palette"] == "pastel":
        return (255 - (255 - base) * rng.uniform(0.75, 1.25)) * rng.uniform(0.93, 1.03)
    return base * rng.uniform(0.8, 1.15)


def draw_ball(img, cx, cy, r, color, defect, rng, marker="black", shadow=False):
    if shadow:  # 흰 작업대 위 그림자(오른쪽 아래로 번짐)
        sh = np.zeros((H, W), np.float32)
        cv2.ellipse(sh, (cx + r // 5, cy + r // 3), (int(r * 1.05), int(r * 0.9)), 0, 0, 360, 1.0, -1)
        img *= (1 - 0.35 * cv2.GaussianBlur(sh, (0, 0), r * 0.25))[..., None]
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    c = obj_color(color, rng)
    yy, xx = np.mgrid[0:H, 0:W]
    shade = 1.15 - 0.45 * np.clip(((xx - cx + r * 0.3) ** 2 + (yy - cy + r * 0.3) ** 2) ** 0.5 / (r * 1.6), 0, 1)
    tex = rng.normal(1, 0.08, (H, W))
    m = mask > 0
    img[m] = np.clip(c[None, :] * (shade * tex)[m][:, None], 0, 255)
    if defect:  # X 표시
        d, t = int(r * 0.6), max(2, r // 6)
        cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), MARKER[marker], t)
        cv2.line(img, (cx - d, cy + d), (cx + d, cy - d), MARKER[marker], t)
    return (cx - r, cy - r, cx + r, cy + r)


def block_polys(cx, cy, side, angle, persp):
    """윗면 꼭짓점과, 옆면까지 포함한 보이는 외곽(볼록 껍질). 윗면은 카메라에 가까워 영상 중심에서 바깥으로 밀려 보인다."""
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    h = side / 2
    top = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ rot.T + (cx, cy)
    base = top + (np.array([W / 2, H / 2]) - (cx, cy)) * persp
    hull = cv2.convexHull(np.concatenate([top, base]).astype(np.float32)).reshape(-1, 2)
    return top, hull


def draw_block(img, top, hull, side, color, defect, rng, marker="black", shadow=False):
    """스펀지 블록 한 개. 반환: 보이는 외곽의 박스."""
    x1, y1 = np.floor(hull.min(0)).astype(int)
    x2, y2 = np.ceil(hull.max(0)).astype(int)
    if shadow:  # 흰 작업대 위 그림자(오른쪽 아래로 번짐)
        sh = np.zeros((H, W), np.float32)
        cv2.fillConvexPoly(sh, (hull + (side * 0.12, side * 0.22)).round().astype(np.int32), 1.0)
        img *= (1 - 0.35 * cv2.GaussianBlur(sh, (0, 0), side * 0.2))[..., None]
    c = obj_color(color, rng)
    m_all = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m_all, hull.round().astype(np.int32), 255)
    m_top = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m_top, top.round().astype(np.int32), 255)
    # 스펀지 질감: 굵은 얼룩(저해상도 잡음을 키움) + 잔 잡음
    bh, bw = y2 - y1 + 1, x2 - x1 + 1
    coarse = cv2.resize(rng.normal(1, 0.10, (max(2, bh // 3), max(2, bw // 3))).astype(np.float32), (bw, bh))
    tex = np.ones((H, W), np.float32)
    ys, xs = slice(max(0, y1), min(H, y2 + 1)), slice(max(0, x1), min(W, x2 + 1))
    tex[ys, xs] = (coarse * rng.normal(1, 0.06, (bh, bw)))[ys.start - y1:ys.stop - y1, xs.start - x1:xs.stop - x1]
    side_m = (m_all > 0) & (m_top == 0)
    img[side_m] = np.clip(c * rng.uniform(0.55, 0.75) * tex[side_m][:, None], 0, 255)  # 옆면은 어둡게
    yy, xx = np.nonzero(m_top)
    shade = 1.08 - 0.16 * ((xx - xx.min()) / max(1, np.ptp(xx)) + (yy - yy.min()) / max(1, np.ptp(yy))) / 2
    img[yy, xx] = np.clip(c * (shade * tex[yy, xx])[:, None], 0, 255)
    # 기공: 같은 색의 어두운 점(검정 마커와 헷갈리지 않게 밝기 45~65%)
    for _ in range(int(side * side / 20)):
        u, v = rng.uniform(0.05, 0.95, 2)
        p = top[0] + u * (top[1] - top[0]) + v * (top[3] - top[0])
        cv2.circle(img, (int(p[0]), int(p[1])), int(rng.integers(1, 3)), tuple(float(x) for x in c * rng.uniform(0.45, 0.65)), -1)
    if defect:  # 굵은 검정 유성매직 X — 손으로 그린 흔들림
        t = max(3, int(side / 7))
        ctr = top.mean(0) + rng.uniform(-0.06, 0.06, 2) * side
        for i, j in ((0, 2), (1, 3)):
            d = (top[j] - top[i]) * 0.38 * rng.uniform(0.85, 1.05)
            d = d @ np.array([[np.cos(k := rng.uniform(-0.2, 0.2)), -np.sin(k)], [np.sin(k), np.cos(k)]]).T
            cv2.line(img, tuple((ctr - d).round().astype(int)), tuple((ctr + d).round().astype(int)), MARKER[marker], t)
        # 옆면에도 X 를 칠했으므로 보이는 옆면에 획 일부(옆면 밖으로는 안 나가게)
        if side_m.sum() > side * 4:
            sy, sx = np.nonzero(side_m)
            k = rng.integers(len(sx))
            stroke = np.zeros((H, W), np.uint8)
            cv2.line(stroke, (int(sx[k]) - t, int(sy[k]) - t), (int(sx[k]) + t, int(sy[k]) + t), 255, max(2, t - 1))
            img[(stroke > 0) & side_m] = MARKER[marker]
    return (int(x1), int(y1), int(x2), int(y2))


def draw_hand(img, cx, cy, s, rng):
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (cx, cy), (int(s * 0.45), int(s * 0.55)), 0, 0, 360, 255, -1)
    ang0 = rng.uniform(-0.3, 0.3)
    for i in range(4):
        a = ang0 - np.pi / 2 + (i - 1.5) * 0.28
        p0 = (int(cx + np.cos(a) * s * 0.4), int(cy + np.sin(a) * s * 0.45))
        p1 = (int(cx + np.cos(a) * s * 1.05), int(cy + np.sin(a) * s * 1.1))
        cv2.line(mask, p0, p1, 255, max(6, int(s * 0.17)))
    a = ang0 + np.pi * (0.1 if rng.random() < 0.5 else 0.9)
    cv2.line(mask, (cx, cy), (int(cx + np.cos(a) * s * 0.8), int(cy + np.sin(a) * s * 0.5)), 255, max(6, int(s * 0.2)))
    skin = np.array([135, 165, 215], np.float32) * rng.uniform(0.75, 1.1)
    m = mask > 0
    img[m] = np.clip(skin[None, :] * rng.normal(1, 0.04, (int(m.sum()), 1)), 0, 255)
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def finish(img, rng):
    j = RENDER["jitter"]
    img = img * RENDER["gain"] * rng.uniform(1 - j, 1 + j) + rng.uniform(-100 * j, 100 * j)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0) if rng.random() < 0.5 else img


def place(rng, objs, r, box=(30, 30, W - 30, H - 30)):
    for _ in range(100):
        cx, cy = int(rng.integers(box[0] + r, box[2] - r)), int(rng.integers(box[1] + r, box[3] - r))
        if all((cx - x) ** 2 + (cy - y) ** 2 > (r + rr + 4) ** 2 for x, y, rr in objs):
            objs.append((cx, cy, r))
            return cx, cy
    return None


def place_block(rng, occupied, ppc, box=(25, 25, W - 25, H - 25), persp=None, touch=0.3):
    """겹치지 않는 블록 자리. touch 비율은 이웃과 맞닿게(틈 0) — 투입구에 쏟은 블록처럼. 반환: (top, hull, side) 또는 None."""
    persp = rng.uniform(0.02, 0.06) if persp is None else persp
    for _ in range(150):
        side = rng.uniform(*BLOCK_CM) * ppc
        r = side * 0.75
        if box[2] - box[0] <= 2 * r or box[3] - box[1] <= 2 * r:
            return None
        cx, cy = rng.uniform(box[0] + r, box[2] - r), rng.uniform(box[1] + r, box[3] - r)
        top, hull = block_polys(cx, cy, side, rng.uniform(0, 90), persp)
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, hull.round().astype(np.int32), 255)
        gap = 0 if rng.random() < touch else int(rng.integers(3, 9))
        if not np.any(cv2.dilate(m, np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)) & occupied):
            occupied |= m
            return top, hull, side
    return None


def random_image(rng, marker="black", table="gray", obj="block", dense=False, ppc=None):
    img, objs, labels = background(rng, table), [], []
    occupied = np.zeros((H, W), np.uint8)
    if rng.random() < 0.4:
        s = int(rng.integers(45, 70))
        p = place(rng, objs, int(s * 1.1), (40, 60, W - 40, H - 30))
        if p:
            box = draw_hand(img, p[0], p[1], s, rng)
            labels.append(("hand", box))
            cv2.rectangle(occupied, box[:2], box[2:], 255, -1)
    ppc = ppc or rng.uniform(*PX_PER_CM)  # 고정하면 카메라 높이가 정해진 실제 작업대와 같다
    # dense: 투입구(왼쪽 절반)에 12~20개를 몰아 쏟은 장면, 60% 가 맞닿음
    n, box, touch = (rng.integers(12, 21), (20, 40, 340, 450), 0.6) if dense else (rng.integers(3, 13), (25, 25, W - 25, H - 25), 0.3)
    for _ in range(n if obj == "block" else rng.integers(3, 11)):
        cls = rng.choice(["red_ok", "blue_ok", "red_defect", "blue_defect"], p=[0.35, 0.35, 0.15, 0.15])
        color, defect = cls.split("_")[0], cls.endswith("defect")
        if obj == "block":
            p = place_block(rng, occupied, ppc, box, touch=touch)
            if p:
                labels.append((cls, draw_block(img, p[0], p[1], p[2], color, defect, rng, marker, table == "white")))
        else:
            r = int(rng.integers(16, 30))
            p = place(rng, objs, r)
            if p:
                labels.append((cls, draw_ball(img, p[0], p[1], r, color, defect, rng, marker, table == "white")))
    return finish(img, rng), labels


def yolo_lines(labels) -> str:
    out = []
    for cls, (x1, y1, x2, y2) in labels:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        out.append(f"{NAMES.index(cls)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return "\n".join(out)  # Roboflow 처럼 마지막 줄바꿈 없음


def make_zip(n: int, out: Path, seed: int, marker: str = "black", table: str = "gray", obj: str = "block",
             dense: bool = False, ppc: float | None = None) -> None:
    rng = np.random.default_rng(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_tr, n_va = int(n * 0.7), int(n * 0.2)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(n):
            split = "train" if i < n_tr else "valid" if i < n_tr + n_va else "test"
            img, labels = random_image(rng, marker, table, obj, dense, ppc)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            z.writestr(f"{split}/images/synth_{i:04d}.jpg", buf.tobytes())
            z.writestr(f"{split}/labels/synth_{i:04d}.txt", yolo_lines(labels))
        z.writestr("data.yaml", "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\n"
                                f"nc: {len(NAMES)}\nnames: {NAMES}\n\nroboflow:\n  workspace: synthetic\n"
                                "  project: ball-sorter-synth\n  version: 0\n  license: none\n  url: none\n")
        z.writestr("README.dataset.txt", "synthetic test data (vision/synth_dataset.py)\n")
    print(out, f"{n}장 (train {n_tr} / valid {n_va} / test {n - n_tr - n_va})")


def make_raw(n: int, out: Path, seed: int, marker: str, table: str, obj: str = "block", dense: bool = False,
             ppc: float | None = None) -> None:
    """capture.py 가 찍은 폴더처럼 jpg 만 둔다. 정답 라벨은 _gt/ 에(autolabel 은 하위 폴더를 안 읽는다)."""
    rng = np.random.default_rng(seed)
    (out / "_gt").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img, labels = random_image(rng, marker, table, obj, dense, ppc)
        cv2.imwrite(str(out / f"synth_{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "_gt" / f"synth_{i:04d}.txt").write_text(yolo_lines(labels))
    (out / "_gt" / "data.yaml").write_text(f"nc: {len(NAMES)}\nnames: {NAMES}\n")
    print(out, f"{n}장 + _gt/")


def octagon(cx, cy, r):
    return [[int(cx + r * np.cos(a)), int(cy + r * np.sin(a))] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]


def make_scene(out: Path, seed: int, marker: str = "black", table: str = "gray", obj: str = "block") -> None:
    """구역이 정해진 장면. 정답 개수와 zones_test.json 을 같이 쓴다."""
    rng = np.random.default_rng(seed)
    zones = {
        "input": [[20, 50], [300, 50], [300, 430], [20, 430]],
        "bowl_red": octagon(400, 130, 70),
        "bowl_blue": octagon(560, 130, 70),
        "bowl_defect": octagon(400, 330, 70),
        "hand_spot": [[500, 250], [635, 250], [635, 470], [500, 470]],
    }
    plan = {"input": {"red_ok": 3, "blue_ok": 2, "red_defect": 1, "blue_defect": 1},
            "bowl_red": {"red_ok": 2}, "bowl_blue": {"blue_ok": 2}, "bowl_defect": {"blue_defect": 1}}
    img = np.empty((H, W, 3), np.float32)
    img[:] = (236, 238, 238) if table == "white" else (190, 195, 200)
    img += rng.normal(0, 5, img.shape)
    for z in ("bowl_red", "bowl_blue", "bowl_defect"):
        c = np.array(zones[z]).mean(0).astype(int)
        cv2.circle(img, tuple(int(v) for v in c), 62, (235, 235, 235), -1)
        cv2.circle(img, tuple(int(v) for v in c), 62, (120, 120, 120), 3)
    for z, want in plan.items():
        pts = np.array(zones[z])
        box = (*pts.min(0), *pts.max(0))
        if z != "input":  # 그릇 안쪽만
            c = pts.mean(0)
            box = (c[0] - 58, c[1] - 58, c[0] + 58, c[1] + 58)
        objs, occupied = [], np.zeros((H, W), np.uint8)
        for cls, k in want.items():
            for _ in range(k):
                color, defect = cls.split("_")[0], cls.endswith("defect")
                if obj == "block":  # 그릇(지름 약 9cm) 안에 블록 2개가 들어가게 cm 당 12px, 자리가 없으면 맞닿게
                    p = (place_block(rng, occupied, 12, tuple(int(v) for v in box), 0.04)
                         or place_block(rng, occupied, 12, tuple(int(v) for v in box), 0.04, touch=1.0))
                    if p is None:
                        raise RuntimeError(f"{z} 에 블록 자리가 없음")
                    draw_block(img, *p, color, defect, rng, marker, table == "white")
                else:
                    p = place(rng, objs, 21, tuple(int(v) for v in box))
                    draw_ball(img, p[0], p[1], 21, color, defect, rng, marker, table == "white")
    draw_hand(img, 568, 375, 55, rng)
    cv2.imwrite(str(out), np.clip(img, 0, 255).astype(np.uint8))
    expected = {z: {k: plan.get(z, {}).get(k, 0) for k in ("red_ok", "blue_ok", "red_defect", "blue_defect")}
                for z in ("input", "bowl_red", "bowl_blue", "bowl_defect")}
    zpath = out.with_name("zones_test.json")
    zpath.write_text(json.dumps({"image_size": [W, H], "camera": "synthetic", "zones": zones}, indent=1))
    out.with_suffix(".expected.json").write_text(json.dumps({"zones": expected, "hand": True}, indent=1))
    print(out, zpath, json.dumps(expected))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "_test" / "block-synth.v0i.yolov8.zip")
    ap.add_argument("--scene", type=Path, help="이 경로에 시험 장면 jpg")
    ap.add_argument("--raw", type=Path, help="이 폴더에 사진만(capture.py 출력 흉내) + _gt/ 정답 라벨")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--marker", choices=list(MARKER), default="black", help="불량 X 색")
    ap.add_argument("--table", choices=["gray", "white"], default="gray", help="white = 흰 작업대 + 그림자")
    ap.add_argument("--object", choices=["block", "ball"], default="block", help="block = 스펀지 블록(기본), ball = 이전 부직포볼")
    ap.add_argument("--dense", action="store_true", help="투입구에 블록 12~20개를 몰아 쏟은 장면(맞닿은 블록 많음)")
    ap.add_argument("--ppc", type=float, help="cm 당 픽셀 고정(카메라 높이 고정). 없으면 사진마다 11~18")
    ap.add_argument("--palette", choices=list(PALETTES), default="pastel", help="pastel = 분홍·초록(기본), strong = 이전 빨강·파랑")
    ap.add_argument("--gain", type=float, default=1.0, help="노출 배율(1.2 = 20%% 밝게, 파스텔이 날아감)")
    ap.add_argument("--jitter", type=float, default=0.2, help="사진마다 밝기 흔들림 비율(노출 고정이면 0.06)")
    a = ap.parse_args()
    COLOR.update(PALETTES[a.palette])
    RENDER.update(palette=a.palette, gain=a.gain, jitter=a.jitter)
    if a.scene:
        a.scene.parent.mkdir(parents=True, exist_ok=True)
        make_scene(a.scene, a.seed + 1000, a.marker, a.table, a.object)
    elif a.raw:
        make_raw(a.n, a.raw, a.seed, a.marker, a.table, a.object, a.dense, a.ppc)
    else:
        make_zip(a.n, a.out, a.seed, a.marker, a.table, a.object, a.dense, a.ppc)


if __name__ == "__main__":
    main()
