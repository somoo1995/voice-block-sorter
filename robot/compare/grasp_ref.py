"""손목 '닫기 직전'·'집은 뒤' 기준을 고정 노출 녹화(grasp 데이터셋, 10-23 10:28~11:05, 노출 250·4600K)로 다시 잰다.

    .venv/bin/python -m robot.compare.grasp_ref            # 지금 기준(wrist_ref.json, 10-22 자동 노출)과 비교만
    .venv/bin/python -m robot.compare.grasp_ref --write    # → robot/coord_pick/wrist_ref_fixed.json(화면 모형은 그대로)

wrist.fit 과 같은 규칙: 닫기 직전 = 그리퍼 동작이 떨어지기 시작하기 한 틱 전(앞 30틱부터 덩어리를 따라감),
집은 뒤 = 닫기 시작 1초(없으면 0.8초) 뒤. 색 = 그 프레임 분홍·초록 마스크 중 큰 쪽, 불량 = 블록 볼록 껍질 안쪽의 검은
픽셀 비율(닫기 1초·⅓초 전·직전 세 장 중앙)이 X_DARK 넘음(39개에서 정상 ≤ 0.01, X 0.13~0.38 로 갈림).
가장자리에 닿거나 중심이 그 묶음 중앙에서 크게 벗어난 덩어리(다른 것을 잡음)는 뺀다.
실행기와 같게 정상은 그 색 마스크, 불량은 그 색 + 조각 묶기로 잰다. 쓰려면 config wrist.ref = "wrist_ref_fixed.json".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from robot.compare.common import closing
from robot.coord_pick import wrist

ROOT = Path(__file__).resolve().parent.parent.parent
N6 = ROOT / "robot" / "logs" / "n6"
OUT = ROOT / "robot" / "coord_pick" / "wrist_ref_fixed.json"


X_DARK = 0.06


def _dark_frac(img, color: str) -> float:
    """목표 색 덩어리(조각 묶음) 볼록 껍질 안쪽(15px 깎음)의 검은 픽셀 비율 — 불량 X 선."""
    import sys
    sys.path.insert(0, str(ROOT))
    from vision import autolabel
    mk = cv2.morphologyEx(autolabel.masks(img, autolabel.load_config())[color].astype(np.uint8), cv2.MORPH_OPEN,
                          np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(cv2.dilate(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))))
    if n < 2:
        return 0.0
    big = max(range(1, n), key=lambda i: int(((lab == i) & (mk > 0)).sum()))
    hull = np.zeros(mk.shape, np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(((lab == big) & (mk > 0)).astype(np.uint8))), 1)
    inner = cv2.erode(hull, np.ones((15, 15), np.uint8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(((gray < 80) & (inner > 0)).sum() / max(1, int(inner.sum())))


def _color(img) -> str:
    import sys
    sys.path.insert(0, str(ROOT))
    from vision import autolabel
    mk = autolabel.masks(img, autolabel.load_config())
    return "red" if int(mk["red"].sum()) >= int(mk["blue"].sum()) else "blue"


def collect() -> list[dict]:
    rec = json.loads((N6 / "grasp_rec.json").read_text())
    meta = {m["episode_index"]: m for m in rec["meta"]}
    caps: dict = {}
    rows = []
    for ep, e in sorted(rec["episodes"].items(), key=lambda kv: int(kv[0])):
        m = meta[int(ep)]
        k = "videos/observation.images.gripper/"
        path = (N6 / "ds_grasp" / "videos" / "observation.images.gripper" / f"chunk-{int(m[k + 'chunk_index']):03d}"
                / f"file-{int(m[k + 'file_index']):03d}.mp4")
        cap = caps.setdefault(path, cv2.VideoCapture(str(path)))
        base = round(float(m[k + "from_timestamp"]) * 30)
        A = np.asarray(e["a"], float)
        gi = closing(A[:, 5], rest=float(np.median(A[:5, 5])) - 7)
        if gi is None:
            continue

        def frame(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, base + i)
            ok, img = cap.read()
            return img if ok else None
        img_c = frame(gi - 1)
        if img_c is None:
            continue
        color = _color(img_c)
        darks = [_dark_frac(f, color) for f in (frame(gi - 30), frame(gi - 10), img_c) if f is not None]
        defect = float(np.median(darks)) > X_DARK
        ds = "pick_defect" if defect else ("pick_red" if color == "red" else "pick_green")
        prev, close = None, None
        for kk in (-30, -20, -10, -5, -1):
            if gi + kk < 0:
                continue
            img = frame(gi + kk)
            b = wrist.blob(img, color, near=prev or (350, 150), merge=defect) if img is not None else None
            if b is not None:
                prev = b["px"]
                if kk == -1:
                    close = b
        after = None
        for kk in (30, 25):
            if gi + kk < len(A):
                img = frame(gi + kk)
                after = wrist.blob(img, color, near=(350, 200), merge=defect) if img is not None else None
                if after is not None:
                    break
        rows.append({"ep": int(ep), "ds": ds, "color": color, "dark": round(float(np.median(darks)), 3),
                     "close": close, "after": after})
    return rows


def dist(rows: list[dict], kind: str) -> dict:
    out = {}
    for ds in ("pick_defect", "pick_red", "pick_green"):
        cc = [r[kind] for r in rows if r["ds"] == ds and r[kind] is not None and not r[kind]["edge"]]
        if not cc:
            continue
        P = np.array([c["px"] for c in cc], float)
        d = np.linalg.norm(P - np.median(P, axis=0), axis=1)
        cc = [c for c, x in zip(cc, d) if x <= max(60.0, 3 * float(np.median(d)))]  # 다른 것을 잡은 덩어리 빼기
        u_, v_, a_ = (np.array([c["px"][0] for c in cc]), np.array([c["px"][1] for c in cc]),
                      np.array([c["area"] for c in cc]))
        out[ds] = {"n": len(cc), "u_med": round(float(np.median(u_)), 1), "v_med": round(float(np.median(v_)), 1),
                   "tol_px": [round(float((np.percentile(u_, 95) - np.percentile(u_, 5)) / 2), 1),
                              round(float((np.percentile(v_, 95) - np.percentile(v_, 5)) / 2), 1)],
                   "area_med": int(np.median(a_)), "area_p10": int(np.percentile(a_, 10)),
                   "aspect_med": round(float(np.median([c["aspect"] for c in cc])), 2)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.compare.grasp_ref", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--rows", action="store_true")
    a = ap.parse_args(argv)
    rows = collect()
    if a.rows:
        for r in rows:
            print(r["ep"], r["ds"], r["dark"], r["close"] and (r["close"]["px"], r["close"]["area"], r["close"]["aspect"]),
                  r["after"] and (r["after"]["px"], r["after"]["area"]))
    old = wrist.load_ref()
    new = {"close": dist(rows, "close"), "after": dist(rows, "after")}
    for kind in ("close", "after"):
        for ds, d in new[kind].items():
            o = old[kind]["by_dataset"].get(ds, {})
            print(f"{kind:5} {ds:11} 고정 노출 n {d['n']:2} 중심 ({d['u_med']}, {d['v_med']}) ±{d['tol_px']} 면적 중앙 {d['area_med']} "
                  f"하위10% {d['area_p10']} | 지금 기준 ({o.get('u_med')}, {o.get('v_med')}) ±{o.get('tol_px')} 중앙 "
                  f"{o.get('area_med')} 하위10% {o.get('area_p10')}")
    if a.write:
        ref = json.loads(json.dumps(old))
        # 면적 하한만, 그리고 예전보다 엄하게는 안 한다(min) — 고정 노출로 작아진 색(초록)만 낮아진다. 분홍 '집은 뒤'는 사람이
        # 깊게 쥐어 고정 노출 녹화 하한(17.9천)이 실물 로봇 값(09:47 14.6천)보다 커서 그대로 쓰면 쥔 블록을 떨어뜨린다.
        # 중심·허용·화면 모형은 10-22 녹화 값(실물 IK 맞춤에서 검증됨) 그대로
        for kind in ("close", "after"):
            for ds, d in new[kind].items():
                o = ref[kind]["by_dataset"][ds]
                o.update(area_med=min(o["area_med"], d["area_med"]), area_p10=min(o["area_p10"], d["area_p10"]),
                         fixed=d)
        ref["_fixed_exposure"] = ("close·after by_dataset 의 area_p10·area_med = min(10-22 녹화, grasp 데이터셋 7b16f024 — 10-23 손목 "
                                  "고정 노출 250·4600K, robot.compare.grasp_ref; fixed = 고정 노출 분포 그대로). 중심·허용·화면 "
                                  "모형은 10-22 녹화 그대로. 쓰려면 config wrist.ref")
        OUT.write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"저장: {OUT.relative_to(ROOT)} — 쓰려면 config wrist.ref = \"{OUT.name}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
