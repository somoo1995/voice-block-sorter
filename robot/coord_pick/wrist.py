"""손목 카메라(그리퍼에 붙은 icSpring)로 집기 확인·보정 — r18·r19.

- 기준("잡기 좋은 모습"): 10-22 녹화 3종(pick_defect·pick_red·pick_green 각 51개)의 손목 영상에서 리더 그리퍼 값이
  떨어지기 시작하기 직전 프레임의 목표 블록 색 덩어리 — 화면 중심·크기(면적)·각도 분포(wrist_ref.json "close").
  집은 뒤 기준("after"): 같은 에피소드의 닫기 시작 0.8~1초 뒤(쥐고 들어 올리는 중) 프레임.
- 카메라 모형: 같은 녹화의 접근 프레임(닫기 0~1초 전)에서 블록 덩어리 중심(픽셀)과 블록 위치를 핀홀 모형(끝 좌표계 안
  카메라 자세 6 + 초점 1 + 중심 2)으로 맞춤. 블록 위치 = 사람이 닫기 시작한 순간의 tcp(같은 순기구학의 상대값이라 위 카메라·
  base 보정 오차가 안 들어간다). → 손목 사진 한 장으로 블록이 tcp 에서 몇 mm 떨어졌는지.
- 실행 중에는 executor 가 단계 사이(송신 프로세스가 없을 때)에 호스트에서 손목 카메라를 열어 한 장 찍고 바로 닫는다.

    python -m robot.coord_pick.wrist fit                 # 녹화로 기준·카메라 모형 → robot/coord_pick/wrist_ref.json
    python -m robot.coord_pick.wrist show 사진.jpg [--color red]   # 사진 한 장 판정(기준과 비교)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REF_PATH = HERE / "wrist_ref.json"
WRIST_CAM = "/dev/v4l/by-id/usb-icSpring_icspring_camera_20240307110322-video-index0"
BLOCK_Z_MM = 15.0      # 블록 가운데 높이(모형을 맞출 때와 같은 약속)
MIN_AREA = 800         # 이보다 작은 덩어리는 블록으로 안 본다(640×480)
MERGE_GAP_PX = 12      # 불량 블록 X 선(손목 화면 hover 약 6~8px, 가까우면 굵어짐) — 틈이 2배(24px) 안인 조각은 한 블록


# ------------------------------------------------------------------ 사진

def capture(dev: str = WRIST_CAM):
    """손목 카메라 한 장(BGR). 다른 프로세스(Studio 등)가 쥐고 있으면 CameraBusy — 열지 않는다."""
    sys.path.insert(0, str(ROOT))
    from vision import camera
    frame, _src = camera.grab_frame("device", dev=dev)
    return frame


PALE_A_MAX, PALE_L_MIN = 124, 110  # 연한 초록(손목 고정 노출 250·4600K 에서 높이 있으면 기본 초록 마스크에 안 잡힘 — 10-23
                                  # grasp 녹화 시작 프레임 초록 17개 중 14개를 놓침): Lab a* < 124 · L > 110


def blob(img, color: str, near=None, merge: bool = False, pale: bool = False) -> dict | None:
    """목표 색 덩어리: near(픽셀)가 있으면 그 점에 가장 가까운 것, 없으면 가장 큰 것.
    {px(중심), area(px²), angle(최소 사각형 각, 0~90), box(x,y,w,h), edge(화면 가장자리에 닿음)}.
    color: red(분홍)·blue(초록)·any. 색 규칙은 vision/autolabel.py 와 같다(손목 화면도 같은 스펀지).
    pale(초록): 기본 마스크에 블록만 한 덩어리가 없으면 연한 초록(Lab a*·L)으로 한 번 더(높은 시작 자세의 손목 사진).
    merge(불량 블록): 검은 X 가 면을 네 조각으로 나눠 조각 하나만 잡히던 것(10-23 10:01·10:07 실물)을, X 선 굵기만큼
    (MERGE_GAP_PX) 떨어진 조각끼리 한 덩어리로 묶어 본다 — 면적 = 조각 합, 중심 = 조각 픽셀 전체 중심, pieces = 조각 수."""
    import cv2
    sys.path.insert(0, str(ROOT))
    from vision import autolabel
    mk = autolabel.masks(img, autolabel.load_config())
    m = mk["red"] | mk["blue"] if color == "any" else mk[color]
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if pale and color == "blue" and not any(int(x) >= MIN_AREA for x in cv2.connectedComponentsWithStats(m)[2][1:, 4]):
        # 기본 초록 마스크에 블록만 한 덩어리가 없을 때만 연한 초록(Lab)으로 — 잡히는 경우는 예전 그대로
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        m = cv2.morphologyEx(((lab[..., 1] < PALE_A_MAX) & (lab[..., 0] > PALE_L_MIN)).astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((5, 5), np.uint8))
    if merge:  # 조각 묶음 번호 = X 선을 메울 만큼 부풀린 마스크의 연결 성분(면적·중심은 원래 마스크 픽셀로)
        k = 2 * MERGE_GAP_PX + 1
        n, glab = cv2.connectedComponents(cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))))
        lab = np.where(m > 0, glab, 0).astype(np.int32)
        ids, cnt = np.unique(lab[lab > 0], return_counts=True)
        st = {int(i): int(c) for i, c in zip(ids, cnt) if c >= MIN_AREA}
        cand = list(st)
        if not cand:
            return None
        cen = {i: np.array([np.mean(np.nonzero(lab == i)[1]), np.mean(np.nonzero(lab == i)[0])]) for i in cand}
        area_of = st.__getitem__
    else:
        n, lab, st_, cen_ = cv2.connectedComponentsWithStats(m)
        cand = [i for i in range(1, n) if st_[i, cv2.CC_STAT_AREA] >= MIN_AREA]
        if not cand:
            return None
        cen = {i: cen_[i] for i in cand}
        area_of = (lambda j: int(st_[j, cv2.CC_STAT_AREA]))
    if near is None:
        i = max(cand, key=area_of)
    else:
        i = min(cand, key=lambda j: float(np.hypot(*(cen[j] - np.asarray(near, float)))))
    area = area_of(i)
    ys, xs = np.nonzero(lab == i)
    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    (_, _), (_, _), ang = cv2.minAreaRect(pts)
    ax, asp = long_axis(pts)
    H, W = img.shape[:2]
    out = {"px": [round(float(cen[i][0]), 1), round(float(cen[i][1]), 1)], "area": area, "angle": round(float(ang) % 90, 1),
           "axis_deg": round(math.degrees(ax) % 180, 1), "aspect": round(asp, 2),
           "box": [x, y, w, h], "edge": x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1}
    if merge:
        out["pieces"] = int(cv2.connectedComponents((lab == i).astype(np.uint8))[0] - 1)
    return out


def long_axis(pts: np.ndarray) -> tuple[float, float]:
    """덩어리 픽셀(N×2)의 긴 축 방향(rad, 화면 x 에서 y 쪽)과 가로세로비(√고유값 비) — 주성분. 90° 주기 모호함이 없다."""
    P = np.asarray(pts, float) - np.mean(pts, axis=0)
    w, V = np.linalg.eigh(np.cov(P.T))
    return math.atan2(V[1, 1], V[0, 1]), math.sqrt(w[1] / max(w[0], 1e-9))


def axis_in_robot(sm: ScreenModel, T: np.ndarray, dz: float, axis_deg: float) -> float:
    """손목 화면의 긴 축 각 → 로봇 좌표 수평 방위(rad). 화면 모형의 J(끝 좌표 mm → px)를 거꾸로."""
    J = np.array([[sm.cu[1] + sm.cu[4] * dz, sm.cu[2] + sm.cu[5] * dz], [sm.cv[1] + sm.cv[4] * dz, sm.cv[2] + sm.cv[5] * dz]])
    a = math.radians(axis_deg)
    t = np.linalg.solve(J, [math.cos(a), math.sin(a)])
    b = T[:2, :2] @ t
    return math.atan2(b[1], b[0])


# ------------------------------------------------------------------ 화면 모형(측정 기반)

FEATURES = ("1", "dx", "dy", "dz", "dx*dz", "dy*dz")


def _feat(d) -> np.ndarray:
    d = np.atleast_2d(np.asarray(d, float))
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    return np.c_[np.ones(len(d)), dx, dy, dz, dx * dz, dy * dz]


class ScreenModel:
    """블록 가운데가 tcp 에서 끝 좌표로 d = (dx, dy, dz) mm 떨어져 있을 때 손목 화면 덩어리 중심 (u, v) — 선형(+ 높이 곱) 회귀.
    녹화에서 사람이 닫기 시작한 순간의 tcp 를 블록 위치로 본다(순기구학의 상대값만 써서 보정 오차가 없다)."""

    def __init__(self, coef_u, coef_v):
        self.cu, self.cv = np.asarray(coef_u, float), np.asarray(coef_v, float)

    def predict(self, d) -> np.ndarray:
        F = _feat(d)
        return np.c_[F @ self.cu, F @ self.cv]

    def solve_xy(self, uv, dz: float) -> np.ndarray:
        """측정 (u, v) 와 높이 차 dz → (dx, dy) (끝 좌표 mm)."""
        a = np.array([[self.cu[1] + self.cu[4] * dz, self.cu[2] + self.cu[5] * dz],
                      [self.cv[1] + self.cv[4] * dz, self.cv[2] + self.cv[5] * dz]])
        b = np.array([uv[0] - self.cu[0] - self.cu[3] * dz, uv[1] - self.cv[0] - self.cv[3] * dz])
        return np.linalg.solve(a, b)


def load_ref(path: Path = REF_PATH) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def judge_close(b: dict | None, ref: dict) -> dict:
    """닫기 직전 사진 판정: 덩어리 중심이 기준(사람이 닫기 직전) 허용 안이고 크기가 기준 하한 이상(= 충분히 내려감)이면 ok."""
    c = ref["close"]
    if b is None:
        return {"ok": False, "why": "블록 덩어리가 안 보임"}
    du, dv = b["px"][0] - c["u"]["med"], b["px"][1] - c["v"]["med"]
    tol_u, tol_v = c["tol_px"]
    small = b["area"] < c["area"]["p10"]
    out = {"du": round(du, 1), "dv": round(dv, 1), "area": b["area"], "area_min": c["area"]["p10"],
           "ok": abs(du) <= tol_u and abs(dv) <= tol_v and not small}
    out["why"] = ("기준 안" if out["ok"] else
                  "덜 내려감(덩어리가 작음)" if small and abs(du) <= tol_u and abs(dv) <= tol_v else
                  f"가운데에서 벗어남(du {du:+.0f}, dv {dv:+.0f}px, 허용 ±{tol_u:.0f}·±{tol_v:.0f})")
    return out


def screen_model(ref: dict) -> ScreenModel:
    m = ref["screen"]
    return ScreenModel(m["coef_u"], m["coef_v"])


def judge(b: dict | None, ref: dict, kind: str, dataset: str, area_scale: float = 1.0, tol_min: float = 0.0) -> dict:
    """kind = "close"(닫기 직전 — 충분히 내려와 가운데) 또는 "after"(쥐고 들어 올린 뒤 — 두 날 사이에 그 크기로 보임).
    기준 = 녹화 그 데이터셋(pick_red·pick_green·pick_defect) 분포: 중심 ±max(5~95% 폭/2, tol_min), 면적 ≥ 하위 10% × area_scale.
    tol_min: 녹화가 좁게 몰린 축(분홍 '집은 뒤' v ±14px)에서 쥔 블록을 못 쥔 것으로 보지 않게(10-23 13:27 실물 dv −18~−25 ×3)."""
    r = ref[kind]["by_dataset"][dataset]
    tol = [max(float(x), float(tol_min)) for x in r["tol_px"]]
    return judge_close(b, {"close": {"u": {"med": r["u_med"]}, "v": {"med": r["v_med"]}, "tol_px": tol,
                                     "area": {"p10": int(r["area_p10"] * area_scale)}}})


def locate(s, q: np.ndarray, b: dict, model: ScreenModel) -> tuple[np.ndarray, np.ndarray]:
    """손목 사진의 덩어리 중심 → 블록 가운데의 작업대 좌표(mm)와 끝 좌표 차이 (dx, dy, dz). s = plan.Setup, q = 찍을 때 자세."""
    T = s.kin.fk(q)
    tcp = (T[:3, 3] + T[:3, :3] @ s.tcp) * 1000
    dz = float((BLOCK_Z_MM - tcp[2]) * T[2, 2])  # tcp 바로 아래 블록 가운데(높이 15)까지 끝 z 방향 거리(+ = 앞)
    dx, dy = model.solve_xy(b["px"], dz)
    P = tcp + T[:3, :3] @ np.array([dx, dy, dz])
    return s.frames.r2w(P[:2]), np.array([dx, dy, dz])


# ------------------------------------------------------------------ 녹화로 기준·모형 맞추기

DATASETS = {"pick_defect": "d9e18d33-5e02-427a-b7bb-fa29c61d3853", "pick_red": "dd6126e6-4ac0-4d56-9400-527135597fb8",
            "pick_green": "07f0afd2-518c-4178-a6fa-1af90b01aad0"}
COLOR = {"pick_defect": "any", "pick_red": "red", "pick_green": "blue"}
MERGE_DS = ("pick_defect",)  # X 조각을 묶어 보는 데이터셋(10-23 — 실행기도 불량 블록은 merge)


def _q(v, ps):
    return {f"p{p}": round(float(np.percentile(v, p)), 1) for p in ps}


def fit(data_dir: Path, s) -> dict:
    """data_dir = robot/logs/r14(rec.json·grasps.json·cam_vs_fk.json·meta_grip.json·video_grip/). s = 보정된 Setup."""
    import cv2
    rec = json.loads((data_dir / "rec.json").read_text())
    grasps = json.loads((data_dir / "grasps.json").read_text())
    meta = json.loads((data_dir / "meta_grip.json").read_text())
    c0 = np.asarray(s.tcp * 1000, float)  # 닫기 시작한 순간 블록 = 그 자세의 tcp(xy), 높이 15mm
    obs: list = []
    K = "videos/observation.images.gripper/"
    close, after = [], []
    for ds, did in DATASETS.items():
        eps = {e["episode_index"]: e for e in meta[did]}
        caps: dict = {}
        for g in [x for x in grasps if x["ds"] == ds]:
            e = eps[g["ep"]]
            fi = int(e[K + "file_index"])
            base = round(e[K + "from_timestamp"] * 30)
            cap = caps.setdefault(fi, cv2.VideoCapture(str(data_dir / "video_grip" / did / f"file-{fi:03d}.mp4")))
            states = rec[did]["episodes"][str(g["ep"])]["s"]
            Tg = s.kin.fk(s.n_to_q(np.array(states[g["gi"]])))
            P = Tg[:3, 3] * 1000 + Tg[:3, :3] @ c0
            P[2] = BLOCK_Z_MM
            for k in (30, 25):  # 닫기 시작 1초·0.8초 뒤 = 쥐고 들어 올리는 중
                i = g["gi"] + k
                if i >= len(states):
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, base + i)
                ok, img = cap.read()
                b = blob(img, COLOR[ds], near=(350, 200), merge=ds in MERGE_DS) if ok else None
                if b is not None:
                    after.append({"ds": ds, "ep": g["ep"], "k": k, **b})
                    break
            prev = None
            for k in (-1, -5, -10, -15, -20, -30, -40, -50):
                i = g["gi"] + k
                if i < 0:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, base + i)
                ok, img = cap.read()
                if not ok:
                    continue
                b = blob(img, COLOR[ds], near=prev or (350, 150), merge=ds in MERGE_DS)
                if b is None:
                    continue
                prev = b["px"]
                if k == -1:
                    close.append({"ds": ds, "ep": g["ep"], **b})
                if not b["edge"]:
                    T = s.kin.fk(s.n_to_q(np.array(states[i])))
                    tcp = (T[:3, 3] + T[:3, :3] @ s.tcp) * 1000
                    obs.append(((P - tcp) @ T[:3, :3], b["px"], ds, g["ep"], k))
    def dist(rows_):
        out = {}
        for ds in DATASETS:
            cc = [c for c in rows_ if c["ds"] == ds]
            if not cc:
                continue
            u_, v_, a_ = (np.array([c["px"][0] for c in cc]), np.array([c["px"][1] for c in cc]),
                          np.array([c["area"] for c in cc]))
            out[ds] = {"n": len(cc), "u_med": round(float(np.median(u_)), 1), "v_med": round(float(np.median(v_)), 1),
                       "tol_px": [round(float((np.percentile(u_, 95) - np.percentile(u_, 5)) / 2), 1),
                                  round(float((np.percentile(v_, 95) - np.percentile(v_, 5)) / 2), 1)],
                       "area_med": int(np.median(a_)), "area_p10": int(np.percentile(a_, 10))}
        return out

    ref_after = {"n": len(after), "by_dataset": dist(after),
                 "_doc": "사람이 닫기 시작하고 0.8~1초 뒤(쥐고 들어 올리는 중) 손목 화면의 블록 덩어리 — 집은 뒤 성공 판정"}
    # 기준 분포(사람이 닫기 직전)
    U = np.array([c["px"][0] for c in close])
    V = np.array([c["px"][1] for c in close])
    A = np.array([c["area"] for c in close])
    G = np.array([c["angle"] for c in close])
    ref_close = {"n": len(close), "u": {"med": round(float(np.median(U)), 1), **_q(U, (5, 25, 75, 95))},
                 "v": {"med": round(float(np.median(V)), 1), **_q(V, (5, 25, 75, 95))},
                 "area": {"med": int(np.median(A)), **_q(A, (10, 25))}, "angle": {"med": round(float(np.median(G)), 1), **_q(G, (25, 75))},
                 # 허용 = 사람 분포의 5~95% 폭의 절반(가운데에서 이만큼까지는 사람도 닫았다)
                 "tol_px": [round(float((np.percentile(U, 95) - np.percentile(U, 5)) / 2), 1),
                            round(float((np.percentile(V, 95) - np.percentile(V, 5)) / 2), 1)],
                 "by_dataset": {}}
    for ds in DATASETS:
        cc = [c for c in close if c["ds"] == ds]
        u_, v_, a_ = (np.array([c["px"][0] for c in cc]), np.array([c["px"][1] for c in cc]), np.array([c["area"] for c in cc]))
        ref_close["by_dataset"][ds] = {"n": len(cc), "u_med": round(float(np.median(u_)), 1), "v_med": round(float(np.median(v_)), 1),
                                       "tol_px": [round(float((np.percentile(u_, 95) - np.percentile(u_, 5)) / 2), 1),
                                                  round(float((np.percentile(v_, 95) - np.percentile(v_, 5)) / 2), 1)],
                                       "area_med": int(np.median(a_)), "area_p10": int(np.percentile(a_, 10))}
    # 화면 모형: 끝 좌표 차이 d → (u, v) 회귀(크게 벗어난 점 두 번 걸러냄)
    D = np.array([o[0] for o in obs])
    UV = np.array([o[1] for o in obs], float)
    keep = np.ones(len(D), bool)
    for _ in range(3):
        F = _feat(D[keep])
        cu, *_ = np.linalg.lstsq(F, UV[keep, 0], rcond=None)
        cv, *_ = np.linalg.lstsq(F, UV[keep, 1], rcond=None)
        err = np.linalg.norm(ScreenModel(cu, cv).predict(D) - UV, axis=1)
        keep = err < max(25.0, 2.5 * float(np.median(err[keep])))
    sm = ScreenModel(cu, cv)
    err = np.linalg.norm(sm.predict(D) - UV, axis=1)
    # 교차 검증: 에피소드 짝·홀로 나눠 맞추고 서로 시험
    ep = np.array([o[3] for o in obs])
    cvs = []
    for part in (0, 1):
        tr, te = keep & (ep % 2 == part), ep % 2 != part
        F = _feat(D[tr])
        m2 = ScreenModel(np.linalg.lstsq(F, UV[tr, 0], rcond=None)[0], np.linalg.lstsq(F, UV[tr, 1], rcond=None)[0])
        cvs.append(np.linalg.norm(m2.predict(D[te]) - UV[te], axis=1))
    cvs = np.concatenate(cvs)
    # 1mm 옮기면 몇 px(높이 차 dz 별)
    sens = {}
    for dz in (20, 35, 50):
        a = np.array([[cu[1] + cu[4] * dz, cu[2] + cu[5] * dz], [cv[1] + cv[4] * dz, cv[2] + cv[5] * dz]])
        sens[str(dz)] = np.round(a, 2).tolist()
    span = {k: [round(float(v), 1) for v in (np.percentile(D[keep][:, i], 5), np.percentile(D[keep][:, i], 95))]
            for i, k in enumerate(("dx", "dy", "dz"))}
    model = {"coef_u": [round(float(v), 5) for v in cu], "coef_v": [round(float(v), 5) for v in cv], "features": list(FEATURES),
             "n": int(keep.sum()), "n_all": len(D), "residual_px": {"med": round(float(np.median(err[keep])), 1),
                                                                     **_q(err[keep], (75, 90))},
             "cross_val_px": {"med": round(float(np.median(cvs)), 1), **_q(cvs, (75, 90))},
             "px_per_mm": sens, "range_mm": span}
    return {"close": ref_close, "after": ref_after, "screen": model, "_doc": "python -m robot.coord_pick.wrist fit 이 만든다(10-22 녹화). close = 사람이 "
            "그리퍼를 닫기 직전 손목 화면의 목표 블록 덩어리 분포(px, 640×480). screen = 블록 가운데가 tcp 에서 끝 좌표 (dx,dy,dz)mm "
            "떨어졌을 때 덩어리 중심 (u,v) 회귀 계수(특징 features)"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.coord_pick.wrist", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", default=str(HERE.parent / "logs" / "r14"))
    f.add_argument("-o", "--out", default=str(REF_PATH))
    f.add_argument("--only", choices=tuple(DATASETS), help="그 데이터셋의 close·after 기준만 지금 파일(--out)에 바꿔 넣는다"
                   "(화면 모형·다른 데이터셋 그대로 — 10-23 불량 X 조각 묶기)")
    sh = sub.add_parser("show")
    sh.add_argument("image")
    sh.add_argument("--color", default="red")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        from robot.coord_pick.plan import Setup, load_config
        s = Setup.load(load_config(), use_studio=False)
        ref = fit(Path(a.data), s)
        if a.only:
            new, ref = ref, json.loads(Path(a.out).read_text(encoding="utf-8"))
            for kind in ("close", "after"):
                print(f"  {kind} {a.only}: {ref[kind]['by_dataset'].get(a.only)} → {new[kind]['by_dataset'][a.only]}")
                ref[kind]["by_dataset"][a.only] = new[kind]["by_dataset"][a.only]
            ref.setdefault("_patched", []).append(f"{a.only}: close·after 다시(merge={a.only in MERGE_DS})")
        Path(a.out).write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
        c, m = ref["close"], ref["screen"]
        print(f"기준(닫기 직전 {c['n']}장): 중심 u {c['u']['med']} (5~95% {c['u']['p5']}~{c['u']['p95']}) · v {c['v']['med']} "
              f"({c['v']['p5']}~{c['v']['p95']}) · 면적 중앙 {c['area']['med']} (하위 10% {c['area']['p10']}) · 각 {c['angle']['med']}° · 허용 ±{c['tol_px']}")
        for ds, d in c["by_dataset"].items():
            print(f"  닫기 직전 {ds}: {d}")
        for ds, d in ref["after"]["by_dataset"].items():
            print(f"  집은 뒤   {ds}: {d}")
        print(f"화면 모형: 점 {m['n']}/{m['n_all']}개, 잔차 중앙 {m['residual_px']['med']}px (90% {m['residual_px']['p90']}) · "
              f"교차 검증 중앙 {m['cross_val_px']['med']}px (90% {m['cross_val_px']['p90']}) · px/mm(dz별 2×2) {m['px_per_mm']} · 범위 {m['range_mm']}")
        print(f"저장: {a.out}")
        return 0
    import cv2
    ref = load_ref()
    img = cv2.imread(a.image)
    b = blob(img, a.color)
    print(json.dumps({"blob": b, "judge": judge_close(b, ref) if ref else None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
