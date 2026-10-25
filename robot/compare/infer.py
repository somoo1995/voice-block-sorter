"""3단계: 정책 오프라인 추론 — Studio 컨테이너 OpenVINO, CPU 장치(GPU 는 OVMS 전용). 로봇·카메라 없음.

    .venv/bin/python -m robot.compare.infer act_red act_green act_defect [--groups own,swap,mirror,empty,photo]
    .venv/bin/python -m robot.compare.infer smolvla_red --groups own,mirror,empty,photo --ds pick_red --limit-empty 20

작업 묶음(group) — '정책이 영상을 보고 움직이나' 시험(10-22 밤 추가 요청)
- own   : 녹화 첫 장면 그대로(위·손목 사진·첫 관절값).
- swap  : 위 사진·관절값은 그대로, 손목 사진만 같은 녹화의 25 에피소드 뒤 장면 것(홈 자세 손목 카메라는 블록이 아니라
          교실을 비춘다) — 예측이 위 사진 장면의 목표와 손목 사진 장면의 목표 중 어느 쪽을 따라가나.
- mirror: 위 사진의 테이프 안쪽만 좌우로 뒤집음(블록 자리만 좌우가 바뀐 같은 장면, 축 = 테이프 안쪽 가운데).
- empty : 위 사진의 테이프 안쪽 블록을 지움(채도·어두운 X 마스크 inpaint — 블록 없는 작업대).
- bgswap: 위 사진의 작업대 너머 교실 배경(BG_Y 위)만 짝 장면 것 — 블록·손목·관절은 이 장면. 예측이 배경을 따라가나.
- inswap: 위 사진의 테이프 안쪽(블록)만 짝 장면 것 — 그릇 안 블록·팔·배경·손목·관절은 이 장면. 블록을 따라가나.
- calib : 위 사진 = 블록 없는 실제 작업대(Studio calibrate 녹화 첫 장면, scenes.py 밖에서 뽑음 → n2/scenes/
          calib_00_front.jpg), 손목 사진·관절값 = 각 녹화 장면 것 — 예측이 그 손목 사진 에피소드의 목표로 가나.
- photo : 손목 사진·관절값이 없는 사진(known4·좌표 집기 front) — 관절 = config home_n, 손목 = 녹화 세 장면의 홈 자세
          손목 사진(SUB_GRIPS) 각각으로 세 번.
결과: robot/logs/n2/jobs_<정책>.json(작업 목록) + infer_<정책>.npz(작업마다 첫 묶음·이어 붙인 동작·추론 시간).
컨테이너에는 아무것도 쓰지 않는다(코드는 -c, 사진은 stdin, 결과는 stdout).
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time

import cv2
import numpy as np

from robot.compare.common import OUT, ROOT, TASK_OF_DS
from robot.coord_pick.plan import load_config

CONTAINER = "physical-ai-studio-xpu"
CPY = "/app/application/backend/.venv/bin/python"
POLICIES = {"act_red": ("708a44b7-ff34-4302-93b9-763dc15066b0", None),
            "act_green": ("b3407a54-4071-4c77-9bab-3b6d930a021a", None),
            "act_defect": ("ef64c0f9-0773-4b77-a119-e1872ab3e70a", None),
            "smolvla_red": ("d887e672-7c59-4777-81ca-c8a1dcdb2238", "pick_red")}
SUB_GRIPS = ("rec_pick_red_00", "rec_pick_green_25", "rec_pick_defect_40")
BG_Y = 112          # 위 카메라 사진에서 이 줄 위 = 작업대 너머 교실(의자·사람). 작업대 끝은 118~125px
MAX_STEPS = {"act": 300, "smolvla": 250}  # 10초·8.3초 — 실물 첫 닫힘 ACT 2.0~4.9초, SmolVLA 8.1초(10-22 실행 기록)


def tape_inner(img: np.ndarray) -> np.ndarray:
    """주황 테이프 안쪽(투입 구역) 마스크 — 사진마다 찾는다(녹화 오전과 12:1x 다시 붙인 뒤 테이프 자리가 달라 zones.json 을
    쓰면 반전 때 테이프 이음매가 생긴다). 가장 큰 테이프 덩어리의 볼록 껍질 − 테이프(블록이 테이프를 가려 끊겨도 된다)
    에서 가장 큰 덩어리를 7px 줄인다."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (3, 100, 100), (22, 255, 255))
    m[:110], m[340:] = 0, 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    hull = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(np.argwhere(lab == k)[:, ::-1].astype(np.int32)), 255)
    inside = hull & ~cv2.dilate(m, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(inside)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return cv2.erode(((lab == k) * 255).astype(np.uint8), np.ones((7, 7), np.uint8))


def mirror_axis(img: np.ndarray) -> float:
    xs = np.nonzero(tape_inner(img).any(0))[0]
    return float(xs.min() + xs.max()) / 2


def mirror(img: np.ndarray) -> tuple[np.ndarray, float]:
    """테이프 안쪽만 좌우로 뒤집는다(테이프·그릇·로봇은 그대로). 반환 = (사진, 축 x px)."""
    inner = tape_inner(img) > 0
    xc = mirror_axis(img)
    h, w = inner.shape
    out = img.copy()
    ys, xs = np.nonzero(inner)
    src = np.round(2 * xc - xs).astype(int)
    okm = (src >= 0) & (src < w)
    okm[okm] = inner[ys[okm], src[okm]]
    out[ys[okm], xs[okm]] = img[ys[okm], src[okm]]
    return out, xc


def mirror_px(px, xc: float) -> list[float]:
    return [2 * xc - px[0], px[1]]


def empty(img: np.ndarray) -> np.ndarray:
    """테이프 안쪽의 블록(식탁보 색에서 벗어난 색·어두운 X)을 지운다 — 기준 = 테이프 안쪽 Lab 중앙값(식탁보 자체가
    푸르스름해 채도 기준은 전부를 블록으로 봤다). 지운 자리는 둘레 식탁보의 가중 평균(정규화 흐림)으로 채운다
    (inpaint 는 블록 색이 번져 옅은 얼룩이 남았다)."""
    inner = tape_inner(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int16)
    med = np.median(lab[inner > 0], axis=0)
    dev = np.abs(lab[..., 1] - med[1]) + np.abs(lab[..., 2] - med[2])
    blk = (((dev > 9) | (lab[..., 0] < med[0] - 45)) & (inner > 0)).astype(np.uint8) * 255
    blk = cv2.dilate(cv2.morphologyEx(blk, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)), np.ones((11, 11), np.uint8))
    blk &= inner
    if not blk.any():
        return img
    keep = ((inner > 0) & (blk == 0)).astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * keep[..., None], (0, 0), 25)
    den = cv2.GaussianBlur(keep, (0, 0), 25)[..., None]
    bg = np.clip(num / np.maximum(den, 1e-3), 0, 255).astype(np.uint8)
    out = img.copy()
    out[blk > 0] = bg[blk > 0]
    return out


def build_jobs(policy: str, groups: set[str], ds_only: str | None, limit_empty: int | None) -> tuple[list, dict, dict]:
    doc = json.loads((OUT / "scenes.json").read_text())
    det = json.loads((OUT / "det.json").read_text())
    rec = [s for s in doc["scenes"] if s["set"] == "rec" and (ds_only is None or s["ds"] == ds_only)]
    allrec = [s for s in doc["scenes"] if s["set"] == "rec"]
    photos = [s for s in doc["scenes"] if s["set"] in ("known4", "front")]
    by_id = {s["id"]: s for s in doc["scenes"]}
    home = list(load_config()["home_n"])
    fronts: dict[str, np.ndarray] = {}
    grips: dict[str, str] = {}
    jobs = []

    def other(i: int) -> dict:  # 짝 장면 = 같은 녹화에서 25 에피소드 뒤(다른 시각·배치, 목표 색은 같음)
        same = [s for s in allrec if s["ds"] == rec[i]["ds"]]
        return same[(same.index(rec[i]) + 25) % len(same)]

    def add(group, sid, fkey, fimg, gid, state, objects, extra=None):
        if fkey not in fronts:
            fronts[fkey] = fimg
        grips.setdefault(gid, by_id[gid]["grip"])
        jobs.append({"group": group, "id": sid, "front": fkey, "grip": gid, "state": state, "objects": objects,
                     **(extra or {})})

    for i, s in enumerate(rec):
        img = cv2.imread(str(ROOT / s["front"]))
        objs = det[s["id"]]["yolo"]
        base = {"ds": s["ds"], "human_w": s.get("human_w")}
        if "own" in groups:
            add("own", s["id"], s["id"], img, s["id"], s["state"], objs, base)
        if "swap" in groups:  # 위 사진 = 이 장면, 손목 사진 = 짝 장면 → 예측이 어느 쪽 목표를 따라가나
            o = other(i)
            add("swap", s["id"], s["id"], img, o["id"], s["state"], objs,
                {**base, "swap": o["id"], "swap_human_w": o.get("human_w")})
        if "mirror" in groups:
            mimg, xc = mirror(img)
            mobjs = [{**ob, "px": mirror_px(ob["px"], xc)} if ob.get("zone") == "input" else ob for ob in objs]
            add("mirror", s["id"], s["id"] + "#mirror", mimg, s["id"], s["state"], mobjs, {**base, "mirror_xc": xc})
        if "bgswap" in groups:  # 위 사진의 작업대 위쪽 교실 배경(y < BG_Y)만 짝 장면 것 — 손목·관절·블록은 이 장면
            o = other(i)
            mix = img.copy()
            mix[:BG_Y] = cv2.imread(str(ROOT / o["front"]))[:BG_Y]
            add("bgswap", s["id"], s["id"] + "#bg", mix, s["id"], s["state"], objs,
                {**base, "swap": o["id"], "swap_human_w": o.get("human_w")})
        if "inswap" in groups:  # 테이프 안쪽(블록)만 짝 장면 것 — 그릇 안 블록·팔·배경·손목·관절은 이 장면
            o = other(i)
            oimg = cv2.imread(str(ROOT / o["front"]))
            inner = (tape_inner(img) > 0) & (tape_inner(oimg) > 0)
            mix = img.copy()
            mix[inner] = oimg[inner]
            add("inswap", s["id"], s["id"] + "#in", mix, s["id"], s["state"], det[o["id"]]["yolo"],
                {**base, "swap": o["id"], "swap_human_w": o.get("human_w")})
        if "calib" in groups:  # 블록 없는 실제 작업대(calibrate 녹화 첫 장면) + 이 장면의 손목 사진·관절값
            if "calib_00" not in fronts:
                fronts["calib_00"] = cv2.imread(str(OUT / "scenes" / "calib_00_front.jpg"))
            add("calib", s["id"], "calib_00", None, s["id"], s["state"], [], base)
        if "empty" in groups and (limit_empty is None or i < limit_empty):
            add("empty", s["id"], s["id"] + "#empty", empty(img), s["id"], s["state"],
                [ob for ob in objs if ob.get("zone") != "input"], base)
    if "photo" in groups:
        subs = SUB_GRIPS if policy.startswith("act") else SUB_GRIPS[:1]
        for s in photos:
            img = cv2.imread(str(ROOT / s["front"]))
            for g in subs:
                add("photo", s["id"], s["id"], img, g, home, det[s["id"]]["yolo"], {"sub": g})
            if s["set"] == "known4" and "mirror" in groups:
                objs = det[s["id"]]["yolo"]
                mimg, xc = mirror(img)
                mobjs = [{**ob, "px": mirror_px(ob["px"], xc)} if ob.get("zone") == "input" else ob for ob in objs]
                add("photo_mirror", s["id"], s["id"] + "#mirror", mimg, subs[0], home, mobjs,
                    {"sub": subs[0], "mirror_xc": xc})
    return jobs, fronts, grips


def payload(jobs, fronts, grips) -> tuple[bytes, list, list]:
    fkeys, gkeys = list(fronts), list(grips)
    arr = {"nf": np.array(len(fkeys)), "ng": np.array(len(gkeys)),
           "jobs": np.array([[fkeys.index(j["front"]), gkeys.index(j["grip"])] for j in jobs], np.int32),
           "states": np.array([j["state"] for j in jobs], np.float32)}
    for i, k in enumerate(fkeys):
        arr[f"f{i}"] = np.frombuffer(cv2.imencode(".jpg", fronts[k], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(),
                                     np.uint8)
    for i, k in enumerate(gkeys):
        arr[f"g{i}"] = np.frombuffer((ROOT / grips[k]).read_bytes(), np.uint8)
    buf = io.BytesIO()
    np.savez(buf, **arr)
    return buf.getvalue(), fkeys, gkeys


def run(policy: str, groups: set[str], threads: int, ds_only: str | None, limit_empty: int | None,
        tag: str = "") -> dict:
    model_id, task = POLICIES[policy]
    jobs, fronts, grips = build_jobs(policy, groups, ds_only, limit_empty)
    data, fkeys, _ = payload(jobs, fronts, grips)
    for k in fkeys:  # 합성한 위 사진(좌우 반전·블록 지움)은 그림·확인용으로 남긴다
        if "#" in k:
            cv2.imwrite(str(OUT / "scenes" / (k.replace("#", "_") + ".jpg")), fronts[k], [cv2.IMWRITE_JPEG_QUALITY, 90])
    code = (ROOT / "robot" / "compare" / "_in_container.py").read_text()
    steps = MAX_STEPS["smolvla" if policy.startswith("smolvla") else "act"]
    cmd = ["docker", "--context", "default", "exec", "-i", "-u", "appuser", "-w", "/tmp", CONTAINER,
           "nice", "-n", "10", CPY, "-I", "-c", code, f"/app/storage/models/{model_id}/exports/openvino", task or "-",
           str(threads), str(steps)]
    t0 = time.time()
    p = subprocess.run(cmd, input=data, capture_output=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode(errors="replace")[-3000:])
        raise SystemExit(f"{policy}: 컨테이너 추론 실패 rc={p.returncode}")
    name = policy + (f"_{tag}" if tag else "")
    (OUT / f"infer_{name}.npz").write_bytes(p.stdout)
    (OUT / f"jobs_{name}.json").write_text(json.dumps(jobs, ensure_ascii=False))
    lines = [l for l in p.stderr.decode(errors="replace").splitlines() if "추론" in l]
    return {"policy": policy, "jobs": len(jobs), "wall_s": round(time.time() - t0, 1), "last": lines[-1] if lines else ""}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.compare.infer", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policies", nargs="+", choices=list(POLICIES))
    ap.add_argument("--groups", default="own,swap,mirror,empty,photo")
    ap.add_argument("--ds", choices=list(TASK_OF_DS), help="녹화 장면을 이 데이터셋만(SmolVLA 시간 줄이기)")
    ap.add_argument("--limit-empty", type=int)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="", help="결과 파일 이름 꼬리(infer_<정책>_<tag>.npz)")
    a = ap.parse_args(argv)
    for pol in a.policies:
        print(json.dumps(run(pol, set(a.groups.split(",")), a.threads, a.ds, a.limit_empty, a.tag), ensure_ascii=False),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
