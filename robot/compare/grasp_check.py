"""집기 정책 오프라인 검사(모델 ID 하나) — 녹화 시작 장면에서 첫 청크가 블록 쪽으로 내려가 쥐나, 블록을 지우거나
손목 사진을 좌우로 뒤집으면 반응이 바뀌나(10-22 '장면 외우기' 시험과 같은 방식). 로봇·카메라 0, 추론은 Studio 컨테이너 CPU.

    .venv/bin/python -m robot.compare.grasp_check <모델ID> [--threads 8] [--limit N]

재료(읽기만): robot/logs/n6/grasp_rec.json(에피소드 상태·동작), robot/logs/n6/ds_grasp/videos(첫 프레임).
없으면 먼저 robot/logs/n6/README 처럼 컨테이너에서 꺼낸다(--fetch 가 해 준다).

장면마다 6가지 입력(관절값은 모두 녹화 첫 값):
  orig        위·손목 사진 그대로
  grip_empty  손목 사진에서 블록을 지움(색 덩어리 + 둘레 검은 테두리·X 를 메움)
  front_empty 위 사진 투입 구역의 블록을 지움
  both_empty  둘 다 지움
  grip_mirror 손목 사진만 좌우 반전(블록이 끝 좌표 y 반대쪽에 보임)
  front_mirror 위 사진만 좌우 반전
판정(첫 청크 100틱 = 3.3초, 녹화 사람은 중앙 3.0초에 쥐기 시작):
  - 내려감: 시작 높이 − 청크 최저 높이(mm), 닫음: 그리퍼가 시작보다 6 넘게 닫힘
  - 블록 쪽: 최저점과 실제 쥔 자리(사람 FK) 거리, 시작 때 거리에서 얼마나 줄었나
  - 영상을 보나: 장면 사이 상관(예측 이동 ↔ 실제 블록 방향, 끝 좌표 x·y), 손목 반전 때 끝 좌표 y 이동이 뒤집히나
    (기울기 −1 = 따라감, +1 = 사진 무시), 블록을 지우면 내려가 닫는 비율이 줄어드나
⚠ 학습에 쓴 장면이다(검증 5% 는 시드 없이 뽑혀 모름) — orig 성적은 낙관적, 지움·반전이 영상 의존을 본다.
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from robot.compare.common import closing, setup

ROOT = Path(__file__).resolve().parent.parent.parent
N6 = ROOT / "robot" / "logs" / "n6"
DS = "7b16f024-92c5-4edc-b974-9284b2b86d00"
CONTAINER = "physical-ai-studio-xpu"
CPY = "/app/application/backend/.venv/bin/python"
DOCKER = ["docker", "--context", "default"]
VARIANTS = ("orig", "grip_empty", "front_empty", "both_empty", "grip_mirror", "front_mirror")


def fetch(ds: str = DS) -> None:
    """컨테이너 데이터셋 → robot/logs/n6(영상 tar, 에피소드 JSON). 읽기만."""
    dst = N6 / "ds_grasp"
    dst.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(DOCKER + ["exec", CONTAINER, "tar", "-C", f"/app/storage/datasets/{ds}", "-cf", "-", "."],
                         capture_output=True, check=True).stdout
    subprocess.run(["tar", "-xf", "-", "-C", str(dst)], input=tar, check=True)
    code = (N6 / "dump_ds.py").read_text()
    out = subprocess.run(DOCKER + ["exec", "-i", CONTAINER, CPY, "-I", "-", f"/app/storage/datasets/{ds}"],
                         input=code.encode(), capture_output=True, check=True).stdout
    (N6 / "grasp_rec.json").write_bytes(out)


def _frame(path: Path, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * 30))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"디코드 실패 {path} {t}")
    return img


def _color_mask(img) -> np.ndarray:
    sys.path.insert(0, str(ROOT))
    from vision import autolabel
    mk = autolabel.masks(img, autolabel.load_config())
    return cv2.morphologyEx((mk["red"] | mk["blue"]).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def _erase(img, keep_poly=None, min_area=150) -> np.ndarray:
    """색 덩어리(블록)를 둘레 테두리·X 까지 덮어 메운다. keep_poly(px 다각형)가 있으면 그 안의 덩어리만."""
    m = _color_mask(img)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    hole = np.zeros(m.shape, np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < min_area:
            continue
        if keep_poly is not None and cv2.pointPolygonTest(keep_poly, (float(cen[i][0]), float(cen[i][1])), True) < -10:
            continue
        hole |= (lab == i).astype(np.uint8)
    if not hole.any():
        return img.copy()
    # 조각(X 로 갈린 것)을 이어 볼록 껍질로 — 검은 X·테두리까지 메우게
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    n2, lab2 = cv2.connectedComponents(cv2.dilate(hole, k))
    fill = np.zeros_like(hole)
    for i in range(1, n2):
        pts = cv2.findNonZero((lab2 == i).astype(np.uint8) & hole)
        if pts is not None:
            cv2.fillConvexPoly(fill, cv2.convexHull(pts), 1)
    fill = cv2.dilate(fill, np.ones((9, 9), np.uint8))
    return cv2.inpaint(img, fill * 255, 7, cv2.INPAINT_TELEA)


def scenes(limit: int | None = None) -> list[dict]:
    rec = json.loads((N6 / "grasp_rec.json").read_text())
    meta = {m["episode_index"]: m for m in rec["meta"]}
    zones = json.loads((ROOT / "vision" / "zones.json").read_text())["zones"]
    inp = np.array(zones["input"], np.int32)
    out = []
    for ep, e in sorted(rec["episodes"].items(), key=lambda kv: int(kv[0]))[:limit]:
        m = meta[int(ep)]
        vid = {}
        for cam in ("front", "gripper"):
            k = f"videos/observation.images.{cam}/"
            vid[cam] = _frame(N6 / "ds_grasp" / "videos" / f"observation.images.{cam}" / f"chunk-{int(m[k + 'chunk_index']):03d}"
                              / f"file-{int(m[k + 'file_index']):03d}.mp4", float(m[k + "from_timestamp"]))
        A = np.asarray(e["a"], float)
        gi = closing(A[:, 5], rest=float(np.median(A[:5, 5])) - 7)
        f, g = vid["front"], vid["gripper"]
        imgs = {"orig": (f, g), "grip_empty": (f, _erase(g)), "front_empty": (_erase(f, inp), g),
                "grip_mirror": (f, cv2.flip(g, 1)), "front_mirror": (cv2.flip(f, 1), g)}
        imgs["both_empty"] = (imgs["front_empty"][0], imgs["grip_empty"][1])
        out.append({"ep": int(ep), "state": e["s"][0], "human": e["a"][:100], "gi": gi,
                    "grasp_n": e["s"][gi] if gi is not None else None, "imgs": imgs})
    return out


def infer(model_id: str, sc: list[dict], threads: int) -> tuple[list[dict], np.lib.npyio.NpzFile]:
    fr, gr, jobs = [], [], []
    for s in sc:
        for v in VARIANTS:
            f, g = s["imgs"][v]
            fr.append(np.frombuffer(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), np.uint8))
            gr.append(np.frombuffer(cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes(), np.uint8))
            jobs.append({"ep": s["ep"], "v": v})
    arr = {"nf": np.array(len(fr)), "ng": np.array(len(gr)),
           "jobs": np.array([[i, i] for i in range(len(jobs))], np.int32),
           "states": np.array([s["state"] for s in sc for _ in VARIANTS], np.float32)}
    arr.update({f"f{i}": x for i, x in enumerate(fr)})
    arr.update({f"g{i}": x for i, x in enumerate(gr)})
    buf = io.BytesIO()
    np.savez(buf, **arr)
    code = (ROOT / "robot" / "compare" / "_in_container.py").read_text()
    cmd = DOCKER + ["exec", "-i", "-u", "appuser", "-w", "/tmp", CONTAINER, "nice", "-n", "10", CPY, "-I", "-c", code,
                    f"/app/storage/models/{model_id}/exports/openvino", "-", str(threads), "100"]
    p = subprocess.run(cmd, input=buf.getvalue(), capture_output=True)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode(errors="replace")[-3000:])
        raise SystemExit(f"컨테이너 추론 실패 rc={p.returncode}")
    return jobs, np.load(io.BytesIO(p.stdout))


def _traj(s, acts) -> dict:
    """동작(정규화 6, 틱별) → 끝 좌표계(시작 자세) 기준 이동·높이·그리퍼."""
    A = np.asarray(acts, float)
    q0 = s.n_to_q(A[0])
    T0 = s.kin.fk(q0)
    P = np.array([s.tcp_pos(s.n_to_q(a)) for a in A])
    return {"P": P, "R0": T0[:3, :3], "g": A[:, 5]}


def metrics(s, state, acts, grasp_n) -> dict:
    t = _traj(s, acts)
    P, g = t["P"], t["g"]
    p0 = s.tcp_pos(s.n_to_q(np.asarray(state, float)))
    low = int(np.argmin(P[:, 2]))
    out = {"descend_mm": round(float(p0[2] - P[low, 2]), 1), "low_z": round(float(P[low, 2]), 1),
           "closes": bool(float(state[5]) - float(g.min()) > 6),
           "move_end": np.round((P[low] - p0) @ t["R0"], 1).tolist()}  # 최저점까지 이동(시작 끝 좌표계 x·y·z)
    if grasp_n is not None:
        pg = s.tcp_pos(s.n_to_q(np.asarray(grasp_n, float)))
        out.update(d_start=round(float(np.linalg.norm((p0 - pg)[:2])), 1),
                   d_low=round(float(np.linalg.norm((P[low] - pg)[:2])), 1),
                   true_end=np.round((pg - p0) @ t["R0"], 1).tolist())
    return out


def _slope(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return None if len(x) < 3 or np.ptp(x) < 1e-6 else round(float(np.polyfit(x, y, 1)[0]), 2)


def _r(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    return None if len(x) < 3 or x.std() < 1e-6 or y.std() < 1e-6 else round(float(np.corrcoef(x, y)[0, 1]), 2)


def evaluate(sc, jobs, res) -> dict:
    s = setup()
    by = {}
    for n, j in enumerate(jobs):
        sce = next(x for x in sc if x["ep"] == j["ep"])
        by.setdefault(j["v"], {})[j["ep"]] = metrics(s, sce["state"], res[f"{n}_first"], sce["grasp_n"])
    human = {x["ep"]: metrics(s, x["state"], x["human"], x["grasp_n"]) for x in sc}
    eps = [x["ep"] for x in sc if x["grasp_n"] is not None]

    def agg(rows: dict) -> dict:
        v = [rows[e] for e in eps]
        return {"descend_mm_med": round(float(np.median([r["descend_mm"] for r in v])), 1),
                "closes": sum(r["closes"] for r in v), "n": len(v),
                "d_start_med": round(float(np.median([r["d_start"] for r in v])), 1),
                "d_low_med": round(float(np.median([r["d_low"] for r in v])), 1),
                "toward_ok": sum(r["d_low"] < r["d_start"] for r in v),
                "r_x": _r([r["true_end"][0] for r in v], [r["move_end"][0] for r in v]),
                "r_y": _r([r["true_end"][1] for r in v], [r["move_end"][1] for r in v])}
    out = {"human_first100": agg(human), **{v: agg(by[v]) for v in VARIANTS}}
    o, m = by["orig"], by["grip_mirror"]
    out["grip_mirror_y_slope"] = _slope([o[e]["move_end"][1] for e in eps], [m[e]["move_end"][1] for e in eps])
    out["grip_mirror_x_slope"] = _slope([o[e]["move_end"][0] for e in eps], [m[e]["move_end"][0] for e in eps])
    for v in ("grip_empty", "front_empty", "both_empty", "front_mirror"):
        out[f"{v}_shift_mm_med"] = round(float(np.median([np.linalg.norm(np.subtract(by[v][e]["move_end"][:2],
                                                                                     o[e]["move_end"][:2])) for e in eps])), 1)
    out["rows"] = {v: by[v] for v in VARIANTS}
    out["rows"]["human"] = human
    return out


def verdict(ev: dict) -> list[str]:
    o, h = ev["orig"], ev["human_first100"]
    lines = [f"녹화 시작 장면 {o['n']}개 · 첫 청크(3.3초): 내려감 중앙 {o['descend_mm_med']}mm(사람 {h['descend_mm_med']}) · 닫음 "
             f"{o['closes']}/{o['n']}(사람 {h['closes']}) · 블록 쪽으로 다가감 {o['toward_ok']}/{o['n']} · 최저점–쥔 자리 중앙 "
             f"{o['d_low_med']}mm(시작 {o['d_start_med']}, 사람 {h['d_low_med']})",
             f"블록 방향 따라가나(장면 사이 상관, 끝 좌표 x·y): 정책 r {o['r_x']}·{o['r_y']} / 사람 {h['r_x']}·{h['r_y']}",
             f"손목 좌우 반전 → 끝 좌표 y 이동 기울기 {ev['grip_mirror_y_slope']}(−1 = 따라감, +1 = 무시), x {ev['grip_mirror_x_slope']}"
             f" · 바뀐 거리 중앙 front_mirror {ev['front_mirror_shift_mm_med']}mm",
             f"블록 지움 → 닫음 손목 {ev['grip_empty']['closes']}·위 {ev['front_empty']['closes']}·둘 다 "
             f"{ev['both_empty']['closes']} /{o['n']}, 내려감 중앙 {ev['grip_empty']['descend_mm_med']}·"
             f"{ev['front_empty']['descend_mm_med']}·{ev['both_empty']['descend_mm_med']}mm, 최저점 이동 중앙 "
             f"{ev['grip_empty_shift_mm_med']}·{ev['front_empty_shift_mm_med']}·{ev['both_empty_shift_mm_med']}mm"]
    sees = (ev["grip_mirror_y_slope"] is not None and ev["grip_mirror_y_slope"] < 0.3) or \
           ev["both_empty"]["closes"] <= o["closes"] // 2
    lines.append("결론: " + ("손목 영상에 반응한다(반전·지움에서 바뀜)" if sees else
                            "영상 반응이 약하다 — 시작 자세(관절값)만 보고 외운 동작일 가능성. 실물 전 IK 대비 기대 낮춤"))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.compare.grasp_check", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_id")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--fetch", action="store_true", help="데이터셋을 컨테이너에서 다시 꺼낸다(읽기만)")
    a = ap.parse_args(argv)
    if a.fetch or not (N6 / "grasp_rec.json").exists():
        fetch()
    t0 = time.time()
    sc = scenes(a.limit)
    for s in sc[:3]:  # 지운·뒤집은 사진 확인용
        for v in ("grip_empty", "front_empty", "grip_mirror"):
            f, g = s["imgs"][v]
            cv2.imwrite(str(N6 / f"check_ep{s['ep']:02d}_{v}.jpg"), np.hstack([f, g]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    jobs, res = infer(a.model_id, sc, a.threads)
    lat = np.concatenate([res[k] for k in res.files if k.endswith("_lat")])
    ev = evaluate(sc, jobs, res)
    ev.update(model_id=a.model_id, wall_s=round(time.time() - t0, 1), cpu_latency_ms_med=round(float(np.median(lat)) * 1000))
    out = N6 / f"check_{a.model_id[:8]}.json"
    out.write_text(json.dumps(ev, ensure_ascii=False, default=float))
    for line in verdict(ev):
        print(line)
    print(f"CPU 추론 중앙 {ev['cpu_latency_ms_med']}ms · {ev['wall_s']}초 · 자세히 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
