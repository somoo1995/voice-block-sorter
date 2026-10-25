"""1단계: 비교에 쓸 장면을 모은다(파일 읽기만 — 카메라·로봇·추론 없음).

    .venv/bin/python -m robot.compare.scenes          # → robot/logs/n2/scenes.json + scenes/*.jpg

장면 묶음
- rec   : 녹화 3종(pick_defect·pick_red·pick_green) 에피소드마다 첫 프레임(위·손목 카메라)·첫 관절값 + 사람이 실제로
          집은 자리(그리퍼 닫힘 시작의 FK, robot/logs/r14/grasps.json).
- known4: 구성을 아는 장면 30장(정상 분홍·초록 1 + 불량 분홍·초록 1, vision/data/raw/known4) — 위 카메라만.
- front : 좌표 집기 실물 때 찍은 위 카메라 원본(robot/logs/20261022/front_*.jpg) — 위 카메라만.
- real  : 오늘 실물 정책 실행(ACT·SmolVLA) — 실행 직전 검출 사진(runs/20261022/snap_*, 박스가 그려진 사진이라 정책
          입력으로는 안 씀) + 실제 관절 기록(ticks.jsonl) → 실물에서 실제로 닫은 자리.
손목 사진·관절값이 없는 장면은 3단계에서 녹화 첫 장면의 홈 자세 값으로 채운다(대용 — 영향은 rec 로 따로 잰다).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R14 = ROOT / "robot" / "logs" / "r14"
OUT = ROOT / "robot" / "logs" / "n2"
DATASETS = {"pick_defect": "d9e18d33-5e02-427a-b7bb-fa29c61d3853", "pick_red": "dd6126e6-4ac0-4d56-9400-527135597fb8",
            "pick_green": "07f0afd2-518c-4178-a6fa-1af90b01aad0"}
# 실물 실행 기록의 모델 → 정책 이름(robot/logs/20261022/*.runtime.json 의 export_dir)
MODELS = {"ef64c0f9": "act_defect", "708a44b7": "act_red", "b3407a54": "act_green", "d887e672": "smolvla_red"}
KST = dt.timezone(dt.timedelta(hours=9))
FPS = 30


class _Videos:
    def __init__(self):
        self.caps: dict[Path, cv2.VideoCapture] = {}

    def frame(self, path: Path, index: int) -> np.ndarray:
        cap = self.caps.get(path)
        if cap is None:
            cap = self.caps[path] = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"디코드 실패: {path} #{index}")
        return img


def _save(img: np.ndarray, name: str) -> str:
    p = OUT / "scenes" / name
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(p.relative_to(ROOT))


def rec_scenes() -> list[dict]:
    rec = json.loads((R14 / "rec.json").read_text())
    meta = json.loads((R14 / "meta.json").read_text())
    meta_g = json.loads((R14 / "meta_grip.json").read_text())
    grasps = {(g["ds"], g["ep"]): g for g in json.loads((R14 / "grasps.json").read_text())}
    v = _Videos()
    out = []
    for ds, did in DATASETS.items():
        mg = {e["episode_index"]: e for e in meta_g[did]}
        for e in sorted(meta[did], key=lambda e: e["episode_index"]):
            ep = e["episode_index"]
            key = "videos/observation.images.{}/"
            fr = v.frame(R14 / "video" / did / f"file-{int(e[key.format('front') + 'file_index']):03d}.mp4",
                         round(e[key.format("front") + "from_timestamp"] * FPS))
            g_e = mg[ep]
            gr = v.frame(R14 / "video_grip" / did / f"file-{int(g_e[key.format('gripper') + 'file_index']):03d}.mp4",
                         round(g_e[key.format("gripper") + "from_timestamp"] * FPS))
            ep_rec = rec[did]["episodes"][str(ep)]
            g = grasps.get((ds, ep))
            out.append({"id": f"rec_{ds}_{ep:02d}", "set": "rec", "ds": ds, "ep": ep,
                        "front": _save(fr, f"rec_{ds}_{ep:02d}_front.jpg"),
                        "grip": _save(gr, f"rec_{ds}_{ep:02d}_grip.jpg"),
                        "state": ep_rec["s"][0], "frames": len(ep_rec["s"]),
                        "human_w": g["g"]["w"] if g else None, "human_gi": g["gi"] if g else None})
    return out


def photo_scenes() -> list[dict]:
    out = []
    for p in sorted((ROOT / "vision" / "data" / "raw" / "known4").glob("k4_*.jpg")):
        out.append({"id": f"known4_{p.stem[3:]}", "set": "known4", "front": str(p.relative_to(ROOT)), "grip": None,
                    "state": None})
    for p in sorted((ROOT / "robot" / "logs" / "20261022").glob("front_*.jpg")):
        out.append({"id": f"front_{p.stem[6:]}", "set": "front", "front": str(p.relative_to(ROOT)), "grip": None,
                    "state": None})
    return out


def _ticks(path: Path) -> tuple[float | None, list]:
    t0, rows = None, []
    for line in path.read_text().splitlines():
        if not line.startswith("["):
            continue
        ts, q = json.loads(line)
        t0 = ts if t0 is None else t0
        rows.append([round(ts - t0, 3), q])
    return t0, rows


def real_runs() -> list[dict]:
    """오늘 정책 실물 실행: 관절 기록이 1초 넘게 있는 것만. 실행 직전(120초 안) 검출 사진을 붙인다."""
    logs = ROOT / "robot" / "logs" / "20261022"
    snaps = []
    for p in sorted((ROOT / "runs" / "20261022").glob("snap_*.jpg")):
        hms, ms = p.stem.split("_")[1:]
        snaps.append((dt.datetime.strptime(f"20261022{hms}", "%Y%m%d%H%M%S").replace(tzinfo=KST).timestamp()
                      + int(ms) / 1000, p))
    out = []
    for rt in sorted(logs.glob("*.runtime.json")):
        stem = rt.name[:-len(".runtime.json")]
        if "_pick_" not in f"_{stem}":
            continue
        model = next((m for m in MODELS if f"models/{m}" in rt.read_text()), None)
        tk = logs / f"{stem}.ticks.jsonl"
        if model is None or not tk.exists():
            continue
        t0, rows = _ticks(tk)
        if len(rows) < FPS:
            continue
        before = [(t, p) for t, p in snaps if t0 - 120 <= t <= t0]
        snap = before[-1] if before else None
        out.append({"id": f"real_{stem}", "set": "real", "policy": MODELS[model], "task": stem.split("_", 1)[1],
                    "start": dt.datetime.fromtimestamp(t0, KST).strftime("%H:%M:%S"), "ticks": rows,
                    "snap": str(snap[1].relative_to(ROOT)) if snap else None,
                    "snap_age_s": round(t0 - snap[0], 1) if snap else None})
    return out


def main() -> int:
    (OUT / "scenes").mkdir(parents=True, exist_ok=True)
    rec = rec_scenes()
    photos = photo_scenes()
    real = real_runs()
    doc = {"made": dt.datetime.now(KST).isoformat(timespec="seconds"), "scenes": rec + photos, "real": real}
    (OUT / "scenes.json").write_text(json.dumps(doc, ensure_ascii=False))
    print(f"녹화 첫 장면 {len(rec)} · 사진 {len(photos)}(known4 {sum(p['set'] == 'known4' for p in photos)}) · "
          f"실물 실행 {len(real)}(직전 사진 있음 {sum(r['snap'] is not None for r in real)})")
    for r in real:
        print(f"  {r['start']} {r['policy']:<12} ticks {len(r['ticks']):>4} 사진 {r['snap']} ({r['snap_age_s']}s 전)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
