"""Studio 녹화 데이터셋(LeRobot)에서 YOLO 학습 사진 뽑기 — 에피소드마다 첫·끝 프레임(+ 간격). 팔이 가리면 건너뛴다.

  python vision/frames_from_dataset.py bs_pick_red                                 # 첫·끝 → vision/data/raw/bs_pick_red/
  python vision/frames_from_dataset.py bs_handoff --every 1 --zones vision/zones.json   # 1초 간격도(손 장면), 가림 검사
  python vision/frames_from_dataset.py --list                                      # 데이터셋·카메라·에피소드 수

읽기만 한다. Studio API 의 GET 만 쓴다(/api/projects, /api/dataset/{id}/episodes[/{n}], /api/dataset/{id}/video/{경로}).
영상 파일은 vision/data/_video_cache/ 에 받아 OpenCV 로 디코드한다. Studio 볼륨·DB 는 건드리지 않는다.
녹화 화면이 열려 있는 동안(영상 파일을 쓰는 중)은 돌리지 않는다 — 정책 하나 녹화를 끝내고 ← 로 나온 뒤.

해상도는 영상 그대로(640x480), 크기 조절·색 보정 없이 JPEG q95(capture.py 와 같음)로 저장한다.
파일 이름: <데이터셋>_ep<에피소드 3자리>_f<에피소드 안 프레임 5자리>.jpg, 옆에 frames.json(뽑은 것·건너뛴 것·이유).

가림 검사(첫 프레임 = 팔이 시작 자세인 기준 장면):
  --zones 를 주면  input 구역 안에서 기준과 달라진 덩어리 중 가장 큰 것이 구역의 --occlude 비율(기본 5%) 이상이면 가림.
                  블록 하나를 옮긴 자리는 1~2% 라 걸리지 않고, 팔은 그보다 훨씬 크다.
  없으면          관절값(리더 팔 action)이 첫 프레임에서 --rest-tol(기본 8) 이상 벗어나면 팔이 나가 있는 것으로 본다.
흐림 검사: 선명도(라플라시안 분산)가 첫 프레임의 --min-sharp 배(기본 0.6) 미만이면 건너뛴다(움직임 흐림).
중복 검사: 바로 앞에 저장한 사진과 바뀐 픽셀이 --dedup %(기본 0.3) 미만이면 같은 장면으로 보고 건너뛴다.
  녹화 가이드는 블록 3개를 깔고 3에피소드를 손대지 않고 이어 찍으므로, 에피소드 끝 장면과 다음 첫 장면이 같다.
  (10-20 녹화 실측: 팔이 홈인 같은 장면 0.5초 차이 0.00~0.04%, 물체 하나 옮긴 장면 1.4~2.9%)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from . import camera, zones as zonelib
except ImportError:
    import camera
    import zones as zonelib

VISION = Path(__file__).resolve().parent
CACHE = VISION / "data" / "_video_cache"


# ---------------------------------------------------------------- Studio API (GET 만)

def api(path: str):
    """GET 한 번. Studio 는 데이터셋 요청을 동기로 처리해 바쁠 때 수십 초 멈출 수 있다(10-21 밤 1분 30초) → 한 번 더."""
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(camera.STUDIO + path, timeout=60) as r:
                return json.load(r)
        except (TimeoutError, OSError) as e:
            if attempt == 2:
                sys.exit(f"Studio 응답 없음({e}) — Studio 가 녹화·학습 중이면 끝난 뒤 다시: {path}")
            time.sleep(10)


def datasets() -> list[dict]:
    out = []
    for p in api("/api/projects"):
        for d in p.get("datasets") or []:
            out.append({**d, "project": p["name"]})
    return out


def find_dataset(key: str) -> dict:
    found = [d for d in datasets() if key in (d["name"], d["id"])]
    if not found:
        sys.exit(f"데이터셋 없음: {key} — --list 로 이름 확인")
    if len(found) > 1:
        sys.exit(f"같은 이름이 여러 개: {[(d['project'], d['id']) for d in found]} — id 로 줄 것")
    return found[0]


_fetched: dict[str, Path] = {}


def fetch_video(ds_id: str, rel: str) -> Path:
    """영상 파일을 캐시에 받는다. 앞 1바이트 Range 요청으로 etag 를 보고, 바뀌었을 때만 전부 다시 받는다.
    한 번 실행에 파일마다 한 번만 묻는다(에피소드 수십 개가 같은 파일을 쓴다)."""
    if rel in _fetched:
        return _fetched[rel]
    url = f"{camera.STUDIO}/api/dataset/{ds_id}/video/{urllib.parse.quote(rel)}"
    dst = CACHE / ds_id / rel
    tag_file = dst.with_name(dst.name + ".etag")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as r:
        etag = r.headers.get("etag", "")
    if dst.exists() and tag_file.exists() and tag_file.read_text() == etag:
        _fetched[rel] = dst
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    part.replace(dst)
    tag_file.write_text(etag)
    _fetched[rel] = dst
    return dst


# ---------------------------------------------------------------- 검사

def sharpness(gray, mask=None) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap[mask > 0].var() if mask is not None else lap.var())


def occlusion(gray, ref, zone_mask) -> float:
    """기준(팔 시작 자세)과 달라진 덩어리 중 가장 큰 것의 면적 / input 구역 면적."""
    # 밝기 차 15: 팔은 작업대와 밝기가 비슷한 부분이 있어 30 으로는 조각나 작게 잡힌다(10-20 녹화 실측)
    diff = cv2.absdiff(cv2.GaussianBlur(gray, (5, 5), 0), cv2.GaussianBlur(ref, (5, 5), 0))
    changed = ((diff > 15) & (zone_mask > 0)).astype(np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(changed)
    biggest = stats[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0
    return float(biggest) / max(1, int(np.count_nonzero(zone_mask)))


def changed_pct(gray, prev) -> float:
    """두 장면에서 밝기가 30 넘게 바뀐 픽셀 비율(%)."""
    d = cv2.absdiff(cv2.GaussianBlur(gray, (5, 5), 0), cv2.GaussianBlur(prev, (5, 5), 0))
    return float((d > 30).mean() * 100)


def joint_dist(actions: list, k: int) -> float | None:
    """관절값이 첫 프레임에서 가장 많이 벗어난 정도(관절 단위, 그리퍼 제외)."""
    if not actions or k >= len(actions):
        return None
    a0, ak = np.array(actions[0][:5], float), np.array(actions[k][:5], float)
    return float(np.abs(ak - a0).max())


# ---------------------------------------------------------------- 뽑기

def parse_episodes(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return sorted(i for i in out if 0 <= i < n)


def plan_frames(length: int, fps: float, every: float | None) -> list[tuple[int, str]]:
    ks = [(0, "first")]
    if every:
        step = max(1, round(every * fps))
        ks += [(k, "every") for k in range(step, length - 1, step)]
    ks.append((length - 1, "last"))
    return ks


def extract(a) -> dict:
    ds = find_dataset(a.dataset)
    name = re.sub(r"[^\w.-]+", "_", ds["name"])
    out = Path(a.out) if a.out else VISION / "data" / "raw" / name
    out.mkdir(parents=True, exist_ok=True)
    infos = api(f"/api/dataset/{ds['id']}/episodes")
    eps = parse_episodes(a.episodes, len(infos))
    zones = None
    jobs: dict[Path, list] = {}   # 영상 파일 → [(전역 프레임 번호, 에피소드, 에피소드 안 번호, 종류, 관절 거리)]
    for e in eps:
        ep = api(f"/api/dataset/{ds['id']}/episodes/{e}")
        keys = list(ep["videos"])
        key = next((k for k in keys if k == f"observation.images.{a.camera}" or k.endswith(a.camera)), None)
        if key is None:
            sys.exit(f"카메라 '{a.camera}' 없음 — 이 데이터셋의 카메라: {keys}")
        v, fps = ep["videos"][key], ep["fps"]
        path = fetch_video(ds["id"], v["path"])
        base = round(v["start"] * fps)
        for k, kind in plan_frames(ep["length"], fps, a.every):
            jobs.setdefault(path, []).append((base + k, e, k, kind, joint_dist(ep.get("actions"), k)))
    records, refs, prev = [], {}, None
    for path, items in jobs.items():
        items.sort()
        cap = cv2.VideoCapture(str(path))
        pos, want = 0, {g: rest for g, *rest in items}
        last = max(want)
        while pos <= last:
            if pos not in want:
                if not cap.grab():
                    break
                pos += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            e, k, kind, jd = want[pos]
            pos += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if zones is None and a.zones:
                zones = zonelib.load_zones(a.zones, (frame.shape[1], frame.shape[0]))
            zmask = None
            if zones and "input" in zones:
                zmask = np.zeros(gray.shape, np.uint8)
                cv2.fillPoly(zmask, [zones["input"]], 255)
            rec = {"episode": e, "frame": k, "kind": kind, "joint_dist": None if jd is None else round(jd, 1),
                   "sharp": round(sharpness(gray, zmask), 1)}
            if kind == "first":
                refs[e] = (gray, rec["sharp"])
            elif e in refs:
                ref_gray, ref_sharp = refs[e]
                if zmask is not None:
                    rec["occlusion"] = round(occlusion(gray, ref_gray, zmask), 4)
                    if rec["occlusion"] >= a.occlude:
                        rec["skip"] = "팔이 투입 구역을 가림"
                elif jd is not None and jd >= a.rest_tol:
                    rec["skip"] = "팔이 시작 자세가 아님(관절)"
                if "skip" not in rec and rec["sharp"] < a.min_sharp * ref_sharp:
                    rec["skip"] = "흐림"
            if "skip" not in rec and prev is not None and a.dedup > 0:
                rec["changed_pct"] = round(changed_pct(gray, prev), 2)
                if rec["changed_pct"] < a.dedup:
                    rec["skip"] = "앞 사진과 같은 장면"
            if "skip" not in rec:
                prev = gray
                fn = out / f"{name}_ep{e:03d}_f{k:05d}.jpg"
                cv2.imwrite(str(fn), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rec["file"] = fn.name
            records.append(rec)
        cap.release()
    missing = sum(len(v) for v in jobs.values()) - len(records)
    summary = {"dataset": ds["name"], "dataset_id": ds["id"], "camera": a.camera, "episodes": len(eps),
               "saved": sum("file" in r for r in records), "skipped": sum("skip" in r for r in records),
               "not_decoded": missing, "out": str(out)}
    (out / "frames.json").write_text(json.dumps({"created": datetime.now().astimezone().isoformat(timespec="seconds"),
                                                 **summary, "args": vars(a), "frames": records},
                                                ensure_ascii=False, indent=1, default=str))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", help="Studio 데이터셋 이름 또는 id")
    ap.add_argument("--list", action="store_true", help="데이터셋·카메라·에피소드 수")
    ap.add_argument("--camera", default="front", help="카메라 이름(위 카메라 = front)")
    ap.add_argument("--episodes", help="예: 0-9,12 (기본 전부)")
    ap.add_argument("--every", type=float, help="첫·끝 사이에 이 초 간격으로도 뽑기(손 장면 등)")
    ap.add_argument("--zones", help="zones.json — 주면 input 구역 가림 검사, 없으면 관절값으로")
    ap.add_argument("--occlude", type=float, default=0.05, help="가림 기준(input 구역 대비 가장 큰 변화 덩어리)")
    ap.add_argument("--rest-tol", type=float, default=8.0, help="관절 기준(첫 프레임에서 벗어난 최대값)")
    ap.add_argument("--min-sharp", type=float, default=0.6, help="흐림 기준(첫 프레임 선명도 대비)")
    ap.add_argument("--dedup", type=float, default=0.3, help="앞에 저장한 사진과 바뀐 픽셀 %% 가 이보다 작으면 건너뜀(0 = 끔)")
    ap.add_argument("--out", help="저장 폴더(기본 vision/data/raw/<데이터셋>)")
    a = ap.parse_args()
    if a.list:
        for d in datasets():
            eps = api(f"/api/dataset/{d['id']}/episodes")
            cams = list(api(f"/api/dataset/{d['id']}/episodes/0")["videos"]) if eps else []
            print(f"{d['project']}/{d['name']}  id={d['id']}  에피소드 {len(eps)}  프레임 {sum(e['length'] for e in eps)}  카메라 {cams}")
        return
    if not a.dataset:
        ap.print_help()
        return
    print(json.dumps(extract(a), ensure_ascii=False))


if __name__ == "__main__":
    main()
