"""2단계: 모든 장면 위 카메라 사진에 YOLO(vision/models/ball_int8, NPU)와 색 규칙(HSV) 검출을 돌린다. 카메라는 안 연다.

    .venv/bin/python -m robot.compare.detect_all      # → robot/logs/n2/det.json, robot/logs/n2/yolo/*.jpg(박스 사진)

실물 실행 직전 사진(runs/20261022/snap_*)은 이미 박스가 그려진 사진이라 YOLO 를 다시 돌린 값은 참고만 하고, 블록 자리·
색은 사람 확인(truth_fix.json)으로 맞춘다.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2

from robot.compare.common import OUT, ROOT
from robot.coord_pick import detect_hsv


def main() -> int:
    os.environ.setdefault("YOLO_OFFLINE", "1")  # Ultralytics 통계 전송 끔(위키 '오프라인' 절)
    sys.path.insert(0, str(ROOT))
    from vision.detect import detect_objects

    doc = json.loads((OUT / "scenes.json").read_text())
    items = [(s["id"], s["front"]) for s in doc["scenes"]] + [(r["id"], r["snap"]) for r in doc["real"] if r["snap"]]
    out, t0 = {}, time.time()
    (OUT / "yolo").mkdir(exist_ok=True)
    for sid, path in items:
        d = detect_objects(str(ROOT / path), out_dir=OUT / "yolo")
        img = cv2.imread(str(ROOT / path))
        h = detect_hsv.detect_hsv(img)
        out[sid] = {"image": path, "yolo": d["objects"], "detections": d["detections"], "hand": d["hand"],
                    "yolo_snap": d["image_path"], "latency_ms": d["latency_ms"], "device": d["device"],
                    "hsv": [{k: o[k] for k in ("cls", "px", "zone", "parts")} for o in h["objects"]]}
    (OUT / "det.json").write_text(json.dumps(out, ensure_ascii=False))
    lat = sorted(v["latency_ms"] for v in out.values())
    print(f"검출 {len(out)}장 {time.time() - t0:.0f}초, 장치 {sorted({v['device'] for v in out.values()})}, "
          f"지연 중앙 {lat[len(lat) // 2]:.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
