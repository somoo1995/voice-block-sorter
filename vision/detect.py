"""계약서 3절 detect_objects() — 위 카메라 한 장 → OpenVINO NPU 추론 → 구역별 개수 + 블록 목록(objects).

  from vision.detect import detect_objects        # 에이전트 (프로젝트 .venv)
  python vision/detect.py [--image x.jpg] [--model DIR] [--device intel:npu]

모델: vision/models/ball_int8_openvino_model (train.py 가 만듦. 물체는 블록이지만 폴더 이름은 식별자라 ball_ 그대로). 없으면 COCO 사전학습 모델로 같은 코드 경로를 돈다
(그때는 COCO_ALIASES 로 이름만 바꿔 센다). 모델은 프로세스당 한 번만 읽고 컴파일한다.
"""
from __future__ import annotations

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # 빠진 패키지를 몰래 설치하지 않게
os.environ.setdefault("YOLO_VERBOSE", "false")  # Ultralytics 로그가 stdout 의 JSON 에 섞이지 않게
os.environ.setdefault("YOLO_OFFLINE", "1")  # import 때 DNS 확인·predict 마다 통계 전송을 끈다(시연은 오프라인)

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

try:
    from . import camera, zones as zonelib
except ImportError:
    import camera
    import zones as zonelib

VISION = Path(__file__).resolve().parent
ROOT = VISION.parent
BALL_CLASSES = ("red_ok", "blue_ok", "red_defect", "blue_defect")
SKIN_HAND_FRAC = float(os.environ.get("BALL_SKIN_HAND_FRAC", "0.08"))  # 손 자리 안 피부색 비율, 0 이하면 끔
# 구역별 신뢰도 문턱(값은 여기 한 곳). 가장 낮은 값으로 한 번 추론하고, 블록 중심이 든 구역의 문턱으로 거른다.
# 10-23 실물: 불량 그릇 흰 접시 위 연한 초록X 가 0.25~0.4 라 0.4 에서 빠져 "옮긴 개수"가 0 이 됐다 → 그릇만 낮춘다.
# 투입 구역·손 자리·구역 밖·손 박스는 CONF_DEFAULT 그대로(집을 블록을 헛것으로 고르지 않게).
CONF_DEFAULT = 0.4
CONF_BY_ZONE = {"bowl_red": 0.25, "bowl_blue": 0.25, "bowl_defect": 0.25}
# CONF_DEFAULT 아래 띠의 박스가 이미 받은(점수 높은) 박스와 이만큼 겹치면 같은 블록으로 보고 버린다.
# NMS(IoU 0.7) 를 지난 같은 블록 겹박스가 0.25~0.4 에 있다(10-22 녹화 불량 그릇 18개 중 3개, IoU 0.67~0.71).
DUP_IOU = 0.5
BALL_MODEL = VISION / "models" / "ball_int8_openvino_model"
COCO_MODEL = VISION / "models" / "coco_int8_openvino_model"
# 사전학습(COCO) 모델로 경로를 검증할 때만 쓰는 이름 바꾸기. 블록 모델은 클래스 이름이 곧 계약 키라 안 쓴다.
COCO_ALIASES = {"sports ball": "red_ok", "orange": "red_ok", "apple": "red_defect", "person": "hand"}
# 박스 사진에 쓰는 이름·색(사람이 보는 것). red_* = 분홍, blue_* = 초록. OpenCV 글꼴이 한글을 못 써서 영어로.
SHOW = {"red_ok": "pink", "red_defect": "pink X", "blue_ok": "green", "blue_defect": "green X", "hand": "hand"}
BOX_COLORS = {"red_ok": (203, 102, 255), "blue_ok": (60, 200, 60), "red_defect": (140, 0, 170),
              "blue_defect": (0, 110, 0), "hand": (0, 170, 255)}
# objects[].cell: 구역 박스를 3×3 으로 나눈 칸(카메라 화면 기준). 가운데 칸은 "가운데", 나머지는 "<열> <행>"
CELL_COLS, CELL_ROWS = ("왼쪽", "가운데", "오른쪽"), ("위", "가운데", "아래")

OV_CACHE = VISION / ".ov_cache"

_models: dict[tuple[str, str], object] = {}


def enable_ov_cache() -> None:
    """Ultralytics 가 만드는 ov.Core 에 CACHE_DIR 을 넣는다(Ultralytics 에는 그 옵션이 없다).

    NPU 컴파일 결과를 디스크에 남겨 새 프로세스의 첫 detect_objects() 가 컴파일을 다시 하지 않게 한다.
    """
    import openvino as ov

    if getattr(ov.Core, "_ball_cache", False):
        return

    class Core(ov.Core):
        _ball_cache = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_property({"CACHE_DIR": str(OV_CACHE)})

    ov.Core = Core


def default_model() -> Path:
    env = os.environ.get("BALL_VISION_MODEL")
    if env:
        return Path(env)
    return BALL_MODEL if BALL_MODEL.exists() else COCO_MODEL


def load_model(model: Path | str, device: str):
    """YOLO(OpenVINO 폴더) — 첫 predict 때 그 장치로 컴파일되고 이후 재사용된다."""
    key = (str(model), device)
    if key not in _models:
        from ultralytics import YOLO

        if not str(model).rstrip("/").endswith("_openvino_model"):
            raise ValueError(f"OpenVINO 폴더(이름이 _openvino_model 로 끝나야 함)가 아님: {model}")
        enable_ov_cache()
        _models[key] = YOLO(str(model), task="detect")
    return _models[key]


def execution_device(yolo) -> str | None:
    """실제로 컴파일된 장치(Ultralytics 는 없는 장치를 조용히 AUTO 로 바꾼다)."""
    try:
        devs = yolo.predictor.model.backend.ov_compiled_model.get_property("EXECUTION_DEVICES")
        return devs if isinstance(devs, str) else ",".join(devs)  # 장치 하나면 문자열로 온다
    except Exception:
        return None


def contract_key(name: str, is_ball_model: bool) -> str | None:
    if name in BALL_CLASSES or name == "hand":
        return name
    return None if is_ball_model else COCO_ALIASES.get(name)


def cell_name(pt, rect) -> str:
    """rect(x1,y1,x2,y2) 안의 점이 3×3 중 어느 칸인가: '왼쪽 위' '가운데 위' … '가운데' … '오른쪽 아래'."""
    x1, y1, x2, y2 = rect
    c = min(2, max(0, int(3 * (pt[0] - x1) / max(1e-6, x2 - x1))))
    r = min(2, max(0, int(3 * (pt[1] - y1) / max(1e-6, y2 - y1))))
    return "가운데" if (r, c) == (1, 1) else f"{CELL_COLS[c]} {CELL_ROWS[r]}"


def draw(img, dets: list[dict], zones, header: str):
    img = zonelib.draw_zones(img, zones)
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        c = BOX_COLORS.get(d["key"], (180, 180, 180))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        label = f"{SHOW.get(d['key'], d['cls'])} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, header, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img



def skin_hand(frame, zones) -> dict:
    """모델이 손을 못 잡을 때 보완: 손 자리 안 피부색 비율로 판정(10-22 실물 handoff 사진 72장 중 69장 맞음).

    주황 테이프는 피부색 범위에 걸리므로 뺀다. hand_spot 이 없으면 판정하지 않는다.
    """
    if "hand_spot" not in zones:
        return {"present": False, "conf": 0.0}
    import numpy as np
    try:
        from . import autolabel
    except ImportError:
        import autolabel
    h, w = frame.shape[:2]
    spot = np.zeros((h, w), np.uint8)
    cv2.fillPoly(spot, [zones["hand_spot"].astype(np.int32)], 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    tape = cv2.dilate(cv2.inRange(hsv, (3, 110, 110), (25, 255, 255)), np.ones((5, 5), np.uint8)) > 0
    cfg = autolabel.load_config(autolabel.CONFIG, "black", "pastel")
    skin = (autolabel.masks(frame, cfg)["skin"] > 0) & (spot > 0) & ~tape
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    frac = float(skin.sum()) / max(int(spot.sum()), 1)
    return {"present": frac > SKIN_HAND_FRAC, "conf": round(min(frac * 4, 1.0), 3), "source": "skin"}


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def min_conf(zone: str | None, conf: float | None = None) -> float:
    """그 구역 블록에 쓰는 신뢰도 문턱. conf 를 주면 모든 구역에 그 값 하나(예전 방식·비교 시험용)."""
    return conf if conf is not None else CONF_BY_ZONE.get(zone, CONF_DEFAULT)


def detect_objects(image_path: str | None = None, *, model: str | Path | None = None,
                   zones_path: str | Path | None = None, device: str = "intel:npu", conf: float | None = None,
                   source: str = "auto", out_dir: str | Path | None = None) -> dict:
    """계약서 3절. image_path 가 없으면 위 카메라에서 한 장 찍고 바로 닫는다.

    conf 가 None 이면 구역별 문턱(CONF_BY_ZONE, 나머지 CONF_DEFAULT)으로 거른다."""
    t0 = time.perf_counter()
    if image_path:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"사진을 못 읽음: {image_path}")
        frame_source = "file"
    else:
        frame, frame_source = camera.grab_frame(source)
    capture_ms = (time.perf_counter() - t0) * 1000

    model = Path(model) if model else default_model()
    yolo = load_model(model, device)
    pred_conf = conf if conf is not None else min(CONF_DEFAULT, *CONF_BY_ZONE.values())
    r = yolo.predict(frame, device=device, conf=pred_conf, imgsz=640, verbose=False)[0]
    names = r.names
    is_ball_model = set(BALL_CLASSES) <= set(names.values())

    h, w = frame.shape[:2]
    zones_path = Path(zones_path or zonelib.ZONES_JSON)
    if not zones_path.exists():
        raise FileNotFoundError(f"{zones_path} 없음 — 먼저 python vision/zones.py 로 구역을 찍을 것")
    zones = zonelib.load_zones(zones_path, (w, h))
    counts = {z: {k: 0 for k in BALL_CLASSES} for z in zonelib.BOWL_ZONES}
    hand = {"present": False, "conf": 0.0}
    dets, objects, kept = [], [], []
    rows = zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
    for (x1, y1, x2, y2), c, p in sorted(rows, key=lambda t: -t[2]):  # 점수 높은 순(겹박스 거르기가 이 순서를 쓴다)
        name = names[int(c)]
        key = contract_key(name, is_ball_model)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        zone = zonelib.zone_of(center, zones) if key in BALL_CLASSES else None
        if p < min_conf(zone, conf):  # NMS 는 점수 높은 박스부터라 낮은 박스를 더 받아도 높은 박스는 그대로다
            continue
        if conf is None and p < CONF_DEFAULT and any(box_iou((x1, y1, x2, y2), b) >= DUP_IOU for b in kept):
            continue
        kept.append((x1, y1, x2, y2))
        if key in BALL_CLASSES:
            if zone:
                counts[zone][key] += 1
            # objects: 5구역 전부(손 위에 놓인 블록 = hand_spot)에서 찾고, 없으면 outside + 화면 전체 3×3
            ozone = zonelib.zone_of(center, zones, zonelib.ZONES)
            rect = (*zones[ozone].min(0), *zones[ozone].max(0)) if ozone else (0, 0, w, h)
            objects.append({"cls": key, "zone": ozone or "outside", "cell": cell_name(center, rect),
                            "px": [round(center[0]), round(center[1])], "conf": round(p, 3)})
        elif key == "hand":
            zone = zonelib.zone_of(center, zones, ["hand_spot"])
            if zone and p > hand["conf"]:
                hand = {"present": True, "conf": round(p, 3)}
        dets.append({"cls": name, "key": key, "conf": round(p, 3), "zone": zone,
                     "box": [round(v, 1) for v in (x1, y1, x2, y2)]})

    if not hand["present"] and SKIN_HAND_FRAC > 0:
        hand = skin_hand(frame, zones)

    speed = {k: round(v, 2) for k, v in r.speed.items()}
    latency = round(sum(speed.values()), 2)
    exec_dev = execution_device(yolo) or device.split(":")[-1].upper()
    now = datetime.now().astimezone()
    conf_txt = f"{conf:g}" if conf is not None else f"{CONF_DEFAULT:g}/{min(CONF_BY_ZONE.values()):g}"
    header = f"{now:%H:%M:%S} {model.name} {exec_dev} {latency:.1f}ms  dets={len(dets)}  conf {conf_txt}"
    out = Path(out_dir) if out_dir else ROOT / "runs" / f"{now:%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    snap = out / f"snap_{now:%H%M%S}_{now.microsecond // 1000:03d}.jpg"
    cv2.imwrite(str(snap), draw(frame.copy(), dets, zones, header), [cv2.IMWRITE_JPEG_QUALITY, 90])
    try:
        image_rel = str(snap.relative_to(ROOT))
    except ValueError:
        image_rel = str(snap)

    return {
        "ts": now.isoformat(timespec="seconds"),
        "zones": counts,
        "hand": hand,
        "objects": objects,
        "image_path": image_rel,
        "device": exec_dev,
        "latency_ms": latency,
        # 계약 밖 참고값
        "speed_ms": speed,
        "capture_ms": round(capture_ms, 1),
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "source": frame_source,
        "model": model.name,
        "detections": dets,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image")
    ap.add_argument("--model")
    ap.add_argument("--zones")
    ap.add_argument("--device", default="intel:npu")
    ap.add_argument("--conf", type=float, default=None, help="모든 구역에 문턱 하나(없으면 구역별 CONF_BY_ZONE)")
    ap.add_argument("--source", choices=["auto", "device", "studio"], default="auto")
    ap.add_argument("--out")
    ap.add_argument("--repeat", type=int, default=1, help="같은 프로세스에서 n번(두 번째부터 컴파일 없음)")
    a = ap.parse_args()
    for _ in range(a.repeat):
        res = detect_objects(a.image, model=a.model, zones_path=a.zones, device=a.device, conf=a.conf,
                             source=a.source, out_dir=a.out)
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
