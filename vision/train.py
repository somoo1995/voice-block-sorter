"""YOLO 데이터셋(폴더 또는 Roboflow "YOLOv8" zip) → 학습(xpu) → OpenVINO INT8 export(imgsz 640 고정, NPU 용).

수업 venv 로 실행만 한다(torch xpu·nncf 가 거기 있음, 설치 금지):
  PY=~/openvino-project/.venv/bin/python
  $PY vision/train.py --dir vision/datasets/auto_day1                         # autolabel/review 가 내보낸 폴더
  $PY vision/train.py --zip ~/Downloads/ball-sorter.v1i.yolov8.zip            # Roboflow zip (대안)
  $PY vision/train.py --export-only vision/models/yolov8n.pt --name coco      # .pt 를 INT8 로만 (COCO 검증용)
  결과: models/ball_int8_openvino_model (+ train_summary.json)

⚑ Roboflow "YOLOv8" export 는 모델이 아니라 데이터셋(jpg + 라벨 txt + data.yaml)이다. 모델은 여기서 만든다.
"""
from __future__ import annotations

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # 수업 venv 에 몰래 설치하지 않게

import argparse
import json
import shutil
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

VISION = Path(__file__).resolve().parent
MODELS = VISION / "models"
DATASETS = VISION / "datasets"
EXPECTED = {"red_ok", "blue_ok", "red_defect", "blue_defect", "hand"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def safe_extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            if m.startswith("/") or ".." in Path(m).parts:
                sys.exit(f"zip 안에 위험한 경로: {m}")
        z.extractall(dest)


def prepare_dataset(zip_path: Path, reextract: bool) -> tuple[Path, list[str]]:
    """zip 을 vision/datasets/<이름>/ 에 풀고 fix_yaml."""
    dest = DATASETS / zip_path.name.removesuffix(".zip")
    if reextract and dest.exists():
        shutil.rmtree(dest)
    if not (dest / "data.yaml").exists():
        dest.mkdir(parents=True, exist_ok=True)
        safe_extract(zip_path, dest)
    return fix_yaml(dest, zip_path)


def fix_yaml(dest: Path, origin: Path) -> tuple[Path, list[str]]:
    """data.yaml 을 찾아 절대경로 data_fixed.yaml 을 만든다(원본 data.yaml 은 그대로)."""
    found = list(dest.rglob("data.yaml"))
    if not found:
        sys.exit(f"data.yaml 없음 — autolabel.py/review.py 로 내보낸 폴더나 Roboflow 'YOLOv8' 형식인지 확인: {origin}")
    root = found[0].parent
    src = yaml.safe_load(found[0].read_text())
    names = src["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    # Roboflow 는 train: ../train/images 처럼 한 단계 위를 가리킨다 → 실제 있는 폴더로 고정
    fixed = {"path": str(root)}
    for key, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (root / folder / "images").is_dir():
            fixed[key] = f"{folder}/images"
    if "val" not in fixed:
        fixed["val"] = fixed["train"]
        print("⚠ valid 폴더 없음 → train 으로 검증(점수가 부풀려짐)")
    fixed.update(nc=len(names), names=names)
    out = root / "data_fixed.yaml"
    out.write_text(yaml.safe_dump(fixed, allow_unicode=True, sort_keys=False))
    return out, names


def dataset_stats(data_yaml: Path) -> dict:
    """분할별 사진·라벨·클래스별 박스 수. Roboflow 라벨은 끝 줄바꿈이 없어 wc -l 로 세면 틀린다."""
    d = yaml.safe_load(data_yaml.read_text())
    root, names, stats = Path(d["path"]), d["names"], {}
    for key in ("train", "val", "test"):
        if key not in d or (key == "val" and d["val"] == d["train"]):
            continue
        img_dir = root / d[key]
        lbl_dir = img_dir.parent / "labels"
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        boxes = {n: 0 for n in names}
        empty = 0
        for p in imgs:
            lf = lbl_dir / f"{p.stem}.txt"
            lines = [ln.split() for ln in lf.read_text().splitlines() if ln.strip()] if lf.exists() else []
            if not lines:
                empty += 1
            for ln in lines:
                boxes[names[int(ln[0])]] += 1
        stats[key] = {"images": len(imgs), "no_label": empty, "boxes": boxes}
    return stats


def coco128_yaml() -> Path:
    """COCO 모델 INT8 보정용 coco128 (vision/datasets 에 받음, Ultralytics datasets_dir 는 안 건드림)."""
    root = DATASETS / "coco128"
    if not (root / "images").exists():
        DATASETS.mkdir(parents=True, exist_ok=True)
        zp = DATASETS / "coco128.zip"
        print("coco128 내려받는 중 …", flush=True)
        urllib.request.urlretrieve(COCO128_URL, zp)
        safe_extract(zp, DATASETS)
        zp.unlink()
    import ultralytics

    names = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg/datasets/coco128.yaml").read_text())["names"]
    out = root / "data_fixed.yaml"
    out.write_text(yaml.safe_dump({"path": str(root), "train": "images/train2017", "val": "images/train2017",
                                   "names": names}, sort_keys=False))
    return out


def npu_check(model_dir: Path) -> dict:
    """export 한 xml 을 NPU 에 직접 컴파일·1회 추론 (정적 shape 확인)."""
    import numpy as np
    import openvino as ov

    core = ov.Core()
    m = core.read_model(next(model_dir.glob("*.xml")))
    shape = m.input().get_partial_shape()
    res = {"input_shape": str(shape), "devices": core.available_devices}
    if "NPU" not in core.available_devices or shape.is_dynamic:
        res["npu"] = "skip (NPU 없음 또는 동적 shape)"
        return res
    t = time.perf_counter()
    cm = core.compile_model(m, "NPU")
    res["npu_compile_ms"] = round((time.perf_counter() - t) * 1000)
    x = np.random.rand(*shape.to_shape()).astype(np.float32)
    cm(x)
    t = time.perf_counter()
    for _ in range(20):
        cm(x)
    res["npu_infer_ms"] = round((time.perf_counter() - t) * 1000 / 20, 2)
    return res


def export(pt: Path, name: str, data: Path, split: str, fp16: bool) -> dict:
    from ultralytics import YOLO

    MODELS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = [("int8", dict(quantize=8, data=str(data), split=split))]
    if fp16:
        jobs.append(("fp16", dict(quantize=16)))
    result = {}
    for tag, kw in jobs:
        t = time.perf_counter()
        out = Path(YOLO(str(pt)).export(format="openvino", imgsz=640, dynamic=False, batch=1, device="cpu", **kw))
        dst = MODELS / f"{name}_{tag}_openvino_model"
        if dst.exists():  # 지우지 않고 옆으로 치운다
            (MODELS / "_old").mkdir(exist_ok=True)
            shutil.move(str(dst), str(MODELS / "_old" / f"{dst.name}.{ts}"))
        shutil.move(str(out), str(dst))
        size = sum(f.stat().st_size for f in dst.glob("*.bin"))
        result[tag] = {"path": str(dst.relative_to(VISION.parent)), "export_s": round(time.perf_counter() - t, 1),
                       "bin_mb": round(size / 1e6, 2), **npu_check(dst)}
        print(f"[export {tag}] {result[tag]}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", type=Path, help="YOLO 데이터셋 폴더(train/ valid/ data.yaml) — autolabel·review 내보내기")
    g.add_argument("--zip", type=Path, help="Roboflow YOLOv8 데이터셋 zip")
    g.add_argument("--export-only", type=Path, metavar="PT", help="학습 없이 이 .pt 를 INT8 export")
    ap.add_argument("--name", help="models/<name>_int8_openvino_model (기본 ball / export-only 는 pt 이름)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="xpu", help="학습 장치: xpu(Intel GPU) | cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epoch-val", action="store_true",
                    help="에포크마다 검증(기본 끔: xpu 검증은 NMS 시간 초과로 점수가 잘린다, 10-21 실측)")
    ap.add_argument("--val-device", default="cpu", help="학습 뒤 최종 검증 장치")
    ap.add_argument("--base", type=Path, default=MODELS / "yolov8n.pt")
    ap.add_argument("--data", type=Path, help="INT8 보정 데이터 yaml (기본: 학습 데이터 / export-only 는 coco128)")
    ap.add_argument("--fp16", action="store_true", help="비교용 FP16 모델도 export (bench.py 용)")
    ap.add_argument("--reextract", action="store_true", help="zip 을 다시 풀기")
    ap.add_argument("--allow-classes", action="store_true", help="클래스 이름이 계약과 달라도 진행")
    a = ap.parse_args()

    try:
        import nncf  # noqa: F401
        import torch
    except ImportError as e:
        sys.exit(f"{e} — 수업 venv 로 실행: ~/openvino-project/.venv/bin/python vision/train.py …")
    if not a.export_only and a.device == "xpu" and not torch.xpu.is_available():
        sys.exit("torch xpu 를 못 씀 — 수업 venv(~/openvino-project/.venv)로 실행했는지 확인")

    summary: dict = {"created": datetime.now().astimezone().isoformat(timespec="seconds"), "python": sys.executable}
    if a.export_only:
        name = a.name or a.export_only.stem
        data = a.data or coco128_yaml()
        summary.update(source=str(a.export_only), calib=str(data))
        summary["export"] = export(a.export_only, name, data, "val", a.fp16)
    else:
        name = a.name or "ball"
        if a.dir:
            if not a.dir.expanduser().is_dir():
                sys.exit(f"폴더 없음: {a.dir}")
            data, names = fix_yaml(a.dir.expanduser().resolve(), a.dir)
        else:
            data, names = prepare_dataset(a.zip.expanduser(), a.reextract)
        missing, extra = EXPECTED - set(names), set(names) - EXPECTED
        print(f"클래스 {names}")
        if (missing or extra) and not a.allow_classes:
            sys.exit(f"클래스 이름이 계약과 다름 — 없음 {sorted(missing)} / 남음 {sorted(extra)} "
                     f"(Roboflow 클래스 이름을 고치거나 --allow-classes)")
        stats = dataset_stats(data)
        print("데이터:", json.dumps(stats, ensure_ascii=False), flush=True)

        from ultralytics import YOLO

        run = f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
        model = YOLO(str(a.base))
        t = time.perf_counter()
        # val=False 여도 마지막 에포크는 검증한다(그 점수는 xpu 라 잘릴 수 있음 → 아래 CPU 검증이 기준)
        model.train(data=str(data), epochs=a.epochs, imgsz=640, batch=a.batch, device=a.device,
                    workers=a.workers, project=str(VISION / "runs_train"), name=run, exist_ok=False,
                    val=a.epoch_val)
        train_s = time.perf_counter() - t
        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"
        print(f"학습 {train_s / 60:.1f}분 → {best}", flush=True)

        m = YOLO(str(best)).val(data=str(data), imgsz=640, batch=a.batch, device=a.val_device, plots=False,
                                project=str(VISION / "runs_train"), name=f"{run}_val")
        per_class = {}
        for i, c in enumerate(m.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            per_class[m.names[int(c)]] = {"P": round(float(p), 3), "R": round(float(r), 3),
                                          "mAP50": round(float(ap50), 3), "mAP50-95": round(float(ap), 3)}
        summary.update(
            dataset=str(data), classes=names, stats=stats, run_dir=str(save_dir), best=str(best),
            epochs=a.epochs, device=a.device, val_device=a.val_device, train_min=round(train_s / 60, 1),
            val={"P": round(float(m.box.mp), 3), "R": round(float(m.box.mr), 3),
                 "mAP50": round(float(m.box.map50), 3), "mAP50-95": round(float(m.box.map), 3)},
            per_class=per_class,
        )
        print("검증:", json.dumps(summary["val"]), json.dumps(per_class, ensure_ascii=False), flush=True)
        # 보정은 사진이 더 많은 train 분할로
        summary["export"] = export(best, name, a.data or data, "train", a.fp16)

    dst = MODELS / f"{name}_int8_openvino_model"
    (dst / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("완료:", dst)


if __name__ == "__main__":
    main()
