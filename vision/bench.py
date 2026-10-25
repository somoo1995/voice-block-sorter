"""같은 모델의 CPU·GPU·NPU 속도 비교표 (발표용 숫자).

  python vision/bench.py                     # models/ 의 블록 모델 INT8(폴더 ball_int8_…) (없으면 COCO INT8 + FP16)
  python vision/bench.py --model vision/models/ball_int8_openvino_model --model vision/models/ball_fp16_openvino_model

장치마다:
  컴파일      캐시 없이 첫 컴파일 (detect_objects 첫 호출이 내는 값)
  지연        LATENCY 힌트, 요청 1개씩 동기 추론 n회 → 중앙값·p90 ms, FPS = 1000/평균
  처리량      THROUGHPUT 힌트, 비동기 요청 여러 개를 몇 초 동안 → 초당 장 수
  e2e         Ultralytics predict(640x480 사진, 전처리+추론+후처리) 평균 ms — detect_objects 와 같은 경로
결과: vision/bench_results.md / .json
"""
from __future__ import annotations

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_OFFLINE", "1")  # import 때 DNS 확인·predict 마다 통계 전송을 끈다(시연은 오프라인)

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import openvino as ov

VISION = Path(__file__).resolve().parent
DEVICES = ["CPU", "GPU", "NPU"]


def power_source() -> str:
    for p in Path("/sys/class/power_supply").glob("*"):
        if (p / "type").exists() and (p / "type").read_text().strip() == "Mains":
            return "AC" if (p / "online").read_text().strip() == "1" else "배터리"
    return "?"


def ovms_state() -> str:
    try:
        return subprocess.run(["systemctl", "--user", "is-active", "ball-ovms"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
    except OSError:
        return "?"


def bench_raw(core: ov.Core, xml: Path, device: str, n: int, seconds: float) -> dict:
    model = core.read_model(xml)
    x = np.random.rand(*model.input().get_shape()).astype(np.float32)

    t = time.perf_counter()
    cm = core.compile_model(model, device, {"PERFORMANCE_HINT": "LATENCY"})
    compile_ms = (time.perf_counter() - t) * 1000
    req = cm.create_infer_request()
    for _ in range(10):
        req.infer({0: x})
    lat = []
    for _ in range(n):
        t = time.perf_counter()
        req.infer({0: x})
        lat.append((time.perf_counter() - t) * 1000)
    del req, cm

    cm = core.compile_model(model, device, {"PERFORMANCE_HINT": "THROUGHPUT"})
    nreq = cm.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    q = ov.AsyncInferQueue(cm, nreq)
    for _ in range(nreq):  # 워밍업
        q.start_async({0: x})
    q.wait_all()
    started, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        q.start_async({0: x})  # 빈 요청이 날 때까지 기다렸다 넣는다
        started += 1
    q.wait_all()
    tput = started / (time.perf_counter() - t0)
    lat.sort()
    return {"compile_ms": round(compile_ms), "lat_median_ms": round(statistics.median(lat), 2),
            "lat_p90_ms": round(lat[int(len(lat) * 0.9)], 2), "lat_mean_ms": round(statistics.fmean(lat), 2),
            "fps_sync": round(1000 / statistics.fmean(lat), 1), "fps_throughput": round(tput, 1), "nireq": nreq}


def bench_e2e(model_dir: Path, device: str, image, n: int) -> float:
    from ultralytics import YOLO

    y = YOLO(str(model_dir), task="detect")
    dev = f"intel:{device.lower()}"
    for _ in range(3):
        y.predict(image, device=dev, imgsz=640, verbose=False)
    tot = []
    for _ in range(n):
        r = y.predict(image, device=dev, imgsz=640, verbose=False)[0]
        tot.append(sum(r.speed.values()))
    return round(statistics.fmean(tot), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", type=Path)
    ap.add_argument("--devices", nargs="+", default=DEVICES)
    ap.add_argument("--n", type=int, default=200, help="지연 측정 횟수")
    ap.add_argument("--seconds", type=float, default=5, help="처리량 측정 시간")
    ap.add_argument("--e2e-n", type=int, default=50)
    ap.add_argument("--image", help="e2e 용 사진 (기본: 640x480 으로 줄인 Ultralytics bus.jpg)")
    ap.add_argument("--out", type=Path, default=VISION / "bench_results")
    a = ap.parse_args()

    models = a.model or ([VISION / "models/ball_int8_openvino_model"]
                         if (VISION / "models/ball_int8_openvino_model").exists()
                         else [VISION / "models/coco_int8_openvino_model", VISION / "models/coco_fp16_openvino_model"])
    import cv2

    if a.image:
        image = cv2.imread(a.image)
    else:
        import ultralytics

        image = cv2.resize(cv2.imread(str(Path(ultralytics.__file__).parent / "assets/bus.jpg")), (640, 480))

    core = ov.Core()
    env = {
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "openvino": ov.__version__.split("-")[0],
        "devices": {d: core.get_property(d, "FULL_DEVICE_NAME") for d in core.available_devices},
        "power": power_source(),
        "ball_ovms": ovms_state(),
    }
    rows = []
    for m in models:
        xml = next(m.glob("*.xml"))
        for d in a.devices:
            if d not in core.available_devices:
                continue
            print(f"{m.name} {d} …", flush=True)
            row = {"model": m.name, "device": d, **bench_raw(core, xml, d, a.n, a.seconds)}
            row["e2e_ms"] = bench_e2e(m, d, image, a.e2e_n)
            rows.append(row)
            print(json.dumps(row), flush=True)

    lines = [
        f"# 비전 모델 장치별 속도 ({env['date'][:16]})",
        "",
        f"- 입력 1x3x640x640, OpenVINO {env['openvino']}, 전원 {env['power']}, OVMS {env['ball_ovms']}",
        "- " + " · ".join(f"{k} = {v}" for k, v in env["devices"].items()),
        "- 지연 = 요청 1개 동기 추론(LATENCY 힌트), 처리량 = 비동기 요청 여러 개(THROUGHPUT 힌트), "
        "e2e = Ultralytics 전처리+추론+후처리(detect_objects 경로)",
        "",
        "| 모델 | 장치 | 첫 컴파일 ms | 지연 중앙값 ms | 지연 p90 ms | FPS (동기) | FPS (처리량) | e2e ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['model'].removesuffix('_openvino_model')} | {r['device']} | {r['compile_ms']:,} | "
                     f"{r['lat_median_ms']} | {r['lat_p90_ms']} | {r['fps_sync']} | {r['fps_throughput']} | {r['e2e_ms']} |")
    md = "\n".join(lines) + "\n"
    a.out.with_suffix(".md").write_text(md)
    a.out.with_suffix(".json").write_text(json.dumps({"env": env, "rows": rows}, ensure_ascii=False, indent=1))
    print("\n" + md)


if __name__ == "__main__":
    main()
