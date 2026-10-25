"""NPU 동시 사용 시험: Whisper(OVMS, NPU) 받아쓰기 + YOLOv8n(별도 프로세스, NPU) 추론 + LLM(OVMS, GPU) 생성을 겹친다.

시연 중 비전 미리보기(YOLO 연속 추론)가 STT·LLM 과 겹칠 때 둘 다 버티는지, 지연이 얼마나 느는지 본다.
마지막 구간들은 YOLO 를 GPU 30fps 로 돌린다 — Studio 의 ACT 정책(OpenVINO GPU, 다른 프로세스)이 LLM 과 겹치는 상황의 대용.
구간마다 조합을 바꿔 N초씩 돌리고, 혼자 돌린 구간과 비교한다. 다른 세션이 OVMS 를 같이 쓴 구간은 조용해질 때를 기다려
다시 돌리고(--retries), 끝내 섞이면 '오염'으로 표시한다. OVMS 가 죽으면 거기서 멈춘다.

사용: .venv/bin/python bench/npu_concurrency.py [--secs 30] [--retries 4] [--skip N] [--model DIR] [--image JPG]
결과: bench/out/npu_concurrency.json (+ 화면 표)
"""
from __future__ import annotations

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

import argparse
import json
import multiprocessing as mp
import re
import statistics
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import requests

from bench_llm import SPEED_PROMPT, stream_once
from bench_stt import cer, transcribe

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "out"
BASE = "http://127.0.0.1:8100/v3"
STT_MODEL, LLM_MODEL = "whisper", "qwen3-4b"
# 비전 폴더는 읽기만 한다(모델·시험 사진). 컴파일 캐시는 내 폴더에 둔다.
DEF_MODEL = ROOT / "vision" / "models" / "synth_int8_openvino_model"
DEF_IMAGE = ROOT / "vision" / "_test" / "scene.jpg"
OV_CACHE = ROOT / ".ov_cache" / "yolo"
PREVIEW, MAX, GPU30 = ("NPU", 10), ("NPU", 0), ("GPU", 30)  # (장치, 목표 fps — 0 은 제한 없음)
PHASES = [  # (이름, stt, yolo: None|(장치, fps), llm)
    ("STT 혼자", True, None, False),
    ("YOLO 미리보기 혼자", False, PREVIEW, False),
    ("YOLO 최대속도 혼자", False, MAX, False),
    ("LLM 혼자", False, None, True),
    ("STT + YOLO 미리보기", True, PREVIEW, False),
    ("STT + YOLO 최대속도", True, MAX, False),
    ("STT + LLM", True, None, True),
    ("YOLO 미리보기 + LLM", False, PREVIEW, True),
    ("STT + YOLO 미리보기 + LLM", True, PREVIEW, True),
    ("STT + YOLO 최대속도 + LLM", True, MAX, True),
    ("YOLO GPU 30fps 혼자", False, GPU30, False),
    ("YOLO GPU 30fps + LLM", False, GPU30, True),
    ("STT + YOLO GPU 30fps + LLM", True, GPU30, True),
]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None


def yolo_worker(model_dir, image, device, fps, secs, ready, start, q):
    """별도 프로세스: 모델 적재·예열 → ready → start 신호부터 secs 동안 predict 반복."""
    device = f"intel:{device.lower()}"
    import cv2
    import openvino as ov

    class Core(ov.Core):  # vision/detect.py 의 enable_ov_cache 와 같은 방식, 캐시 위치만 내 폴더
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.set_property({"CACHE_DIR": str(OV_CACHE)})

    ov.Core = Core
    from ultralytics import YOLO

    frame = cv2.imread(str(image))
    y = YOLO(str(model_dir), task="detect")
    for _ in range(5):
        y.predict(frame, device=device, imgsz=640, conf=0.4, verbose=False)
    try:
        dev = y.predictor.model.backend.ov_compiled_model.get_property("EXECUTION_DEVICES")
        dev = dev if isinstance(dev, str) else ",".join(dev)
    except Exception:
        dev = "?"
    ready.set()
    start.wait()
    period = 1 / fps if fps else 0
    t_end = time.perf_counter() + secs
    infer, wall, fails, errors, nbox = [], [], 0, [], []
    nxt = time.perf_counter()
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        try:
            r = y.predict(frame, device=device, imgsz=640, conf=0.4, verbose=False)[0]
            infer.append(r.speed["inference"])
            nbox.append(len(r.boxes))
            wall.append((time.perf_counter() - t0) * 1000)
        except Exception as e:  # 실패도 세고 계속
            fails += 1
            if len(errors) < 3:
                errors.append(repr(e)[:160])
            time.sleep(0.5)
        if period:
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))
    q.put({"device": dev, "infer_ms": infer, "wall_ms": wall, "fails": fails, "errors": errors,
           "boxes": sorted(set(nbox)), "secs": secs})


def stt_loop(rows, end, out):
    i = 0
    while time.time() < end:
        row = rows[i % len(rows)]
        i += 1
        try:
            text, dt = transcribe(BASE, STT_MODEL, OUT / row["file"])
            out["lat_s"].append(dt)
            out["cer"].append(cer(row["text"], text))
        except Exception as e:
            out["fails"] += 1
            if len(out["errors"]) < 3:
                out["errors"].append(repr(e)[:160])
            time.sleep(1)


def llm_loop(end, out):
    while time.time() < end:
        try:
            r = stream_once(LLM_MODEL, SPEED_PROMPT, max_tokens=160)
            out["ttft_s"].append(r["ttft_s"])
            out["tok_s"].append(r["decode_tok_s"])
        except Exception as e:
            out["fails"] += 1
            if len(out["errors"]) < 3:
                out["errors"].append(repr(e)[:160])
            time.sleep(1)


def foreign_now(me, idle_ms=1500):
    """지금 OVMS(:8100)와 주고받는 중인 다른 프로세스 {pid: 명령줄}.

    UI 처럼 쉬는 keep-alive 연결을 계속 물고 있는 프로세스가 있어서, 마지막 송수신이 idle_ms 안인 연결만 센다.
    """
    out = subprocess.run(["ss", "-Htnpi", "state", "established", "( dport = :8100 )"],
                         capture_output=True, text=True).stdout
    seen = {}
    for conn, info in re.findall(r"^(\S.*)\n\s+(.*)$", out, re.M):  # 연결 줄 + 탭으로 시작하는 정보 줄
        pid = re.search(r"pid=(\d+)", conn)
        last = [int(v) for v in re.findall(r"last(?:snd|rcv):(\d+)", info)]
        if not pid or not last or min(last) > idle_ms:
            continue
        pid = pid.group(1)
        if int(pid) not in me:
            try:
                seen[pid] = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()[:120]
            except OSError:
                seen[pid] = "?"
    return seen


def foreign_watch(end, me, seen):
    """다른 세션이 OVMS 를 쓰는지 0.5초마다 본다 — 있으면 그 구간은 오염."""
    while time.time() < end:
        seen.update(foreign_now(me))
        time.sleep(0.5)


def training_running():
    """비전 학습(train.py)이 돌면 GPU·CPU 를 같이 써서 숫자가 흐려진다 — 시연 땐 안 도는 부하."""
    for p in Path("/proc").glob("[0-9]*"):
        try:
            if b"train.py" in (p / "cmdline").read_bytes():
                return True
        except OSError:
            pass
    return False


def wait_quiet(quiet_s=5, max_s=900):
    """다른 세션의 OVMS 접속·비전 학습이 quiet_s 초 연속 없을 때까지 기다린다(최대 max_s)."""
    t0 = last = time.time()
    while time.time() - t0 < max_s:
        if foreign_now({os.getpid()}) or training_running():
            last = time.time()
        elif time.time() - last >= quiet_s:
            return True
        time.sleep(0.5)
    return False


def ovms_alive():
    try:
        return len(requests.get(f"{BASE}/models", timeout=5).json()["data"])
    except Exception:
        return 0


def run_phase(name, stt, yolo, llm, a, rows, ctx):
    proc = q = None
    if yolo:
        ready, start, q = ctx.Event(), ctx.Event(), ctx.Queue()
        proc = ctx.Process(target=yolo_worker, args=(a.model, a.image, *yolo, a.secs, ready, start, q))
        proc.start()
        if not ready.wait(120):
            proc.kill()
            raise RuntimeError("YOLO 프로세스가 120초 안에 준비되지 않음")
    end = time.time() + a.secs
    me = {os.getpid()} | ({proc.pid} if proc else set())
    s_out = {"lat_s": [], "cer": [], "fails": 0, "errors": []}
    l_out = {"ttft_s": [], "tok_s": [], "fails": 0, "errors": []}
    foreign = {}
    ths = [threading.Thread(target=foreign_watch, args=(end, me, foreign))]
    if stt:
        ths.append(threading.Thread(target=stt_loop, args=(rows, end, s_out)))
    if llm:
        ths.append(threading.Thread(target=llm_loop, args=(end, l_out)))
    if proc:
        start.set()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if training_running():
        foreign["train"] = "train.py"
    res = {"phase": name, "foreign": foreign, "ovms_models_after": ovms_alive()}
    if stt:
        res["stt"] = {"n": len(s_out["lat_s"]), "fails": s_out["fails"], "errors": s_out["errors"],
                      "median_s": statistics.median(s_out["lat_s"]) if s_out["lat_s"] else None,
                      "p90_s": pct(s_out["lat_s"], 0.9),
                      "cer_mean": statistics.fmean(s_out["cer"]) if s_out["cer"] else None}
    if llm:
        res["llm"] = {"n": len(l_out["ttft_s"]), "fails": l_out["fails"], "errors": l_out["errors"],
                      "ttft_median_s": statistics.median(l_out["ttft_s"]) if l_out["ttft_s"] else None,
                      "tok_s_median": statistics.median(l_out["tok_s"]) if l_out["tok_s"] else None}
    if proc:
        y = q.get(timeout=60)
        proc.join(30)
        res["yolo"] = {"target": f"{yolo[0]} {yolo[1] or '최대'}fps", "device": y["device"], "n": len(y["infer_ms"]), "fails": y["fails"],
                       "errors": y["errors"], "fps": round(len(y["wall_ms"]) / y["secs"], 1),
                       "infer_median_ms": statistics.median(y["infer_ms"]) if y["infer_ms"] else None,
                       "infer_p90_ms": pct(y["infer_ms"], 0.9),
                       "wall_median_ms": statistics.median(y["wall_ms"]) if y["wall_ms"] else None,
                       "wall_p90_ms": pct(y["wall_ms"], 0.9), "boxes": y["boxes"]}
    return res


def fmt(v, unit="", nd=0):
    return "-" if v is None else f"{v:.{nd}f}{unit}"


def table(results):
    lines = ["| 구간 | STT 중앙값·p90 (회·실패·CER) | YOLO 추론 중앙값·p90 ms (fps·실패·장치) | LLM TTFT·tok/s (회·실패) | 오염 |",
             "|---|---|---|---|---|"]
    for r in results:
        s, y, l = r.get("stt"), r.get("yolo"), r.get("llm")
        cs = (f"{fmt(s['median_s'] * 1000 if s['median_s'] else None, 'ms')}·{fmt(s['p90_s'] * 1000 if s['p90_s'] else None, 'ms')}"
              f" ({s['n']}·{s['fails']}·{fmt(s['cer_mean'], nd=3)})") if s else ""
        cy = (f"{fmt(y['infer_median_ms'], nd=1)}·{fmt(y['infer_p90_ms'], nd=1)} "
              f"({y['fps']}fps·{y['fails']}·{y['device']})") if y else ""
        cl = (f"{fmt(l['ttft_median_s'], 's', 3)}·{fmt(l['tok_s_median'], nd=1)} ({l['n']}·{l['fails']})") if l else ""
        lines.append(f"| {r['phase']} | {cs} | {cy} | {cl} | {'있음' if r['foreign'] else ''} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--secs", type=int, default=30)
    ap.add_argument("--model", type=Path, default=DEF_MODEL)
    ap.add_argument("--image", type=Path, default=DEF_IMAGE)
    ap.add_argument("--retries", type=int, default=4, help="다른 세션이 끼어든 구간을 다시 돌리는 횟수")
    ap.add_argument("--skip", type=int, default=0, help="앞 구간 N개를 건너뛴다(이어서 돌릴 때)")
    a = ap.parse_args()
    OV_CACHE.mkdir(parents=True, exist_ok=True)
    rows = json.loads((OUT / "tts.json").read_text())["rows"]
    transcribe(BASE, STT_MODEL, OUT / rows[0]["file"])  # 예열
    stream_once(LLM_MODEL, "안녕", max_tokens=8)
    ctx = mp.get_context("spawn")
    started = datetime.now().astimezone()
    doc = {"started": started.isoformat(timespec="seconds"), "secs": a.secs, "model": str(a.model),
           "image": str(a.image), "phases": []}
    for name, stt, yolo, llm in PHASES[a.skip:]:
        for attempt in range(1, a.retries + 2):  # 오염되면 조용해질 때를 기다려 다시
            wait_quiet()
            print(f"{datetime.now():%H:%M:%S} {name} (시도 {attempt}) …", flush=True)
            r = run_phase(name, stt, yolo, llm, a, rows, ctx)
            r["attempt"] = attempt
            print(json.dumps(r, ensure_ascii=False), flush=True)
            if not r["foreign"] or not r["ovms_models_after"]:
                break
        doc["phases"].append(r)
        (OUT / "npu_concurrency.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        if not r["ovms_models_after"]:
            print(f"OVMS 가 죽음 — '{name}' 에서 멈춤", flush=True)
            break
        time.sleep(2)
    kern = subprocess.run(["journalctl", "-k", "--no-pager", "-S", started.strftime("%Y-%m-%d %H:%M:%S"),
                           "-g", "intel_vpu|ivpu|i915|xe |GPU HANG|reset"], capture_output=True, text=True).stdout
    doc["kernel"] = [ln for ln in kern.splitlines() if "-- No entries --" not in ln][-20:]
    (OUT / "npu_concurrency.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print("\n" + table(doc["phases"]))
    print("\n커널 로그(NPU·GPU):", doc["kernel"] or "없음")


if __name__ == "__main__":
    main()
