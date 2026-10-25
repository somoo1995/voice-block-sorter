"""자원표 측정 — 구간마다 부하(LLM·리랭크·STT·TTS·YOLO NPU·ACT CPU/NPU)를 켜고 1초마다 메모리·CPU·GPU·NPU 를 잰다.

GPU 는 OVMS LLM·리랭크만 쓴다(10-22 밤 규칙). ACT CPU 는 Studio 컨테이너 안(docker exec, stdin 스크립트), ACT NPU 는
컨테이너에서 꺼낸 IR(/tmp/actx)을 이 프로세스가 NPU 로 돌린다. 정책 간격은 실물과 같은 1.7초(50틱)에 한 번.
사용: .venv/bin/python bench/resmix.py [--tag night3] [--short]
"""
from __future__ import annotations

import argparse, json, os, statistics, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("SUPERTONIC_INTRA_OP_THREADS", "4")
os.environ.setdefault("SUPERTONIC_INTER_OP_THREADS", "1")
import requests  # noqa: E402

OVMS = "http://127.0.0.1:8100/v3"
CONTAINER = "physical-ai-studio-xpu"
ACT_MODEL = "708a44b7-ff34-4302-93b9-763dc15066b0"  # pick_red ACT(네 정책 모두 같은 ACT 구조)
ACT_IR = Path("/tmp/actx/act.xml")
ACT_EVERY = 1.7  # SyncExecution request_threshold 0.5 × chunk 100 = 50틱 / 30fps
CLK = os.sysconf("SC_CLK_TCK")
LAT: dict[str, list[float]] = {}
ERR: dict[str, int] = {}
LOCK = threading.Lock()


def rec(kind: str, ms: float | None):
    with LOCK:
        if ms is None:
            ERR[kind] = ERR.get(kind, 0) + 1
        else:
            LAT.setdefault(kind, []).append(ms)


def timed(kind, fn):
    t0 = time.perf_counter()
    try:
        fn()
        rec(kind, (time.perf_counter() - t0) * 1000)
    except Exception as e:  # noqa: BLE001
        rec(kind, None)
        print(f"[{kind}] {type(e).__name__}: {str(e)[:120]}", flush=True)


# ---- 부하 ----
PROMPT = ("너는 블록 선별 로봇의 안내 담당이다. 작업대에 분홍 블록 2개, 초록 블록 1개, 불량 블록 1개가 있다. "
          "불량은 검은 X 표시가 있는 블록이다. 지금까지 불량 1개를 불량 그릇으로 옮겼다. 다음에 할 일을 세 문장으로 설명해라. /no_think")


def llm_loop(stop):
    while not stop.is_set():
        timed("llm", lambda: requests.post(f"{OVMS}/chat/completions", timeout=60, json={
            "model": "qwen3-4b", "max_tokens": 160, "temperature": 0,
            "messages": [{"role": "user", "content": PROMPT}]}).raise_for_status())


DOCS = [f"작업 기준서 {i}조: 불량 블록은 검은 X 가 있는 블록이며 불량 그릇에 넣는다. 정상 블록은 색깔별 그릇에 넣는다." * 3 for i in range(10)]


def rerank_loop(stop):
    while not stop.is_set():
        timed("rerank", lambda: requests.post(f"{OVMS}/rerank", timeout=60, json={
            "model": "qwen3-reranker", "query": "불량 블록은 어디에 넣나요?", "documents": DOCS}).raise_for_status())
        timed("embed", lambda: requests.post(f"{OVMS}/embeddings", timeout=60, json={
            "model": "qwen3-embedding", "input": ["불량 블록은 어디에 넣나요?"]}).raise_for_status())
        stop.wait(1.0)


def stt_loop(stop):
    from ui import stt
    wavs = sorted((ROOT / "ui" / "samples").glob("*.wav"))
    i = 0
    while not stop.is_set():
        w = wavs[i % len(wavs)]; i += 1
        timed("stt", lambda: stt.transcribe(str(w)))
        stop.wait(0.5)


def tts_loop(stop):
    from ui import tts
    while not stop.is_set():
        out = []
        timed("tts", lambda: out.append(tts.speak("불량 블록 한 개를 불량 그릇으로 옮겼어요. 이제 분홍 블록을 옮길게요.")))
        for p in out:
            if p:
                Path(p).unlink(missing_ok=True)  # 내가 만든 합성 파일만 지운다
        stop.wait(1.0)


def yolo_loop(stop):
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "vision" / "models" / "ball_int8_openvino_model"), task="detect")
    img = str(ROOT / "robot" / "logs" / "20261022" / "front_174916.jpg")
    model.predict(img, device="intel:npu", verbose=False)
    while not stop.is_set():
        timed("yolo", lambda: model.predict(img, device="intel:npu", verbose=False))
        stop.wait(0.1)  # 미리보기 10fps 이하 권장(SETUP.md)


def act_npu_loop(stop):
    import numpy as np, openvino as ov
    core = ov.Core()
    cm = core.compile_model(core.read_model(str(ACT_IR)), "NPU", {"PERFORMANCE_HINT": "LATENCY"})
    r = cm.create_infer_request()
    rng = np.random.default_rng(0)
    inp = {"state": rng.normal(0, 30, (1, 6)).astype(np.float32),
           "images.front": rng.random((1, 3, 384, 384), dtype=np.float32),
           "images.gripper": rng.random((1, 3, 384, 384), dtype=np.float32)}
    r.infer(inp)
    while not stop.is_set():
        timed("act_npu", lambda: r.infer(inp))
        stop.wait(ACT_EVERY)


ACT_CPU_SRC = r'''
import sys, time, json, numpy as np
from physicalai.inference import InferenceModel
mid, secs, every = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
m = InferenceModel(export_dir=f"/app/storage/models/{mid}/exports/openvino", backend="openvino", device="CPU")
rng = np.random.default_rng(0)
o = {"state": np.array([[0.07, -52.23, 5.15, 50.82, 2.61, 49.96]], dtype=np.float32),
     "images.front": rng.integers(0, 256, (1, 480, 640, 3), dtype=np.uint8),
     "images.gripper": rng.integers(0, 256, (1, 480, 640, 3), dtype=np.uint8)}
m.predict_action_chunk(o)
end = time.time() + secs
while time.time() < end:
    t0 = time.perf_counter(); m.predict_action_chunk(o); print(json.dumps({"ms": (time.perf_counter() - t0) * 1000}), flush=True)
    time.sleep(every)
'''


def act_cpu_loop(stop, secs):
    p = subprocess.Popen(["docker", "--context", "default", "exec", "-i", CONTAINER, "/app/application/backend/.venv/bin/python",
                          "-I", "-", ACT_MODEL, str(secs), str(ACT_EVERY)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    p.stdin.write(ACT_CPU_SRC); p.stdin.close()
    for line in p.stdout:
        if line.startswith("{"):
            rec("act_cpu", json.loads(line)["ms"])
    p.wait()


LOOPS = {"llm": llm_loop, "rerank": rerank_loop, "stt": stt_loop, "tts": tts_loop, "yolo": yolo_loop, "act_npu": act_npu_loop}

# ---- 측정 ----
DRM_CARD = Path("/sys/class/drm/card1")
NPU_DEV = Path("/sys/class/accel/accel0/device")


def proc_groups():
    ovms = int(subprocess.run(["systemctl", "--user", "show", "ball-ovms", "-p", "MainPID", "--value"], capture_output=True, text=True).stdout.strip() or 0)
    g: dict[str, list[int]] = {"ovms": [ovms] if ovms else [], "studio": [], "act_cpu": [], "bench": [os.getpid()]}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if cmd.startswith("/app/application/backend/.venv/bin/python -I -"):
            g["act_cpu"].append(int(d.name))
        elif "/app/application/backend/.venv" in cmd or "physicalai-studio" in cmd:
            g["studio"].append(int(d.name))
    return g


def pid_stat(pid):
    try:
        st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        rss = int(Path(f"/proc/{pid}/status").read_text().split("VmRSS:")[1].split()[0])
        return int(st[11]) + int(st[12]), rss
    except (OSError, IndexError, ValueError):
        return None


def drm_info(pid):
    """그 프로세스의 i915·intel_vpu fdinfo(클라이언트 id 로 중복 제거). 다른 uid 면 못 읽는다(None)."""
    gpu_mem = gpu_ns = npu_mem = 0
    seen = set()
    try:
        fds = list(Path(f"/proc/{pid}/fdinfo").iterdir())
    except OSError:
        return None
    for f in fds:
        try:
            t = f.read_text()
        except OSError:
            continue
        if "drm-driver" not in t:
            continue
        kv = dict(line.split(":", 1) for line in t.splitlines() if ":" in line)
        key = (kv.get("drm-driver", "").strip(), kv.get("drm-client-id", "").strip())
        if key in seen:
            continue
        seen.add(key)
        if key[0] == "i915":
            gpu_mem += int(kv.get("drm-resident-system0", "0 KiB").split()[0])
            gpu_ns += sum(int(kv.get(k, "0 ns").split()[0]) for k in ("drm-engine-render", "drm-engine-compute", "drm-engine-copy", "drm-engine-video", "drm-engine-video-enhance"))
        elif key[0] == "intel_vpu":
            npu_mem += int(kv.get("drm-resident-memory", "0 KiB").split()[0])
    return gpu_mem, gpu_ns, npu_mem


def rd(p, cast=int):
    try:
        return cast(Path(p).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def meminfo():
    m = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0])
    return m


def sampler(stop, out, phase_ref):
    prev = None
    while not stop.is_set():
        t = time.time()
        g = proc_groups()
        cur = {"t": t, "rc6": rd(DRM_CARD / "gt/gt0/rc6_residency_ms"), "npu_us": rd(NPU_DEV / "npu_busy_time_us"), "grp": {}}
        for name, pids in g.items():
            jif = rss = gmem = gns = nmem = 0
            for p in pids:
                s = pid_stat(p)
                if s:
                    jif += s[0]; rss += s[1]
                d = drm_info(p)
                if d:
                    gmem += d[0]; gns += d[1]; nmem += d[2]
            cur["grp"][name] = (jif, rss, gmem, gns, nmem, len(pids))
        mi = meminfo()
        if prev:
            dt = t - prev["t"]
            row = {"phase": phase_ref[0], "t": round(t, 1), "avail_gb": round(mi["MemAvailable"] / 1048576, 2),
                   "swap_used_gb": round((mi["SwapTotal"] - mi["SwapFree"]) / 1048576, 2),
                   "gpu_busy_pct": None if cur["rc6"] is None else round(max(0.0, 100 - (cur["rc6"] - prev["rc6"]) / (dt * 10)), 1),
                   "gpu_mhz": rd(DRM_CARD / "gt_act_freq_mhz"),
                   "npu_busy_pct": None if cur["npu_us"] is None else round((cur["npu_us"] - prev["npu_us"]) / (dt * 1e4), 1),
                   "npu_mem_mb": (rd(NPU_DEV / "npu_memory_utilization") or 0) // 1048576}
            for name, v in cur["grp"].items():
                pv = prev["grp"].get(name, v)
                row[name] = {"cpu_cores": round(max(0, v[0] - pv[0]) / CLK / dt, 2), "rss_mb": v[1] // 1024,
                             "gpu_mem_mb": v[2] // 1024, "gpu_busy_pct": round(max(0, v[3] - pv[3]) / (dt * 1e7), 1),
                             "npu_mem_mb": v[4] // 1024, "n": v[5]}
            out.append(row)
        prev = cur
        stop.wait(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="night3")
    ap.add_argument("--short", action="store_true", help="구간 길이를 1/3 로(시험용)")
    ap.add_argument("--phases", help="구간을 직접: 이름:초:부하+부하,... (예 정책+TTS:45:act_cpu+tts)")
    a = ap.parse_args()
    k = 1 / 3 if a.short else 1
    phases = [("대기", 40, []), ("LLM", 60, ["llm"]), ("리랭크·임베딩", 40, ["rerank"]), ("STT NPU", 40, ["stt"]),
              ("TTS CPU", 30, ["tts"]), ("YOLO NPU 10fps", 30, ["yolo"]),
              ("ACT CPU 단독", 40, ["act_cpu"]), ("ACT NPU 단독", 40, ["act_npu"]),
              ("시연 피크(ACT CPU)", 90, ["llm", "rerank", "stt", "tts", "yolo", "act_cpu"]),
              ("시연 피크(ACT NPU)", 90, ["llm", "rerank", "stt", "tts", "yolo", "act_npu"])]
    if a.phases:
        phases = [(n, int(sec), [x for x in lp.split("+") if x]) for n, sec, lp in (p.split(":") for p in a.phases.split(","))]
    rows, phase_ref, res = [], ["시작"], {}
    sstop = threading.Event()
    th = threading.Thread(target=sampler, args=(sstop, rows, phase_ref), daemon=True)
    th.start()
    for name, secs, loops in phases:
        secs = max(10, int(secs * k))
        LAT.clear(); ERR.clear()
        phase_ref[0] = name
        stop = threading.Event()
        ts = []
        for lp in loops:
            if lp == "act_cpu":
                ts.append(threading.Thread(target=act_cpu_loop, args=(stop, secs), daemon=True))
            else:
                ts.append(threading.Thread(target=LOOPS[lp], args=(stop,), daemon=True))
        t0 = time.time()
        print(f"[resmix] {time.strftime('%H:%M:%S')} {name} {secs}s {loops}", flush=True)
        for t in ts:
            t.start()
        time.sleep(secs)
        stop.set()
        for t in ts:
            t.join(timeout=90)
        lat = {kk: {"n": len(v), "median_ms": round(statistics.median(v), 1), "p95_ms": round(sorted(v)[int(0.95 * (len(v) - 1))], 1),
                    "max_ms": round(max(v), 1)} for kk, v in LAT.items() if v}
        res[name] = {"secs": round(time.time() - t0, 1), "lat": lat, "err": dict(ERR)}
        print(f"[resmix]   {json.dumps(lat, ensure_ascii=False)} err={dict(ERR)}", flush=True)
        phase_ref[0] = "사이"
        time.sleep(3)
    sstop.set(); th.join(timeout=5)
    # 구간별 요약: 그룹마다 RSS 최대·CPU 평균/최대, GPU·NPU 바쁨 평균/최대
    summ = {}
    for name, *_ in phases:
        rs = [r for r in rows if r["phase"] == name]
        if not rs:
            continue
        s = {"avail_gb_min": min(r["avail_gb"] for r in rs), "swap_gb_max": max(r["swap_used_gb"] for r in rs)}
        for key in ("gpu_busy_pct", "npu_busy_pct", "gpu_mhz", "npu_mem_mb"):
            vals = [r[key] for r in rs if r[key] is not None]
            if vals:
                s[key] = [round(statistics.mean(vals), 1), max(vals)]
        for grp in ("ovms", "studio", "act_cpu", "bench"):
            gv = [r[grp] for r in rs if grp in r]
            if gv:
                s[grp] = {"rss_mb_max": max(x["rss_mb"] for x in gv), "cpu_cores": [round(statistics.mean(x["cpu_cores"] for x in gv), 2), max(x["cpu_cores"] for x in gv)],
                          "gpu_mem_mb_max": max(x["gpu_mem_mb"] for x in gv), "gpu_busy_pct": [round(statistics.mean(x["gpu_busy_pct"] for x in gv), 1), max(x["gpu_busy_pct"] for x in gv)],
                          "npu_mem_mb_max": max(x["npu_mem_mb"] for x in gv)}
        res[name]["res"] = s
    out = ROOT / "bench" / "out" / f"resmix_{a.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"phases": res, "rows": rows}, ensure_ascii=False, indent=1))
    for name, v in res.items():
        print(f"== {name}: {json.dumps(v.get('res', {}), ensure_ascii=False)}")
    print(f"[resmix] 끝 {time.strftime('%H:%M:%S')} → {out}")


if __name__ == "__main__":
    main()
