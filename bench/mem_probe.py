"""모델을 하나씩 단독으로 OVMS 에 올려 적재 시간·메모리를 잰다(포트 8101, 유닛 ball-ovms-probe).

사용: .venv/bin/python bench/mem_probe.py [이름 ...]   (기본: 전부)
결과: bench/out/mem_probe.json
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "out"
UNIT = "ball-ovms-probe"
PORT = 8101
BASE = f"http://127.0.0.1:{PORT}"
CG = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{UNIT}.service")
W = "models/OpenVINO/whisper-large-v3-turbo-int8-ov"

SPECS = {
    "qwen3-embedding": ({"base_path": "models/OpenVINO/Qwen3-Embedding-0.6B-int8-ov"}, "embed"),
    "qwen3-reranker": ({"base_path": "models/OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov"}, "rerank"),
    "whisper": ({"base_path": W}, "stt"),
    "whisper-gpu": ({"base_path": W, "graph_path": "graph_gpu.pbtxt"}, "stt"),
    "qwen3-4b": ({"base_path": "models/OpenVINO/Qwen3-4B-int4-ov"}, "llm"),
    "qwen3-8b": ({"base_path": "models/OpenVINO/Qwen3-8B-int4-ov"}, "llm"),
}


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        d[k] = int(v.split()[0]) / 2**20  # GiB
    return {"avail": d["MemAvailable"], "swap_used": d["SwapTotal"] - d["SwapFree"], "shmem": d["Shmem"]}


def cgstat():
    try:
        st = dict(l.split() for l in (CG / "memory.stat").read_text().splitlines())
        peak = int((CG / "memory.peak").read_text()) / 2**30
        return {"anon": int(st["anon"]) / 2**30, "shmem": int(st["shmem"]) / 2**30,
                "file": int(st["file"]) / 2**30, "peak": peak}
    except FileNotFoundError:
        return None


def active():
    r = subprocess.run(["systemctl", "--user", "is-active", UNIT], capture_output=True, text=True)
    return r.stdout.strip() == "active"


def warm(name, kind):
    t0 = time.perf_counter()
    if kind == "llm":
        r = requests.post(f"{BASE}/v3/chat/completions", json={
            "model": name, "messages": [{"role": "user", "content": "빨간 공 옮겨 줘. 한 문장으로 답해."}],
            "max_tokens": 64, "chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
    elif kind == "embed":
        r = requests.post(f"{BASE}/v3/embeddings", json={"model": name, "input": ["빨간 공을 상자에 담는다"]}, timeout=600)
    elif kind == "rerank":
        r = requests.post(f"{BASE}/v3/rerank", json={"model": name, "query": "빨간 공",
                                                    "documents": ["빨간 공을 상자에 담는다", "파란 공"]}, timeout=600)
    else:
        with open(ROOT / "bench/out/tts_05.wav", "rb") as f:
            r = requests.post(f"{BASE}/v3/audio/transcriptions", files={"file": f},
                              data={"model": name, "language": "ko"}, timeout=600)
    r.raise_for_status()
    return time.perf_counter() - t0


def probe(name):
    spec, kind = SPECS[name]
    cfg = ROOT / "logs" / "probe_config.json"
    cfg.write_text(json.dumps({"model_config_list": [], "mediapipe_config_list": [dict(name=name, **{
        k: (str(ROOT / v) if k == "base_path" else v) for k, v in spec.items()})]}, indent=2))
    before = meminfo()
    env = dict(os.environ, OVMS_UNIT=UNIT, OVMS_PORT=str(PORT))
    subprocess.run([str(ROOT / "start_ovms.sh"), str(cfg)], env=env, check=True)
    t0 = time.perf_counter()
    ready = False
    min_avail = before["avail"]
    while time.perf_counter() - t0 < 900:
        min_avail = min(min_avail, meminfo()["avail"])
        if not active():
            break
        try:
            if requests.get(f"{BASE}/v2/health/ready", timeout=2).status_code == 200:
                ready = True
                break
        except requests.RequestException:
            pass
        time.sleep(1)
    load_s = time.perf_counter() - t0
    row = {"name": name, "ready": ready, "load_s": load_s, "before": before, "min_avail_during_load": min_avail}
    if ready:
        row["after_load"] = {"mem": meminfo(), "cg": cgstat()}
        row["warm_s"] = warm(name, kind)
        row["after_infer"] = {"mem": meminfo(), "cg": cgstat()}
        a = row["after_infer"]
        print(f"{name:16} 적재 {load_s:5.1f}s | OVMS anon {a['cg']['anon']:.2f}G shmem(GPU/NPU) {a['cg']['shmem']:.2f}G "
              f"peak {a['cg']['peak']:.2f}G | 시스템 가용 {before['avail']:.1f}→{a['mem']['avail']:.1f}G "
              f"(적재 중 최저 {min_avail:.1f}G) | 첫 추론 {row['warm_s']:.2f}s", flush=True)
    else:
        print(f"{name:16} 실패({load_s:.0f}s) — logs/{UNIT}.log 확인", flush=True)
    subprocess.run(["systemctl", "--user", "stop", UNIT])
    time.sleep(3)
    return row


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or list(SPECS)
    path = OUT / "mem_probe.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for n in names:
        results[n] = probe(n)
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
