"""녹화 관절 기록을 홈 복귀 정지 감지기에 재생해 본다 — 잘못 걸림(작업 중 걸림)이 0 인지, 걸리는 시점이 에피소드 끝과 얼마나 다른지.

    python -m robot.home_stop_replay                       # Studio 의 모든 데이터셋
    python -m robot.home_stop_replay bs_pick_defect bs_handoff
    python -m robot.home_stop_replay --near 10 --still 2    # 임계값을 바꿔 보기(기본 = policies.json early_stop)

데이터는 Studio 컨테이너 안에서 parquet 를 읽기만 한다(볼륨·DB 변경 없음). 에피소드 끝 뒤에는 마지막 자세를
pad 초 붙여 "정책이 홈에서 버팀"을 흉내 낸다. 판정:
- 잘못 걸림: 마지막으로 홈 근처에 들어온 시각보다 먼저 걸림(= 작업 도중 홈 근처에서 멈춘 곳에서 걸림)
- 안 걸림: 끝 자세가 near 밖(덜 돌아옴) → 실제 실행에서는 timeout 이 끝낸다
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from robot import home_stop
from robot import run_policy as rp

_DUMP = r"""
import glob, json, os, sys
import pandas as pd
want = set(sys.argv[1:])
out = {}
for root in sorted(glob.glob("/app/storage/datasets/*")):
    if want and os.path.basename(root) not in want:
        continue
    eps = {}
    for f in sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True)):
        df = pd.read_parquet(f, columns=["episode_index", "observation.state"])
        for ep, g in df.groupby("episode_index"):
            eps[int(ep)] = [[round(float(x), 3) for x in v] for v in g["observation.state"]]
    fps = json.load(open(os.path.join(root, "meta/info.json"))).get("fps", 30)
    out[os.path.basename(root)] = {"fps": fps, "episodes": eps}
print(json.dumps(out))
"""


def datasets(cfg: dict) -> dict[str, str]:
    """{데이터셋 id: 이름} (Studio API GET)."""
    def items(d):
        return d.get("items", d) if isinstance(d, dict) else d
    out = {}
    for p in items(rp._get(cfg, "/api/projects")):
        for d in rp._get(cfg, f"/api/projects/{p['id']}").get("datasets", []):
            out[d["id"]] = d.get("name", d["id"])
    return out


def load(cfg: dict, names: list[str]) -> dict:
    ids = datasets(cfg)
    pick = [i for i, n in ids.items() if not names or n in names or i in names]
    r = subprocess.run(rp._docker(cfg) + ["exec", "-i", "-u", "appuser", "-w", "/tmp", cfg["container"], rp.CPY,
                                          "-I", "-", *pick], input=_DUMP, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-500:])
    data = json.loads(r.stdout)
    return {ids.get(k, k): v for k, v in data.items()}


def replay(states: list[list[float]], fps: float, params: dict, pad_s: float = 3.0) -> dict:
    near = params["near"]
    home = states[0]
    dist = [max(abs(q[j] - home[j]) for j in range(5)) for q in states]
    t_home = None
    if dist[-1] <= near:
        i = len(dist) - 1
        while i > 0 and dist[i - 1] <= near:
            i -= 1
        t_home = i / fps
    det = home_stop.HomeStopDetector(**{k: params[k] for k in home_stop.DEFAULTS})
    full = states + [states[-1]] * int(pad_s * fps)
    for i, q in enumerate(full):
        if det.feed(i / fps, q):
            break
    tr = det.trigger
    t_end = (len(states) - 1) / fps
    return {"t_end": t_end, "t_home": t_home, "t_trig": tr.t if tr else None,
            "false": tr is not None and (t_home is None or tr.t < t_home - 1e-9),
            "end_dist": round(dist[-1], 2), "max_dist": round(max(dist), 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*")
    for k in ("leave", "near", "still", "hold_s"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float)
    ap.add_argument("--pad", type=float, default=3.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    cfg = rp.load_config()
    params = rp._early_stop_settings({"early_stop": {**cfg["defaults"].get("early_stop", {}), "enabled": True}})
    for k in ("leave", "near", "still", "hold_s"):
        if getattr(a, k) is not None:
            params[k] = getattr(a, k)
    data = load(cfg, a.datasets)
    print(f"임계값 {json.dumps({k: params[k] for k in home_stop.DEFAULTS})} · 끝 뒤 버팀 {a.pad:g}초")
    total = false = none = 0
    for name, d in data.items():
        rows = [replay(s, d["fps"], params, a.pad) for _, s in sorted(d["episodes"].items(), key=lambda x: int(x[0]))]
        trig = [r for r in rows if r["t_trig"] is not None and not r["false"]]
        f = [r for r in rows if r["false"]]
        n = [r for r in rows if r["t_trig"] is None]
        total, false, none = total + len(rows), false + len(f), none + len(n)
        vs_end = sorted(r["t_trig"] - r["t_end"] for r in trig)
        after = sorted(r["t_trig"] - r["t_home"] for r in trig if r["t_home"] is not None)
        med = (lambda xs: xs[len(xs) // 2]) if trig else None
        print(f"{name}: 에피소드 {len(rows)} · 걸림 {len(trig)} · 잘못 걸림 {len(f)} · 안 걸림 {len(n)}"
              + (f" · 끝 대비 {med(vs_end):+.2f}s({vs_end[0]:+.2f}~{vs_end[-1]:+.2f}) · 홈 도착 뒤 {med(after):.2f}s" if trig else ""))
        if a.verbose:
            for i, r in enumerate(rows):
                print(f"  {i:>3} 길이 {r['t_end']:5.1f} 끝 거리 {r['end_dist']:5.1f} 홈 도착 {r['t_home']} 걸림 {r['t_trig']}"
                      f"{' ← 잘못!' if r['false'] else ''}")
    print(f"합계: 에피소드 {total} · 잘못 걸림 {false} · 안 걸림 {none}")
    return 1 if false else 0


if __name__ == "__main__":
    sys.exit(main())
