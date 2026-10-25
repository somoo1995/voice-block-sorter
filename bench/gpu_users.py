"""i915 fdinfo 로 프로세스별 GPU 엔진 사용률 표본(기본 3초). 읽기만. 사용: python3 -I bench/gpu_users.py [초]

크롬(--type=gpu-process)·gnome-shell 이 렌더 엔진을, OVMS 가 연산 엔진(compute)을 얼마나 쓰는지 본다(SETUP.md §8-14, 10-23).
같은 클라이언트를 여러 fd 로 열어도 한 번만 센다. Studio 컨테이너 안은 uid 가 달라 docker exec -i <컨테이너> python3 -I - < bench/gpu_users.py
"""
import os, re, time, collections, sys
def snap():
    out = {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            name = open(f"/proc/{pid}/comm").read().strip()
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            kind = next((c.decode()[7:] for c in cmd if c.startswith(b"--type=")), "")
            fds = os.listdir(f"/proc/{pid}/fdinfo")
        except OSError:
            continue
        for fd in fds:
            try:
                t = open(f"/proc/{pid}/fdinfo/{fd}").read()
            except OSError:
                continue
            if "drm-driver:\ti915" not in t:
                continue
            m = re.search(r"drm-client-id:\s*(\d+)", t)
            eng = {g.group(1): int(g.group(2)) for g in re.finditer(r"drm-engine-([\w-]+):\s*(\d+) ns", t)}
            out[m.group(1)] = (f"{name}{'/' + kind if kind else ''}[{pid}]", eng)
    return out
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 3
a = snap(); t0 = time.time(); time.sleep(secs); b = snap(); dt = time.time() - t0
rows = collections.defaultdict(collections.Counter)
for cid, (n, e) in b.items():
    if cid in a:
        for eng, v in e.items():
            rows[n][eng] += v - a[cid][1].get(eng, 0)
print(f"표본 {dt:.1f}초 · i915 클라이언트 {len(b)}개(같은 클라이언트는 한 번만)")
for n, c in sorted(rows.items()):
    busy = {e: f"{v / dt / 1e7:.1f}%" for e, v in c.items() if v > 0}
    print(" ", n, busy or "0")
