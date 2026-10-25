"""run_policy 의 대기 루프·stop·시간 상한을 로봇·Studio 없이 시험한다.

    python -m robot.selftest

`physicalai run` 대신 이름만 같은 가짜 프로세스(호스트)를 띄우고, 컨테이너 조사 스크립트(_PROBE)도
호스트에서 그대로 돌린다. 로봇·카메라·컨테이너·robot/.state 는 건드리지 않는다(임시 폴더 사용).
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import robot.run_policy as rp

FAKE = r"""
import json, os, signal, subprocess, sys, time
# 카메라 publisher 흉내: 이 실행이 끝나면 FAKE_CAM_IDLE 초 뒤에 스스로 끝난다(명령줄에 _publisher_worker → _PROBE 가 camera_publisher 로 셈)
PUB = "import sys, time\nppid = sys.argv[2]\nwhile True:\n    try:\n        if 'State:\\tZ' in open('/proc/' + ppid + '/status').read():\n            break\n    except OSError:\n        break\n    time.sleep(0.05)\ntime.sleep(float(sys.argv[3]))\n"  # 부모가 끝나면(좀비 포함)
subprocess.Popen([sys.executable, "-I", "-c", PUB, "_publisher_worker", str(os.getpid()), os.environ.get("FAKE_CAM_IDLE", "1.0")],
                 start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
def on_int(*_):
    print("SIGINT: 끝내는 중", flush=True)
    time.sleep(float(os.environ.get("FAKE_SLOW_INT", "0.5")))
    print("Run complete: 7 steps", flush=True)
    sys.exit(0)
signal.signal(signal.SIGINT, signal.SIG_IGN if os.environ.get("FAKE_HANG") else on_int)
HOME = [0.0, -52.3, 4.8, 50.9, 2.2, 50.1]
# FAKE_TICKS: 관절 1 의 홈 대비 위치(정규화)를 시각(초)별로 이은 구간 — JsonlCallback 과 같은 틱 줄을 30Hz 로 찍는다
SCEN = {
    "task":  [(0, 0), (1.0, 0), (3.0, 50), (5.0, 0)],                                   # 홈 대기 → 나감 → 돌아와 멈춤
    "pause": [(0, 0), (1.0, 0), (2.0, 50), (2.5, 5), (3.1, 5), (4.1, 50), (5.1, 0)],    # 도중 홈 근처 0.6초 멈춤
    "away":  [(0, 0), (1.0, 0), (2.5, 30)],                                              # 안 돌아옴
    "still": [(0, 0)],                                                                   # 안 움직임
    "quick": [(0, 0), (0.3, 0), (0.9, 40), (1.5, 0)],                                    # 짧은 작업(연속 실행 시험)
}
def pos(seg, t):
    if t >= seg[-1][0]:
        return seg[-1][1]
    for (t1, v1), (t2, v2) in zip(seg, seg[1:]):
        if t1 <= t <= t2:
            return v1 + (v2 - v1) * (t - t1) / max(t2 - t1, 1e-9)
seg = SCEN.get(os.environ.get("FAKE_TICKS", ""))
t0 = time.time()
i = 0
while time.time() - t0 < float(os.environ.get("FAKE_DUR", "1")):
    if seg:
        q = list(HOME)
        q[1] += pos(seg, time.time() - t0) + (0.2 if i % 2 else -0.2)   # ±0.2 흔들림
        print(json.dumps({"type": "tick", "session_id": "fake", "step": i, "timestamp": time.time(),
                          "joint_positions": q, "action_sent": q, "loop_duration_s": 0.01,
                          "sleep_time_s": 0.02, "stale_obs": False}), flush=True)
        i += 1
    time.sleep(1 / 30 if seg else 0.05)
print("Run complete: 30 steps", flush=True)
"""


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="robot-selftest-"))
    fake = tmp / "bin" / "physicalai"  # 조사 스크립트가 argv[1] 이 .../physicalai, argv[2] 가 run 인 것을 찾는다
    fake.parent.mkdir()
    fake.write_text(FAKE, encoding="utf-8")
    rp.STATE_DIR, rp.LOG_DIR = tmp / "state", tmp / "logs"
    rp.STATE_DIR.mkdir()

    def local_probe(cfg, action="list", paths=(), timeout=20):
        r = subprocess.run([sys.executable, "-I", "-c", rp._PROBE, rp.MARKER, action, *paths],
                           capture_output=True, text=True, timeout=timeout)
        return __import__("json").loads(r.stdout)

    real_preflight, real_validate = rp.preflight, rp._validate
    rp._probe = local_probe
    # docker exec 흉내: 가짜 physicalai 가 끝난 뒤 껍데기(sh)가 FAKE_EXEC_DELAY(기본 2.1)초 더 있다가 돌아온다
    rp._argv = lambda plan, run_id: ["env", f"{rp.MARKER}={run_id}", "sh", "-c",
                                     f'"{sys.executable}" "{fake}" run; rc=$?; sleep ${{FAKE_EXEC_DELAY:-2.1}}; exit $rc']
    rp.preflight = lambda plan, **kw: {"problems": [], "info": {}}
    rp.load_config = lambda: {}

    early = {**rp.home_stop.DEFAULTS, "near": 8.0, "still": 3.0, "return_far": 6.0}
    returns: list = []
    rp._return_home = lambda plan, q0, run_id, log_path: (returns.append(q0), {"ok": True, "reason": "done"})[1]

    def plan(wall=30.0, es=None):
        return {"name": "pick_red", "cfg": {}, "settings": {"camera_idle_timeout_s": 0.5},
                "model": {"id": "fake"}, "timeout_s": 1.0, "wall_limit_s": wall,
                "runtime_cli": {"runtime": {}}, "config_json": "{}", "run_args": [],
                "early_stop": es, "return_home": True, "runtime_return": es is None}

    def run(wall=30.0, env=None, during=None, es=None):
        for k in ("FAKE_HANG", "FAKE_TICKS", "FAKE_SLOW_INT"):
            os.environ.pop(k, None)
        os.environ.update(env or {})
        if during:
            threading.Timer(during[0], during[1]).start()
        returns.clear()
        t0 = time.monotonic()
        res = rp._run(plan(wall, es), rp._now())
        res["_returns"] = len(returns)
        return res, time.monotonic() - t0

    def stop_file():
        (rp.STATE_DIR / "stop").write_text("kill:pending", encoding="utf-8")  # 다른 프로세스의 stop() 이 남기는 흔적

    cases = [
        ("정상 종료", dict(env={"FAKE_DUR": "1"}), "duration_elapsed", True),
        ("stop() 즉시 중단", dict(env={"FAKE_DUR": "20"}, during=(1.0, rp.stop)), "stopped", False),
        ("stop(home=True)", dict(env={"FAKE_DUR": "20"}, during=(1.0, lambda: rp.stop(home=True))), "stopped", False),
        ("다른 프로세스의 stop", dict(env={"FAKE_DUR": "20"}, during=(1.0, stop_file)), "stopped", False),
        ("벽시계 상한 → SIGINT", dict(wall=1.5, env={"FAKE_DUR": "20"}), "timeout", False),
        ("SIGINT 무시 → SIGKILL", dict(wall=1.0, env={"FAKE_DUR": "60", "FAKE_HANG": "1"}), "timeout", False),
    ]
    failed = 0
    for title, kw, want_reason, want_ok in cases:
        res, took = run(**kw)
        good = res["reason"] == want_reason and res["ok"] == want_ok
        failed += not good
        print(f"{'OK ' if good else 'BAD'} {title:<22} reason={res['reason']:<16} ok={res['ok']!s:<5} "
              f"rc={res['returncode']} steps={res['steps']} {took:.1f}s")
    # 홈 복귀 정지 감지(조기 종료)와 끝 자세가 멀 때만 복귀
    early_cases = [
        ("조기 종료: 홈 복귀 정지", dict(env={"FAKE_DUR": "20", "FAKE_TICKS": "task"}), "home_reached", True, 0,
         lambda r, took: 5.5 < r["early_stop"]["t_s"] < 6.6 and took < 9),  # 홈 도착 5.0초 뒤 0.5~1.6초
        ("조기 종료: 도중 0.6초 멈춤 무시", dict(env={"FAKE_DUR": "20", "FAKE_TICKS": "pause"}), "home_reached", True, 0,
         lambda r, took: r["early_stop"]["t_s"] > 6.0),
        ("안 돌아옴 → timeout·복귀", dict(env={"FAKE_DUR": "4", "FAKE_TICKS": "away"}), "duration_elapsed", True, 1,
         lambda r, took: not r["early_stop"]["triggered"] and r["end_dist"] > 25),
        ("안 움직임 → 감지 안 함", dict(env={"FAKE_DUR": "3", "FAKE_TICKS": "still"}), "duration_elapsed", True, 0,
         lambda r, took: not r["early_stop"]["moved"] and r["end_dist"] < 1),
        ("stop() → 복귀 안 함", dict(env={"FAKE_DUR": "20", "FAKE_TICKS": "away"}, during=(3.0, rp.stop)), "stopped", False, 0,
         lambda r, took: True),
        ("stop(home=True) → 복귀", dict(env={"FAKE_DUR": "20", "FAKE_TICKS": "away"},
                                     during=(3.0, lambda: rp.stop(home=True))), "stopped", False, 1, lambda r, took: True),
    ]
    for title, kw, want_reason, want_ok, want_ret, extra in early_cases:
        res, took = run(es=early, **kw)
        good = (res["reason"] == want_reason and res["ok"] == want_ok and res["_returns"] == want_ret
                and extra(res, took))
        failed += not good
        es_ = res.get("early_stop") or {}
        print(f"{'OK ' if good else 'BAD'} {title:<22} reason={res['reason']:<16} ok={res['ok']!s:<5} "
              f"감지 {es_.get('t_s')}s 끝 거리 {res.get('end_dist')} 복귀 {res['_returns']}회 {took:.1f}s")
    # 결과 먼저 돌려주기 + 다음 실행은 이전 실행이 풀릴 때까지 기다리기
    conflicts = {"preflight": [], "after_return": []}

    def checking_preflight(plan, **kw):
        busy = [p for p in local_probe({}, "list")["procs"] if p["kind"] in ("run", "camera_publisher")]
        if busy:
            conflicts["preflight"].append(busy)
        return {"problems": [], "info": {}}

    rp.preflight = checking_preflight
    os.environ.pop("FAKE_DUR", None)
    res, took = run(env={"FAKE_DUR": "20", "FAKE_TICKS": "quick"}, es=early)
    tail_alive = res["run_id"] in rp._TAILS
    good = res["reason"] == "home_reached" and res["tail_s"] < 1.6 and tail_alive and res["camera_released"]
    failed += not good
    print(f"{'OK ' if good else 'BAD'} {'결과 먼저(exec 안 기다림)':<22} 끝 → 반환 {res['tail_s']}s(카메라 놓기 1.0 포함, exec 2.1 안 기다림), "
          f"반환 때 exec 껍데기 아직 있음 {tail_alive}")
    t0 = time.monotonic()
    rows = []
    for i in range(5):
        res, took = run(env={"FAKE_DUR": "20", "FAKE_TICKS": "quick"}, es=early)
        cam_busy = [p for p in local_probe({}, "list")["procs"] if p["kind"] == "camera_publisher"]
        if cam_busy:
            conflicts["after_return"].append(cam_busy)  # 에이전트의 detect 가 바로 카메라를 열 수 있어야 한다
        rows.append((res["reason"], took, res["tail_s"], len(rp._TAILS)))
    total = time.monotonic() - t0
    good = (all(r[0] == "home_reached" for r in rows) and not conflicts["preflight"] and not conflicts["after_return"])
    failed += not good
    print(f"{'OK ' if good else 'BAD'} {'연속 5회':<22} 잠금·실행 겹침 {len(conflicts['preflight'])} · 반환 직후 카메라 사용 중 "
          f"{len(conflicts['after_return'])} · 회당 {', '.join(f'{r[1]:.1f}' for r in rows)}s · 합계 {total:.1f}s · "
          f"다음 실행 시작 때 이전 exec 껍데기가 남아 있던 회 {sum(1 for r in rows if r[3] > 0)}")
    # 이전 실행이 끝나는 중(실행 프로세스가 아직 있음) → 다음 실행이 기다렸다가 시작
    linger = subprocess.Popen(["env", f"{rp.MARKER}=old-run", sys.executable, str(fake), "run"],
                              env={**os.environ, "FAKE_DUR": "1.5", "FAKE_TICKS": ""}, stdout=subprocess.DEVNULL)
    threading.Thread(target=linger.wait, daemon=True).start()  # 바로 거둔다(좀비로 남지 않게)
    time.sleep(0.3)
    conflicts["preflight"].clear()
    res, took = run(env={"FAKE_DUR": "20", "FAKE_TICKS": "quick"}, es=early)
    waited = float(open(res["log_path"], encoding="utf-8").readline().split("waited_previous_s=", 1)[1].split()[0])
    good = res["reason"] == "home_reached" and waited >= 1.0 and not conflicts["preflight"]
    failed += not good
    print(f"{'OK ' if good else 'BAD'} {'끝나는 중인 이전 실행 기다림':<22} 기다린 시간 {waited}s(이전 실행 1.2초 + 카메라 1.0초) · 겹침 "
          f"{len(conflicts['preflight'])}")
    # stop() 은 SIGINT 를 받고 끝나는 중인 실행도 바로 끊는다
    t_stop = []

    def stop_soon():
        t_stop.append(time.monotonic())
        rp.stop()

    os.environ["FAKE_SLOW_INT"] = "8"
    holder = {}

    def watch():  # 조기 종료 SIGINT 를 보낸 뒤(끝나는 중) stop()
        while not any(ln.startswith("# 홈 복귀 정지 감지") for ln in _read(holder.get("log"))):
            time.sleep(0.05)
        time.sleep(0.3)
        stop_soon()

    def _read(path):
        try:
            return open(path, encoding="utf-8").read().splitlines()
        except (OSError, TypeError):
            return []

    orig_run = rp._run

    def run_and_watch(plan_, started):
        holder["log"] = str(rp.LOG_DIR / f"{started:%Y%m%d}" / f"{started:%H%M%S}_{plan_['name']}.log")
        threading.Thread(target=watch, daemon=True).start()
        return orig_run(plan_, started)

    rp._run = run_and_watch
    try:
        res, took = run(env={"FAKE_DUR": "20", "FAKE_TICKS": "quick", "FAKE_SLOW_INT": "8"}, es=early)
    finally:
        rp._run = orig_run
        os.environ.pop("FAKE_SLOW_INT", None)
    after = time.monotonic() - t_stop[0] if t_stop else 99
    good = res["reason"] == "stopped" and after < 2.5
    failed += not good
    print(f"{'OK ' if good else 'BAD'} {'stop() → 끝나는 중인 실행':<22} reason={res['reason']} · stop 뒤 {after:.2f}s 만에 반환"
          f"(SIGINT 를 받고 8초 끌던 가짜를 SIGKILL)")
    for th_ in list(rp._TAILS.values()):
        th_.join(timeout=5)
    rp.preflight = lambda plan, **kw: {"problems": [], "info": {}}
    # --print_config 검사 캐시: 같은 날 같은 설정 해시면 건너뜀, 바뀌면·실패면·다음 날이면 다시
    calls: list = []
    rp._validate = lambda p: (calls.append(p["name"]), (p.get("_ok", True), "가짜 print_config"))[1]
    p1 = {"name": "pick_red", "model": {"id": "m1"}, "config_json": '{"port": "/dev/ttyACM0"}', "run_args": ["--run.duration_s=21"]}
    r = [rp._validate_cached(p1)[2], rp._validate_cached(p1)[2]]
    r.append(rp._validate_cached({**p1, "config_json": '{"port": "/dev/ttyACM1"}'})[2])   # USB 번호가 바뀜
    r.append(rp._validate_cached({**p1, "model": {"id": "m2"}})[2])                        # 모델이 바뀜
    bad = {**p1, "name": "pick_blue", "_ok": False}
    r += [rp._validate_cached(bad)[2], rp._validate_cached(bad)[2]]                         # 실패는 기록 안 함
    real_now = rp._now
    rp._now = lambda: real_now() + dt.timedelta(days=1)
    r.append(rp._validate_cached(p1)[2])                                                      # 다음 날
    rp._now = real_now
    rp.robot_preflight = lambda cfg, port, paths: ([], {})
    rp._validate = lambda p: (calls.append("preflight"), (True, "가짜"))[1]
    pp = {**p1, "name": "handoff", "cfg": {}, "follower": {"port": "/dev/ttyACM0"}, "container_paths": [], "cameras": []}
    cached_pf = [real_preflight(pp, use_cache=True)["info"]["validate_cached"],
                 real_preflight(pp, use_cache=True)["info"]["validate_cached"],
                 "validate_cached" in real_preflight(pp)["info"]]                             # dry-run·--link 경로는 캐시 안 씀
    rp._validate = real_validate
    want = [False, True, False, False, False, False, False]
    good = r == want and cached_pf == [False, True, False] and calls.count("preflight") == 2
    failed += not good
    print(f"{'OK ' if good else 'BAD'} {'print_config 캐시':<22} 건너뜀 {r} (기대 {want}) · preflight 경로 {cached_pf} · "
          f"검사 실제 호출 {len(calls)}회")
    # 사전 점검 중에 온 stop: 프로세스를 띄우지 않아야 한다
    rp.preflight = lambda plan, **kw: (rp.stop(), {"problems": [], "info": {}})[1]
    res, took = run(env={"FAKE_DUR": "20"})
    good = res["reason"] == "stopped" and res["returncode"] is None
    failed += not good
    print(f"{'OK ' if good else 'BAD'} {'사전 점검 중 stop':<22} reason={res['reason']:<16} rc={res['returncode']} {took:.1f}s")
    time.sleep(1.2)  # 마지막 가짜 카메라 publisher 가 스스로 끝날 시간
    for th_ in list(rp._TAILS.values()):
        th_.join(timeout=5)
    # 정책별 추론 설정(execution)이 결과·dry-run 요약(_public)과 실제 런타임 설정에 다 들어가나(10-23: 요약에서 빠져 'sync 0.5')
    ex_ = rp.execution_settings({"execution": {"mode": "async", "request_threshold": 0.85, "smoother_frames": 8}})
    src_ = rp._policy_source("/x", "CPU", None, ex_)["init_args"]
    good = (rp._public({"execution": ex_}).get("execution") == ex_
            and src_["execution"] == {"class_path": "physicalai.runtime.AsyncExecution", "init_args": {"request_threshold": 0.85}}
            and src_["action_queue"]["init_args"]["smoother"]["init_args"] == {"duration_frames": 8}
            and rp._policy_source("/x", "CPU", None)["init_args"]["execution"]["init_args"] == {"request_threshold": 0.5})
    failed += not good
    print(f"{'OK ' if good else 'BAD'} {'추론 설정 execution':<22} 요약 {rp._public({'execution': ex_}).get('execution')} · 런타임 "
          f"{src_['execution']['class_path'].rsplit('.', 1)[-1]} {src_['execution']['init_args']} · 섞기 "
          f"{src_['action_queue']['init_args']['smoother']['init_args']}")
    left = local_probe({}, "list")["procs"]
    print(f"남은 가짜 프로세스: {left} · 거두는 중인 exec {list(rp._TAILS)}")
    return 1 if failed or left or rp._TAILS else 0


if __name__ == "__main__":
    sys.exit(main())
