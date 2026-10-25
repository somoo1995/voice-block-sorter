"""좌표 집기 자체 시험 — 로봇·카메라·Studio 없이(--offline 과 같은 기본 범위).

숫자 배치·실행기 시험은 예시 작업대(config_example.json), 그릇 옮기기 시험은 10-22 실측 config.json 으로.

    python -m robot.coord_pick.selftest
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from robot.coord_pick import digits
from robot.coord_pick.__main__ import main as cli
from robot.coord_pick.geometry import Frames, apply_h, fit_homography, fit_rigid2d
from robot.coord_pick.kinematics import ARM, JointMap, jaw_gap_mm
from robot.coord_pick.plan import (EXAMPLE_CONFIG_PATH, Setup, blocks_from_detection, demo_scene, demo_sort_scene,
                                   load_config, plan_digit, plan_sort, trajectory)

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((bool(ok), name, detail))
    print(f"{'OK ' if ok else 'BAD'} {name:<40} {detail}")


def main() -> int:
    t0 = time.perf_counter()
    s = Setup.load(load_config(EXAMPLE_CONFIG_PATH), use_studio=False)
    kin = s.kin

    T = kin.fk(np.zeros(5))
    check("FK: 영점 자세 끝 위치", np.allclose(T[:3, 3] * 1000, [391.4, 0.0, 226.5], atol=0.5),
          f"{np.round(T[:3, 3] * 1000, 1).tolist()} mm")

    jm = JointMap({k: (0, 4095) for k in ARM}, {k: 1 for k in ARM}, {k: 0.0 for k in ARM})
    n = np.array([10.0, -52.3, 4.8, 50.9, 2.2])
    check("정규화 ↔ 각: 범위 0~4095 면 1.8°/단위", abs(jm.deg_per_unit("elbow_flex") - 1.8) < 1e-3
          and np.allclose(jm.to_norm(jm.to_rad(n)), n), f"{jm.deg_per_unit('elbow_flex'):.4f}°")

    worst, fails = 0.0, 0
    for x in range(140, 281, 35):
        for y in range(-100, 101, 50):
            for z in (15, 50):
                q, info = s.solve(np.array([x, y, z], float), 0.0, s.n_to_q(np.array(s.cfg["home_n"])))
                if q is None:
                    fails += 1
                    continue
                Tq = kin.fk(q)
                worst = max(worst, float(np.linalg.norm(Tq[:3, 3] * 1000 - [x, y, z])))
    check("IK: 앞쪽 격자 50점(위에서 수직)", fails == 0 and worst < 2.0, f"실패 {fails}, 최대 오차 {worst:.2f}mm")

    rng = np.random.default_rng(1)
    H0 = np.array([[1.1, 0.05, 30], [-0.02, -1.2, 420], [1e-4, 2e-4, 1]])
    px = rng.uniform(50, 600, (6, 2))
    mm = apply_h(H0, px)
    H = fit_homography(px[:4], mm[:4])
    err4 = float(np.abs(apply_h(H, px) - mm).max())
    noisy = px + rng.normal(0, 1.0, px.shape)
    err6 = float(np.linalg.norm(apply_h(fit_homography(noisy, mm), px) - mm, axis=1).max())
    check("호모그래피: 4점 정확 · 6점 1px 잡음", err4 < 1e-6 and err6 < 5, f"{err4:.1e} / {err6:.2f}mm")

    fr = Frames(150, -110, 90)
    r_pts = rng.uniform(100, 300, (4, 2))
    tx, ty, yaw = fit_rigid2d(r_pts, fr.r2w(r_pts))
    check("평면 강체 맞추기", abs(tx - 150) < 1e-6 and abs(ty + 110) < 1e-6 and abs(math.degrees(yaw) - 90) < 1e-6,
          f"({tx:.1f}, {ty:.1f}, {math.degrees(yaw):.1f}°)")

    counts = {d: len(digits.cells(d)) for d in digits.FONT}
    check("숫자 표 칸 수", counts == {"0": 12, "1": 8, "2": 11, "3": 11, "4": 9, "5": 11, "6": 12, "7": 7, "8": 13, "9": 12},
          " ".join(f"{d}:{c}" for d, c in counts.items()))

    blocks = demo_scene(s, 9)
    p = plan_digit(s, "7", blocks)
    check("계획: 숫자 7 가짜 장면 — 실패 0", not p["failures"] and len(p["order"]) == 7,
          f"블록 {len(p['order'])}개, {p['total_s']:.0f}초, 경고 {len(p['warnings'])}")
    # 내려간 끝이 정말 블록·칸 위치인가(FK 로 되짚기)
    off = []
    downs =[st for st in p["steps"] if st.what.endswith(" 내려감")]
    puts = [st for st in p["steps"] if st.what.endswith(" 내려놓음")]
    for st, o in zip(downs, p["order"]):
        tip = kin.fk(s.n_to_q(st.n))[:3, 3] * 1000
        off.append(np.linalg.norm(tip - s.w_to_r3(o["block"]["w"], s.cfg["heights_mm"]["grasp"])))
    for st, o in zip(puts, p["order"]):
        tip = kin.fk(s.n_to_q(st.n))[:3, 3] * 1000
        off.append(np.linalg.norm(tip - s.w_to_r3(o["target_w"], s.cfg["heights_mm"]["place"])))
    check("계획: 집는 점·놓는 점을 FK 로 되짚기", len(off) == 2 * len(p["order"]) and max(off) < 1.0, f"최대 {max(off):.2f}mm")
    tr = trajectory(p["steps"], s.cfg["motion"]["rate_hz"])
    step_deg = float(np.abs(np.diff(tr[:, :5], axis=0)).max()) * 1.8
    lim = s.cfg["motion"]["max_deg_s"] / s.cfg["motion"]["rate_hz"]
    check("궤적: 한 틱 관절 변화 ≤ 속도 한도", np.isfinite(tr).all() and step_deg <= lim * 1.05,
          f"최대 {step_deg:.2f}° ≤ {lim:.2f}°/틱")

    det_obj = {"objects": [{"cls": b["cls"], "zone": "input", "cell": "가운데", "px": b["px"], "conf": 0.9} for b in blocks]
               + [{"cls": "hand", "zone": "hand_spot", "cell": "가운데", "px": [10, 10], "conf": 0.8}]}
    det_box = {"detections": [{"cls": b["cls"], "key": b["cls"], "conf": 0.9,
                               "box": [b["px"][0] - 20, b["px"][1] - 18, b["px"][0] + 20, b["px"][1] + 18]} for b in blocks]}
    a, b2 = blocks_from_detection(det_obj), blocks_from_detection(det_box)
    check("검출 결과 읽기(objects·detections)", len(a) == len(b2) == len(blocks)
          and np.allclose([x["px"] for x in a], [x["px"] for x in b2]), f"{len(a)}개, 손 제외")

    # 입회 보정 3단계 흉내: 테이프 점 3개를 팔 끝으로 짚은 관절값 → base 자세가 config 와 같게 나오나
    mm4 = np.array(s.cfg["table"]["markers_mm"], float)
    touches = []
    for i in range(4):
        pr = s.w_to_r3(mm4[i], 0.0)
        q, info = kin.ik(pr / 1000, yaw=None, q0=np.radians([math.degrees(math.atan2(pr[1], pr[0])), 0, 30, 65, 0]))
        touches.append({"marker": i, "n": s.q_to_n(q, 50.0).round(3).tolist(), "_ok": info["ok"]})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([{k: v for k, v in t.items() if k != "_ok"} for t in touches], f)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli(["--offline", "--config", str(EXAMPLE_CONFIG_PATH), "calib-base", f.name, "--contact", "frame"])
    Path(f.name).unlink()
    b = s.cfg["table"]["base"]
    want = f'"x": {float(b["x"])}, "y": {float(b["y"])}, "yaw_deg": {float(b["yaw_deg"])}'
    check("보정: 팔 끝 짚기 4점 → 로봇 위치", rc == 0 and all(t["_ok"] for t in touches) and want in buf.getvalue(),
          buf.getvalue().strip().splitlines()[-1][:80])

    executor_tests(s)
    sort_tests()

    bad = [r for r in results if not r[0]]
    print(f"\n{len(results) - len(bad)}/{len(results)} 통과 · {time.perf_counter() - t0:.1f}초")
    return 1 if bad else 0


def executor_tests(s: Setup) -> None:
    """실행기(executor.py·stream.py) — 가짜 SharedRobot 으로만. 실제 컨테이너·로봇에는 아무것도 보내지 않는다."""
    import copy
    import os
    import subprocess
    import threading

    from robot import run_policy as rp
    from robot.coord_pick import executor as ex
    from robot.coord_pick import fake, stream

    tmp = Path(tempfile.mkdtemp(prefix="coord_pick_selftest_"))
    base = load_config(EXAMPLE_CONFIG_PATH)
    det = {"objects": [{"cls": b["cls"], "px": b["px"], "conf": 0.9} for b in demo_scene(s, 9)]}

    def run(digit, cfg=None, **kw):
        return ex.arrange_digit(digit, detection=copy.deepcopy(det), config=cfg or copy.deepcopy(base), log_dir=tmp, **kw)

    # 1) 정상: 숫자 7·1 계획 → 느린 속도로 늘림 → 가짜 로봇(지연 + 어깨 처짐 3.6° + 블록을 쥔 그리퍼)으로 끝까지
    for digit, kw in (("7", {}), ("1", {}), ("1", {"small": True})):
        r = run(digit, **kw)
        sf = ex.safety(base)
        lim = ex.limits(s, sf, r["speed_deg_s"])
        p = plan_digit(s, digit, blocks_from_detection(copy.deepcopy(det)), small=kw.get("small", False))
        traj, marks = trajectory(p["steps"], base["motion"]["rate_hz"], with_marks=True)
        tg, mk = ex.retime(traj, marks, lim["step_body"], lim["step_grip"])
        sim = ex.simulate(tg, mk, lim, tau=0.08, bias=(1, 2.0), grip_block=55.0)
        steps = sum(e["ev"] == "step" for e in sim["events"])
        name = f"실행기: 숫자 {digit}{'(작은)' if kw else ''} dry-run + 지연·처짐 가짜 로봇"
        check(name, r["ok"] and r["mode"] == "dry_run" and not r["executed"] and sim["reason"] == "done"
              and sim["sent"] == len(tg) and steps == len(mk),
              f"블록 {r['blocks']}개 {r['time_s']}초, 보냄 {sim['sent']}, 단계 {steps}, 최대 오차 {max(sim['max_err']):.1f}")

    # 틱당 변화가 한도 안인가(속도 상한)
    step = np.abs(np.diff(tg[:, :5], axis=0)).max(axis=0)
    check("실행기: 틱당 관절 변화 ≤ 15°/s", bool((step <= np.array(lim["step_body"]) * 1.001).all()),
          f"최대 {(step * 1.8).max():.3f}°/틱 ≤ {15 / 30:.3f}")

    # 2) 추종 오차 초과 → 그 틱 근처에서 중단, 마지막 송신 = 측정 자세(몸통)·마지막 그리퍼 명령
    sim = ex.simulate(tg, mk, lim, stuck=(1, 300))
    rob = sim["robot"]
    last = rob.sends[-1]
    # 멈춘 관절(1)의 명령이 멈춘 값에서 한도보다 멀어진 첫 틱 → 그 뒤 lag + track_ticks + 2 틱 안에 멈춰야 한다
    frozen = rob.q[1]
    cmd1 = np.array([sd[1] for sd in rob.sends[:-1]])
    far = [i for i in range(300, len(cmd1)) if abs(cmd1[max(0, i - lim["lag_ticks"])] - frozen) > lim["track_max"][1]]
    late = sim["detail"]["tick"] - far[0] if far else 999
    check("실행기: 추종 오차 초과 → 중단·그 자리 유지", sim["reason"] == "tracking" and sim["detail"]["joint"] == 1
          and late <= lim["lag_ticks"] + lim["track_ticks"] + 2
          and np.allclose(last[:5], rob.q[:5], atol=0.05) and abs(last[5] - rob.sends[-2][5]) < 1e-9,
          f"관절 1 이 틱 300 에 멈춤 → 오차가 한도를 넘은 틱 {far[0] if far else '-'}, 중단 틱 {sim['detail']['tick']}"
          f"(+{late}), 마지막 송신 = 측정 자세")
    sim = ex.simulate(tg, mk, lim, stale_from=200)
    sends = sim["robot"].sends
    check("실행기: 상태 끊김(관측 시각 멈춤) → 중단", sim["reason"] == "stale_state" and sim["sent"] < 200 + 20
          and np.allclose(sends[-1], sends[-2]), f"{sim['reason']} 보냄 {sim['sent']}, 마지막 명령 그대로 유지")

    # 3) 경계 밖·조건 밖 → 거부(아무것도 안 보냄)
    c = copy.deepcopy(base)
    c["heights_mm"]["grasp"] = 9  # 계획(집게 끝 ≥ 0)은 되지만 집게 끝 2.7mm < 실행기 바닥 5mm
    r = run("7", c)
    kinds = {v["kind"] for v in r["violations"]}
    check("실행기: 바닥 아래 목표 → 거부", not r["ok"] and r["reason"] == "refused" and "floor" in kinds, f"{sorted(kinds)}")
    c = copy.deepcopy(base)
    org = np.array(c["digit"]["origin_mm"], float)
    c["safety"]["forbid_extra_mm"] = [(org + [[-25, -25], [25, -25], [25, 25], [-25, 25]]).tolist()]
    r = run("7", c)
    kinds = {v["kind"] for v in r["violations"]}
    check("실행기: 금지 구역(그릇·손 자리) 위 → 거부", not r["ok"] and "forbidden:extra1" in kinds, f"{sorted(kinds)}")
    c = copy.deepcopy(base)
    c["safety"]["radius_mm"] = [110, 250]
    r = run("7", c)
    kinds = {v["kind"] for v in r["violations"]}
    check("실행기: 반경 밖 → 거부", not r["ok"] and "radius" in kinds, f"{sorted(kinds)}")
    c = copy.deepcopy(base)
    c["gripper"]["close"] = 45
    r = run("7", c)
    kinds = {v["kind"] for v in r["violations"]}
    check("실행기: 그리퍼를 블록보다 더 쥐는 값 → 거부", not r["ok"] and {"gripper", "gripper_close"} & kinds, f"{sorted(kinds)}")
    r = run("7", color="red")  # 가짜 장면의 분홍은 4개뿐
    check("실행기: 블록이 모자라 모양을 못 채움 → ok=False", not r["ok"] and r["blocks"] < 7
          and any("필요한데" in f for f in r["failures"]), r["failures"][-1][:60] if r["failures"] else "")
    clock = fake.FakeClock()
    rob = fake.FakeSharedRobot(tg[0], clock)
    jump = tg[:50].copy()
    jump[30:, 1] += 5  # 한 틱에 9°
    res = stream.stream(rob, jump, [], **lim, emit=lambda e: None, clock=clock.now, sleep=clock.sleep)
    rob2 = fake.FakeSharedRobot(tg[0] + [0, 12, 0, 0, 0, 0], clock)  # 홈에서 21° 떨어짐
    res2 = stream.stream(rob2, tg, mk, **lim, emit=lambda e: None, clock=clock.now, sleep=clock.sleep)
    nan = tg[:10].copy()
    nan[5, 2] = np.nan
    res3 = stream.stream(fake.FakeSharedRobot(tg[0], clock), nan, [], **lim, emit=lambda e: None, clock=clock.now,
                         sleep=clock.sleep)
    check("실행기: 틱 한도 넘는 점프·홈 아님·NaN → 하나도 안 보냄",
          (res["reason"], res2["reason"], res3["reason"]) == ("step_limit", "not_at_home", "step_limit")
          and not rob.sends and not rob2.sends, f"{res['reason']} · {res2['reason']} · {res3['reason']}")
    saved = os.environ.pop("BALL_ROBOT", None)
    real_called = []
    orig = ex._run_real
    ex._run_real = lambda *a, **k: real_called.append(1) or {"reason": "done"}
    try:
        r1 = run("7", dry_run=False)                       # BALL_ROBOT 없음
        os.environ["BALL_ROBOT"] = "1"
        r2 = run("7", dry_run=False)                       # 보정 전(calibrated=false)·금지 구역 없음
        c = copy.deepcopy(base)
        c["calibrated"] = True
        c["safety"]["forbid_extra_mm"] = [[[-200, -200], [-150, -200], [-150, -150], [-200, -150]]]
        r3 = run("7", c, dry_run=False)                    # 문이 다 열림 → (가짜) 실제 실행 호출
    finally:
        ex._run_real = orig
        os.environ.pop("BALL_ROBOT", None)
        if saved is not None:
            os.environ["BALL_ROBOT"] = saved
    check("실행기: BALL_ROBOT=1·확인·보정·금지 구역이 다 있어야 송신",
          r1["reason"] == "refused" and "BALL_ROBOT" in r1["error"] and r2["reason"] == "refused"
          and "calibrated" in r2["error"] and r3["mode"] == "real" and real_called == [1],
          f"없음 → {r1['reason']} · 보정 전 → {r2['reason']} · 다 있음 → 실행 호출 {len(real_called)}회")

    # 정책이 끝난 뒤 복귀(run_policy 2.3 이 부르는 stream.container_return) — 실시간 가짜 로봇
    from robot.run_policy import RETURN_LIMITS
    home = np.array(base["home_n"], float)
    quiet = io.StringIO()
    outs = {}
    for label, start, fault in (("far", home + [0, 20, -15, 10, 0, 5], {}), ("near", home + [0, 3, 0, 0, 0, 0], {}),
                                ("stuck", home + [0, 20, 0, 0, 0, 0], {"stuck": (1, 5)})):
        rob = fake.FakeSharedRobot(start, fake.RealClock(), **fault)
        payload = {"run_id": "t", "target": home.tolist(), "limits": RETURN_LIMITS, "already_tol": 6.0}
        with contextlib.redirect_stdout(quiet):
            rc = stream.container_return(payload, robot_factory=lambda rob=rob: rob)
        outs[label] = (rc, rob)
    rc_far, rob_far = outs["far"]
    steps_far = np.abs(np.diff(np.array(rob_far.sends)[:, :5], axis=0)).max() if len(rob_far.sends) > 1 else 0
    check("복귀: 끝 자세가 멀면 시작 자세로(틱 한도 안)", rc_far == 0 and np.abs(rob_far.q[:5] - home[:5]).max() < 0.5
          and steps_far <= RETURN_LIMITS["step_body"] * 1.001,
          f"보냄 {len(rob_far.sends)}틱({len(rob_far.sends) / 30:.1f}초), 남은 거리 {np.abs(rob_far.q[:5] - home[:5]).max():.2f}")
    check("복귀: 이미 홈 근처면 안 보냄 · 걸리면 중단", outs["near"][0] == 0 and not outs["near"][1].sends
          and outs["stuck"][0] == 2, f"근처 → 보냄 {len(outs['near'][1].sends)} · 걸림 → 종료 코드 {outs['stuck'][0]}")

    # 4) stop() → 즉시 멈춤: 실제와 같은 감독 루프(supervise) + 별도 프로세스의 가짜 실행(실시간 30Hz)
    #    rp.stop() 은 컨테이너에도 신호를 보내므로 여기서는 stop() 이 남기는 표시만 같은 방식으로 세운다.
    lim_rt = dict(lim)
    short, short_marks = tg[:600], [m for m in mk if m[0] < 600]
    state_dir, rp.STATE_DIR = rp.STATE_DIR, tmp / "state"  # 프로젝트 .state(stop 파일)를 건드리지 않게
    rp._stop_requested.clear()

    def child(fake_kw, run_id):
        payload = {"run_id": run_id, "robot_name": "fake", "targets": short, "marks": short_marks, "limits": lim_rt}
        env = {**os.environ, rp.MARKER: run_id}
        argv = [sys.executable, "-I", "-", "coord_pick", run_id]
        return argv, ex.build_script(payload, fake_kwargs=fake_kw), env

    for mode, want in (("kill", -9), ("home", 2)):
        run_id = f"selftest-{mode}"
        argv, script, env = child({}, run_id)
        log = ex.StageLog(tmp / f"stop_{mode}.jsonl")
        t_stop = []

        def press(mode=mode):
            time.sleep(1.5)
            t_stop.append(time.monotonic())
            rp._stop_kind[0] = f"{mode}:pending"   # rp.stop() 이 신호를 못 보냈을 때 남기는 것과 같다
            rp._stop_requested.set()

        probe = {}

        def look(run_id=run_id):
            time.sleep(0.8)  # 실행 중일 때 run_policy._PROBE 가 이 프로세스를 "run" 으로 세는지(stop() 대상인지)
            try:
                out = subprocess.run([sys.executable, "-I", "-c", rp._PROBE, rp.MARKER, "list"], capture_output=True,
                                     text=True, timeout=20)
                probe.update(json.loads(out.stdout))
            except Exception as e:  # noqa: BLE001
                probe["error"] = f"{type(e).__name__}: {e} {out.stderr[-200:] if 'out' in locals() else ''}"

        th = [threading.Thread(target=press), threading.Thread(target=look)]
        for t in th:
            t.start()
        try:
            sup = ex.supervise(argv, script, wall_s=60, log=log, env=env,
                               send_signal=lambda sig, proc: os.kill(proc.pid, getattr(__import__("signal"), sig)))
        finally:
            for t in th:
                t.join()
            rp._stop_requested.clear()
            log.close()
        lag = time.monotonic() - t_stop[0]
        mine = [p for p in probe.get("procs", []) if p["run_id"] == run_id]
        end = sup["end"] or {}
        ok = sup["stopped"] and sup["returncode"] == want and lag < 2.0 and mine and mine[0]["kind"] == "run"
        if mode == "home":
            ok = ok and end.get("reason") == "interrupted" and end.get("sent", 999) < len(short)
        check(f"실행기: stop({'home=True' if mode == 'home' else ''}) → 즉시 멈춤",
              ok, f"종료 코드 {sup['returncode']}, stop 뒤 {lag:.2f}초, 단계 {sup['steps_done']}, "
                  f"_PROBE 분류 {mine[0]['kind'] if mine else '없음 ' + str(probe)[:200]}"
                  + (f", {end.get('reason')}" if mode == "home" else ""))
    rp.STATE_DIR = state_dir
    rp._stop_kind[0] = "kill:sent"
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def sort_tests() -> None:
    """그릇 옮기기(r14·r15) — 10-22 실측 config.json, Studio·로봇·카메라 없이."""
    import copy
    import os

    from robot.coord_pick import detect_hsv
    from robot.coord_pick import executor as ex

    real = load_config()
    # 총괄이 실물 시험 중 config.json 을 직접 고친다(10-22 16:29~) → 시험은 보정값만 쓰고 손잡이는 이 값으로 고정
    real["calibrated"] = False
    real["heights_mm"].update(grasp=14, grasp_finger=6)
    real["safety"].update(floor_mm=3, forbid_below_mm=100, forbid_margin_mm=20)
    real["motion"].update(settle_s=0.3, cart_mm_s=50)
    real["wrist"] = {"enabled": False}  # 손목 확인은 wrist_tests 에서 가짜 세계로만(실제 손목 카메라를 열지 않게)
    real["grasp"] = {"mode": "ik"}      # 하이브리드(act)·공중 속도는 hybrid_tests 에서만 켠다(10-23)
    real["safety"]["air_deg_s"] = None
    s = Setup.load(real, use_studio=False)
    tmp = Path(tempfile.mkdtemp(prefix="coord_pick_sort_"))

    # 기울인 IK: 무작위 자세(바깥으로 기운 것) → tcp 위치·기울기·집게 방향으로 되풀기
    rng = np.random.default_rng(3)
    ok = n = 0
    worst = 0.0
    while n < 80:
        q = np.radians([rng.uniform(-40, 40), rng.uniform(0, 75), rng.uniform(-90, 20), rng.uniform(30, 90),
                        rng.uniform(-60, 90)])
        T = s.kin.fk(q)
        p = T[:3, 3] + T[:3, :3] @ s.tcp
        if T[2, 2] > -0.3 or T[:2, 2] @ (p[:2] - s.kin.pan_origin[:2]) < 0:
            continue
        n += 1
        tilt, yaw = math.acos(-T[2, 2]), math.atan2(T[1, 0], T[0, 0])
        az = math.degrees(math.atan2(p[1], p[0]))
        for q0 in (np.radians([az, 0, 30, 65, 0]), np.radians([az, 55, -60, 55, 0])):
            qs, info = s.kin.ik(p, yaw=yaw, q0=q0, tilt=tilt, tcp=s.tcp)
            if info["ok"]:
                ok += 1
                worst = max(worst, info["pos_err_mm"])
                break
    check("기울인 IK: 무작위 자세 80개 되풀기", ok >= 76 and worst < 2.0, f"성공 {ok}/80, 최대 위치 오차 {worst:.2f}mm")
    check("집게 틈(메시 표)", abs(jaw_gap_mm(63) - 66.2) < 0.1 and abs(jaw_gap_mm(50) - 15.8) < 0.1
          and abs(jaw_gap_mm(51.5) - 22.7) < 0.5, f"50 → {jaw_gap_mm(50):.1f} · 63 → {jaw_gap_mm(63):.1f}mm")

    def run(color, **kw):
        return ex.sort_blocks(color, config=copy.deepcopy(kw.pop("cfg", real)), log_dir=tmp, **kw)

    r = run("red,blue,defect", demo=True)
    p = plan_sort(s, demo_sort_scene(s), classes=("red_ok", "blue_ok", "red_defect", "blue_defect"))
    tcp_ok = []
    for o, st in zip(p["order"], [st for st in p["steps"] if st.what.endswith(" 내려감")]):
        tcp_ok.append(np.linalg.norm(s.tcp_pos(s.n_to_q(st.n)) - s.w_to_r3(o["block"]["w"], o["grasp_z"])) < 1.0)
    check("그릇 옮기기: 시연 배치 4개 dry-run", r["ok"] and r["blocks"] == 4 and not r["violations"] and not r["failures"]
          and r["sim"]["ok"] and all(tcp_ok) and len(tcp_ok) == 4 and p["checks"]["집게 끝 최저 높이 mm"] >= 5,
          f"{r['time_s']}초(15°/s), 위반 {len(r['violations'])}, 집는 tcp 가 블록 위치 {sum(tcp_ok)}/4, "
          f"집게 끝 최저 {p['checks']['집게 끝 최저 높이 mm']}mm")
    holds = [st for st in p["steps"] if st.what.endswith("전 멈춤")]
    check("그릇 옮기기: 집기·놓기 전 멈춤(motion.settle_s)", len(holds) == 8 and all(abs(st.dur_s - 0.3) < 1e-9 for st in holds)
          and all(np.allclose(st.n, p["steps"][p["steps"].index(st) - 1].n) for st in holds),
          f"{len(holds)}번 × {real['motion']['settle_s']}초, 그 자리 그대로")
    hand = [st.what for st in p["steps"] if "hand_spot 위라 올라감" in st.what]
    check("그릇 옮기기: 손 자리(금지 구역) 위는 높이 올려 건넘", bool(hand) and not r["violations"], hand[0] if hand else "없음")

    # 목적지 그릇만 허용: 같은 궤적을 허용 없이 보면 그릇 위반이 나와야 한다
    traj, marks = trajectory(p["steps"], real["motion"]["rate_hz"], with_marks=True)
    sf = ex.safety(real)
    polys, _ = ex.forbidden_polygons(s, sf)
    no_allow = {v["kind"] for v in ex.check_workspace(s, traj, sf, polys)}
    floor = p["heights"]["release"] - real["bowls"]["release_margin_mm"]
    with_allow = ex.check_workspace(s, traj, sf, polys, {o["bowl"]: floor for o in p["order"]})
    check("그릇 옮기기: 금지 구역에서 목적지 그릇만 뺌", {"forbidden:bowl_red", "forbidden:bowl_blue",
                                                 "forbidden:bowl_defect"} <= no_allow and not with_allow,
          f"허용 없이 {sorted(no_allow)} → 허용하면 {len(with_allow)}")

    # 검출 결과: 투입 구역(zone input) 블록만, 로봇에 가까운 것부터
    blocks = demo_sort_scene(s)
    det = {"objects": [dict(b) for b in blocks] + [{"cls": "red_ok", "zone": "bowl_red", "px": [379, 226], "conf": 0.9},
                                                   {"cls": "hand", "zone": "hand_spot", "px": [200, 300], "conf": 0.9}]}
    r1 = run("red", detection=copy.deepcopy(det), max_blocks=1)
    near = min((b for b in blocks if b["cls"] == "red_ok"), key=lambda b: float(np.linalg.norm(s.frames.w2r(s.px_to_w(b["px"])[0]))))
    check("그릇 옮기기: 구역 input 만·가까운 것부터(max_blocks=1)", r1["ok"] and r1["blocks"] == 1
          and np.allclose(r1["plan"][0]["from_mm"], s.px_to_w(near["px"])[0], atol=0.2)
          and any("구역 밖" in w for w in r1["warnings"]), f"{r1['plan'][0]['from_mm']} · {r1['warnings'][-1][:40]}")

    # 손 자리 가장자리 블록(10-22 16:38 실물 거부 자리): 벌린 집게가 손 자리 위로 나가지 않는 방향을 고른다
    edge = {"objects": [{"cls": "red_ok", "zone": "input", "px": s.w_to_px(np.array([[286.8, 26.5]]))[0].round(1).tolist(),
                         "conf": 0.9}]}
    re = run("red", detection=edge)
    check("그릇 옮기기: 손 자리 가장자리 블록 — 집게가 금지 구역 위로 안 나감", re["ok"] and not re["violations"],
          f"{re['plan'][0]['from_mm'] if re.get('plan') else '-'} · 위반 {len(re['violations'])}")

    r2 = run("red", demo=True, lift_mm=40)
    lift_low = plan_sort(s, demo_sort_scene(s), classes=("red_ok",), lift_mm=40)["checks"]["집게 끝 최저 높이 mm"]
    check("그릇 옮기기: 공중 리허설 +40mm → 집게 끝이 블록(30mm) 위", r2["ok"] and lift_low > 35,
          f"집게 끝 최저 {lift_low}mm, {r2['time_s']}초")

    # 문: calibrated=false 면 공중 리허설만 실제 송신(r15)
    saved = os.environ.pop("BALL_ROBOT", None)
    called = []
    orig = ex._run_real
    ex._run_real = lambda *a, **k: called.append(k.get("name")) or {"reason": "done", "run_id": "x"}
    try:
        os.environ["BALL_ROBOT"] = "1"
        g1 = run("red", demo=True, dry_run=False)
        g2 = run("red", demo=True, dry_run=False, lift_mm=40)
        c = copy.deepcopy(real)
        c["calibrated"] = True
        g3 = run("red", demo=True, dry_run=False, cfg=c)
        os.environ.pop("BALL_ROBOT")
        g4 = run("red", demo=True, dry_run=False, lift_mm=40)
    finally:
        ex._run_real = orig
        os.environ.pop("BALL_ROBOT", None)
        if saved is not None:
            os.environ["BALL_ROBOT"] = saved
    check("그릇 옮기기: 보정 확정 전엔 공중 리허설만 송신",
          g1["reason"] == "refused" and "calibrated" in g1["error"] and g2["mode"] == "real" and g3["mode"] == "real"
          and g4["reason"] == "refused" and "BALL_ROBOT" in g4["error"] and len(called) == 2,
          f"보정 전 일반 → {g1['reason']} · 보정 전 리허설 → {g2['mode']} · 보정 후 → {g3['mode']} · BALL_ROBOT 없음 → {g4['reason']}")

    # 색 검출 어댑터: 같은 색 가까운 조각은 합치고(중심 = 조각 중심 평균), 다른 색·먼 것은 따로
    boxes = [{"cls": "red_ok", "xyxy": [100, 100, 112, 124]}, {"cls": "red_ok", "xyxy": [114, 100, 126, 124]},
             {"cls": "blue_ok", "xyxy": [128, 100, 150, 124]}, {"cls": "red_defect", "xyxy": [200, 100, 216, 124]},
             {"cls": "red_ok", "xyxy": [219, 102, 232, 120]}, {"cls": "hand", "xyxy": [0, 0, 50, 50]}]
    m = detect_hsv.merge_boxes(boxes)
    reds = sorted((o for o in m if o["cls"].startswith("red")), key=lambda o: o["px"][0])
    check("색 검출 어댑터: 쪼갠 조각 합치기", len(m) == 3 and reds[0]["parts"] == 2 and reds[0]["px"] == [113.0, 112.0]
          and reds[1]["cls"] == "red_defect" and reds[1]["parts"] == 2,
          " · ".join(f"{o['cls']}{o['px']}×{o['parts']}" for o in m))

    # 짚기 보정: 고정 집게 끝으로 짚은 4점(가짜) → base 를 되찾나
    mm4 = np.array(real["table"]["markers_mm"], float)
    contact = np.array([3.0, 0.0, 6.3]) / 1000
    touches = []
    for i in range(4):
        pr = s.w_to_r3(mm4[i], 0.0)
        az = math.degrees(math.atan2(pr[1], pr[0]))
        for tilt in (25, 35, 45, 55, 15, 5):  # 10-22 실제 짚기도 4~52° 기울였다
            q, info = s.kin.ik(pr / 1000, q0=np.radians([az, 55, -60, 55, 0]), tilt=math.radians(tilt), tcp=contact,
                               tol_mm=0.1, iters=120)
            if info["ok"]:
                break
        touches.append({"marker": i, "n": s.q_to_n(q, 50.0).round(4).tolist(), "_ok": info["ok"]})
    f = tmp / "touches.json"
    f.write_text(json.dumps([{k: v for k, v in t.items() if k != "_ok"} for t in touches]))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli(["--offline", "calib-base", str(f)])
    b = real["table"]["base"]
    got = json.loads(buf.getvalue().strip().splitlines()[-1].split("넣을 값:", 1)[1])
    check("짚기 보정: 고정 집게 끝 4점(기울여 짚음) → 로봇 위치", rc == 0 and all(t["_ok"] for t in touches)
          and abs(got["x"] - b["x"]) < 1 and abs(got["y"] - b["y"]) < 1 and abs(got["yaw_deg"] - b["yaw_deg"]) < 0.3,
          f"{got} (설정 {b['x']}, {b['y']}, {b['yaw_deg']}°)")
    home_tests(real, tmp)
    wrist_tests(real, tmp)
    hybrid_tests(real, tmp)
    field_tests(real, tmp)
    unsort_tests(real, tmp)
    act_start_tests(tmp)
    servo_tests(real, tmp)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def wrist_tests(real: dict, tmp: Path) -> None:
    """손목 카메라 확인·보정(r18·r19) — 가짜 세계(wrist_fake.FakeWorld: 송신·손목 사진을 흉내)."""
    import copy
    import os

    import cv2

    from robot.coord_pick import executor as ex
    from robot.coord_pick import wrist
    from robot.coord_pick.wrist_fake import FakeWorld

    ref = wrist.load_ref()
    rr = (ref or {}).get("close", {}).get("by_dataset", {}).get("pick_red", {})
    check("손목: 녹화 기준·화면 모형(wrist_ref.json)", ref is not None and 300 < rr.get("u_med", 0) < 380
          and ref["screen"]["cross_val_px"]["med"] < 45 and "pick_red" in ref.get("after", {}).get("by_dataset", {}),
          f"분홍 닫기 직전 중심 ({rr.get('u_med')}, {rr.get('v_med')}) 면적 {rr.get('area_med')} · 화면 모형 교차 검증 "
          f"{(ref or {}).get('screen', {}).get('cross_val_px', {}).get('med')}px")
    img = np.full((480, 640, 3), 225, np.uint8)
    cv2.rectangle(img, (40, 40), (200, 200), (200, 140, 230), -1)      # 큰 이웃
    cv2.rectangle(img, (300, 120), (380, 200), (200, 140, 230), -1)    # 목표(기준 가까이)
    b_big, b_near = wrist.blob(img, "red"), wrist.blob(img, "red", near=(340, 155))
    check("손목: 기준에 가까운 덩어리 고르기", b_big["px"] == [120.0, 120.0] and b_near["px"] == [340.0, 160.0],
          f"큰 것 {b_big['px']} · 가까운 것 {b_near['px']}")

    cfg = copy.deepcopy(real)
    cfg["calibrated"] = True
    s = Setup.load(cfg, use_studio=False)
    blocks = demo_sort_scene(s)
    near = min((b for b in blocks if b["cls"] == "red_ok"), key=lambda b: float(np.linalg.norm(s.frames.w2r(s.px_to_w(b["px"])[0]))))
    det_w = s.px_to_w(near["px"])[0]

    def go(offset, **kw):
        true_w = det_w + np.array(offset, float)
        fw = FakeWorld(s, true_w, **kw)
        fw.install(ex)
        saved = os.environ.get("BALL_ROBOT")
        os.environ["BALL_ROBOT"] = "1"
        try:
            r = ex.sort_blocks("red", dry_run=False, detection={"objects": copy.deepcopy(blocks)}, max_blocks=1,
                               config=copy.deepcopy(cfg), wrist_check=True, log_dir=tmp,
                               redetect=lambda: {"objects": [dict(near, px=s.w_to_px(true_w)[0].tolist())]})
        finally:
            fw.uninstall()
            if saved is None:
                os.environ.pop("BALL_ROBOT", None)
            else:
                os.environ["BALL_ROBOT"] = saved
        return r, fw

    r, fw = go((20, -12))
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])
    check("손목 확인: 20mm 어긋난 블록 → 맞춤 → 닫기 전 기준 안 → 쥠 → 놓기", r["ok"] and fw.grasps and fw.grasps[-1]["held"]
          and fw.grasps[-1]["dist_mm"] < 10 and len(at) == 1,
          f"맞춤 {[x.get('err_px') for x in at[0].get('align', [])]} · 닫을 때 블록과 {fw.grasps[-1]['dist_mm'] if fw.grasps else '-'}mm · "
          f"구간 {len(fw.sends)} · 손목 사진 {fw.shots}")
    # 10-23 속도: 멈춤으로 끝나는 구간만 송신기 끝 대기를 safety.hold_tail_s 로(기본 없음 = 0.5 그대로)
    from robot.coord_pick.plan import Step
    st_h = [Step("x 닫기 직전 위로", np.zeros(6), 1.0), Step("x 닫기 직전 위 멈춤", np.zeros(6), 1.0)]
    st_m = [Step("x 내려감", np.zeros(6), 1.0)]
    check("끝 대기 줄이기(safety.hold_tail_s): 멈춤으로 끝난 구간만 · 기본은 그대로",
          ex._tail_settle({"hold_tail_s": 0.1}, st_h) == {"settle_s": 0.1} and ex._tail_settle({"hold_tail_s": 0.1}, st_m) == {}
          and ex._tail_settle({}, st_h) == {} and "hold_tail_s" not in ex.SAFETY_DEFAULTS)
    # 10-23 속도: 방향 사진을 hover 맞춤 첫 장으로(wrist.reuse_dir_photo) — 같은 결과에 손목 사진 한 장 덜
    nshot = lambda f: f.shots if isinstance(f.shots, int) else len(f.shots)
    shots0, at0 = nshot(fw), at
    cfg.setdefault("wrist", {})["reuse_dir_photo"] = True
    try:
        r2, fw2 = go((20, -12))
    finally:
        cfg["wrist"].pop("reuse_dir_photo", None)
    at2 = (r2.get("wrist") or [{}])[0].get("attempts", [{}])
    check("방향 사진 다시 쓰기(wrist.reuse_dir_photo): 20mm 어긋난 블록도 쥠 · 손목 사진 한 장 덜",
          r2["ok"] and fw2.grasps and fw2.grasps[-1]["held"] and fw2.grasps[-1]["dist_mm"] < 10 and len(at2) == 1
          and nshot(fw2) == shots0 - 1 and not ex.WRIST_DEFAULTS["reuse_dir_photo"],
          f"사진 {shots0} → {nshot(fw2)} · 닫을 때 블록과 {fw2.grasps[-1]['dist_mm'] if fw2.grasps else '-'}mm · "
          f"맞춤 {[x.get('err_px') for x in at2[0].get('align', [])]}")
    # 17:29 실물: 화면 감도가 모형의 1.4~1.9배 → 감쇠·실측 배율로 수렴, hover~들어 올림 동안 롤 고정(내려가며 안 돈다)
    r, fw = go((-15, 22), px_gain=1.8)
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])
    steps = at[0].get("align", []) if at else []
    errs = [float(np.linalg.norm(x["err_px"])) for x in steps]
    seg = [x for x in fw.rolls if "·1]" in x[0] and ("맞춤 이동" in x[0] or "내려감" in x[0] or "들어 올림" in x[0])]
    roll_span = max(x[2] for x in seg) - min(x[1] for x in seg) if seg else 99
    check("손목 확인: 화면 감도 1.8배 → 진동 없이 수렴 · hover~들어 올림 롤 고정", r["ok"] and len(steps) <= 5
          and errs[-1] < errs[0] and roll_span < 8 and fw.grasps and fw.grasps[-1]["dist_mm"] < 10,
          f"오차 {[round(e_) for e_ in errs]}px · 배율 {[x.get('gain_k') for x in steps]} · hover 뒤 롤 폭 {roll_span:.1f}° · "
          f"닫을 때 {fw.grasps[-1]['dist_mm'] if fw.grasps else '-'}mm")
    # r23: hover 화면이 내려간 뒤 위치를 못 맞힘(17:32 실물) → 닫기 직전 높이 + 10mm 에서 녹화 기준에 직접 맞춤
    r, fw = go((20, -12), hover_bias_px=(35.0, -25.0), px_gain=1.5)
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])
    al = at[0].get("align", []) if at else []
    near_ok = [x for x in al if x.get("level") == "near" and x.get("ok")]
    logged = []
    for f_ in sorted(tmp.glob("*sort_red*_real.jsonl"))[-1:]:
        logged = [json.loads(l_) for l_ in f_.read_text(encoding="utf-8").splitlines()]
    has_gain = any(x.get("stage") == "align" and "gain_k" in x for x in logged)
    check("손목 확인: hover 어긋남(35·−25px) → 닫기 직전 높이에서 다시 맞춰 기준 안 → 쥠, gain_k 기록", r["ok"] and near_ok
          and len(at) == 1 and fw.grasps and fw.grasps[-1]["dist_mm"] < 8 and has_gain,
          f"hover {[x['err_px'] for x in al if x.get('level') == 'hover']} → near {[x['err_px'] for x in al if x.get('level') == 'near']}"
          f" · 배율 {[x['gain_k'] for x in al]} · 닫을 때 {fw.grasps[-1]['dist_mm'] if fw.grasps else '-'}mm · 기록 gain_k {has_gain}")
    # 17:54 실물: 먼 줄 블록(검출 142.7,115.4 → 실제 143,129)을 맞춰 집은 뒤 맞추기 전 계획의 나르는 기울기가 안 풀려 거부됐다
    #  → 든 뒤로는 지금 자세에서 다시 풀어 그릇에 놓아야 한다
    one = [{"cls": "red_ok", "zone": "input", "px": s.w_to_px(np.array([142.7, 115.4]))[0].round(1).tolist(), "conf": 0.9}]
    true_w3 = np.array([143.0, 129.0])
    fw = FakeWorld(s, true_w3, px_gain=1.5)
    fw.install(ex)
    os.environ["BALL_ROBOT"] = "1"
    try:
        r = ex.sort_blocks("red", dry_run=False, detection={"objects": copy.deepcopy(one)}, max_blocks=1,
                           config=copy.deepcopy(cfg), wrist_check=True, log_dir=tmp,
                           redetect=lambda: {"objects": [dict(one[0], px=s.w_to_px(true_w3)[0].tolist())]})
    finally:
        fw.uninstall()
        os.environ.pop("BALL_ROBOT", None)
    w0 = (r.get("wrist") or [{}])[0]
    check("손목 확인: 먼 줄 블록(17:54) — 집은 뒤 놓기를 다시 풀어 그릇에 놓음", r["ok"] and fw.grasps and fw.grasps[-1]["held"]
          and any("그릇에 놓기" in n for n in fw.sends) and fw.dropped,
          f"맞춘 자리 {w0.get('final_mm')} · 놓는 점 {w0.get('drop_mm')} 나르는 높이 {w0.get('carry_mm')} · 구간 {len(fw.sends)} · "
          f"{str(r.get('error'))[:60] if not r['ok'] else ''}")
    # 블록 방향(r21): 긴 축이 작업대 x(0°) — 작업대 축 집게 중 0° 는 긴 축과 나란해 못 쥔다
    r, fw = go((10, -6), true_yaw_w=0.0)
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])
    yr = at[0].get("yaw") or {}
    seg = [x for x in fw.rolls if "·1]" in x[0] and ("맞춤 이동" in x[0] or "내려감" in x[0] or "들어 올림" in x[0])]
    span = max(x[2] for x in seg) - min(x[1] for x in seg) if seg else 99
    check("블록 방향: 위 카메라 방향 없이 → hover 손목 사진으로 집게를 돌려 짧은 변을 끼움", r["ok"] and fw.grasps
          and abs(fw.grasps[-1]["rho_deg"]) >= 60 and span < 8,
          f"hover 각 {yr.get('rho_deg')}° → 돌림 {yr.get('rotated')}° · 닫을 때 긴 축−집게 {fw.grasps[-1]['rho_deg'] if fw.grasps else '-'}°"
          f" · 돌린 뒤 롤 폭 {span:.1f}°")
    det_yaw = [dict(b_, yaw_w=0.15, aspect=1.6) if b_ is near else b_ for b_ in blocks]
    true_w2 = det_w + np.array([10.0, -6.0])
    fw = FakeWorld(s, true_w2, true_yaw_w=0.0)
    fw.install(ex)
    os.environ["BALL_ROBOT"] = "1"
    try:
        r = ex.sort_blocks("red", dry_run=False, detection={"objects": copy.deepcopy(det_yaw)}, max_blocks=1,
                           config=copy.deepcopy(cfg), wrist_check=True, log_dir=tmp,
                           redetect=lambda: {"objects": [dict(near, px=s.w_to_px(true_w2)[0].tolist())]})
    finally:
        fw.uninstall()
        os.environ.pop("BALL_ROBOT", None)
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])
    yr = at[0].get("yaw") or {}
    check("블록 방향: 위 카메라 긴 축(9° 오차) → 처음부터 수직 집게, hover 확인만", r["ok"] and fw.grasps
          and abs(fw.grasps[-1]["rho_deg"]) >= 60 and abs(yr.get("rotated", 0)) < 1,
          f"hover 각 {yr.get('rho_deg')}° 돌림 {yr.get('rotated')}° · 닫을 때 {fw.grasps[-1]['rho_deg'] if fw.grasps else '-'}°")
    from robot.coord_pick import detect_hsv
    img = np.full((480, 640, 3), 225, np.uint8)
    c0 = s.w_to_px(det_w)[0]
    c1 = s.w_to_px(det_w + [20 * math.cos(0.5), 20 * math.sin(0.5)])[0]
    ang = math.degrees(math.atan2(c1[1] - c0[1], c1[0] - c0[0]))
    cv2.fillConvexPoly(img, cv2.boxPoints(((float(c0[0]), float(c0[1])), (30.0, 18.0), ang)).astype(np.int32), (200, 140, 230))
    mo = detect_hsv.measure_yaw(img, [{"cls": "red_ok", "px": c0.tolist()}], s)[0]
    check("블록 방향: 위 카메라 사진 → 작업대 긴 축", abs(((mo.get("yaw_w", 9) - 0.5 + math.pi / 2) % math.pi) - math.pi / 2) < math.radians(8),
          f"그린 긴 축 28.6° → 잰 {math.degrees(mo.get('yaw_w', float('nan'))):.1f}° (가로세로비 {mo.get('aspect')})")
    r, fw = go((20, -12), fail_first=True)
    at = (r.get("wrist") or [{}])[0].get("attempts", [])
    check("손목 확인: 못 쥠(집은 뒤 손목 사진) → 홈 → 위 카메라로 다시 찾아 재시도 → 성공", r["ok"] and len(at) == 2
          and at[0]["held"] is False and at[1]["held"] is True and "홈으로(다시)" in " ".join(fw.sends),
          f"시도 {len(at)}번: {[a_.get('grip_measured') for a_ in at]} · 구간 {len(fw.sends)}")
    r, fw = go((70, 60))  # 손목 화면 밖으로 크게 벗어남 → 맞춤이 한 번에 25mm 까지만 → 기준 밖이면 닫지 않는다
    closed = len(fw.grasps)
    check("손목 확인: 기준 밖이면 닫지 않음(크게 어긋남)", (not r["ok"] and closed == 0) or (r["ok"] and fw.grasps[-1]["held"]),
          f"ok={r['ok']} 닫음 {closed}번 · {str(r.get('error'))[:60]}")
    c2 = copy.deepcopy(cfg)
    c2["heights_mm"].update(grasp=5, grasp_finger=6)
    s2 = Setup.load(c2, use_studio=False)
    col = s2.column(det_w, 0.0, (50, 5), s2.n_to_q(np.array(c2["home_n"])), tilt_max=s2.tilt["grasp"], yaws=(0.0,),
                    finger_target=6, close_n=c2["gripper"]["close"])
    check("집기 높이: grasp 가 낮으면 grasp_finger 까지 올려 푼다", col is not None and col["zs"][-1] > 5
          and min(i["finger_low_mm"] for i in col["infos"]) >= 5.5,
          f"grasp 5 → tcp {col['zs'][-1] if col else '-'}mm(집게 끝 ≥ 6)")


def hybrid_tests(real: dict, tmp: Path) -> None:
    """10-23 하이브리드: 공중 구간 빠르게(retime) · 집기 끝 판정(grasp_stop) · ACT 집기 갈래(가짜 정책·가짜 유지 구독자)."""
    import copy
    import os

    from robot import grasp_stop
    from robot import run_policy as rp
    from robot.coord_pick import executor as ex
    from robot.coord_pick.plan import plan_sort, trajectory
    from robot.coord_pick.wrist_fake import FakeWorld

    # 1) 공중 구간: 끄면 예전과 같은 틱, 켜면 공중만 빠르게(가감속 한도 안), 놓기·집기 근처는 그대로
    cfg = copy.deepcopy(real)
    s = Setup.load(cfg, use_studio=False)
    sf = ex.safety(cfg)
    p = plan_sort(s, demo_sort_scene(s), classes=("red_ok",), max_blocks=1)
    traj, marks, fast = trajectory(p["steps"], 30, with_marks=True, with_fast=True)
    lim = ex.limits(s, sf, sf["exec_deg_s"])
    t_off, _ = ex.retime(traj, marks, lim["step_body"], lim["step_grip"])
    t_off2, _ = ex.retime(traj, marks, lim["step_body"], lim["step_grip"], fast=fast, air=None)
    sf_on = {**sf, "air_deg_s": 30, "air_acc_deg_s2": 150}
    air = ex.air_limits(s, sf_on, sf["exec_deg_s"])
    t_on, m_on = ex.retime(traj, marks, lim["step_body"], lim["step_grip"], fast=fast, air=air)
    dpu = np.array([s.jm.deg_per_unit(k) for k in ARM])
    v = np.abs(np.diff(t_on[:, :5], axis=0)) * dpu * 30
    vmax = v.max(axis=1)
    quick = (vmax[1:] > sf["exec_deg_s"] + 0.5) & (vmax[:-1] > sf["exec_deg_s"] + 0.5)
    acc = np.abs(np.diff(vmax)) * 30
    near_lbl = [i for i, lb in m_on if "놓음" in lb or "집음" in lb]
    slow_ok = all(vmax[max(0, i - 3):i].max(initial=0) <= sf["exec_deg_s"] + 0.01 for i in near_lbl)
    check("공중 구간: 끄면 예전 틱 그대로 · 켜면 공중만 30°/s(가속 150°/s² 안) · 집음·놓음 직전은 15°/s",
          np.array_equal(t_off, t_off2) and len(t_on) < 0.75 * len(t_off) and vmax.max() <= 30.01
          and (acc[quick].max() if quick.any() else 0) <= 151 and slow_ok
          and all(a[0] <= b[0] for a, b in zip(m_on, m_on[1:])),
          f"틱 {len(t_off)} → {len(t_on)} · 최대 {vmax.max():.1f}°/s · 공중 가속 최대 "
          f"{(acc[quick].max() if quick.any() else 0):.0f}°/s²")
    c_on = copy.deepcopy(cfg)
    c_on["safety"].update(air_deg_s=30, air_acc_deg_s2=150)
    c_on["motion"].update(air_mm_s=120, air_deg_s=60)
    d0 = ex.sort_blocks("red", demo=True, max_blocks=1, config=copy.deepcopy(cfg), log_dir=tmp)
    d1 = ex.sort_blocks("red", demo=True, max_blocks=1, config=c_on, log_dir=tmp)
    check("공중 구간: 블록 1개 dry-run 시간 줄어듦(공간 위반 0·모의 실행 통과)", d0["ok"] and d1["ok"]
          and d1["time_s"] < 0.75 * d0["time_s"] and not d1["violations"],
          f"{d0['time_s']}초 → {d1['time_s']}초")

    # 2) 집기 끝 판정: 닫고 → 들고 → 멈춤에서만
    def ticks(close=True, lift=True, hold=1.0):
        out, t = [], 0.0
        def add(z, g, dur):
            nonlocal t
            for _ in range(int(dur * 30)):
                out.append((t, [0, z, 0, 0, 0, g]))
                t += 1 / 30
        add(90, 63, 0.8)
        for k in range(45):
            out.append((t, [0, 90 - 85 * k / 44, 0, 0, 0, 63])); t += 1 / 30
        for k in range(10):
            out.append((t, [0, 5, 0, 0, 0, 63 - (13 if close else 0) * k / 9])); t += 1 / 30
        gz = 50 if close else 63
        for k in range(27):
            out.append((t, [0, 5 + (40 if lift else 0) * k / 26, 0, 0, 0, gz])); t += 1 / 30
        add(45 if lift else 5, gz, hold)
        return out

    def run_det(seq):
        d = grasp_stop.GraspStopDetector(lambda q: q[1])
        hit = next((t for t, q in seq if d.feed(t, q)), None)
        return hit, d
    hit, d = run_det(ticks())
    miss1, _ = run_det(ticks(close=False))
    miss2, _ = run_det(ticks(lift=False))
    check("집기 끝 판정: 닫고·들고·0.5초 멈춤에서 끝, 안 닫거나 안 들면 끝 아님", hit is not None and miss1 is None
          and miss2 is None and d.trigger.lift_mm >= 20,
          f"끝 {hit:.2f}s(들림 {d.trigger.lift_mm if d.trigger else '-'}mm) · 안 닫음 {miss1} · 안 듦 {miss2}")
    es = rp._early_stop_settings({"early_stop": {"enabled": True, "mode": "grasp", "lift_mm": 25}})
    check("run_policy: early_stop mode grasp 설정", es["mode"] == "grasp" and es["lift_mm"] == 25 and "return_far" not in es,
          json.dumps(es))
    rec = Path(__file__).resolve().parent.parent / "logs" / "r14" / "rec.json"
    gr = Path(__file__).resolve().parent.parent / "logs" / "r14" / "grasps.json"
    if rec.exists() and gr.exists():   # 녹화 사람 집기(블록 위 → 쥐고 → 들기) 뒤에 1초 정지를 붙이면 끝 판정이 나야 한다
        R = json.loads(rec.read_text())
        G = json.loads(gr.read_text())
        ids = {"pick_red": "dd6126e6-4ac0-4d56-9400-527135597fb8", "pick_green": "07f0afd2-518c-4178-a6fa-1af90b01aad0"}
        zf = grasp_stop.tcp_z_fn()
        ok_n = tot = 0
        for g_ in [x for x in G if x["ds"] in ids][::5]:
            S = R[ids[g_["ds"]]]["episodes"][str(g_["ep"])]["s"]
            gi = g_["gi"]
            lift_i = next((i for i in range(gi, len(S)) if zf(S[i]) >= zf(S[gi]) + 40), None)
            if lift_i is None:
                continue
            seq = S[max(0, gi - 45):lift_i + 1] + [S[lift_i]] * 30
            det = grasp_stop.GraspStopDetector(zf)
            tot += 1
            ok_n += next((True for k, q_ in enumerate(seq) if det.feed(k / 30, q_)), False)
        check("집기 끝 판정: 녹화 사람 집기 + 1초 정지(실제 기구학)", tot > 0 and ok_n == tot, f"{ok_n}/{tot}")

    # 3) ACT 집기 갈래(가짜 세계): 정책이 쥐면 IK 내려가 쥐기 없이 나르기, 못 쥐면 그 시도에서 IK 집기, 유지 구독자 못 붙으면 정책 안 돌림
    cfg = copy.deepcopy(real)
    cfg["calibrated"] = True
    cfg["grasp"] = {"mode": "act", "start_z_mm": 90, "timeout_s": 8}
    s = Setup.load(cfg, use_studio=False)
    blocks = demo_sort_scene(s)
    near = min((b for b in blocks if b["cls"] == "red_ok"), key=lambda b: float(np.linalg.norm(s.frames.w2r(s.px_to_w(b["px"])[0]))))
    true_w = s.px_to_w(near["px"])[0] + np.array([8.0, -5.0])

    class FakeKeeper:
        made = []

        def __init__(self, ok=True):
            self.ok, self.started, self.stopped = ok, 0, 0
            FakeKeeper.made.append(self)

        def start(self):
            self.started += 1
            return self.ok

        def stop(self):
            self.stopped += 1

    def go(policy_ok=True, keeper_ok=True):
        fw = FakeWorld(s, true_w)
        calls = []

        def policy(name, timeout_s=None, return_home=None):
            q0 = fw.q
            z0 = float(s.tcp_pos(q0)[2])
            calls.append({"name": name, "timeout_s": timeout_s, "return_home": return_home, "start_z": round(z0, 1)})
            if not policy_ok:   # 블록 옆에서 닫고 들지 못함(시간 끝)
                fw.grasps.append({"dist_mm": 25.0, "held": False, "rho_deg": None})
                qe, _ = s.solve(s.w_to_r3(true_w + [20, 0], 12), 0.0, q0, tilt=None, tilt_max=s.tilt["grasp"])
                fw.q = qe
                return {"reason": "duration_elapsed", "end_q": s.q_to_n(qe, 50.4).tolist(), "early_stop": {"closed": True,
                        "lifted": False, "z_low_mm": 12.0}, "log_path": "x"}
            qe, _ = s.solve(s.w_to_r3(true_w, 45), 0.0, q0, tilt=None, tilt_max=s.tilt["grasp"])
            fw.q = qe
            fw.grasps.append({"dist_mm": 0.0, "held": True, "rho_deg": None})
            fw.dropped = False
            return {"reason": "grasp_done", "end_q": s.q_to_n(qe, 50.4).tolist(),
                    "early_stop": {"closed": True, "lifted": True, "z_low_mm": 4.0}, "log_path": "x"}

        FakeKeeper.made.clear()
        fw.install(ex)
        os.environ["BALL_ROBOT"] = "1"
        try:
            r = ex.sort_blocks("red", dry_run=False, detection={"objects": copy.deepcopy(blocks)}, max_blocks=1,
                               config=copy.deepcopy(cfg), wrist_check=True, log_dir=tmp, act_runner=policy,
                               keeper_factory=lambda: FakeKeeper(keeper_ok),
                               redetect=lambda: {"objects": [dict(near, px=s.w_to_px(true_w)[0].tolist())]})
        finally:
            fw.uninstall()
            os.environ.pop("BALL_ROBOT", None)
        return r, fw, calls
    r, fw, calls = go()
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])
    kp = FakeKeeper.made
    check("하이브리드: 블록 위 90mm 까지 IK → 집기 정책(복귀 없음) → 그 자리부터 IK 로 그릇에 놓기",
          r["ok"] and r.get("grasp_mode") == "act" and len(calls) == 1 and calls[0]["return_home"] is False
          and abs(calls[0]["start_z"] - 90) < 3 and not any("내려감" in n for n in fw.sends)
          and any("그릇에 놓기" in n for n in fw.sends) and fw.dropped and kp and kp[0].started == 1 and kp[0].stopped >= 1,
          f"시작 높이 {calls[0]['start_z'] if calls else '-'}mm · 구간 {fw.sends} · 판정 {at[0].get('act', {}).get('reason')}")
    r, fw, calls = go(policy_ok=False)
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])
    check("하이브리드: 정책이 못 쥠 → 벌리고 올라가 같은 시도에서 IK 집기 → 놓기", r["ok"] and len(calls) == 1
          and any("정책 못 쥠" in n for n in fw.sends) and any("집고 들어 올림" in n for n in fw.sends) and len(at) == 1
          and fw.grasps[-1]["held"],
          f"구간 {len(fw.sends)} · {at[0].get('why') if at else ''} · {str(r.get('error') or '')[:160]}")
    r, fw, calls = go(keeper_ok=False)
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])
    check("하이브리드: 유지 구독자 못 붙임 → 정책 안 돌리고 IK 집기", r["ok"] and not calls
          and "유지 구독자" in str(at[0].get("act", {}).get("skipped")),
          f"정책 호출 {len(calls)} · {at[0].get('act')}")


def field_tests(real: dict, tmp: Path) -> None:
    """10-23 실물에서 난 일(가짜 로봇·가짜 손목): 09:46 블록 위 90mm 기울기 거부 · act 시작 자세 · 09:50 앞 그릇에서 떠나는
    구간 거부 · 거부 뒤 자동 홈 복귀 · 왼쪽/오른쪽/가운데 고르기."""
    import copy
    import os
    import re

    from robot.coord_pick import executor as ex
    from robot.coord_pick.plan import select_which
    from robot.coord_pick.wrist_fake import FakeWorld

    # sort_tests 가 real 을 10-22 이전 시험값(grasp 14·금지 높이 100 등)으로 바꿔 두므로, 실물 재현은 config.json 을 새로 읽고
    # 10-23 실물 때 값으로 고정한다(총괄이 config 를 바꿔도 시험이 그대로이게)
    cfg = copy.deepcopy(load_config())
    cfg["calibrated"] = True
    cfg["heights_mm"].update(grasp=5, approach=50, carry=70, grasp_finger=2, finger_min=0)
    cfg["safety"].update(floor_mm=0, forbid_below_mm=60, forbid_margin_mm=20, grip_min_n=48, air_deg_s=30, air_acc_deg_s2=150)
    cfg["bowls"].update(rim_mm=20, release_above_mm=50, release_margin_mm=30)
    cfg["wrist"] = {"enabled": True}
    cfg["grasp"] = {"mode": "ik", "policy": "grasp", "approach_safe_mm": 90, "start_z_mm": 90, "timeout_s": 8,
                    "act_align_iters": 1, "fallback_ik": True}
    cfg.pop("which_flip", None)
    s = Setup.load(cfg, use_studio=False)

    class Keeper:
        def start(self):
            return True

        def stop(self):
            pass

    class World(FakeWorld):
        """블록 여러 개(구간 이름 [k/n·…] 의 k 번째 블록이 손목에 보임) + 이름에 fault 가 든 구간은 추종 오차로 멈춤."""

        def __init__(self, ws, color="red", yaws=None, fault=None):
            super().__init__(s, ws[0], color, true_yaw_w=None if yaws is None else yaws[0], aspect=1.6)
            self.ws, self.yaws, self.fault = [np.asarray(w, float) for w in ws], yaws, fault

        def run_real(self, s_, targets, marks, lim, log, *, name, **kw):
            m = re.match(r"\[(\d+)/", name)
            if m:
                k = int(m.group(1)) - 1
                if self.grasps and getattr(self, "k", k) != k:  # 다음 블록 — 앞 블록은 그릇에 있다(쥔 것 없음)
                    self.grasps.append({"dist_mm": None, "held": False, "rho_deg": None})
                self.k = k
                self.true_w = self.ws[k]
                self.true_yaw_w = None if self.yaws is None else self.yaws[k]
            if self.fault and self.fault in name:
                self.sends.append(name)
                return {"reason": "tracking", "run_id": "fake-fault", "error": "추종 오차(시험)"}
            return super().run_real(s_, targets, marks, lim, log, name=name, **kw)

    def go(ws, yaws, *, color="red", mode="ik", c=None, fault=None, n=None, **kw):
        fw = World(ws, color, yaws, fault)
        cls = "red_ok" if color == "red" else "blue_ok"
        objs = [{"cls": cls, "px": s.w_to_px(np.asarray(w, float))[0].tolist(), "conf": 0.9, "zone": "input",
                 **({} if yaws is None else {"yaw_w": yaws[i], "aspect": 1.6})} for i, w in enumerate(ws)]
        calls = []

        def policy(name, timeout_s=None, return_home=None):
            calls.append(round(float(s.tcp_pos(fw.q)[2]), 1))
            qe, _ = s.solve(s.w_to_r3(fw.true_w, 45), 0.0, fw.q, tilt=None, tilt_max=s.tilt["grasp"])
            fw.q = qe
            fw.grasps.append({"dist_mm": 0.0, "held": True, "rho_deg": None})
            fw.dropped = False
            return {"reason": "grasp_done", "end_q": s.q_to_n(qe, 50.4).tolist(),
                    "early_stop": {"closed": True, "lifted": True, "z_low_mm": 4.0}, "log_path": "x"}
        cc = copy.deepcopy(c or cfg)
        cc["grasp"]["mode"] = mode
        fw.install(ex)
        os.environ["BALL_ROBOT"] = "1"
        try:
            r = ex.sort_blocks(color, dry_run=False, detection={"objects": copy.deepcopy(objs)}, max_blocks=n or len(ws),
                               config=cc, wrist_check=True, log_dir=tmp, act_runner=policy, keeper_factory=Keeper,
                               redetect=lambda: {"objects": copy.deepcopy(objs)}, **kw)
        finally:
            fw.uninstall()
            os.environ.pop("BALL_ROBOT", None)
        logs = [json.loads(x) for x in Path(r["log_path"]).read_text(encoding="utf-8").splitlines()]
        return r, fw, calls, logs

    far, far_yaw = (71.8, 103.5), math.radians(-140.3)   # 09:46 실물 분홍(계획 기울기 15°)
    # 1) 09:46: approach_safe 90 에서 계획 기울기가 안 됨 → 기울기를 바꾸거나 낮춰 도착(거부 없이), 로그 arrive
    r, fw, _, logs = go([far], [far_yaw])
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])[0]
    arr = at.get("arrive") or {}
    check("09:46 블록 위 90mm: 계획 기울기가 안 되면 기울기를 바꿔·낮춰 도착 → 집고 놓기(로그 arrive)",
          r["ok"] and arr.get("used_mm") is not None and arr.get("tried") and any(e["stage"] == "arrive" for e in logs),
          f"{arr.get('want_mm')} → {arr.get('used_mm')}mm · 기울기 {arr.get('plan_tilt_deg')} → {arr.get('tilt_deg')}° · "
          f"{str(r.get('error') or '')[:120]}")
    # 2) act 시작 자세: 계획 기울기가 안 되는 칸은 가까운 기울기로 시작, 그래도 안 되는 칸(작업대 (241, 149) 긴 축 0°)은 IK 집기
    r, fw, calls, logs = go([far], [far_yaw], mode="act")
    st = next((e for e in logs if e["stage"] == "act_start"), {})
    check("act 시작 자세: 90mm 에서 계획 기울기가 안 되면 가까운 기울기로 시작 → 정책", r["ok"] and len(calls) == 1
          and abs(calls[0] - 90) < 3 and st.get("tilt_deg") != st.get("plan_tilt_deg") and st,
          f"정책 시작 높이 {calls} · 기울기 {st.get('plan_tilt_deg')} → {st.get('tilt_deg')}° · {str(r.get('error') or '')[:100]}")
    r, fw, calls, logs = go([(241.0, 149.0)], [0.0], mode="act")
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])[0]
    check("act 시작 자세가 안 풀리는 칸 → 그 블록은 IK 집기(fallback_ik, 로그 act_skip)",
          r["ok"] and not calls and "IK 집기" in str((at.get("act") or {}).get("skipped"))
          and any(e["stage"] == "act_skip" for e in logs) and any("집고 들어 올림" in n_ for n_ in fw.sends),
          f"ok {r['ok']} · 정책 호출 {len(calls)} · {(at.get('act') or {}).get('skipped')} · 구간 {fw.sends} · "
          f"{str(r.get('error') or '')[:100]}")
    # 3) 09:50: 초록 2개 — 1번을 그릇에 놓은 자리(집게 끝이 금지 높이 아래)에서 2번 hover 로 떠나는 구간이 거부되지 않음
    blue2 = [(150.0, 95.0), (225.0, 120.0)]
    r, fw, _, logs = go(blue2, [math.radians(-150), math.radians(-146)], color="blue")
    hov = [n_ for n_ in fw.sends if n_ == "[2/2·1] 초록 hover"]
    check("09:50 두 번째 블록: 앞 그릇 위에서 떠나는 hover 구간 허용 → 2개 다 놓음",
          r["ok"] and len(hov) == 1 and sum("그릇에 놓기" in n_ for n_ in fw.sends) == 2,
          f"ok {r['ok']} · {len(fw.sends)}구간 · hover {hov} · 놓기 {[n_ for n_ in fw.sends if '놓기' in n_]} · "
          f"{str(r.get('error') or '')[:160]}")
    # 4) 보내기 전 거부 → 그 자리에서 올라가 홈으로(저절로). 보내던 중 멈춤(추종 오차)은 움직이지 않는다
    orig_ws, k = ex.check_workspace, {"n": 0}

    def ws_once(*a, **kw_):
        k["n"] += 1
        v = orig_ws(*a, **kw_)
        return v + ([{"kind": "forbidden:시험", "count": 1, "first_tick": 0, "detail": "시험용 거부"}] if k["n"] == 3 else [])
    ex.check_workspace = ws_once
    try:
        r, fw, _, logs = go([far], [far_yaw])
    finally:
        ex.check_workspace = orig_ws
    rc = r.get("recover") or {}
    home_off = float(np.abs(s.q_to_n(fw.q, 0.0)[:5] - np.array(real["home_n"][:5])).max()) if fw.q is not None else 99
    check("보내기 전 거부 → 그 자리에서 수직으로 올라가 홈으로(로그 recover)", not r["ok"] and rc.get("ok")
          and fw.sends[-1] == "거부 뒤 올라가 홈으로" and home_off < 0.5 and "홈으로 돌아감" in str(r.get("error"))
          and any(e["stage"] == "recover" for e in logs),
          f"올라간 높이 {rc.get('lift_to_mm')} · 홈과 {home_off:.2f} · {str(r.get('error') or '')[-80:]}")
    r, fw, _, _ = go([far], [far_yaw], fault="내려감")
    check("보내던 중 멈춤(추종 오차) → 저절로 움직이지 않음(팔은 그 자리)", not r["ok"] and not r.get("recover")
          and fw.sends[-1].endswith("내려감") and "팔은 그 자리" in str(r.get("error")),
          f"{fw.sends[-1]} · {str(r.get('error') or '')[-60:]}")
    # 5) 왼쪽·오른쪽·가운데: 위 카메라 u 순서(뒤집기 config which_flip 한 곳), blocks=N 이면 그 방향에서 N 개
    three = [{"cls": "red_ok", "px": [px, 210.0], "zone": "input"} for px in (250.0, 170.0, 330.0)] + \
            [{"cls": "blue_ok", "px": [120.0, 200.0], "zone": "input"}]
    pick = {w_: [b["px"][0] for b in select_which(copy.deepcopy(three), ("red_ok",), w_, n_)[1]]
            for w_, n_ in (("left", 1), ("right", 2), ("middle", 1))}
    flip = [b["px"][0] for b in select_which(copy.deepcopy(three), ("red_ok",), "left", 1, flip=True)[1]]
    rest = select_which(copy.deepcopy(three), ("red_ok",), "left", 1)[0]
    check("which: left = u 가장 작은 것, right 2개 = 큰 것부터, middle, flip 이면 맞바꿈, 안 고른 같은 색은 이웃",
          pick == {"left": [170.0], "right": [330.0, 250.0], "middle": [250.0]} and flip == [330.0]
          and sum(b["cls"] == "obstacle" for b in rest) == 2 and rest[3]["cls"] == "blue_ok",
          f"{pick} · 뒤집음 {flip}")
    det = {"objects": [{"cls": "red_ok", "px": s.w_to_px(np.array(w))[0].tolist(), "conf": 0.9, "zone": "input"}
                       for w in ((120.0, 110.0), (200.0, 90.0), (270.0, 120.0))]}
    us = sorted(o["px"][0] for o in det["objects"])
    d_l = ex.sort_blocks("red", detection=copy.deepcopy(det), which="left", config=copy.deepcopy(cfg), log_dir=tmp)
    d_r = ex.sort_blocks("red", detection=copy.deepcopy(det), which="right", max_blocks=2, config=copy.deepcopy(cfg),
                         log_dir=tmp)
    c_f = copy.deepcopy(cfg)
    c_f["which_flip"] = True
    d_f = ex.sort_blocks("red", detection=copy.deepcopy(det), which="left", config=c_f, log_dir=tmp)
    check("sort_blocks(which=…) dry-run: 고른 블록만 계획(left 1개·right 2개 큰 u 부터·which_flip)",
          d_l["ok"] and d_l["blocks"] == 1 and d_l["which"]["chosen_px"][0][0] == us[0]
          and d_r["ok"] and d_r["blocks"] == 2 and [p_[0] for p_ in d_r["which"]["chosen_px"]] == [us[2], us[1]]
          and d_f["ok"] and d_f["which"]["chosen_px"][0][0] == us[2],
          f"left {d_l['which']['chosen_px']} · right {d_r['which']['chosen_px']} · flip {d_f['which']['chosen_px']}"
          f" · {d_l.get('error') or ''}{d_r.get('error') or ''}")
    # 5b) 위·아래·가까운·먼·옆: 위 카메라 v·로봇까지 거리·다른 색 블록과 거리로, 모르는 값·옆 기준 없음은 거부
    pts = {"A": (120.0, 150.0), "B": (200.0, 60.0), "C": (275.0, 110.0)}
    det3 = {"objects": [{"cls": "red_ok", "px": s.w_to_px(np.array(w))[0].tolist(), "conf": 0.9, "zone": "input"}
                        for w in pts.values()] + [{"cls": "blue_ok", "px": s.w_to_px(np.array([250.0, 115.0]))[0].tolist(),
                                                   "conf": 0.9, "zone": "input"}]}
    px_of = {k: s.w_to_px(np.array(w))[0].tolist() for k, w in pts.items()}
    r_of = {k: float(np.linalg.norm(s.frames.w2r(np.array(w)))) for k, w in pts.items()}
    want = {"top": min(px_of, key=lambda k: px_of[k][1]), "bottom": max(px_of, key=lambda k: px_of[k][1]),
            "near": min(r_of, key=r_of.get), "far": max(r_of, key=r_of.get)}
    got = {}
    for w_ in ("top", "bottom", "near", "far"):
        d_ = ex.sort_blocks("red", detection=copy.deepcopy(det3), which=w_, config=copy.deepcopy(cfg), log_dir=tmp)
        cp = d_["which"]["chosen_px"][0] if d_.get("which") else None
        got[w_] = next((k for k, v in px_of.items() if cp and abs(v[0] - cp[0]) < 0.5 and abs(v[1] - cp[1]) < 0.5), None)
    d_nt = ex.sort_blocks("red", detection=copy.deepcopy(det3), which={"next_to": "blue"}, config=copy.deepcopy(cfg),
                          log_dir=tmp)
    nt = next((k for k, v in px_of.items() if d_nt.get("which") and abs(v[0] - d_nt["which"]["chosen_px"][0][0]) < 0.5), None)
    d_none = ex.sort_blocks("red", detection=copy.deepcopy(det3), which={"next_to": "red_defect"}, config=copy.deepcopy(cfg),
                            log_dir=tmp)
    d_bad = ex.sort_blocks("red", detection=copy.deepcopy(det3), which="upper", config=copy.deepcopy(cfg), log_dir=tmp)
    check("which: top·bottom(위 카메라 v)·near·far(로봇까지 거리)·next_to(그 색 블록 옆) · 모르는 값·옆 기준 없음은 거부",
          got == want and nt == "C" and d_nt["ok"] and not d_none["ok"] and d_none["reason"] == "refused"
          and not d_bad["ok"] and d_bad["reason"] == "refused" and "rule" in d_nt["which"],
          f"{got} (기대 {want}) · 초록 옆 {nt} · 불량 옆 {d_none['reason']} · 'upper' {d_bad['reason']}")
    # 6) 불량 블록: 검은 X 가 면을 네 조각으로 → merge 면 한 덩어리(중심 = 블록 가운데), 옆 블록(130px)은 안 묶음
    import cv2

    from robot.coord_pick import wrist
    img = np.full((480, 640, 3), 225, np.uint8)
    cv2.rectangle(img, (220, 100), (380, 190), (150, 220, 160), -1)
    cv2.line(img, (220, 100), (380, 190), (20, 20, 20), 8)
    cv2.line(img, (220, 190), (380, 100), (20, 20, 20), 8)
    cv2.rectangle(img, (510, 100), (600, 190), (150, 220, 160), -1)
    one = wrist.blob(img, "blue", near=(300, 145))
    mg = wrist.blob(img, "blue", near=(300, 145), merge=True)
    check("불량 X 조각: merge 면 네 조각을 한 덩어리로(중심 = 블록 가운데), 옆 블록은 따로",
          one is not None and mg is not None and mg["pieces"] == 4 and abs(mg["px"][0] - 300) < 4
          and abs(mg["px"][1] - 145) < 4 and mg["area"] > 2.5 * one["area"] and mg["box"][0] + mg["box"][2] < 400,
          f"조각 하나 {one and (one['px'], one['area'])} → 묶음 {mg and (mg['px'], mg['area'], mg['pieces'])}")
    ref = wrist.load_ref()
    r9 = ref["after"]["by_dataset"]["pick_green"]
    fake_b = {"px": [r9["u_med"], r9["v_med"]], "area": int(r9["area_p10"] * 0.8)}
    check("집은 뒤 면적 하한 × wrist.after_area_scale(기본 1.0 그대로, 0.7 이면 녹화 하위 10%의 70%)",
          not wrist.judge(fake_b, ref, "after", "pick_green")["ok"]
          and wrist.judge(fake_b, ref, "after", "pick_green", area_scale=0.7)["ok"]
          and ex.WRIST_DEFAULTS["after_area_scale"] == 1.0, f"면적 {fake_b['area']} · 하한 {r9['area_p10']}")
    # 6b) act 시작을 녹화처럼(블록에서 로봇 쪽 33·옆 −17mm, 120mm, 22° 근처) — 손목 맞춤 없이 정책, 정책이 제어 시간을 다 써도
    #     닫고 들었으면 쥔 것으로(녹화 17/39 는 끝에 0.5초 가만히 있지 않았다)
    c_o = copy.deepcopy(cfg)
    c_o["grasp"].update(start_z_mm=120, start_back_mm=33, start_side_mm=-17, start_tilt_deg=22)
    r, fw, calls, logs = go([(200.0, 90.0)], [0.0], mode="act", c=c_o)
    st = next((e for e in logs if e["stage"] == "act_start"), {})
    off = float(np.linalg.norm(np.subtract(st.get("start_mm", [0, 0]), st.get("block_mm", [0, 0]))))
    aligns = [e for e in logs if e["stage"] == "align" and e.get("level") == "hover"]
    check("act 시작 비켜 세우기: 블록에서 37mm·120mm 에서 정책(손목 맞춤 안 함) → 그 자리부터 나르기",
          r["ok"] and len(calls) == 1 and abs(calls[0] - 120) < 3 and 30 < off < 45 and not aligns,
          f"시작 높이 {calls} · 블록과 {off:.0f}mm · 기울기 {st.get('tilt_deg')}° · 맞춤 {len(aligns)}")
    orig_go_pol = ex.grasp_act.run_grasp_policy

    def timeout_policy(policy, timeout_s, log, runner=None):
        r_ = orig_go_pol(policy, timeout_s, log, runner=runner)
        return {**r_, "ok": False, "reason": "duration_elapsed",
                "early_stop": {**(r_.get("early_stop") or {}), "closed": True, "lifted": True}}
    ex.grasp_act.run_grasp_policy = timeout_policy
    try:
        r, fw, calls, logs = go([(200.0, 90.0)], [0.0], mode="act", c=c_o)
    finally:
        ex.grasp_act.run_grasp_policy = orig_go_pol
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])[0]
    check("act: 제어 시간이 먼저 끝나도 닫고 들었으면 쥔 것으로(손목 '집은 뒤' 사진이 가림) → IK 집기 없이 놓기",
          r["ok"] and at.get("held") and not any("집고 들어 올림" in n_ for n_ in fw.sends),
          f"쥠 {at.get('held')} · {(at.get('act') or {}).get('reason')} · {str(r.get('error') or '')[:100]}")
    # 7) 10:07: 손 자리 41mm 옆으로 떨어진 불량 → 금지 구역 때문에 거부되면 그 블록에서 끝내고(알림) 저절로 홈
    r, fw, _, _ = go([(51.8, 32.6)], [0.0], color="blue")
    check("손 자리 가까이 떨어진 블록 → '손 자리 가까이' 알림으로 끝 + 저절로 홈", not r["ok"]
          and "손 자리 가까이" in str(r.get("error")) and (r.get("recover") or {}).get("ok")
          and fw.sends[-1] == "거부 뒤 올라가 홈으로", f"{str(r.get('error') or '')[:150]}")


def unsort_tests(real: dict, tmp: Path) -> None:
    """그릇 → 투입 구역 되돌리기(10-23): 가짜 장면 계획 · 집는 그릇 금지 구역 예외 범위 · 빈 칸 고르기 · 겹침 건너뛰기 ·
    가짜 로봇 손목 실행 · 색별 불량(blue_defect)."""
    import copy
    import os

    from robot.coord_pick import executor as ex
    from robot.coord_pick.plan import (BOWL_ZONES, demo_unsort_scene, free_cells, input_cells, plan_sort, select_unsort,
                                       zones_w)
    from robot.coord_pick.wrist_fake import FakeWorld

    cfg = copy.deepcopy(load_config())  # sort_tests 가 real 을 예전 시험값으로 바꿔 두므로 새로 읽고 10-23 값으로 고정
    cfg["calibrated"] = True
    cfg["heights_mm"].update(grasp=5, approach=50, carry=70, place=10, grasp_finger=2, finger_min=0)
    # 금지 구역 여유 10 = 10-23 오전 총괄이 바꾼 실물 값(20 이면 로봇 쪽 칸 7~9 는 놓을 때 손가락이 손 자리 여유에 걸려 빠진다)
    cfg["safety"].update(floor_mm=0, forbid_below_mm=60, forbid_margin_mm=10, grip_min_n=48, air_deg_s=30, air_acc_deg_s2=150)
    cfg["wrist"] = {"enabled": True}
    cfg["grasp"] = {"mode": "ik", "approach_safe_mm": 90}
    cfg.pop("unsort", None)
    s = Setup.load(cfg, use_studio=False)
    zw = zones_w(s)
    allc = ("red_ok", "blue_ok", "red_defect", "blue_defect")
    d = ex.unsort_blocks(demo=True, config=copy.deepcopy(cfg), log_dir=tmp)
    why = {x["why"].split("(")[0] for x in d.get("skipped", [])}
    check("되돌리기 dry-run(가짜 장면): 그릇 블록 → 빈 칸, 겹친 것·먼 것은 이유와 함께 남김, 공간 위반 0",
          d["ok"] and d["blocks"] >= 2 and not d["violations"] and "겹쳐 있어 못 꺼냄" in why
          and all(x["to"].startswith("칸 ") for x in d["plan"]),
          f"{[(x['color'], x['from'], x['to']) for x in d['plan']]} · 못 꺼냄 {[x['why'][:20] for x in d['skipped']]}")
    # 집는 그릇 예외는 그 그릇만: 계획 궤적을 예외 없이 검사하면 그 그릇 위반이 나고, 예외를 주면 0
    blocks = select_unsort(s, demo_unsort_scene(s), allc, zw)
    cells = free_cells(s, blocks, 60, zw)
    p = plan_sort(s, blocks, classes=allc, unsort=True, cells=cells)
    sf = ex.safety(cfg)
    traj = trajectory(p["steps"], 30)
    polys, _ = ex.forbidden_polygons(s, sf)
    srcs = {o["src"] for o in p["order"]}
    v0 = {v["kind"] for v in ex.check_workspace(s, traj, sf, polys, None)}
    v1 = ex.check_workspace(s, traj, sf, polys, {k: 0.0 for k in srcs})
    check("집는 그릇 금지 구역 예외: 예외 없으면 그 그릇 위반 · 그 그릇만 허용하면 위반 0(다른 그릇·손 자리는 그대로)",
          {f"forbidden:{k}" for k in srcs} <= v0 and not v1 and all(o["src"] in BOWL_ZONES for o in p["order"]),
          f"예외 없음 {sorted(v0)} · 예외 {sorted(srcs)} → {len(v1)}")
    yaws = [round(math.degrees(o["drop_yaw_w"])) % 180 for o in p["order"]]
    check("놓기: 집게축 90°(블록 긴 축 0° = 테이프 가로)·낮게(집게 끝이 탁자 위)", yaws and all(y == 90 for y in yaws)
          and all(o["place_z"] < 20 for o in p["order"]), f"집게축 {yaws} · 놓는 높이 {[round(o['place_z'], 1) for o in p['order']]}")
    # 빈 칸: 투입 구역 블록 가운데 60mm 안 칸은 빼고, 로봇에 가까운 칸부터
    c5 = dict(input_cells(s))[5]
    occ = [{"cls": "blue_ok", "px": s.w_to_px(c5)[0].tolist(), "zone": "input"}]
    fc = [n for n, _ in free_cells(s, occ, 60, zw)]
    r_of = {n: float(np.linalg.norm(s.frames.w2r(c))) for n, c in input_cells(s)}
    check("빈 칸 고르기: 블록이 있는 칸 5(와 60mm 안 칸)는 빼고 로봇에 가까운 칸부터", 5 not in fc and len(fc) >= 6
          and all(r_of[a_] <= r_of[b_] + 1e-6 for a_, b_ in zip(fc, fc[1:])), f"빈 칸 {fc}")
    # 가짜 로봇 손목 실행: 불량 그릇 초록X 1개 → 칸에 놓기(그 그릇 예외가 없으면 hover 부터 거부된다)
    bd = next(b for b in demo_unsort_scene(s) if b["cls"] == "blue_defect")
    w = s.px_to_w(bd["px"])[0]
    # 가짜 세계 '쥠' 허용 16mm(기본 12) — 이 시험은 그릇 예외·칸 놓기 흐름을 본다(먼 불량 그릇에서 맞춤 잔차 12.9mm)
    fw = FakeWorld(s, w, "blue", true_yaw_w=math.radians(-150), aspect=1.8, grip_ok_mm=16.0)
    obj = {**bd, "yaw_w": math.radians(-150), "aspect": 1.8}
    fw.install(ex)
    os.environ["BALL_ROBOT"] = "1"
    try:
        r = ex.unsort_blocks(False, "blue_defect", detection={"objects": [obj]}, config=copy.deepcopy(cfg), log_dir=tmp,
                             wrist_check=True, redetect=lambda: {"objects": [dict(obj)]})
    finally:
        fw.uninstall()
        os.environ.pop("BALL_ROBOT", None)
    wr = (r.get("wrist") or [{}])[0]
    check("되돌리기 실행(가짜 로봇·손목): 불량 그릇 초록X → 손목 확인 집기 → 빈 칸에 놓고 홈",
          r["ok"] and wr.get("cell") is not None and any("에 놓기" in n_ for n_ in fw.sends) and fw.dropped,
          f"{wr.get('to')} · 구간 {len(fw.sends)} · {str(r.get('error') or '')[:120]}")
    # 색별 불량: blue_defect 는 초록X 만(분홍X 는 그대로)
    two = [{"cls": "red_defect", "px": s.w_to_px(np.array([120.0, 90.0]))[0].tolist(), "conf": 0.9, "zone": "input"},
           {"cls": "blue_defect", "px": s.w_to_px(np.array([230.0, 90.0]))[0].tolist(), "conf": 0.9, "zone": "input"}]
    db = ex.sort_blocks(["blue_defect"], detection={"objects": copy.deepcopy(two)}, config=copy.deepcopy(cfg), log_dir=tmp)
    dw = ex.sort_blocks("blue_defect", detection={"objects": copy.deepcopy(two)}, which="left", config=copy.deepcopy(cfg),
                        log_dir=tmp)
    check("색별 불량: colors=['blue_defect'] 은 초록X 만 불량 그릇으로(분홍X 그대로) · which 와 같이",
          db["ok"] and db["blocks"] == 1 and db["plan"][0]["color"] == "초록 불량" and db["plan"][0]["to"] == "불량 그릇"
          and dw["ok"] and dw["blocks"] == 1, f"{[(x['color'], x['to']) for x in db['plan']]} · which {dw.get('which')}")
    # 11:37 실물: 먼 줄 초록X 를 닫기 전 자세(측정값)에서 hover 로 들어 올림 — 기울기 15° 고정이면 IK 실패, 가까운 기울기면 됨
    from robot.compare.cell_table import replay_1137
    rp_ = replay_1137(Setup.load(load_config(), use_studio=False), load_config())
    check("11:37 들어 올림(실물 마지막 자세): 계획 기울기 고정이면 실패 · 허용 안 가까운 기울기로 풀면 됨(n29)",
          not rp_["before"] and rp_["after"], f"앞 {rp_['why_before'][:70]} · 뒤 {rp_['tilt_after']}°")


def act_start_tests(tmp: Path) -> None:
    """10-23 12:59 실물 뒤: ACT 시작 장면 맞춤(손목 화면 긴 축·가운데) · 정책이 바닥을 누르면 바로 끝 · 바닥에서 빼고 IK 로."""
    import copy
    import os

    from robot import grasp_stop
    from robot.coord_pick import executor as ex
    from robot.coord_pick.wrist_fake import FakeWorld

    # 1) 바닥 누름: 닫기 전 tcp −3mm 아래 1.0초 → floor. 사람처럼 −8mm 를 0.3초 찍고 닫아 들면 grasp
    def det_run(seq):
        d = grasp_stop.GraspStopDetector(lambda q: q[1], close_drop=6, lift_mm=50, still=99, hold_s=0, floor_mm=-3,
                                         floor_hold_s=1.0, floor_hard_mm=-15)
        hit = next((t_ for t_, q in seq if d.feed(t_, q)), None)
        return hit, d
    press = [(k / 30, [0, 90 - k * 3 if k < 32 else -6.6, 0, 0, 0, 63]) for k in range(300)]
    human = ([(k / 30, [0, 90 - k * 3.2, 0, 0, 0, 63]) for k in range(31)] + [(1.0 + k / 30, [0, -8, 0, 0, 0, 63]) for k in range(9)]
             + [(1.3 + k / 30, [0, -8 + k * 3, 0, 0, 0, 63 - 13 * min(1, k / 6)]) for k in range(40)])
    h1, d1 = det_run(press)
    h2, d2 = det_run(human)
    check("정책 바닥 누름(12:59): 닫기 전 tcp −3mm 아래 1초면 바로 끝(floor) · 사람처럼 잠깐 찍고 닫아 들면 정상 끝",
          d1.trigger is not None and d1.trigger.kind == "floor" and h1 < 2.5 and d2.trigger is not None
          and d2.trigger.kind == "grasp", f"누름 → {h1:.2f}s {d1.trigger.kind if d1.trigger else None} · 사람 → "
                                          f"{d2.trigger.kind if d2.trigger else None}")
    # 2) 시작 장면 맞춤(image): 위 카메라 방향이 40° 틀려 손목 화면 긴 축이 벗어남 → 롤을 돌려 가로로, 가운데를 녹화 자리로
    cfg = copy.deepcopy(load_config())
    cfg["calibrated"] = True
    cfg["heights_mm"].update(grasp=5, approach=50, carry=70, place=10, grasp_finger=2, finger_min=0)
    cfg["safety"].update(floor_mm=0, forbid_below_mm=60, forbid_margin_mm=10, grip_min_n=48, air_deg_s=30, air_acc_deg_s2=150)
    cfg["wrist"] = {"enabled": True}
    cfg["grasp"] = {"mode": "act", "start_z_mm": 135, "start_back_mm": 33, "start_side_mm": -17, "start_tilt_deg": 22,
                    "start_align": "image", "approach_safe_mm": 90, "timeout_s": 8}
    s = Setup.load(cfg, use_studio=False)
    w, true_yaw = np.array([200.0, 90.0]), math.radians(10.0)

    class Keeper:
        def start(self):
            return True

        def stop(self):
            pass

    def go(det_yaw, policy):
        fw = FakeWorld(s, w, "red", true_yaw_w=true_yaw, aspect=1.8)
        obj = {"cls": "red_ok", "px": s.w_to_px(w)[0].tolist(), "conf": 0.9, "zone": "input", "yaw_w": det_yaw, "aspect": 1.8}
        calls = []

        def pol(name, timeout_s=None, return_home=None):
            calls.append(round(float(s.tcp_pos(fw.q)[2]), 1))
            return policy(fw)
        fw.install(ex)
        os.environ["BALL_ROBOT"] = "1"
        try:
            r = ex.sort_blocks("red", dry_run=False, detection={"objects": [obj]}, max_blocks=1, config=copy.deepcopy(cfg),
                               wrist_check=True, log_dir=tmp, act_runner=pol, keeper_factory=Keeper,
                               redetect=lambda: {"objects": [dict(obj)]})
        finally:
            fw.uninstall()
            os.environ.pop("BALL_ROBOT", None)
        return r, fw, calls

    def grasping(fw):
        qe, _ = s.solve(s.w_to_r3(w, 45), 0.0, fw.q, tilt=None, tilt_max=s.tilt["grasp"])
        fw.q = qe
        fw.grasps.append({"dist_mm": 0.0, "held": True, "rho_deg": None})
        fw.dropped = False
        return {"reason": "grasp_done", "end_q": s.q_to_n(qe, 50.4).tolist(),
                "early_stop": {"closed": True, "lifted": True, "z_low_mm": 4.0}, "log_path": "x"}
    # 화면 모형으로 그리는 가짜 손목도 이 높이에선 각이 줄어 보인다(실물 12:59: 계산 rho 65.6° 인데 화면 축 3.1°) → 75° 틀린다
    r, fw, calls = go(true_yaw + math.radians(75), grasping)
    at = (r.get("wrist") or [{}])[0].get("attempts", [{}])[0]
    sa = at.get("start_align") or {}
    last_off = (sa.get("turns") or [{}])[-1].get("off_deg", sa.get("off_deg"))
    check("ACT 시작 장면 맞춤: 손목 화면 긴 축이 벗어나면 롤을 돌려 가로로(실제로 돈 각으로 배율) · 가운데를 녹화 자리로 → 정책",
          r["ok"] and len(calls) == 1 and sa.get("turns") and abs(last_off) <= 20 and sa.get("centered")
          and abs(calls[0] - 135) < 4,
          f"긴 축 {sa.get('off_deg')}° → {[x['off_deg'] for x in sa.get('turns', [])]} · 가운데 {sa.get('err_px')} · 정책 시작 {calls}")
    # 3) 정책이 닫지 않고 바닥을 누르다 끝남(floor_stop) → 그 자세에서 수직으로 빼고 IK 집기(12:59 는 계획이 거부됐다)

    end_1259 = [-3.199, 37.827, -39.976, 43.004, 38.022, 62.515]   # 12:59 실물 정책 끝 관절값(tcp 로봇 [356.7, 55.9, −6.6])

    def pressing(fw):
        fw.q = s.n_to_q(np.array(end_1259))
        return {"reason": "floor_stop", "end_q": end_1259,
                "early_stop": {"closed": False, "lifted": False, "stop_kind": "floor"}, "log_path": "x"}
    r, fw, calls = go(true_yaw, pressing)
    # (이어진 IK 집기가 가짜 세계에서 쥐는지는 이 시험의 몫이 아니다 — 바닥에서 빼는 구간이 거부 없이 나가고 IK 로 이어지나)
    # 4) 연한 초록: 기본 초록 마스크에 안 잡히는 옅은 초록 덩어리 → pale 이면 Lab 으로 잡고, 진한 초록은 예전 그대로
    import cv2

    from robot.coord_pick import wrist
    img = np.full((480, 640, 3), 225, np.uint8)
    cv2.rectangle(img, (150, 190), (260, 250), (205, 228, 212), -1)  # 연한 초록(BGR)
    b0, b1 = wrist.blob(img, "blue", near=(200, 220)), wrist.blob(img, "blue", near=(200, 220), pale=True)
    img2 = np.full((480, 640, 3), 225, np.uint8)
    cv2.rectangle(img2, (150, 190), (260, 250), (150, 220, 160), -1)
    b2, b3 = wrist.blob(img2, "blue", near=(200, 220)), wrist.blob(img2, "blue", near=(200, 220), pale=True)
    check("연한 초록(높은 시작 자세): 기본 마스크에 없을 때만 Lab 으로 잡음 · 진한 초록은 예전 그대로",
          b0 is None and b1 is not None and abs(b1["px"][0] - 205) < 5 and b2 is not None and b3 == b2,
          f"기본 {b0 and b0['px']} · pale {b1 and (b1['px'], b1['area'])} · 진한 초록 같음 {b3 == b2}")
    # 5) '집은 뒤' 가운데 허용 하한: 분홍 녹화 v 허용이 ±14px 로 좁아 13:27 실물에서 쥔 블록(dv −18~−25)을 세 번 못 쥠으로 봄
    # (총괄이 13:30 wrist_ref_fixed.json 분홍 허용을 60·30 으로 넓혔다 — 시험은 10-22 원래 값 ±50·±14 로 고정)
    rr = {"u_med": 349.9, "v_med": 262.3, "tol_px": [50.2, 14.2], "area_p10": 7404}
    ref_ = {"after": {"by_dataset": {"pick_red": rr}}}
    bb = {"px": [rr["u_med"] - 3, rr["v_med"] - 22], "area": int(rr["area_p10"] * 1.5)}
    check("'집은 뒤' 가운데 허용 하한(wrist.after_tol_min_px): 기본 0 은 예전 그대로(dv −22 못 쥠) · 30 이면 쥠",
          not wrist.judge(bb, ref_, "after", "pick_red")["ok"] and wrist.judge(bb, ref_, "after", "pick_red", tol_min=30)["ok"]
          and ex.WRIST_DEFAULTS["after_tol_min_px"] == 0.0, f"분홍 허용 {rr['tol_px']}")
    check("정책이 바닥을 누르다 끝남(12:59 끝 자세) → 수직으로 빼고(25mm) IK 집기로 이어짐(계획 거부 없이)",
          any(n_.endswith("정책 못 쥠 → IK 집기") for n_ in fw.sends) and any("닫기 직전 위로" in n_ for n_ in fw.sends)
          and "정책 못 쥠 → IK 집기: refused" not in str(r.get("error") or ""),
          f"{[n_.split('] ')[-1] for n_ in fw.sends][:6]} · {str(r.get('error') or '')[:100]}")


def servo_tests(real: dict, tmp: Path) -> None:
    """손목 실시간 맞춤 시험판(servo.py, grasp.servo) — 꺼져 있으면 계획이 그대로인지, 가짜 세계에서 한 번에 내려가는지,
    블록을 잃으면 지금 방식으로 넘어가는지, 컨테이너 쪽 루프의 틱 한도·호스트 침묵."""
    import copy
    import hashlib
    import os

    from robot.coord_pick import executor as ex
    from robot.coord_pick import servo
    from robot.coord_pick.wrist_fake import FakeWorld

    cfg = copy.deepcopy(real)
    cfg["calibrated"] = True
    s = Setup.load(cfg, use_studio=False)
    blocks = demo_sort_scene(s)
    near = min((b for b in blocks if b["cls"] == "red_ok"), key=lambda b: float(np.linalg.norm(s.frames.w2r(s.px_to_w(b["px"])[0]))))
    det_w = s.px_to_w(near["px"])[0]

    def go(offset, grasp_extra, **kw):
        c2 = copy.deepcopy(cfg)
        c2.setdefault("grasp", {}).update(grasp_extra)
        true_w = det_w + np.array(offset, float)
        fw = FakeWorld(s, true_w, **kw)
        hashes = []
        orig = fw.run_real

        def rr(s_, targets, marks, lim, log, *, name, **k):
            hashes.append((name, hashlib.sha1(np.ascontiguousarray(np.asarray(targets, float)).tobytes()).hexdigest()))
            return orig(s_, targets, marks, lim, log, name=name, **k)
        fw.run_real = rr
        fw.install(ex)
        saved = os.environ.get("BALL_ROBOT")
        os.environ["BALL_ROBOT"] = "1"
        try:
            r = ex.sort_blocks("red", dry_run=False, detection={"objects": copy.deepcopy(blocks)}, max_blocks=1,
                               config=c2, wrist_check=True, log_dir=tmp,
                               redetect=lambda: {"objects": [dict(near, px=s.w_to_px(true_w)[0].tolist())]})
        finally:
            fw.uninstall()
            if saved is None:
                os.environ.pop("BALL_ROBOT", None)
            else:
                os.environ["BALL_ROBOT"] = saved
        return r, fw, hashes

    # 1) 꺼짐: grasp.servo 없음 = false — 보낸 궤적(바이트) 전부 같다
    r0, fw0, h0 = go((20, -12), {})
    r1, fw1, h1 = go((20, -12), {"servo": False})
    check("손목 실시간 맞춤 꺼짐(grasp.servo 없음·false): 보낸 궤적 바이트 단위로 같음 · 손목 사진 수 같음",
          r0["ok"] and h0 == h1 and len(h0) > 3 and fw0.shots == fw1.shots,
          f"구간 {len(h0)}개 · 사진 {fw0.shots}/{fw1.shots}")
    # 2) 켬(가짜 로봇): hover 맞춤 이동·닫기 직전 위로 없이 한 번에 닫기 직전 높이 → 쥠
    r2, fw2, h2 = go((20, -12), {"servo": True}, px_gain=1.6)
    at = (r2.get("wrist") or [{}])[0].get("attempts", [{}])[0]
    names = [n for n, _ in h2]
    sv = at.get("servo") or {}
    check("손목 실시간 맞춤 켬(가짜, 화면 감도 1.6배): 20mm 어긋난 블록 → 한 번에 내려가 쥠 · hover 맞춤 이동·닫기 직전 위로 없음",
          r2["ok"] and fw2.grasps and fw2.grasps[-1]["held"] and fw2.grasps[-1]["dist_mm"] < 10 and sv.get("ok")
          and not any("맞춤 이동" in n or "닫기 직전 위로" in n for n in names),
          f"서보 {sv} · 닫을 때 블록과 {fw2.grasps[-1]['dist_mm'] if fw2.grasps else '-'}mm · 구간 {names}")
    # 3) 블록을 잃음(사진 4장째부터 빈 화면) → 멈추고 hover 로 올라와 지금 방식으로 쥠
    orig_read = servo.FakeCam.read
    cnt = {"n": 0}

    def lose(self):
        cnt["n"] += 1
        img = orig_read(self)
        return img if cnt["n"] <= 3 else np.full_like(img, 225)
    servo.FakeCam.read = lose
    try:
        r3, fw3, h3 = go((20, -12), {"servo": True})
    finally:
        servo.FakeCam.read = orig_read
    at3 = (r3.get("wrist") or [{}])[0].get("attempts", [{}])[0]
    sv3 = at3.get("servo") or {}
    names3 = [n for n, _ in h3]
    check("손목 실시간 맞춤: 블록을 잃으면(빈 사진 3장) 멈추고 hover 로 올라와 지금 방식(닫기 직전 위로)으로 쥠",
          r3["ok"] and fw3.grasps and fw3.grasps[-1]["held"] and sv3.get("ok") is False and "연속 없음" in str(sv3.get("why"))
          and any("닫기 직전 위로" in n for n in names3), f"서보 {sv3} · 구간 {names3[:6]}")

    # 4) 컨테이너 쪽 루프: 틱 한도 안에서 따라가고, 호스트가 조용하면 그 자리, 끝 요청이면 done
    class _Obs:
        def __init__(self, q, ts):
            self.joint_positions, self.timestamp = q, ts

    class _Bot:
        def __init__(self, q):
            self.q, self.ts = np.asarray(q, float), 0

        def get_observation(self):
            self.ts += 1
            return _Obs(self.q.copy(), self.ts)

        def send_action(self, cmd, goal_time=None):
            self.q = np.asarray(cmd, float).copy()
    clk = {"t": 0.0}
    msgs = [{"target": [10, 0, 0, 0, 0, 50]}] + [None] * 40 + [{"end": 1}]
    bot = _Bot([0, 0, 0, 0, 0, 50])
    log_q: list = []
    res = servo.servo_loop(bot, lambda: msgs.pop(0) if msgs else None, rate_hz=30, step_body=0.5, step_grip=1.0,
                           track_max=5.0, track_ticks=3, lag_ticks=3, stale_s=0.5, start_tol=5.0,
                           start=[0, 0, 0, 0, 0, 50], max_s=10, host_timeout_s=0.6, settle_s=0.1,
                           emit=lambda ev: log_q.append(ev), clock=lambda: clk["t"],
                           sleep=lambda d: clk.__setitem__("t", clk["t"] + d))
    check("서보 컨테이너 루프: 틱마다 한도(0.5) 안에서 목표로 · 호스트 0.6초 조용하면 그 자리 · 끝 요청 → done",
          res["ok"] and res["reason"] == "done" and 5.0 < bot.q[0] < 10.0 and log_q[-1]["ev"] == "end",
          f"끝 관절0 {bot.q[0]:.2f}(목표 10, 0.6초 × 30틱 × 0.5 = 9 에서 멈춤) · {res['reason']}")
    msgs2 = [{"target": [1, 0, 0, 0, 0, 50]}]
    clk["t"] = 0.0
    res2 = servo.servo_loop(_Bot([0, 0, 0, 0, 0, 50]), lambda: msgs2.pop(0) if msgs2 else None, rate_hz=30, step_body=0.5,
                            step_grip=1.0, track_max=5.0, track_ticks=3, lag_ticks=3, stale_s=0.5, start_tol=5.0,
                            start=[0, 0, 0, 0, 0, 50], max_s=10, host_timeout_s=0.6, settle_s=0.1, emit=lambda ev: None,
                            clock=lambda: clk["t"], sleep=lambda d: clk.__setitem__("t", clk["t"] + d))
    check("서보 컨테이너 루프: 호스트가 3초(5배) 넘게 조용하면 끝(host_silent)", res2["reason"] == "host_silent", res2["reason"])


def home_tests(real: dict, tmp: Path) -> None:
    """executor home(go_home) — 16:18 실물 리허설이 not_at_home 으로 거부된 자세에서 홈으로(가짜 로봇)."""
    import copy
    import os
    import subprocess

    from robot.coord_pick import executor as ex
    from robot.coord_pick import fake, stream

    s = Setup.load(real, use_studio=False)
    sf = ex.safety(real)
    lim = ex.limits(s, sf, sf["home_deg_s"])
    home = np.array(real["home_n"], float)
    measured = np.array([1.0, -35.92, 2.91, 50.04, 4.91, 55.26])  # 10-22 16:18 not_at_home(off 16.38)
    quiet = io.StringIO()

    def ret(start, **extra):
        clock = fake.FakeClock()
        t0 = clock.now()
        rob = fake.FakeSharedRobot(start, clock)
        payload = {"run_id": "t", "target": home.tolist(), "limits": lim, "keep_gripper": True, "already_tol": 1.0,
                   "max_off": round(sf["home_max_deg"] / 1.8, 2), **extra}
        events = []
        with contextlib.redirect_stdout(quiet):
            orig = stream.emit_json
            stream.emit_json = events.append
            try:
                rc = stream.container_return(payload, robot_factory=lambda: rob, clock=clock.now, sleep=clock.sleep)
            finally:
                stream.emit_json = orig
        return rc, rob, events, clock.now() - t0

    rc, rob, ev, t_end = ret(measured)
    sends = np.array(rob.sends)
    step = np.abs(np.diff(sends[:, :5], axis=0)).max() * 1.8 * 30 if len(sends) > 1 else 0
    check("홈 복귀: 16:18 거부 자세 → 홈(8°/s, 그리퍼 그대로)", rc == 0 and np.abs(rob.q[:5] - home[:5]).max() < 0.3
          and np.allclose(sends[:, 5], measured[5]) and step <= 8.01,
          f"{len(sends)}틱 {t_end:.1f}초, 최대 {step:.2f}°/s, 그리퍼 {sends[-1, 5]:.2f}, 측정 {ev[1]['q'] if len(ev) > 1 else '-'}")
    far = measured.copy()
    far[1] += 70 / 1.8
    rc2, rob2, ev2, _ = ret(far)
    rc3, rob3, _, _ = ret(measured, keep_gripper=False, target=np.r_[home[:5], real["gripper"]["open"]].tolist())
    rc4, rob4, ev4, _ = ret(home + [0, 0.5, 0, 0, 0, 0])
    check("홈 복귀: 60° 넘게 멀면 거부 · --open · 이미 홈",
          rc2 == 2 and not rob2.sends and ev2[-1]["reason"] == "too_far" and rc3 == 0
          and abs(rob3.sends[-1][5] - real["gripper"]["open"]) < 1e-6 and rc4 == 0 and not rob4.sends
          and ev4[-1]["reason"] == "already_home",
          f"먼 자세 → {ev2[-1]['reason']}(보냄 0) · 벌림 → 그리퍼 {rob3.sends[-1][5]:.0f} · 홈 근처 → {ev4[-1]['reason']}")

    # 컨테이너에 넣는 스크립트 그대로(별도 프로세스, 가짜 로봇·실시간) — entry 가 container_return 인지
    payload = {"run_id": "selftest-home", "robot_name": "fake", "limits": {**lim, "step_body": [v * 4 for v in lim["step_body"]]},
               "target": home.tolist(), "keep_gripper": True, "already_tol": 1.0, "max_off": 33.33,
               "fake_start": (home + [0, 4, -3, 2, 0, 3]).tolist()}
    r = subprocess.run([sys.executable, "-I", "-", "coord_pick", "selftest-home"], input=ex.build_script(
        payload, fake_kwargs={}, entry="container_return"), capture_output=True, text=True, timeout=60)
    end = [json.loads(x) for x in r.stdout.splitlines() if x.startswith("{")][-1]
    check("홈 복귀: 컨테이너용 스크립트(container_return) 실행", r.returncode == 0 and end.get("reason") == "done",
          f"종료 {r.returncode}, {end.get('reason')}, 보냄 {end.get('sent')}")

    saved = os.environ.pop("BALL_ROBOT", None)
    called = []
    orig = ex._run_real
    ex._run_real = lambda *a, **k: called.append(k) or {"reason": "done", "run_id": "x", "run": {"end": {}},
                                                         "measured": {"q": measured.tolist()}}
    try:
        d = ex.go_home(config=copy.deepcopy(real), log_dir=tmp)
        n1 = ex.go_home(dry_run=False, config=copy.deepcopy(real), log_dir=tmp)
        os.environ["BALL_ROBOT"] = "1"
        y = ex.go_home(dry_run=False, open_gripper=True, config=copy.deepcopy(real), log_dir=tmp)
    finally:
        ex._run_real = orig
        os.environ.pop("BALL_ROBOT", None)
        if saved is not None:
            os.environ["BALL_ROBOT"] = saved
    k = called[0] if called else {}
    check("홈 복귀: go_home 문(BALL_ROBOT=1)·인자", d["ok"] and d["mode"] == "dry_run" and n1["reason"] == "refused"
          and y["ok"] and y["mode"] == "real" and k.get("entry") == "container_return"
          and k["extra"]["keep_gripper"] is False and k["extra"]["target"][5] == real["gripper"]["open"]
          and abs(k["extra"]["max_off"] - 33.33) < 0.01 and len(called) == 1,
          f"dry-run → {d['reason']} · BALL_ROBOT 없음 → {n1['reason']} · 있음 → {y['mode']}·{y['reason']}, max_off {k.get('extra', {}).get('max_off')}")


if __name__ == "__main__":
    sys.exit(main())
