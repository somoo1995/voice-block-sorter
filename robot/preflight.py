"""녹화·시연 전 점검 — 한 줄로 전부 본다. 아무것도 바꾸지 않는다(읽기만, 카메라도 열지 않는다).

    .venv/bin/python -m robot.preflight             # 시연 기준
    .venv/bin/python -m robot.preflight --record    # 녹화·학습 날 기준
    .venv/bin/python -m robot.preflight --quick     # 정책 설정 검사(--print_config, 정책당 약 4초) 생략
    .venv/bin/python -m robot.preflight --json      # 기계용

보는 것: 팔로워·리더 USB(by-id) · 카메라 2대(by-id, 누가 열고 있는지) · Studio 컨테이너·API·세션·장치 잠금 ·
컨테이너 안 장치 · 학습 대기열 · policies.json 4정책 모델 연결 · OVMS 4모델 · 대시보드 :7870 ·
트레이너 trainer-pc:8001 · 디스크 · 전원.

같은 상태도 기준에 따라 빨강(안됨)·노랑(주의)이 갈린다.
  시연      정책 4개 모델 연결 · Studio 세션 없음이 필수. 리더 팔·트레이너는 주의만.
  --record  리더 팔·트레이너가 필수. 모델 미연결·Studio 세션(녹화 중)은 문제 아님.
종료 코드: 안됨이 하나라도 있으면 1.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import unicodedata
import urllib.request
from pathlib import Path

import robot.run_policy as rp

OK, WARN, FAIL, SKIP = "정상", "주의", "안됨", "건너뜀"
PROJECT_DIR = rp.ROBOT_DIR.parent
DEVICES_CACHE = rp.STATE_DIR / "devices.json"  # Studio 가 꺼져 있을 때 쓸 "마지막으로 본 기대 장치"
OVMS_URL = "http://127.0.0.1:8100"
DASH_URL = f"http://127.0.0.1:{os.environ.get('UI_PORT', '7870')}/"
TRAINER_URL = os.environ.get("TRAINER_URL", "http://127.0.0.1:8001")
OVMS_MODELS = [os.environ.get("LLM_MODEL", "qwen3-4b"), os.environ.get("EMB_MODEL", "qwen3-embedding"),
               os.environ.get("RERANK_MODEL", "qwen3-reranker"), os.environ.get("STT_MODEL", "whisper")]
LLMS = ("qwen3-4b", "qwen3-8b")
STUDIO_COMPOSE_DIR = "~/dev/projects/physical-ai-studio/application/docker"
TRAINER_START = ("트레이너 PC 에서: cd ~/dev/projects/physical-ai-studio/application/backend && "
                 "setsid nohup ~/.local/bin/uv run --no-sync physicalai-trainer --host 0.0.0.0 --port 8001 "
                 "> /tmp/pai-trainer.log 2>&1 < /dev/null & (켜졌는지는 ss -ltn | grep 8001)")
DISK_FAIL_GB, DISK_WARN_GB = 10, 25  # 하루치 ≈ 데이터셋 4 + 모델 4(정책당 체크포인트·export·받은 압축 ≈ 2GB)

# root 로 /proc/*/fd 를 훑어 장치를 연 pid 를 찾는다(컨테이너 프로세스는 uid 가 달라 sudo 가 있어야 보인다)
_SCAN = r"""
import os, sys
rdev, me, out = int(sys.argv[1]), sys.argv[2], []
for pid in filter(str.isdigit, os.listdir("/proc")):
    if pid == me:
        continue
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except OSError:
        continue
    for fd in fds:
        try:
            st = os.stat(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if st.st_rdev == rdev and (st.st_mode & 0o170000) == 0o020000:
            out.append(pid)
            break
print(" ".join(out))
"""


def _row(title: str, status: str, detail: str, fix: str | None = None) -> dict:
    return {"title": title, "status": status, "detail": detail, "fix": fix}


def _http(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _kst(iso: str | None) -> str:
    if not iso:
        return "?"
    t = dt.datetime.fromisoformat(iso)
    return (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).astimezone(rp.KST).strftime("%m-%d %H:%M")


# ---------------------------------------------------------------- 기대 장치 (Studio 환경)

def _devices_from_studio(cfg: dict) -> dict:
    """정책이 쓰는 환경(연결된 모델의 데이터셋 환경)의 팔·카메라 시리얼. 연결된 모델이 없으면 모든 환경."""
    envs: dict[str, str] = {}
    for p in cfg["policies"].values():
        if not p.get("model_id") or p.get("deprecated"):
            continue
        try:
            m = rp._get(cfg, f"/api/models/{p['model_id']}", timeout=5)["model"]
            eid = p.get("environment_id") or (
                rp._get(cfg, f"/api/dataset/{m['dataset_id']}", timeout=5).get("environment_id")
                if m.get("dataset_id") else None)
        except rp.PolicyError:
            continue
        if eid:
            envs[eid] = m["project_id"]
    if not envs:
        for proj in rp._get(cfg, "/api/projects", timeout=5):
            for e in rp._get(cfg, f"/api/projects/{proj['id']}/environments", timeout=5):
                envs[e["id"]] = proj["id"]
    out = {"saved_at": rp._iso(rp._now()), "environments": [], "follower": {}, "leader": {}, "cameras": {}}
    for eid, pid in envs.items():
        e = rp._get(cfg, f"/api/projects/{pid}/environments/{eid}", timeout=5)
        out["environments"].append(e.get("name"))
        for r in e.get("robots", []):
            for role, robot in (("follower", r.get("robot") or {}),
                                ("leader", (r.get("tele_operator") or {}).get("robot") or {})):
                serial = (robot.get("payload") or {}).get("serial_number")
                if serial:
                    out[role][serial] = robot.get("name")
        for c in e.get("cameras", []):
            serial = (c.get("fingerprint") or {}).get("serial")
            if serial:
                out["cameras"][serial] = c.get("name")
    return out


def expected_devices(cfg: dict, studio_up: bool) -> tuple[dict | None, str]:
    if studio_up:
        try:
            dev = _devices_from_studio(cfg)
            rp._write(DEVICES_CACHE, json.dumps(dev, ensure_ascii=False, indent=1))
            return dev, f"Studio 환경 {', '.join(map(str, dev['environments']))}"
        except (rp.PolicyError, KeyError, OSError):
            pass
    if DEVICES_CACHE.exists():
        dev = json.loads(DEVICES_CACHE.read_text(encoding="utf-8"))
        return dev, f"Studio 에서 못 읽어 {dev['saved_at'][:16].replace('T', ' ')} 에 읽은 값"
    return None, "기대 장치 모름(Studio 를 못 읽었고 기록도 없음)"


# ---------------------------------------------------------------- 장치

def _links(base: str) -> list[Path]:
    p = Path(base)
    return sorted(p.iterdir()) if p.is_dir() else []


def _busy(dev: str) -> list[int] | None:
    """dev 를 연 pid. sudo 가 안 되면 None(남의 프로세스는 못 봄)."""
    try:
        r = subprocess.run(["sudo", "-n", "/usr/bin/python3", "-I", "-S", "-c", _SCAN,
                            str(os.stat(dev).st_rdev), str(os.getpid())], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return [int(p) for p in r.stdout.split()] if r.returncode == 0 else None


def _cmd(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return "?"
    words = [w.decode(errors="replace") for w in raw if w]
    return " ".join(os.path.basename(w) if i == 0 else w for i, w in enumerate(words))[:70]


def check_arms(devs: dict | None, src: str, record: bool) -> tuple[list[dict], list[str]]:
    rows, found = [], []
    links = _links("/dev/serial/by-id")
    seen = ", ".join(l.name for l in links) or "/dev/serial/by-id 비어 있음"
    for role, title, required in (("follower", "팔로워 USB", True), ("leader", "리더 USB", record)):
        want = (devs or {}).get(role) or {}
        if not want:
            rows.append(_row(title, SKIP, f"{src} · 보이는 시리얼 장치: {seen}"))
            continue
        for serial, name in want.items():
            hit = next((l for l in links if serial in l.name), None)
            if hit:
                port = os.path.realpath(hit)
                found.append(port)
                rows.append(_row(title, OK, f"{name} {serial} → {port}"))
            else:
                need = "" if required else " (녹화·원격조작 때만 필요)"
                rows.append(_row(title, FAIL if required else WARN, f"{name} {serial} 없음{need} · 보이는 것: {seen}",
                                 f"{name} USB 를 꽂는다 → ls /dev/serial/by-id 에 …{serial}… 이 보여야 한다"))
    return rows, found


def check_cameras(devs: dict | None, src: str, record: bool) -> tuple[list[dict], list[str]]:
    rows, found = [], []
    links = _links("/dev/v4l/by-id")
    want = (devs or {}).get("cameras") or {}
    if not want:
        return [_row("카메라", SKIP, f"{src} · 보이는 것: {', '.join(l.name for l in links) or '없음'}")], found
    for serial, name in want.items():
        title = f"카메라 {name}"
        hit = next((l for l in links if serial in l.name and l.name.endswith("-video-index0")), None)
        if not hit:
            others = ", ".join(l.name.removeprefix("usb-") for l in links if l.name.endswith("-video-index0")) or "없음"
            rows.append(_row(title, FAIL, f"{serial} 없음 · 보이는 카메라: {others}",
                             f"'{name}' 카메라 USB 를 꽂는다 → ls /dev/v4l/by-id 에 …{serial}…-video-index0 이 보여야 한다"))
            continue
        dev = os.path.realpath(hit)
        found.append(dev)
        pids = _busy(dev)
        if pids is None:
            rows.append(_row(title, OK, f"{serial} → {dev} (누가 여는지는 sudo 가 없어 못 봄)"))
        elif not pids:
            rows.append(_row(title, OK, f"{serial} → {dev}, 아무도 안 엶"))
        else:
            who = "; ".join(f"pid {p} {_cmd(p)}" for p in pids)
            if record:
                rows.append(_row(title, OK, f"{serial} → {dev}, 여는 중: {who}"))
            else:
                rows.append(_row(title, WARN, f"{serial} → {dev}, 여는 중: {who}",
                                 "그 프로그램(Studio 미리보기·다른 창)을 닫는다 — 열려 있으면 비전 스냅샷·정책 실행이 카메라를 못 연다"))
    return rows, found


# ---------------------------------------------------------------- Studio

def check_studio_base(cfg: dict) -> tuple[list[dict], bool]:
    r = subprocess.run(rp._docker(cfg) + ["inspect", "-f",
                       "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}|{{.State.StartedAt}}",
                       cfg["container"]], capture_output=True, text=True, timeout=20)
    if r.returncode != 0 or not r.stdout.startswith("running"):
        state = r.stdout.strip().split("|")[0] if r.returncode == 0 else "없음"
        return [_row("Studio 컨테이너", FAIL, f"{cfg['container']} {state}",
                     f"cd {STUDIO_COMPOSE_DIR} && docker --context default compose up -d (기존 compose 그대로)"),
                _row("Studio API", SKIP, "컨테이너가 꺼져 있음")], False
    status, health, started = r.stdout.strip().split("|")
    rows = [_row("Studio 컨테이너", OK, f"{status} ({health or '상태검사 없음'}), {_kst(started)} 기동")]
    try:
        h = rp._get(cfg, "/api/health", timeout=3)
        ok = h.get("status") == "healthy"
        rows.append(_row("Studio API", OK if ok else FAIL, f"{cfg['studio_url']} {h.get('status')}",
                         None if ok else f"docker --context default logs --tail 50 {cfg['container']} 로 원인 확인"))
        return rows, ok
    except rp.PolicyError as e:
        rows.append(_row("Studio API", FAIL, str(e)[:120],
                         f"켠 직후면 잠시 뒤 다시. 아니면 docker --context default logs --tail 50 {cfg['container']}"))
        return rows, False


def check_studio_runtime(cfg: dict, host_paths: list[str], record: bool) -> list[dict]:
    rows = []
    sessions = rp._get(cfg, "/api/runtime/sessions", timeout=5)
    pr = rp._probe(cfg, "list", host_paths, timeout=20)
    names = ", ".join(f"{s.get('follower_name')}({s.get('status')})" for s in sessions)
    runs = [p for p in pr["procs"] if p["kind"] == "run"]
    others = [lk for lk in pr["device_locks"] if lk["owner"] != cfg["robot_name"]]
    ours = [lk for lk in pr["device_locks"] if lk["owner"] == cfg["robot_name"]]
    locks = ", ".join(f"{lk['device']}←{lk['owner']}(pid {lk['pid']})" for lk in pr["device_locks"]) or "없음"
    detail = f"세션 {len(sessions)}개{' ' + names if names else ''} · 장치 잠금 {locks}"
    if record:
        if runs or ours:
            rows.append(_row("Studio 세션·잠금", WARN, detail + " · 에이전트 정책이 팔을 쥐고 있음",
                             "정책 실행이 끝나고 10초 뒤에 Studio 녹화를 연다 (급하면 python -m robot.run_policy --stop)"))
        else:
            rows.append(_row("Studio 세션·잠금", OK, detail + (" (녹화 중이면 정상)" if sessions else "")))
    elif sessions:
        rows.append(_row("Studio 세션·잠금", FAIL, detail,
                         "Studio 의 원격조작·녹화·모델 실행 화면에서 ← 로 나온다 (탭만 닫으면 45초 뒤에 풀린다)"))
    elif others:
        rows.append(_row("Studio 세션·잠금", FAIL, detail,
                         "Studio 세션이 방금 끝났으면 10초 기다렸다 다시. 계속이면 python -m robot.run_policy --status"))
    elif runs:
        rows.append(_row("Studio 세션·잠금", WARN, detail + f" · 정책 실행 중 {runs[0]['run_id']}",
                         "끝나기를 기다리거나 python -m robot.run_policy --stop"))
    else:
        rows.append(_row("Studio 세션·잠금", OK, detail + (" (직전 실행의 소유 프로세스, 10초 뒤 끝남)" if ours else "")))
    if not host_paths:
        rows.append(_row("컨테이너 안 장치", SKIP, "호스트에 꽂힌 팔·카메라가 없음"))
    elif pr["missing_paths"]:
        rows.append(_row("컨테이너 안 장치", FAIL, f"컨테이너 안에 없음: {' '.join(pr['missing_paths'])}",
                         f"사용자: cd {STUDIO_COMPOSE_DIR} && docker --context default compose restart"))
    else:
        rows.append(_row("컨테이너 안 장치", OK, " ".join(host_paths)))
    return rows


def check_training_queue(cfg: dict) -> dict:
    jobs = [j for j in rp._get(cfg, "/api/jobs", timeout=5) if j.get("type") == "training"]
    live = [j for j in jobs if j.get("status") in ("running", "pending")]
    names: dict[str, str] = {}

    def ds(j: dict) -> str:
        did = (j.get("payload") or {}).get("dataset_id")
        if did and did not in names:
            try:
                names[did] = rp._get(cfg, f"/api/dataset/{did}", timeout=5).get("name")
            except rp.PolicyError:
                names[did] = did[:8]
        return names.get(did, "?")

    if not live:
        done = [j for j in jobs if j.get("end_time")]
        last = max(done, key=lambda j: j["end_time"], default=None)
        tail = f" · 마지막: {ds(last)} {last.get('status')} {_kst(last['end_time'])}" if last else ""
        return _row("Studio 학습 대기열", OK, "도는 학습 없음" + tail)
    parts = [f"{ds(j)} {j['status']} {j.get('progress', 0)}%" for j in live]
    return _row("Studio 학습 대기열", OK, " · ".join(parts) + " (Stop 누르지 말 것 — 모델이 안 남는다)")


# ---------------------------------------------------------------- 정책

def check_policies(cfg: dict, studio_up: bool, record: bool, quick: bool) -> list[dict]:
    def one(name: str) -> dict:
        p = cfg["policies"][name]
        title = f"정책 {name}"
        if p.get("deprecated"):  # 10-23 13:35 사용자 결정: 시연 정책은 handoff 하나 — 나머지는 점검하지 않는다
            return _row(title, SKIP, f"폐기 — {p.get('deprecated_why') or '시연에서 안 씀'}")
        if not p.get("model_id"):
            return _row(title, WARN if record else FAIL, "모델 연결 안 됨",
                        f"학습이 끝나면 .venv/bin/python -m robot.run_policy --models → --link {name} <모델ID> (README 5.5)")
        mid = p["model_id"][:8]
        if not studio_up:
            return _row(title, SKIP, f"모델 {mid} 연결됨 — Studio 를 못 읽어 확인 못 함")
        try:
            plan = rp.build_plan(name, require_devices=False)
        except rp.PolicyError as e:
            return _row(title, FAIL, f"{e.code}: {str(e)[:110]}",
                        f".venv/bin/python -m robot.run_policy --models 로 모델을 다시 찾아 --link {name} <모델ID>")
        m = plan["model"]
        detail = f"모델 {mid} · 데이터셋 {m['dataset']} · timeout {plan['timeout_s']:g}s"
        if not quick:
            ok, out = rp._validate(plan)
            if not ok:
                return _row(title, FAIL, detail + " · 설정 검사 실패: " + out.strip().splitlines()[-1][:80],
                            f".venv/bin/python -m robot.run_policy {name} --dry-run 으로 전체 오류 보기")
            detail += " · 설정 검사 통과"
        if m.get("temporary"):
            return _row(title, WARN, detail + " · 임시 모델(블록 선별 모델 아님)",
                        f"학습한 모델로 .venv/bin/python -m robot.run_policy --link {name} <모델ID>")
        return _row(title, OK, detail)

    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(one, rp.NAMES))


# ---------------------------------------------------------------- 로컬 AI·대시보드·트레이너·디스크·전원

def check_ovms() -> dict:
    title = "OVMS 모델"
    try:
        ids = [m["id"] for m in _http(f"{OVMS_URL}/v3/models")["data"]]
        cfgs = _http(f"{OVMS_URL}/v1/config")
    except (OSError, ValueError, KeyError) as e:
        return _row(title, FAIL, f"{OVMS_URL} 응답 없음 ({type(e).__name__})",
                    "~/dev/projects/ball-sorter/start_ovms.sh (첫 적재 7~14초, SETUP.md §1)")
    state = {k: (v.get("model_version_status") or [{}])[0].get("state") for k, v in cfgs.items()}
    llms = [i for i in ids if i in LLMS]
    missing = [w for w in OVMS_MODELS if w not in ids]
    not_ready = [w for w in OVMS_MODELS if w in ids and state.get(w) != "AVAILABLE"]
    listed = ", ".join(f"{i}" + ("" if state.get(i) == "AVAILABLE" else f"({state.get(i)})") for i in ids)
    if len(llms) > 1:
        return _row(title, FAIL, f"LLM 두 개({', '.join(llms)})가 함께 떠 있다 — GPU HANG 위험(SETUP.md §8-3)",
                    "systemctl --user stop ball-ovms && ~/dev/projects/ball-sorter/start_ovms.sh (권장 구성 config.json)")
    if missing or not_ready:
        return _row(title, FAIL, f"없음 {missing} · 준비 안 됨 {not_ready} · 떠 있는 것: {listed}",
                    "적재 중이면 잠시 뒤 다시. 아니면 logs/ball-ovms.log 확인, 구성은 config.json(SETUP.md §1·§4)")
    return _row(title, OK, listed)


def check_dashboard(record: bool) -> dict:
    title = "대시보드"
    try:
        with urllib.request.urlopen(DASH_URL, timeout=3) as r:
            code = r.status
    except OSError as e:
        return _row(title, FAIL, f"{DASH_URL} 응답 없음 ({type(e).__name__})",
                    "~/dev/projects/ball-sorter/ui/start_ui.sh (로그 ui/ui.log)")
    # start_ui.sh 는 systemd-run --user 유닛 ball-ui 로 띄운다 → 그 유닛에 넘긴 환경변수로 모드를 본다
    r = subprocess.run(["systemctl", "--user", "show", "ball-ui", "-p", "ActiveState", "-p", "Environment"],
                       capture_output=True, text=True, timeout=5)
    props = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    env = dict(kv.split("=", 1) for kv in props.get("Environment", "").split() if "=" in kv)
    if props.get("ActiveState") != "active":
        mode = "모름(ball-ui 유닛이 아님)"
    elif env.get("BALL_MOCK") == "1":
        mode = "가짜(BALL_MOCK=1)"
    elif env.get("BALL_ROBOT") == "1":
        mode = "실제 로봇(BALL_ROBOT=1)"
    else:
        mode = f"로봇 가짜(BALL_ROBOT={env.get('BALL_ROBOT', '없음')})"
    detail = f"{DASH_URL} HTTP {code} · 모드 {mode}"
    if not record and env.get("BALL_ROBOT") != "1":
        return _row(title, WARN, detail,
                    "./start_demo.sh --restart (BALL_ROBOT=1 로 다시 띄움)")
    return _row(title, OK, detail)


def check_trainer(record: bool) -> dict:
    title = "트레이너 trainer-pc"
    try:
        h = _http(f"{TRAINER_URL}/health")
        gpus = _http(f"{TRAINER_URL}/devices")
        st = _http(f"{TRAINER_URL}/storage")
    except (OSError, ValueError) as e:
        return _row(title, FAIL if record else WARN,
                    f"{TRAINER_URL} 응답 없음 ({type(e).__name__}){'' if record else ' (시연엔 필요 없음)'}",
                    TRAINER_START)
    gpu = ", ".join(f"{g.get('name')} {'학습 중' if g.get('busy') else '놀고 있음'}" for g in gpus) or "GPU 없음"
    free = st.get("free_bytes", 0) / 1e9
    detail = f"{h.get('status')} · {gpu} · 디스크 여유 {free:.0f}GB"
    if h.get("status") != "healthy" or not gpus:
        return _row(title, FAIL if record else WARN, detail, TRAINER_START)
    if free < DISK_FAIL_GB:
        return _row(title, FAIL if record else WARN, detail,
                    "데스크톱 ~/.local/share/physicalai-trainer/archives 의 다 받아 간 압축 정리(사용자 승인 후)")
    return _row(title, OK, detail)


def check_disk() -> dict:
    free = shutil.disk_usage(PROJECT_DIR).free / 1e9
    try:  # Studio 볼륨(/var/lib/docker)이 다른 디스크면 따로 본다
        other = os.stat("/var/lib/docker").st_dev != os.stat(PROJECT_DIR).st_dev
    except OSError:
        other = False
    detail = f"젠북 {PROJECT_DIR} 여유 {free:.0f}GB" + (" (Studio 볼륨은 다른 디스크 — 따로 확인)" if other else "")
    if free < DISK_FAIL_GB:
        return _row("디스크", FAIL, detail, "큰 것부터 확인: du -xh --max-depth=2 ~ | sort -h | tail (지우기는 승인 후)")
    if free < DISK_WARN_GB:
        return _row("디스크", WARN, detail + " · 하루치(데이터셋·모델 4개) 약 8GB", None)
    return _row("디스크", OK, detail)


def check_power() -> dict:
    ac, bat = None, None
    for d in Path("/sys/class/power_supply").glob("*"):
        try:
            kind = (d / "type").read_text().strip()
            if kind == "Mains":
                ac = bool(ac) or (d / "online").read_text().strip() == "1"
            elif kind == "Battery" and d.name.startswith("BAT"):  # hid-*-battery 는 펜·키보드
                bat = (d / "capacity").read_text().strip()
        except OSError:
            continue
    if ac is None:
        return _row("전원", SKIP, "어댑터 상태를 못 읽음")
    if ac:
        return _row("전원", OK, f"어댑터 연결{f' · 배터리 {bat}%' if bat else ''}")
    return _row("전원", WARN, f"배터리로 동작{f' {bat}%' if bat else ''}",
                "어댑터를 꽂는다 — 녹화·학습 대기가 몇 시간 걸리고, 배터리일 때는 절전 정책이 재울 수 있다")


# ---------------------------------------------------------------- 실행·출력

def run_checks(record: bool, quick: bool) -> dict:
    cfg = rp.load_config()
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        f_studio = ex.submit(check_studio_base, cfg)
        f_tail = [ex.submit(check_ovms), ex.submit(check_dashboard, record), ex.submit(check_trainer, record),
                  ex.submit(check_disk), ex.submit(check_power)]  # 출력 순서대로
        studio_rows, studio_up = f_studio.result()
        devs, src = expected_devices(cfg, studio_up)
        arm_rows, ports = check_arms(devs, src, record)
        cam_rows, cams = check_cameras(devs, src, record)
        f_pol = ex.submit(check_policies, cfg, studio_up, record, quick)
        rows = arm_rows + cam_rows + studio_rows
        if studio_up:
            try:
                rows += check_studio_runtime(cfg, ports + cams, record)
                rows.append(check_training_queue(cfg))
            except (rp.PolicyError, subprocess.SubprocessError, ValueError) as e:
                rows.append(_row("Studio 세션·잠금", FAIL, f"조사 실패: {str(e)[:120]}",
                                 "python -m robot.run_policy --status 로 다시 확인"))
        rows += f_pol.result() + [f.result() for f in f_tail]
    counts = {s: sum(r["status"] == s for r in rows) for s in (FAIL, WARN, OK, SKIP)}
    return {"mode": "record" if record else "demo", "checked_at": rp._iso(rp._now()), "devices_from": src,
            "ready": counts[FAIL] == 0, "counts": counts, "rows": rows}


def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, n: int) -> str:
    return s + " " * max(0, n - _width(s))


def print_report(res: dict) -> None:
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    paint = {FAIL: "\033[31;1m", WARN: "\033[33;1m", OK: "\033[32m", SKIP: "\033[90m"}

    def c(status: str, text: str) -> str:
        return f"{paint[status]}{text}\033[0m" if color else text

    mode = "녹화·학습 기준(--record)" if res["mode"] == "record" else "시연 기준"
    print(f"점검 {res['checked_at'][:16].replace('T', ' ')} · {mode} · 기대 장치: {res['devices_from']}")
    tw = max(_width(r["title"]) for r in res["rows"]) + 2
    print(f"{_pad('상태', 8)}{_pad('항목', tw)}내용")
    for r in res["rows"]:
        print(f"{c(r['status'], _pad(r['status'], 8))}{_pad(r['title'], tw)}{r['detail']}")
    fixes = [r for r in res["rows"] if r["fix"] and r["status"] in (FAIL, WARN)]
    if fixes:
        print("\n고치는 법")
        for r in sorted(fixes, key=lambda r: r["status"] != FAIL):
            print(f"  {c(r['status'], '[' + r['status'] + ']')} {r['title']} — {r['fix']}")
    n = res["counts"]
    verdict = "준비됨" if res["ready"] else "준비 안 됨"
    print(f"\n{c(FAIL if n[FAIL] else OK, verdict)} — 안됨 {n[FAIL]} · 주의 {n[WARN]} · 정상 {n[OK]} · 건너뜀 {n[SKIP]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.preflight", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", action="store_true", help="녹화·학습 날 기준(리더·트레이너 필수, 모델 미연결은 주의)")
    ap.add_argument("--quick", action="store_true", help="정책 설정 검사(--print_config) 생략")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    res = run_checks(a.record, a.quick)
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:
        print_report(res)
    return 0 if res["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
