"""정책 실행기 — Physical AI Studio 컨테이너 안에서 `physicalai run` 으로 ACT 정책을 돌린다.

계약서 3절:
    run_policy(name, timeout_s=40) -> dict
    stop() -> None

CLI (프로젝트 루트에서, 프로젝트 .venv 로):
    python -m robot.run_policy pick_red --dry-run      # 로봇 안 움직임. 실행될 명령·설정만 출력
    python -m robot.run_policy pick_red --yes          # 실제 실행 (사용자 입회 때만)
    python -m robot.run_policy --stop                  # 실행 중인 정책 즉시 중단(그 자리에 멈춤)
    python -m robot.run_policy --status                # 실행 중인 정책·로봇 잠금·Studio 세션
    python -m robot.run_policy --models                # Studio 모델 목록 (policies.json 연결용)
    python -m robot.run_policy --link pick_red <모델ID> # policies.json 에 모델 연결

동작 요약 (README.md 2절에 근거와 함께 자세히):
- 설정은 실행 때마다 Studio API(GET)로 새로 만든다: 환경의 runtime-config 를 받아
  action_source 를 PolicySource(OpenVINO)로 바꾸고, 팔로워 포트를 시리얼(by-id)로 다시 찾는다.
- 컨테이너 안 파일·볼륨에는 아무것도 쓰지 않는다(설정은 --config 에 JSON 문자열로 넘김).
- 끝나는 조건은 시간(--run.duration_s)·stop()·오류뿐이다. ACT 에는 "다 했다" 신호가 없다.
- stop() 은 실행 프로세스를 SIGKILL 한다 → 팔은 마지막 자세에서 토크를 유지한 채 멈춘다.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import yaml

from robot import grasp_stop, home_stop

ROBOT_DIR = Path(__file__).resolve().parent
POLICIES_PATH = ROBOT_DIR / "policies.json"
LOG_DIR = ROBOT_DIR / "logs"
STATE_DIR = ROBOT_DIR / ".state"
KST = dt.timezone(dt.timedelta(hours=9))
NAMES = ("pick_defect", "pick_red", "pick_blue", "handoff", "grasp")

# 컨테이너 안 경로(이미지 0.3.0-dev-0874e5d2 기준)
CPY = "/app/application/backend/.venv/bin/python"
PHYSICALAI = "/app/application/backend/.venv/bin/physicalai"
# 우리 실행에만 붙는 환경변수. 자식(로봇 소유·카메라 publisher 프로세스)도 물려받아서 찾을 수 있다.
MARKER = "BALL_SORTER_RUN"
RETURN_MOVE_S = 2.5  # physicalai.runtime.core._RETURN_DURATION_S

_stop_requested = threading.Event()
_stop_kind = ["kill:sent"]  # stop() 이 남긴 상태 "<kill|home>:<sent|pending>" (신호가 실제 프로세스에 갔는지)


class PolicyError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _now() -> dt.datetime:
    return dt.datetime.now(KST)


def _iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="seconds")


def load_config() -> dict:
    return json.loads(POLICIES_PATH.read_text(encoding="utf-8"))


def _settings(cfg: dict, name: str) -> dict:
    """defaults 위에 정책별 값을 덮은 것."""
    return {**cfg["defaults"], **{k: v for k, v in cfg["policies"][name].items() if v is not None}}


# ---------------------------------------------------------------- Studio API (GET 만)

def _get(cfg: dict, path: str, *, raw: bool = False, timeout: float = 10):
    url = cfg["studio_url"].rstrip("/") + path
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        raise PolicyError("studio_api", f"Studio GET {path} → HTTP {e.code} {e.read()[:200]!r}") from e
    except OSError as e:
        raise PolicyError("studio_down", f"Studio({cfg['studio_url']}) 접속 안 됨: {e}") from e
    return data.decode() if raw else json.loads(data)


# ---------------------------------------------------------------- docker

def _docker(cfg: dict) -> list[str]:
    # 젠북 docker CLI 의 기본 context 는 Docker Desktop(desktop-linux)이라 Studio 가 안 보인다 → context 고정
    return ["docker", "--context", cfg["docker_context"]]


def _container_running(cfg: dict) -> bool:
    r = subprocess.run(_docker(cfg) + ["inspect", "-f", "{{.State.Running}}", cfg["container"]],
                       capture_output=True, text=True, timeout=20)
    return r.returncode == 0 and r.stdout.strip() == "true"


# 컨테이너 안(appuser)에서 돌리는 조사·중단 스크립트. 파일은 쓰지 않는다.
_PROBE = r"""
import json, os, signal, sys, tempfile
marker, action, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
procs = []
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        cmd = [c.decode(errors="replace") for c in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if c]
    except OSError:
        continue
    run_id = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(marker.encode() + b"=")), None)
    if run_id is None:
        continue
    if "--print_config" in cmd:
        kind = "validate"  # 설정 검사(dry-run·preflight). 로봇을 안 쥐므로 실행 중으로 세지도, 끊지도 않는다
    elif len(cmd) > 2 and cmd[1].endswith("/physicalai") and cmd[2] == "run":
        kind = "run"
    elif "coord_pick" in cmd[1:5]:
        kind = "run"  # 좌표 집기 실행기(python -I - coord_pick <run_id>) — stop() 이 같이 끊는다
    elif any("_owner_worker" in c for c in cmd):
        kind = "robot_owner"
    elif any("_publisher_worker" in c for c in cmd):
        kind = "camera_publisher"
    else:
        kind = "other"
    procs.append({"pid": int(pid), "run_id": run_id, "kind": kind})
killed = []
if action.startswith("kill:"):
    sig = getattr(signal, action.split(":", 1)[1])
    for p in procs:
        if p["kind"] == "run":
            try:
                os.kill(p["pid"], sig)
                killed.append(p["pid"])
            except OSError:
                pass
xdg = os.environ.get("XDG_RUNTIME_DIR")
lock_dir = (os.path.join(xdg, "physicalai", "robot-locks") if xdg
            else os.path.join(tempfile.gettempdir(), f"physicalai-{os.getuid()}", "robot-locks"))
locks = []
if os.path.isdir(lock_dir):
    for f in os.listdir(lock_dir):
        try:
            d = json.load(open(os.path.join(lock_dir, f)))
        except Exception:
            continue
        pid = d.get("pid") if isinstance(d, dict) else None
        if not (isinstance(d, dict) and d.get("kind") == "device" and isinstance(pid, int)):
            continue
        try:
            st = open(f"/proc/{pid}/status").read()
        except OSError:
            continue
        # Studio 컨테이너의 PID 1(uv run)은 끝난 자식을 거두지 않아 소유 프로세스가 좀비로 남는다 → 좀비는 잠금 아님
        if "\nState:\tZ" not in st:
            locks.append({"device": d.get("identity"), "owner": d.get("owner_name"), "pid": pid})
print(json.dumps({"procs": procs, "killed": killed, "device_locks": locks,
                  "missing_paths": [p for p in paths if not os.path.exists(p)]}))
"""


def _probe(cfg: dict, action: str = "list", paths: list[str] = (), timeout: float = 20) -> dict:
    argv = _docker(cfg) + ["exec", "-u", "appuser", cfg["container"], CPY, "-I", "-c", _PROBE, MARKER, action, *paths]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise PolicyError("docker", f"컨테이너 조사 실패: {r.stderr.strip()[-300:]}")
    return json.loads(r.stdout)


# ---------------------------------------------------------------- 설정 만들기

def _host_dev_by_serial(kind: str, serial: str) -> str | None:
    """by-id 링크(시리얼)로 지금 붙은 장치 경로를 찾는다. ttyACM·video 번호는 꽂는 순서로 바뀐다."""
    base = Path("/dev/serial/by-id") if kind == "serial" else Path("/dev/v4l/by-id")
    if not base.is_dir():
        return None
    for link in sorted(base.iterdir()):
        if serial and serial in link.name and (kind == "serial" or link.name.endswith("-video-index0")):
            return os.path.realpath(link)
    return None


def episode_stats(cfg: dict, dataset_id: str | None) -> dict | None:
    """학습 데이터 에피소드 길이(초). ACT 는 스스로 멈추지 않으니 timeout_s 는 이것에 맞춘다."""
    if not dataset_id:
        return None
    eps = _get(cfg, f"/api/dataset/{dataset_id}/episodes")
    secs = sorted(e["length"] / e["fps"] for e in eps if e.get("fps"))
    if not secs:
        return None
    p90 = secs[min(len(secs) - 1, int(0.9 * len(secs)))]
    return {"episodes": len(secs), "median_s": round(secs[len(secs) // 2], 1), "p90_s": round(p90, 1),
            "max_s": round(secs[-1], 1), "suggested_timeout_s": int(p90 + 3.999)}


EXECUTION_DEFAULTS = {"mode": "sync", "request_threshold": 0.5, "smoother_frames": None}


def execution_settings(s: dict) -> dict:
    """policies.json 정책 항목의 execution(없으면 Studio 기본 = SyncExecution 0.5 · LerpSmoother 5틱).
    mode "sync"(제어 루프 안에서 추론 — 추론 동안 틱이 멈춤) | "async"(뒤 스레드 추론, 그동안 쓴 틱만큼 새 청크를 건너뜀).
    request_threshold = 남은 동작이 청크의 이 비율 아래면 다시 추론(0.5 = 100틱 청크를 50틱마다, 0.8 = 20틱마다).
    smoother_frames = 새 청크와 남은 동작을 섞는 틱 수(LerpSmoother duration_frames, None = 5)."""
    out = {**EXECUTION_DEFAULTS, **{k: v for k, v in (s.get("execution") or {}).items() if k in EXECUTION_DEFAULTS}}
    if out["mode"] not in ("sync", "async") or not 0.0 <= float(out["request_threshold"]) < 1.0:
        raise PolicyError("bad_execution", f"execution 값이 이상하다: {out}")
    return out


def _policy_source(export_dir: str, device: str, task: str | None, execution: dict | None = None) -> dict:
    """Studio 의 policy_source_fragment()(runtime/config_builder.py)와 같은 내용 + execution(정책별 재추론·섞기)."""
    ex = {**EXECUTION_DEFAULTS, **(execution or {})}
    cls = "physicalai.runtime.AsyncExecution" if ex["mode"] == "async" else "physicalai.runtime.SyncExecution"
    smooth = {} if ex["smoother_frames"] is None else {"duration_frames": int(ex["smoother_frames"])}
    init_args = {
        "model": {"class_path": "physicalai.inference.InferenceModel",
                  "init_args": {"export_dir": export_dir, "backend": "openvino", "device": device}},
        "execution": {"class_path": cls, "init_args": {"request_threshold": float(ex["request_threshold"])}},
        "action_queue": {"class_path": "physicalai.runtime.ChunkedActionQueue",
                         "init_args": {"smoother": {"class_path": "physicalai.runtime.LerpSmoother", "init_args": smooth}}},
    }
    if task:
        init_args["task"] = task
    return {"class_path": "physicalai.runtime.PolicySource", "init_args": init_args}


def build_plan(name: str, timeout_s: float | None = None, *, model_id: str | None = None,
               device: str | None = None, return_home: bool | None = None, require_devices: bool = True) -> dict:
    """실행 계획(설정·명령·점검 재료)을 만든다. 로봇·카메라·컨테이너는 건드리지 않는다(Studio GET 만).

    require_devices=False: 팔로워 USB 가 없어도 거부하지 않고 Studio 에 저장된 포트를 둔다(preflight 의 모델 점검용).
    """
    cfg = load_config()
    if name not in cfg["policies"]:
        raise PolicyError("unknown_policy", f"모르는 정책: {name} (가능: {', '.join(cfg['policies'])})")
    s = _settings(cfg, name)
    model_id = model_id or s.get("model_id")
    if not model_id:
        raise PolicyError("no_model", f"{name} 에 연결된 모델이 없다 — 학습 후 "
                                      f"`python -m robot.run_policy --link {name} <모델ID>` (README 5절)")
    timeout_s = float(timeout_s if timeout_s is not None else s["timeout_s"])
    device = device or s["device"]
    return_home = s["return_to_initial_state"] if return_home is None else return_home
    notes: list[str] = []

    detail = _get(cfg, f"/api/models/{model_id}")  # {"model": {...}, "exports": [...], ...}
    model = detail["model"]
    export = next((e for e in detail.get("exports", []) if e.get("type") == "openvino"), None)
    if export is None:
        raise PolicyError("no_openvino", f"모델 {model_id} 에 OpenVINO export 가 없다 (학습 때 Export 에 OpenVINO 체크)")
    model_cams = [f["name"].removeprefix("images.") for f in export["io_spec"]["input_features"]
                  if f["ftype"] == "VISUAL"]
    dataset = _get(cfg, f"/api/dataset/{model['dataset_id']}") if model.get("dataset_id") else {}
    env_id = s.get("environment_id") or dataset.get("environment_id")
    if not env_id:
        raise PolicyError("no_environment", f"모델 {model_id} 의 환경을 모른다 — policies.json 의 {name}.environment_id 를 적는다")
    project_id = model["project_id"]
    env, ia, follower_info, robot_notes = environment_robot(cfg, project_id, env_id, require_devices=require_devices)
    notes += robot_notes

    # 카메라: 모델 입력 이름(images.<이름>)에 맞춰 환경 카메라를 고른다
    env_cams = ia.get("cameras") or {}
    cam_map = s.get("camera_map") or {}
    cams, cam_info = {}, []
    for m in model_cams:
        src = cam_map.get(m, m)
        if src not in env_cams:
            raise PolicyError("camera_mismatch",
                              f"모델 카메라 '{m}' 에 맞는 환경 카메라가 없다 (환경: {list(env_cams)}). "
                              f"policies.json 의 {name}.camera_map 에 {{\"{m}\": \"<환경 카메라 이름>\"}} 를 적는다")
        c = copy.deepcopy(env_cams[src])
        c["init_args"]["idle_timeout"] = s["camera_idle_timeout_s"]
        cams[m] = c
        fp = c["init_args"]["camera"]["init_args"]["device"]
        dev = _host_dev_by_serial("video", str(fp.get("serial", "")))
        cam_info.append({"model": m, "env": src, "serial": fp.get("serial"), "fingerprint_index": fp.get("index"),
                         "host_dev": dev})
    extra = sorted(set(env_cams) - {cam_map.get(m, m) for m in model_cams})
    if extra:
        notes.append(f"모델이 안 쓰는 환경 카메라는 열지 않음: {extra}")
    ia["cameras"] = cams

    export_dir = f"{model['path']}/exports/openvino"
    execution = execution_settings(s)
    ia["action_source"] = _policy_source(export_dir, device, s.get("task"), execution)
    # 홈 복귀 정지 감지(2.1): 런타임이 매 틱 관절값을 stdout 에 JSON 한 줄로 쓰게 한다(내장 JsonlCallback, 컨테이너에 파일 없음).
    # 켜면 런타임 복귀는 끄고, 끝 자세가 시작에서 멀 때만 우리가 복귀시킨다(2.3).
    early = _early_stop_settings(s)
    if early:
        ia["callbacks"] = list(ia.get("callbacks") or []) + [
            {"class_path": "physicalai.runtime.JsonlCallback", "init_args": {"path": "/dev/stdout"}}]
    runtime_return = bool(return_home) and not early
    runtime_cli = {"runtime": ia}  # _unwrap_runtime_document() 가 만드는 모양 = --config 가 받는 모양
    run_args = [f"--run.duration_s={timeout_s:g}", f"--run.return_to_initial_state={str(runtime_return).lower()}"]
    config_json = json.dumps(runtime_cli, ensure_ascii=False, separators=(",", ":"))
    startup = float(s["startup_budget_s"])
    eps = episode_stats(cfg, model.get("dataset_id"))
    if eps and timeout_s > eps["p90_s"] + 10:
        notes.append(f"timeout_s {timeout_s:g}s 가 학습 에피소드(90% {eps['p90_s']}s)보다 한참 길다 — "
                     f"남는 시간에 정책이 한 번 더 집으러 갈 수 있다. 권장 {eps['suggested_timeout_s']}s")
    return {
        "name": name, "cfg": cfg, "settings": s, "notes": notes,
        "model": {"id": model_id, "name": model.get("name"), "created_at": model.get("created_at"),
                  "dataset": dataset.get("name"), "export_dir": export_dir, "device": device,
                  "temporary": s.get("temporary", False), "note": s.get("note")},
        "environment": {"id": env_id, "name": env.get("name"), "project_id": project_id},
        "follower": follower_info,
        "cameras": cam_info,
        "timeout_s": timeout_s, "return_home": bool(return_home), "episodes": eps,
        "early_stop": early, "runtime_return": runtime_return, "execution": execution,
        # duration_s 는 제어 주기 수로 센다 → 추론이 느려 30Hz 를 못 맞추면 벽시계로는 더 걸린다(그래서 2배)
        "wall_limit_s": startup + 2 * timeout_s + (RETURN_MOVE_S if runtime_return else 0) + 15,
        "runtime_cli": runtime_cli, "config_json": config_json, "run_args": run_args,
        "container_paths": [follower_info["port"]] + [c["host_dev"] for c in cam_info if c["host_dev"]],
    }


def _early_stop_settings(s: dict) -> dict | None:
    """policies.json 의 early_stop(꺼져 있으면 None). mode "home"(기본) = 홈 복귀 정지(home_stop.DEFAULTS + return_far 6),
    "grasp" = 집기 정책의 닫고·들고·멈춤(grasp_stop.DEFAULTS — 복귀하지 않는다, 10-23 하이브리드)."""
    es = s.get("early_stop") or {}
    if not es.get("enabled"):
        return None
    if es.get("mode") == "grasp":
        out = {**grasp_stop.DEFAULTS}
        out.update({k: (None if v is None else float(v)) for k, v in es.items() if k in out})
        return {**out, "mode": "grasp"}
    out = {**home_stop.DEFAULTS, "return_far": 6.0}
    out.update({k: float(v) for k, v in es.items() if k in out})
    return {**out, "mode": "home"}


def environment_robot(cfg: dict, project_id: str, env_id: str, *,
                      require_devices: bool = True) -> tuple[dict, dict, dict, list[str]]:
    """Studio 환경의 runtime-config 에서 팔로워 SharedRobot 설정을 꺼내 이름·포트를 우리 것으로 고친다(GET 만).

    반환 (환경, runtime-config init_args — robot 절은 고친 것, {name, serial, port}, notes).
    좌표 집기 실행기(coord_pick/executor.py)도 이것을 쓴다.
    """
    notes: list[str] = []
    env = _get(cfg, f"/api/projects/{project_id}/environments/{env_id}")
    follower = env["robots"][0]["robot"]
    doc = yaml.safe_load(_get(cfg, f"/api/projects/{project_id}/environments/{env_id}/runtime-config", raw=True))
    ia = doc["init_args"]

    # 로봇: Studio 와 다른 이름의 SharedRobot → Studio 가 팔을 쥐고 있으면 장치 잠금에 걸려 연결 단계에서 실패(움직이기 전)
    robot = ia["robot"]
    if robot["class_path"] != "physicalai.robot.SharedRobot":
        raise PolicyError("unexpected_config", f"Studio 설정의 robot 이 SharedRobot 이 아니다: {robot['class_path']}")
    driver = robot["init_args"]["robot"]
    if driver["init_args"].get("role") != "follower":
        raise PolicyError("unexpected_config", "Studio 설정의 robot 역할이 follower 가 아니다")
    robot["init_args"]["name"] = cfg["robot_name"]
    serial = follower["payload"].get("serial_number", "")
    port = _host_dev_by_serial("serial", serial)
    if port is None and require_devices:
        raise PolicyError("follower_missing", f"팔로워(시리얼 {serial}) USB 가 안 보인다 — /dev/serial/by-id 확인")
    if port is None:
        port = driver["init_args"]["port"]
        notes.append(f"팔로워(시리얼 {serial}) USB 가 안 보여 Studio 에 저장된 포트 {port} 를 그대로 둠")
    elif driver["init_args"]["port"] != port:
        notes.append(f"팔로워 포트 {driver['init_args']['port']} → {port} 로 고침(시리얼 {serial})")
    driver["init_args"]["port"] = port
    return env, ia, {"name": follower.get("name"), "serial": serial, "port": port}, notes


def _argv(plan: dict, run_id: str) -> list[str]:
    cfg = plan["cfg"]
    return (_docker(cfg) + ["exec", "-u", "appuser", "-e", f"{MARKER}={run_id}", "-w", "/tmp", cfg["container"],
                            PHYSICALAI, "run", "--config", plan["config_json"], *plan["run_args"]])


def _shell_command(plan: dict, config_path: Path) -> str:
    argv = _argv(plan, "<run_id>")
    i = argv.index("--config") + 1
    parts = [shlex.quote(a) for a in argv]
    parts[i] = f'"$(cat {shlex.quote(str(config_path))})"'
    return " ".join(parts)


def _validate(plan: dict) -> tuple[bool, str]:
    """컨테이너 안에서 --print_config: 설정을 해석만 하고 끝난다(로봇·카메라 연결 없음)."""
    r = subprocess.run(_argv(plan, "validate") + ["--print_config"], capture_output=True, text=True, timeout=120)
    return r.returncode == 0, (r.stdout if r.returncode == 0 else r.stderr.strip()[-1500:])


def robot_preflight(cfg: dict, port: str, container_paths: list[str]) -> tuple[list, dict]:
    """로봇 쪽 실행 전 점검(정책·좌표 집기 공통): 컨테이너·Studio 세션·실행 중인 것·장치 잠금·컨테이너 안 장치.
    반환 (problems, info). 컨테이너가 꺼져 있으면 거기서 끝."""
    problems, info = [], {}
    if not _container_running(cfg):
        problems.append(("container_down", f"Studio 컨테이너({cfg['container']})가 안 떠 있다"))
        return problems, info
    sessions = _get(cfg, "/api/runtime/sessions")
    info["studio_sessions"] = [{"session": x.get("session_name"), "follower": x.get("follower_name"),
                                "status": x.get("status")} for x in sessions]
    if sessions:
        problems.append(("studio_session",
                         f"Studio 세션이 로봇을 쥐고 있다({', '.join(str(x.get('follower_name')) for x in sessions)}) — "
                         "Studio 의 원격조작·녹화·모델 실행 화면에서 나온 뒤 다시"))
    pr = _probe(cfg, "list", container_paths)
    info["device_locks"] = pr["device_locks"]
    info["our_procs"] = pr["procs"]
    if any(p["kind"] == "run" for p in pr["procs"]):
        problems.append(("already_running", f"이미 정책·좌표 집기가 돌고 있다: {pr['procs']}"))
    my_dev = f"serial:{os.path.basename(port)}"
    for lk in pr["device_locks"]:
        if lk["device"] == my_dev and lk["owner"] != cfg["robot_name"]:
            problems.append(("robot_busy", f"팔로워 {my_dev} 를 다른 소유자({lk['owner']}, pid {lk['pid']})가 잡고 있다"))
    if pr["missing_paths"]:
        problems.append(("device_not_in_container",
                         f"컨테이너 안에 장치가 없다 {pr['missing_paths']} — 컨테이너를 켠 뒤 새로 꽂힌 번호. "
                         "Studio 컨테이너 재시작 필요(사용자)"))
    return problems, info


def _config_hash(plan: dict) -> str:
    """정책·모델·runtime 설정(포트·카메라 장치·콜백 포함)·실행 인자 → 해시. 하나라도 바뀌면 다른 해시."""
    blob = json.dumps({"name": plan["name"], "model": plan["model"]["id"], "config": plan["config_json"],
                       "run_args": plan["run_args"]}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _validate_cached(plan: dict) -> tuple[bool, str, bool]:
    """--print_config 검사를 설정 해시마다 하루 한 번만(0.57초 실측). 통과만 기록하고 실패는 매번 다시 검사한다.
    반환 (ok, 출력 또는 메모, 캐시에서 건너뜀 여부). 기록 = robot/.state/validated.json {해시: {date, at, name, model}}."""
    key, today = _config_hash(plan), _now().date().isoformat()
    path = STATE_DIR / "validated.json"
    try:
        seen = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seen = {}
    if seen.get(key, {}).get("date") == today:
        return True, f"오늘 {seen[key]['at']} 에 같은 설정(해시 {key})을 검사함 — 건너뜀", True
    ok, out = _validate(plan)
    if ok:
        seen = {k: v for k, v in seen.items() if v.get("date") == today}  # 지난 날 기록은 지운다
        seen[key] = {"date": today, "at": _now().strftime("%H:%M:%S"), "name": plan["name"], "model": plan["model"]["id"]}
        _write(path, json.dumps(seen, ensure_ascii=False, indent=1))
    return ok, out, False


def preflight(plan: dict, *, use_cache: bool = False) -> dict:
    """실행 전 점검. problems 가 비어 있어야 실제로 돌린다.
    use_cache=True(실제 실행만): --print_config 설정 검사를 같은 날 같은 설정 해시면 건너뛴다. 로봇 쪽 점검은 매번 한다."""
    problems, info = robot_preflight(plan["cfg"], plan["follower"]["port"], plan["container_paths"])
    if any(c == "container_down" for c, _ in problems):
        return {"problems": problems, "info": info}
    for c in plan["cameras"]:
        if not c["host_dev"]:
            problems.append(("camera_missing", f"카메라 '{c['model']}'(시리얼 {c['serial']}) USB 가 안 보인다"))
    if use_cache:
        ok, out, cached = _validate_cached(plan)
        info["validate_cached"] = cached
    else:
        ok, out = _validate(plan)
    info["validated"] = ok
    info["print_config"] = out
    if not ok:
        problems.append(("config_invalid", f"physicalai 설정 검사 실패: {out[-500:]}"))
    return {"problems": problems, "info": info}


# ---------------------------------------------------------------- 실행

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_config(plan: dict, stem: Path) -> tuple[Path, Path]:
    j, y = stem.with_suffix(".runtime.json"), stem.with_suffix(".runtime.yaml")
    _write(j, plan["config_json"])
    _write(y, yaml.safe_dump(plan["runtime_cli"], sort_keys=False, allow_unicode=True))
    return j, y


def _result(name: str, started: dt.datetime, *, ok: bool = False, returncode: int | None = None,
            log_path: str | None = None, **extra) -> dict:
    ended = _now()
    return {"name": name, "ok": ok, "started": _iso(started), "ended": _iso(ended),
            "duration_s": round((ended - started).total_seconds(), 1), "returncode": returncode,
            "log_path": log_path, **extra}


def run_policy(name: str, timeout_s: float | None = None, *, dry_run: bool = False, model_id: str | None = None,
               device: str | None = None, return_home: bool | None = None) -> dict:
    """정책 하나를 실행하고 끝날 때까지 기다린다.

    timeout_s: 제어 루프 시간(--run.duration_s). 이만큼 돌고 끝난다(정책이 일찍 끝내는 신호는 없다).
               None 이면 policies.json 값(기본 21 = 15초 에피소드 90% + 3초, --link 가 정책별로 덮음). 벽시계 상한 = startup_budget_s + 2×timeout_s + 복귀 2.5초 + 15초,
               넘기면 SIGINT(복귀 후 종료) → 12.5초 안에 안 끝나면 SIGKILL.
    반환: 계약서 키 + reason(duration_elapsed|stopped|timeout|error|refused) · steps · run_id · error.
    """
    started = _now()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_f = open(STATE_DIR / "run.lock", "w")
    try:
        if not dry_run:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return _result(name, started, reason="refused", error="already_running: 이 PC 에서 다른 run_policy 가 실행 중")
        try:
            plan = build_plan(name, timeout_s, model_id=model_id, device=device, return_home=return_home)
        except PolicyError as e:
            return _result(name, started, reason="refused", error=f"{e.code}: {e}")
        if dry_run:
            stem = LOG_DIR / "dryrun" / name
            cpath, ypath = _save_config(plan, stem)
            pf = preflight(plan)
            _write(stem.with_suffix(".print_config.yaml"), pf["info"].get("print_config", ""))
            return _result(name, started, ok=pf["info"].get("validated", False), reason="dry_run",
                           command=_shell_command(plan, cpath), config_path=str(ypath), plan=_public(plan),
                           preflight=pf)
        return _run(plan, started)
    finally:
        lock_f.close()


def _public(plan: dict) -> dict:
    """결과·dry-run 출력용 계획 요약. execution(10-23)이 빠져 dry-run 이 늘 'sync 0.5' 로 찍혔다 — 실제 설정은 계획 전체로 만든다."""
    return {k: plan.get(k) for k in ("name", "model", "environment", "follower", "cameras", "timeout_s", "return_home",
                                     "early_stop", "runtime_return", "execution", "episodes", "wall_limit_s", "notes",
                                     "run_args")}


def _stop_flag() -> str | None:
    """stop() 이 남긴 상태 "<kill|home>:<sent|pending>". 같은 프로세스면 이벤트, 다른 프로세스면 .state/stop 파일."""
    if _stop_requested.is_set():
        return _stop_kind[0]
    f = STATE_DIR / "stop"
    if f.exists():
        return f.read_text(encoding="utf-8").strip() or "kill:pending"
    return None


def _run(plan: dict, started: dt.datetime) -> dict:
    name, cfg = plan["name"], plan["cfg"]
    run_id = f"{started:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    stem = LOG_DIR / f"{started:%Y%m%d}" / f"{started:%H%M%S}_{name}"
    _save_config(plan, stem)
    log_path = stem.with_suffix(".log")
    # 지난 stop 요청은 여기서 지운다. 이 뒤(사전 점검 중 포함)에 온 stop 은 아래 대기 루프가 잡는다.
    _stop_requested.clear()
    (STATE_DIR / "stop").unlink(missing_ok=True)
    # 이전 실행(정책·좌표 집기)이 결과를 먼저 돌려주고 끝나는 중일 수 있다 → 실행 프로세스·카메라가 풀릴 때까지 기다린다
    prev = wait_previous(cfg)
    if not prev["ok"]:
        return _result(name, started, reason="refused", run_id=run_id,
                       error=f"previous_not_released: 이전 실행이 {prev['waited_s']}초 안에 안 끝남 {prev['busy']}")
    pf = preflight(plan, use_cache=True)
    if pf["problems"]:
        _write(log_path, json.dumps(pf, ensure_ascii=False, indent=1))
        codes = "; ".join(f"{c}: {m}" for c, m in pf["problems"])
        return _result(name, started, log_path=str(log_path), reason="refused", error=codes, run_id=run_id)
    if _stop_flag():
        return _result(name, started, log_path=str(log_path), reason="stopped", error="시작 전에 stop 요청",
                       run_id=run_id)

    _write(STATE_DIR / "current.json", json.dumps({"run_id": run_id, "name": name, "started": _iso(started),
                                                   "host_pid": os.getpid(), "log_path": str(log_path)}))
    es = plan.get("early_stop")
    grasp_mode = bool(es) and es.get("mode") == "grasp"
    if grasp_mode:
        det = grasp_stop.GraspStopDetector(grasp_stop.tcp_z_fn(), **{k: es[k] for k in grasp_stop.DEFAULTS})
    else:
        det = home_stop.HomeStopDetector(**{k: es[k] for k in home_stop.DEFAULTS}) if es else None
    hit, complete = threading.Event(), threading.Event()
    tk: dict = {"n": 0, "t_first": None, "q0": None, "q": None, "steps": None}
    timed_out = False
    early_at = None
    t_launch = time.time()
    log = open(log_path, "w", encoding="utf-8")
    tf = open(stem.with_suffix(".ticks.jsonl"), "w", encoding="utf-8")
    log.write(f"# {_iso(started)} {name} run_id={run_id} model={plan['model']['id']} "
              f"timeout_s={plan['timeout_s']:g} wall_limit_s={plan['wall_limit_s']:g} "
              f"early_stop={json.dumps(es) if es else 'off'} waited_previous_s={prev['waited_s']} "
              f"print_config={'cached' if pf['info'].get('validate_cached') else 'checked'}\n")
    log.flush()
    proc = subprocess.Popen(_argv(plan, run_id), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, errors="replace")

    def reader() -> None:
        """컨테이너 출력: 틱 JSON(JsonlCallback) → 감지기·.ticks.jsonl, 나머지 → .log("Run complete" 를 보면 표시)."""
        for line in proc.stdout:
            if line.startswith('{"type": '):
                try:
                    ev = json.loads(line)
                except ValueError:
                    log.write(line)
                    continue
                if ev.get("type") != "tick":
                    tf.write(line)
                    continue
                q, ts = ev.get("joint_positions"), ev.get("timestamp")
                if not q or ts is None:
                    continue
                tf.write(json.dumps([round(ts, 3), [round(v, 2) for v in q]]) + "\n")
                tk["n"] += 1
                if tk["q0"] is None:
                    tk["t_first"], tk["q0"] = ts, q
                tk["q"] = q
                if det is not None and det.feed(ts, q):
                    hit.set()
            else:
                log.write(line)
                log.flush()
                if "Run complete:" in line:
                    try:
                        tk["steps"] = int(line.rsplit("Run complete:", 1)[1].split()[0])
                    except (IndexError, ValueError):
                        pass
                    complete.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    deadline = time.monotonic() + plan["wall_limit_s"]
    stop_sent_at = kill_sent_at = None
    last_probe = 0.0
    rc = None
    kind = ""
    while True:
        try:
            rc = proc.wait(timeout=0.1)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.monotonic()
        # 컨테이너 안 실행 프로세스가 끝났으면 docker exec(약 2.1초 늦게 돌아옴)를 기다리지 않는다.
        # 끝났을 법할 때("Run complete" 를 봤거나 신호를 보낸 뒤)는 0.1초마다, 그 밖에는 1초마다 조사(0.06초)한다.
        if now - last_probe >= (0.1 if (complete.is_set() or stop_sent_at is not None) else 1.0):
            last_probe = now
            try:
                procs = _probe(cfg, "list", timeout=10)["procs"]
            except (PolicyError, subprocess.SubprocessError, ValueError):
                procs = None
            if procs is not None and tk["n"] + int(complete.is_set()) > 0 and not any(
                    p["run_id"] == run_id and p["kind"] == "run" for p in procs):
                break
        mode = _stop_flag()
        kind, _, delivered = (mode or "").partition(":")
        if mode and stop_sent_at is None:
            stop_sent_at = now
            # stop() 이 신호를 못 보냈으면(컨테이너 프로세스가 뜨기 직전) 여기서 보낸다.
            # home 은 두 번 보내면 안 된다 — 두 번째 SIGINT 가 복귀 동작을 도중에 끊는다.
            if kind == "kill" or delivered != "sent":
                _signal_runs(cfg, "SIGINT" if kind == "home" else "SIGKILL")
        elif not mode and stop_sent_at is None and hit.is_set():
            # 홈 복귀 정지 감지 → 정상 종료(SIGINT). 런타임 복귀는 꺼 두었으니 바로 끝난다.
            stop_sent_at = early_at = now
            trig = det.trigger
            if grasp_mode:
                log.write(f"\n# 집기 끝 감지(첫 틱 뒤 {trig.t - tk['t_first']:.2f}s, 닫은 뒤 {trig.lift_mm}mm 들고 멈춤) → SIGINT\n")
            else:
                log.write(f"\n# 홈 복귀 정지 감지(첫 틱 뒤 {trig.t - tk['t_first']:.2f}s, 홈 거리 {trig.dist}) → SIGINT\n")
            log.flush()
            _signal_runs(cfg, "SIGINT")
        elif not mode and not timed_out and now > deadline:
            timed_out, stop_sent_at = True, now
            log.write(f"\n# 벽시계 상한 {plan['wall_limit_s']:g}s 초과 → SIGINT\n")
            log.flush()
            _signal_runs(cfg, "SIGINT")
        # 신호를 보냈는데 안 끝나면 SIGKILL 을 1초마다. SIGINT 는 런타임 복귀가 켜져 있으면 2.5초 + 여유 10초 뒤부터
        grace = 1.0 if kind == "kill" else 10 + (RETURN_MOVE_S if plan.get("runtime_return", True) else 0)
        if stop_sent_at is not None and now - stop_sent_at > grace and (
                kill_sent_at is None or now - kill_sent_at > 1.0):
            kill_sent_at = now
            _signal_runs(cfg, "SIGKILL")
    t_ended = time.monotonic()
    if rc is None and not complete.is_set() and stop_sent_at is None:
        # 신호 없이 "Run complete" 도 없이 끝났다 → 오류일 수 있으니 종료 코드를 기다린다(약 2초)
        try:
            rc = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    released = _wait_cameras_released(cfg, run_id, timeout=float(plan["settings"]["camera_idle_timeout_s"]) + 8)
    _reap_later(run_id, proc, th, log, tf)  # docker exec 의 나머지(약 2초)는 뒤에서 거둔다 — 결과는 먼저 돌려준다
    stopped = bool(_stop_flag())
    stop_kind = (_stop_flag() or "").partition(":")[0]
    if stopped:
        reason = "stopped"
    elif timed_out:
        reason = "timeout"
    elif early_at is not None:
        # 집기: 닫기 전 바닥 누름(grasp_stop floor)은 쥔 게 아니다 → floor_stop(실행기가 IK 로 넘긴다)
        reason = (("floor_stop" if det.trigger is not None and det.trigger.kind == "floor" else "grasp_done")
                  if grasp_mode else "home_reached")
    elif rc == 0 or (rc is None and complete.is_set()):
        reason = "duration_elapsed"
    else:
        reason = "error"
    err = None
    if reason == "error":
        text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = [ln for ln in text.splitlines() if ln.strip()][-3:]
        err = " | ".join(tail)[-400:]
    early_info = None
    end_dist = None
    if tk["q"] is not None:
        end_dist = round(max(abs(a - b) for a, b in zip(tk["q"][:5], tk["q0"][:5])), 2)
    if es and grasp_mode:
        trig = det.trigger
        early_info = {"mode": "grasp", "triggered": trig is not None, "ticks": tk["n"], **det.summary(),
                      "t_s": round(trig.t - tk["t_first"], 2) if trig else None,
                      "load_s": round(tk["t_first"] - t_launch, 2) if tk["t_first"] else None}
    elif es:
        trig = det.trigger
        early_info = {"triggered": trig is not None, "moved": det.left_at is not None, "ticks": tk["n"],
                      "t_s": round(trig.t - tk["t_first"], 2) if trig else None, "dist": trig.dist if trig else None,
                      "load_s": round(tk["t_first"] - t_launch, 2) if tk["t_first"] else None}
    # 복귀(2.3): 조기 종료를 쓰면 런타임 복귀를 껐으므로, 끝 자세가 시작에서 return_far 넘게 떨어졌을 때만 우리가 돌린다.
    # stop()(그 자리 정지)·오류 뒤에는 돌리지 않는다. stop(home=True) 는 돌린다. 집기 정책(grasp)은 돌리지 않는다 —
    # 블록을 든 끝 자세에서 좌표 집기 실행기가 이어서 나른다.
    returned = None
    want = reason in ("duration_elapsed", "timeout", "home_reached") or (reason == "stopped" and stop_kind == "home")
    if (es and not grasp_mode and plan.get("return_home") and want and end_dist is not None
            and end_dist > es["return_far"]):
        returned = _return_home(plan, tk["q0"], run_id, log_path)
    (STATE_DIR / "current.json").unlink(missing_ok=True)
    return _result(name, started, ok=reason in ("duration_elapsed", "home_reached", "grasp_done"),
                   returncode=rc if rc is not None else (0 if complete.is_set() else None),
                   log_path=str(log_path), reason=reason, steps=tk["steps"], run_id=run_id, camera_released=released,
                   error=err, early_stop=early_info, end_dist=end_dist, returned=returned,
                   start_q=tk["q0"], end_q=tk["q"],
                   tail_s=round(time.monotonic() - t_ended, 2), validate_cached=pf["info"].get("validate_cached"))


_TAILS: dict[str, threading.Thread] = {}  # 결과를 돌려준 뒤 docker exec 를 거두는 중인 실행


def _reap_later(run_id: str, proc: subprocess.Popen, reader: threading.Thread, log, tf) -> None:
    """docker exec 가 돌아올 때까지(컨테이너 프로세스가 끝나고 약 2.1초) 뒤에서 기다렸다가 출력·파일을 닫는다.
    10초 넘게 안 돌아오면 클라이언트만 끊는다(컨테이너 안 프로세스는 이미 끝났거나 stop() 이 끊는다)."""
    def tail() -> None:
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
        reader.join(timeout=5)
        with contextlib.suppress(Exception):
            log.write(f"\n# docker exec 종료 rc={rc}\n")
        for f in (log, tf):
            with contextlib.suppress(Exception):
                f.close()
        _TAILS.pop(run_id, None)

    th = threading.Thread(target=tail, daemon=True, name=f"reap-{run_id}")
    _TAILS[run_id] = th
    th.start()


def wait_previous(cfg: dict, timeout: float = 15.0) -> dict:
    """이전 실행(정책·좌표 집기)의 컨테이너 실행 프로세스와 우리 카메라 publisher 가 없어질 때까지 기다린다.
    소유 프로세스(robot_owner)는 기다리지 않는다 — 같은 이름(robot_name)으로 다시 붙는 게 정상 경로다(2.2, 10초 뒤 스스로
    끝남). 다른 소유자(Studio 세션)의 장치 잠금은 robot_preflight 가 거부한다. 반환 {ok, waited_s, busy}."""
    t0 = time.monotonic()
    busy: list = []
    while True:
        try:
            procs = _probe(cfg, "list", timeout=10)["procs"]
        except (PolicyError, subprocess.SubprocessError, ValueError) as e:
            return {"ok": True, "waited_s": round(time.monotonic() - t0, 2), "busy": [], "note": f"조사 실패: {e}"}
        busy = [p for p in procs if p["kind"] in ("run", "camera_publisher")]
        if not busy:
            return {"ok": True, "waited_s": round(time.monotonic() - t0, 2), "busy": []}
        if time.monotonic() - t0 > timeout:
            return {"ok": False, "waited_s": round(time.monotonic() - t0, 2), "busy": busy}
        time.sleep(0.1)


# 복귀 속도: 런타임 복귀(2.5초 선형)와 비슷하게 관절 36°/s 상한, 추종 오차 8°(3틱 연속) 넘으면 그 자리 유지.
RETURN_LIMITS = {"rate_hz": 30.0, "step_body": 36 / 30 / 1.8, "step_grip": 20 / 30, "track_max": 8 / 1.8,
                 "track_ticks": 3, "lag_ticks": 3, "stale_s": 0.5, "start_tol": 1000.0, "settle_s": 0.3}


def _return_home(plan: dict, q0: list, run_id: str, log_path: Path) -> dict:
    """끝 자세 → 시작 자세(q0)를 SharedRobot 으로 천천히(coord_pick/stream.py 의 container_return). 컨테이너에 파일 없음.
    명령줄에 coord_pick 이 들어 있어 _PROBE 가 run 으로 센다 → stop() 이 그대로 끊는다."""
    cfg = plan["cfg"]
    rid = f"{run_id}-ret"
    payload = {"run_id": rid, "robot_name": cfg["robot_name"], "target": list(q0), "limits": RETURN_LIMITS,
               "already_tol": plan["early_stop"]["return_far"],
               "driver": plan["runtime_cli"]["runtime"]["robot"]["init_args"]["robot"]}
    script = ((ROBOT_DIR / "coord_pick" / "stream.py").read_text(encoding="utf-8")
              + f"\nPAYLOAD = json.loads({json.dumps(json.dumps(payload))})\nsys.exit(container_return(PAYLOAD))\n")
    argv = _docker(cfg) + ["exec", "-i", "-u", "appuser", "-e", f"{MARKER}={rid}", "-w", "/tmp", cfg["container"],
                           CPY, "-I", "-", "coord_pick", rid]
    t0 = time.monotonic()
    try:
        r = subprocess.run(argv, input=script, capture_output=True, text=True, timeout=30)
        out, rc = r.stdout, r.returncode
    except subprocess.TimeoutExpired as e:
        _signal_runs(cfg, "SIGKILL")
        out, rc = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), None
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n# 복귀(끝 자세가 시작에서 멂)\n" + out)
    end = None
    for line in out.splitlines():
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("ev") == "end":
            end = ev
    return {"ok": bool(end and end.get("ok")), "reason": (end or {}).get("reason", "error"), "returncode": rc,
            "duration_s": round(time.monotonic() - t0, 1)}


def _signal_runs(cfg: dict, sig: str) -> list[int]:
    try:
        return _probe(cfg, f"kill:{sig}", timeout=10)["killed"]
    except (PolicyError, subprocess.SubprocessError, ValueError) as e:
        print(f"[robot] {sig} 보내기 실패: {e}", file=sys.stderr)
        return []


def _wait_cameras_released(cfg: dict, run_id: str, timeout: float) -> bool:
    """우리 카메라 publisher 가 끝날 때까지 기다린다(비전이 바로 위 카메라를 열 수 있게)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            procs = _probe(cfg, "list", timeout=10)["procs"]
        except (PolicyError, subprocess.SubprocessError, ValueError):
            return False
        if not any(p["run_id"] == run_id and p["kind"] == "camera_publisher" for p in procs):
            return True
        time.sleep(0.1)
    return False


def stop(home: bool = False) -> None:
    """실행 중인 정책(좌표 집기 실행기 포함)을 즉시 중단한다. 다른 프로세스에서 불러도 된다.

    home=False(기본): SIGKILL → 제어 루프가 바로 끊기고 팔은 마지막 자세에서 토크를 유지한 채 멈춘다.
    home=True: SIGINT → 런타임이 끝난 뒤 시작 자세로 복귀 후 종료. 조기 종료(early_stop)를 켠 실행은 끝 자세가
               시작에서 return_far 넘게 떨어졌을 때만 우리가 복귀시키고, 끈 실행은 런타임이 2.5초 복귀한다.
    """
    kind = "home" if home else "kill"
    killed: list[int] = []
    try:
        killed = _signal_runs(load_config(), "SIGINT" if home else "SIGKILL")
    except Exception as e:  # stop 은 절대 예외를 던지지 않는다
        print(f"[robot] stop 신호 실패: {e}", file=sys.stderr)
    # 신호를 보낸 뒤에 상태를 남긴다 → 실행 쪽 대기 루프는 sent/pending 을 보고 다시 보낼지 정한다
    _stop_kind[0] = f"{kind}:{'sent' if killed else 'pending'}"
    _stop_requested.set()
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / "stop").write_text(_stop_kind[0], encoding="utf-8")
    except OSError as e:
        print(f"[robot] stop 기록 실패: {e}", file=sys.stderr)


# ---------------------------------------------------------------- 보조 명령

def status() -> dict:
    cfg = load_config()
    out: dict = {"current": None}
    cur = STATE_DIR / "current.json"
    if cur.exists():
        out["current"] = json.loads(cur.read_text(encoding="utf-8"))
    out["container_running"] = _container_running(cfg)
    if out["container_running"]:
        out["studio_sessions"] = _get(cfg, "/api/runtime/sessions")
        pr = _probe(cfg, "list")
        out["our_procs"], out["device_locks"] = pr["procs"], pr["device_locks"]
    return out


def list_models() -> list[dict]:
    cfg = load_config()
    linked = {p.get("model_id"): n for n, p in cfg["policies"].items() if p.get("model_id")}
    rows = []
    for proj in _get(cfg, "/api/projects"):
        datasets = {}
        for m in _get(cfg, f"/api/projects/{proj['id']}/models"):
            did = m.get("dataset_id")
            if did not in datasets:
                try:
                    datasets[did] = _get(cfg, f"/api/dataset/{did}").get("name")
                except PolicyError:
                    datasets[did] = None
            rows.append({"id": m["id"], "name": m.get("name"), "created_at": m.get("created_at"),
                         "project": proj["name"], "dataset": datasets[did],
                         "backends": m.get("available_backends"), "linked_to": linked.get(m["id"])})
    return sorted(rows, key=lambda r: r["created_at"] or "")


def link(name: str, model_id: str) -> dict:
    """policies.json 의 name 에 모델을 연결한다(이전 파일은 policies.json.bak). 연결 전 dry-run 으로 확인."""
    res = run_policy(name, dry_run=True, model_id=model_id)
    if res.get("reason") == "refused":
        raise PolicyError("link_refused", res["error"])
    if not res["preflight"]["info"].get("validated"):
        raise PolicyError("link_refused", "physicalai 설정 검사 실패 — 연결하지 않음")
    text = POLICIES_PATH.read_text(encoding="utf-8")
    cfg = json.loads(text)
    (POLICIES_PATH.parent / "policies.json.bak").write_text(text, encoding="utf-8")
    p = cfg["policies"][name]
    p["model_id"] = model_id
    p["temporary"] = False
    eps = res["plan"]["episodes"]
    if eps:
        p["timeout_s"] = eps["suggested_timeout_s"]
    p["note"] = (f"{_now():%Y-%m-%d %H:%M} 연결 — 데이터셋 {res['plan']['model']['dataset']}"
                 + (f", 에피소드 {eps['episodes']}개 90% {eps['p90_s']}s → timeout_s {eps['suggested_timeout_s']}" if eps else ""))
    POLICIES_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return res


def _print_dry(res: dict) -> None:
    if res.get("reason") == "refused":
        print(f"거부: {res['error']}")
        return
    p, pf = res["plan"], res["preflight"]
    m = p["model"]
    print(f"== {p['name']} — dry-run (로봇 안 움직임)")
    if m.get("temporary"):
        print(f"⚠ {m.get('note')}")
    print(f"모델      {m['id']} '{m['name']}' (데이터셋 {m['dataset']}, {m['created_at']})")
    print(f"모델 경로 {m['export_dir']} (컨테이너 안, OpenVINO {m['device']})")
    print(f"환경      {p['environment']['name']} ({p['environment']['id']})")
    print(f"팔로워    {p['follower']['name']} 시리얼 {p['follower']['serial']} → {p['follower']['port']}")
    for c in p["cameras"]:
        print(f"카메라    images.{c['model']} ← 환경 '{c['env']}' 시리얼 {c['serial']} → {c['host_dev']}"
              f" (Studio 지문 index {c['fingerprint_index']})")
    ex = p.get("execution") or EXECUTION_DEFAULTS
    print(f"추론      {ex['mode']} · 남은 동작 {ex['request_threshold']:g} 아래면 다시 추론 · 섞기 "
          f"{ex['smoother_frames'] if ex['smoother_frames'] is not None else 5}틱")
    es = p.get("early_stop")
    if es and es.get("mode") == "grasp":
        print(f"시간      제어 루프 상한 {p['timeout_s']:g}s · 집기 끝 감지 켬(그리퍼 {es['close_drop']:g} 넘게 닫힘 → "
              f"{es['lift_mm']:g}mm 넘게 들림 → {es['hold_s']:g}s 정지 ≤ {es['still']:g} 이면 SIGINT) · 복귀 안 함"
              f" · 벽시계 상한 {p['wall_limit_s']:g}s")
        if es.get("floor_mm") is not None:
            print(f"          닫기 전 tcp 가 {es['floor_mm']:g}mm 아래 {es['floor_hold_s']:g}s"
                  + (f"(또는 {es['floor_hard_mm']:g}mm 아래)" if es.get("floor_hard_mm") is not None else "")
                  + " 면 바닥 누름으로 바로 끝(floor_stop → 실행기가 IK 로)")
    elif es:
        print(f"시간      제어 루프 상한 {p['timeout_s']:g}s · 홈 복귀 정지 감지 켬(벗어남 {es['leave']:g} → 홈 {es['near']:g} 안에서 "
              f"{es['hold_s']:g}s 정지 ≤ {es['still']:g} 이면 SIGINT) · 복귀는 끝 자세가 시작에서 {es['return_far']:g} 넘을 때만"
              f"{'' if p['return_home'] else '(복귀 끔)'} · 벽시계 상한 {p['wall_limit_s']:g}s")
    else:
        print(f"시간      제어 루프 {p['timeout_s']:g}s, 끝나면 시작 자세 복귀 {'켬(2.5s)' if p['return_home'] else '끔'},"
              f" 벽시계 상한 {p['wall_limit_s']:g}s")
    if p["episodes"]:
        e = p["episodes"]
        print(f"학습 에피소드 {e['episodes']}개: 중앙 {e['median_s']}s · 90% {e['p90_s']}s · 최대 {e['max_s']}s"
              f" → 권장 timeout_s {e['suggested_timeout_s']}")
    for n in p["notes"]:
        print(f"메모      {n}")
    info = pf["info"]
    print("점검:")
    print(f"  Studio 세션        {info.get('studio_sessions')}")
    print(f"  로봇 장치 잠금     {info.get('device_locks')}")
    print(f"  설정 검사          {'통과 (--print_config)' if info.get('validated') else '실패'}")
    if pf["problems"]:
        print("  지금 실제로 돌리면 거부되는 이유:")
        for code, msg in pf["problems"]:
            print(f"    - {code}: {msg}")
    else:
        print("  실제 실행을 막는 것 없음")
    print("\n--- 실행될 명령 (<run_id> 는 실행 때 생성)")
    print(res["command"])
    print(f"\n--- 설정 ({res['config_path']}, --config 에는 같은 내용의 JSON 이 들어간다)")
    print(Path(res["config_path"]).read_text(encoding="utf-8"), end="")
    print(f"\n--- physicalai 가 해석한 전체 설정: {LOG_DIR / 'dryrun' / (p['name'] + '.print_config.yaml')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.run_policy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help=" | ".join(NAMES))
    ap.add_argument("--dry-run", action="store_true", help="로봇을 움직이지 않고 명령·설정·점검만 출력")
    ap.add_argument("--timeout", type=float, help="제어 루프 시간(초). 기본 policies.json")
    ap.add_argument("--model", help="policies.json 대신 이 모델 ID 로")
    ap.add_argument("--device", help="OpenVINO 장치(GPU|CPU|NPU). 기본 policies.json")
    ap.add_argument("--no-home", action="store_true", help="끝날 때 시작 자세로 돌아가지 않음")
    ap.add_argument("--yes", action="store_true", help="실제 실행 확인 질문 생략")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 으로")
    ap.add_argument("--stop", action="store_true", help="실행 중인 정책 즉시 중단(그 자리에 멈춤)")
    ap.add_argument("--stop-home", action="store_true", help="실행 중인 정책 중단 후 시작 자세로 복귀")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--models", action="store_true", help="Studio 모델 목록")
    ap.add_argument("--link", nargs=2, metavar=("NAME", "MODEL_ID"), help="policies.json 에 모델 연결")
    a = ap.parse_args(argv)

    def show(obj) -> None:
        print(json.dumps(obj, ensure_ascii=False, indent=1))

    if a.stop or a.stop_home:
        stop(home=a.stop_home)
        print("중단 신호 보냄")
        return 0
    if a.status:
        show(status())
        return 0
    if a.models:
        rows = list_models()
        if a.json:
            show(rows)
        else:
            for r in rows:
                print(f"{r['id']}  {r['created_at']}  {r['name']:<16} 데이터셋 {r['dataset']:<20} "
                      f"{','.join(r['backends'] or [])}  {('← ' + r['linked_to']) if r['linked_to'] else ''}")
        return 0
    if a.link:
        try:
            res = link(*a.link)
        except PolicyError as e:
            print(f"연결 안 함 — {e}")
            return 1
        e = res["plan"]["episodes"]
        print(f"연결함: {a.link[0]} → {a.link[1]} (데이터셋 {res['plan']['model']['dataset']})"
              + (f", timeout_s {e['suggested_timeout_s']}(에피소드 90% {e['p90_s']}s + 3)" if e else "")
              + ". 이전 파일 policies.json.bak")
        return 0
    if not a.name:
        ap.print_help()
        return 2
    kw = {"model_id": a.model, "device": a.device, "return_home": False if a.no_home else None}
    if a.dry_run:
        res = run_policy(a.name, a.timeout, dry_run=True, **kw)
        if a.json:
            res.pop("plan", None)
            show(res)
        else:
            _print_dry(res)
        return 0 if res.get("ok") else 1
    if not a.yes:
        ans = input(f"로봇이 실제로 움직입니다({a.name}). 팔로워 전원 스위치에 손을 두셨나요? [y/N] ")
        if ans.strip().lower() != "y":
            print("취소")
            return 1
    signal.signal(signal.SIGINT, lambda *_: stop())  # Ctrl+C = 즉시 중단
    res = run_policy(a.name, a.timeout, **kw)
    show(res)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
