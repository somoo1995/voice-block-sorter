"""Physical AI Studio 런타임 세션 목록·멈춤(10-22 r22). 정책·좌표 집기·홈 맞춤이 studio_session·robot_busy 로 거부될 때만 쓴다.

Studio 는 원격조작 탭을 닫거나 다른 탭으로 옮겨도 45초 idle + 10초 뒤에야 팔을 놓는다(10-22 16:56 로그). 그동안 대시보드 명령이
"로봇을 다른 곳에서 쓰고 있어" 로 거부됐다. 세션 멈춤은 사용자 허락(10-22 17:17). 녹화 세션은 멈추지 않는다(graph._robot_call).

API(Studio 소스 `api/runtime_sessions.py`): GET /api/runtime/sessions → [{session_name, status, follower_name, activity{
follower_source, is_recording, dataset_loaded, …}}] · POST /api/runtime/sessions/{session_name}/stop → 204(SIGTERM 정리, 녹화는
마무리)·500(멈추지 않음).
"""
import json
import os
import urllib.error
import urllib.request

from .config import ROOT


def base_url() -> str:
    """STUDIO_URL, 없으면 robot/policies.json studio_url(로봇 담당과 같은 주소), 없으면 :7860."""
    if os.getenv("STUDIO_URL"):
        return os.environ["STUDIO_URL"].rstrip("/")
    try:
        return json.loads((ROOT / "robot" / "policies.json").read_text(encoding="utf-8"))["studio_url"].rstrip("/")
    except Exception:
        return "http://127.0.0.1:7860"


def list_sessions(timeout: float = 5) -> list[dict]:
    with urllib.request.urlopen(base_url() + "/api/runtime/sessions", timeout=timeout) as r:
        return json.loads(r.read())


def stop_session(name: str, timeout: float = 20) -> int:
    """세션을 멈춘다. HTTP 상태 코드(204 = 멈춤)."""
    req = urllib.request.Request(base_url() + f"/api/runtime/sessions/{urllib.request.quote(name)}/stop", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
