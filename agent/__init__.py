"""ball-sorter 에이전트. UI 는 run_command 하나만 쓴다(계약서 3절). warmup 은 선택(시작 때 불러 두면 첫 명령이 빠름).

NODE_LABELS: 대시보드 그래프 칸 이름(ui/app.py 가 읽어 덧붙인다). coord = 좌표 집기 그릇 옮기기(10-22 r16, 10-23 13:10 부터 옮기기는 늘 이것),
unheard = 받아쓰기 환청을 거른 것(그래프 앞). none = 명령 아님(잡담·마이크 시험·LLM 실패 + 자신 없음, 로봇 안 움직임, 10-22 밤).
home = 홈 자세 명령("홈으로"·"처음 자세로", 규칙, go_home(open_gripper=True), 10-23).
unsort = 그릇에서 꺼내기("다 꺼내 놔"·"초기화해 줘", 규칙, robot unsort_blocks, 작업 기록·결과 보고에 안 넣음, 10-23 리허설 초기화).
get_method()·set_method("act"|"coord"): 집는 방식 기본값(대시보드 표시·토글, 10-22 r21). 10-23 13:10 부터 옮기기는 늘 coord,
    set_method("act") 는 ValueError(학습한 동작은 손 전달만).
speech_seconds(wav_path): 녹음의 말소리 초 — UI 가 run_command(..., stt={"speech_s": …}) 로 넘기면 무음 기준으로도 거른다.
"""

NODE_LABELS = {"coord": "좌표 집기", "unheard": "못 알아들음", "none": "명령 아님", "home": "홈 자세",
               "unsort": "그릇에서 꺼내기"}


def run_command(text: str, thread_id: str = "demo", stt: dict | None = None):
    from .graph import run_command as _run
    return _run(text, thread_id, stt)


def warmup() -> dict:
    from .graph import warmup as _warmup
    return _warmup()


def get_method() -> dict:
    """집는 방식 기본값과 쓸 수 있는지 {default: act|coord, coord_available, coord_backend, robot, home, labels}(10-22 r21)."""
    from .graph import get_method as _get
    return _get()


def set_method(method: str) -> dict:
    """기본 방식을 바꾼다(다음 명령부터). act 는 ValueError(10-23 13:10, 학습한 동작은 손 전달만). 좌표 집기를 못 불러왔으면 coord 도."""
    from .graph import set_method as _set
    return _set(method)


def speech_seconds(wav_path: str) -> float | None:
    from .hearing import speech_seconds as _sp
    return _sp(wav_path)


__all__ = ["run_command", "warmup", "speech_seconds", "get_method", "set_method", "NODE_LABELS"]
