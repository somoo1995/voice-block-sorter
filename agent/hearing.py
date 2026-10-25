"""받아쓰기 환청 거르기(10-22 r16). Whisper 는 조용하거나 잡음만 있는 녹음에서 방송 자막 문구를 지어낸다.

실물 10-22 15:57: 마이크를 허용하는 순간의 1.26초 녹음(말소리 약 0.3초)이 "감사합니다." 로 받아적혀 question 으로 갔다.
- 글: 말 전체가 대표 환청 문구(여럿 이어 붙은 것 포함)·말 없는 소리("음", "아")·문장부호뿐이면 못 들은 것.
  명령 속에 섞인 "감사합니다"("분홍 블록 옮겨 줘 감사합니다")는 거르지 않는다.
- 소리: UI 가 녹음 정보(run_command 의 stt={"speech_s"} 또는 {"wav_path"})를 넘길 때만. 말소리가 STT_MIN_SPEECH_S 보다
  짧거나, 글자 수 ÷ 말소리 초가 STT_MAX_CPS 보다 크면(말한 것보다 글이 길면) 못 들은 것. "멈춰"는 그래프 앞에서 먼저 처리한다.
  10-22 녹음 실측(speech_seconds): 무음·잡음 0.00초, 딸깍 0.12초 | 합성 음성 50개 0.24("멈춰")~1.65초, 초당 글자 최대 15.6 |
  실물 마이크 명령 8개 0.78~2.10초, 초당 글자 3.6~7.9 | 환청 "감사합니다." 0.36초(13.9자/초 — 글 기준으로 거른다).
  "응"이 "음"으로 받아써지면(M1 목소리, ui/samples/demo/README) 거절 대신 "잘 못 들었어요"가 되어 "할까요?" 기다림이 이어진다.
"""
import re

from . import config

UNHEARD_MSG = "잘 못 들었어요. 다시 말씀해 주세요."

# 띄어쓰기·문장부호를 빼고 소문자로 바꾼 글에 맞춘다(ui/stt.py HALLUCINATIONS 는 앞부분 일치라 "감사합니다." 하나는 못 거른다)
_PHRASES = [
    r"(?:오늘도)?(?:끝까지)?시청(?:해주셔서|해주신여러분)?(?:정말)?(?:감사|고맙)(?:합니다|드립니다|습니다)",
    r"(?:정말|너무)?감사(?:합니다|드립니다|해요)", r"고맙(?:습니다|습니당)",
    r"(?:구독과?|구독하고)좋아요(?:부탁드립니다|부탁드려요|눌러주세요)?", r"좋아요(?:와|랑)?구독(?:부탁드립니다|부탁드려요)?",
    r"구독부탁드(?:립니다|려요)", r"알림설정(?:까지)?(?:부탁드립니다)?",
    r"(?:mbc|kbs|sbs|ytn|jtbc|ebs|연합)뉴스(?:[가-힣]{2,4}입니다)?",
    r"(?:한글)?자막(?:제공|제작|by)[a-z가-힣0-9]{0,12}",
    r"다음(?:영상|시간|주)에(?:서)?(?:또)?(?:만나요|뵙겠습니다|만나겠습니다|봬요)", r"안녕히(?:계세요|가세요)",
    r"음+|으+음*|흠+|아+|에+",                          # 말 없는 소리. "어"는 "할까요?" 대답이라 넣지 않는다
]
_HALLU = re.compile("(?:" + "|".join(_PHRASES) + ")+")
_PUNCT = re.compile(r"[\s.,!?~…·:;\"'“”‘’()\[\]♪♬\-]+")


def _norm(text: str) -> str:
    return _PUNCT.sub("", text or "").lower()


def speech_seconds(wav_path: str) -> float | None:
    """녹음에서 말소리 길이(초). 30ms 틀의 세기가 문턱을 넘는 틀 수 × 0.03. 문턱 = 잡음 바닥(p10) × 2.5, 큰 소리(p95) × 0.3,
    0.02 중 큰 값. 첫 0.15초(버튼·마이크 켜는 소리)는 뺀다. 못 읽으면 None(소리 기준을 건너뜀).
    실측은 모듈 설명 참고(README 받아쓰기 환청 절)."""
    try:
        import numpy as np
        import soundfile as sf

        a, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    except Exception:
        return None
    a = a.mean(axis=1)[int(sr * 0.15):]
    fr = max(1, int(sr * 0.03))
    n = len(a) // fr
    if n == 0:
        return 0.0
    e = np.sqrt((a[:n * fr].reshape(n, fr) ** 2).mean(axis=1))
    thr = max(0.02, 2.5 * float(np.percentile(e, 10)), 0.3 * float(np.percentile(e, 95)))
    return round(float((e > thr).sum()) * fr / sr, 2)


def check(text: str, stt: dict | None = None) -> str | None:
    """못 들은 말이면 이유(사람이 읽는 말), 아니면 None."""
    core = _norm(text)
    if not core:
        return "글자 없음(문장부호·음표만)"
    if _HALLU.fullmatch(core):
        return f"환청 문구 “{text.strip()}”"
    if not stt:
        return None
    sp = stt.get("speech_s")
    if sp is None and stt.get("wav_path"):
        sp = speech_seconds(stt["wav_path"])
    if sp is None:
        return None
    if sp < config.STT_MIN_SPEECH_S:
        return f"말소리 {sp:.2f}초뿐"
    n = len(re.findall(r"[가-힣a-z0-9]", core))
    if n / max(sp, 0.05) > config.STT_MAX_CPS:
        return f"말소리 {sp:.2f}초에 글자 {n}개"
    return None
