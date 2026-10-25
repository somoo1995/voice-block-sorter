"""에이전트 설정. 값은 환경변수에서 읽고, 기본값은 젠북 로컬 OVMS(127.0.0.1:8100)."""
import os
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent
ROOT = AGENT_DIR.parent                      # ball-sorter/
RUNS_DIR = ROOT / "runs"
CACHE_DIR = AGENT_DIR / ".cache"
FIXTURES_DIR = AGENT_DIR / "fixtures"

# 프로젝트 루트 .env 가 있으면 읽는다(이미 설정된 환경변수가 우선)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

# 시연 중 외부 통신 0건: Chroma·LangSmith 원격 전송 끔
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")

LOCAL_OVMS = "http://127.0.0.1:8100/v3"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", LOCAL_OVMS).rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
# 기본은 qwen3-4b(10-21 총괄 지시: 지금 떠 있는 모델). 8B 는 LLM_MODEL=qwen3-8b 로 바꾼다.
# 지정한 모델이 OVMS 목록에 없으면 목록에 있는 LLM_FALLBACKS 중 하나로 대신 쓰고 start 이벤트에 남긴다(llm.py)
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-4b")
LLM_FALLBACKS = ["qwen3-4b", "qwen3-8b"]
LLM_THINKING = os.getenv("LLM_THINKING", "0") == "1"
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
# 의도 판별 LLM(64토큰, 정상 0.8초대)만 짧게(10-22 밤). 23:08 OVMS 가 멎자 판별이 30초 기다린 뒤 임베딩 1위(red, 자신 없음)로
# 정책을 돌렸다. 넘기면 router.classify 가 움직이는 의도를 막는다
LLM_CLASSIFY_TIMEOUT_S = float(os.getenv("LLM_CLASSIFY_TIMEOUT_S", "5"))
# with_structured_output 방식: json_schema | function_calling | json_mode
LLM_SO_METHOD = os.getenv("LLM_SO_METHOD", "json_schema")

# 임베딩·리랭크는 OVMS 에만 있다. LLM 을 임시로 외부 Gemma 서버(/v1)로 돌려도 로컬 OVMS 를 쓴다.
EMB_BASE_URL = os.getenv(
    "EMB_BASE_URL", LLM_BASE_URL if LLM_BASE_URL.endswith("/v3") else LOCAL_OVMS
).rstrip("/")
EMB_MODEL = os.getenv("EMB_MODEL", "qwen3-embedding")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-reranker")

# 의도 라우터: 예문과의 코사인 유사도 최고값이 THRESHOLD 이상이고 2등 의도와의 차가 MARGIN 이상이면 확정
ROUTER_THRESHOLD = float(os.getenv("ROUTER_THRESHOLD", "0.80"))
ROUTER_MARGIN = float(os.getenv("ROUTER_MARGIN", "0.04"))
# 1위가 question 일 때만 쓰는 낮은 기준(질문은 로봇을 안 움직인다). 10-21 밤 의도 9개·102문장: 1위가 question 인데
# 질문이 아닌 문장 0개(질문 7개는 여유 0.003~0.18) → 여유 0.02 로 "작업 순서?"·"손은 어디에?"(0.029)도 LLM 없이
ROUTER_Q_THRESHOLD = float(os.getenv("ROUTER_Q_THRESHOLD", "0.78"))
ROUTER_Q_MARGIN = float(os.getenv("ROUTER_Q_MARGIN", "0.02"))

# 반복 노드
# 정책 1회 시간은 넘기지 않는다 → robot/policies.json 정책별 값(10-22: 15초 녹화 → 기본 21초). ACT 는 그 시간만큼
# 돌고 끝나서, 40 같은 큰 값을 넘기면 남는 시간에 블록을 하나 더 집으러 간다(로봇 담당 발견, 계약서 3절 timeout_s=None)
RETRY_PER_ITEM = int(os.getenv("RETRY_PER_ITEM", "1"))     # 한 개를 놓쳤을 때 다시 시도하는 횟수
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "20"))        # 명령 하나당 정책 실행 상한
HAND_WAIT_S = float(os.getenv("HAND_WAIT_S", "15"))
HAND_POLL_S = float(os.getenv("HAND_POLL_S", "0.7"))
# 정책이 camera_released=False 로 끝났거나 카메라 열기에 실패했을 때 다시 찍기 전 기다리는 시간
CAMERA_WAIT_S = float(os.getenv("CAMERA_WAIT_S", "1.5"))
# 좌표 집기(10-22 r16): 1 이면 intent defect·red·blue(와 plan 의 그 단계)를 ACT 정책 대신 robot/coord_pick sort_blocks 로.
# 실물 시험에서 pick_red 가 "분홍 1 + 초록 1" 장면을 3/3 놓쳤다(색보다 위치를 배움). handoff 는 그대로 ACT
# 10-23 13:10 사용자 결정: 옮기기는 이 값과 상관없이 늘 좌표 집기(graph.METHOD). 이 값은 시작 스크립트 호환으로만 남는다
BALL_COORD = os.getenv("BALL_COORD") == "1"
# 장면 답 LLM 다듬기(10-23): 기본 끔. 코드가 만든 템플릿 답을 그대로 말한다. 09:35 GPU HANG 이 장면 답 다듬기 중에 났다(총괄)
SCENE_POLISH = os.getenv("AGENT_SCENE_POLISH", "0") == "1"
# 홈 자세 명령(10-23 intent home): 홈에서 너무 멀면(too_far, 로봇 기본 safety.home_max_deg 60°) 이 각도로 한 번 더 보낸다.
# 사람이 옆에서 말로 시키는 명령이라 허용(총괄). 홈 맞춤(HOME_ALIGN)·동작 뒤 홈(HOME_AFTER)은 그대로 로봇 기본값
HOME_CMD_MAX_DEG = float(os.getenv("HOME_CMD_MAX_DEG", "150"))
# 정책·좌표 집기 직전 홈 맞춤(10-22 r20, robot.coord_pick.executor.go_home, BALL_ROBOT=1 일 때만). 16:53 실물: 앞 명령이 '멈춤'으로
# 끝나 팔이 홈 밖에 있었고, handoff 의 조기 종료(시작 자세로 돌아와 멈춤)가 안 걸려 남은 시간에 다음 블록을 집으러 갔다.
# always = 매번(go_home 이 재서 홈 1.8° 안이면 안 보냄, 실측 4.0~4.2초 / 옮기면 5.3~6.6초)
# after_stop = 이 프로세스 첫 동작·멈춤·오류·시간 초과 뒤에만(정상으로 끝난 정책 다음은 건너뜀) / off = 안 함
HOME_ALIGN = os.getenv("HOME_ALIGN", "always")
# 동작이 끝나면 집게를 최대로 열고 홈으로(10-22 r22, go_home(open_gripper=True)). 집게에 블록이 끼어 남거나 홈이 아닌 자세로
# 끝나는 일이 실물에서 여러 번 있었다. open(기본) | off. '멈춤'으로 끝나면 그 자리 유지(다음 명령 전에 HOME_ALIGN)
HOME_AFTER = os.getenv("HOME_AFTER", "open")
# 방금 홈 복귀가 됐으면(이 초 안, 그 뒤 로봇이 안 움직임) 다음 정책 전 홈 맞춤은 다시 재지 않는다(이미 홈 측정 4.0~4.2초)
HOME_FRESH_S = float(os.getenv("HOME_FRESH_S", "120"))
# Studio 세션이 팔을 쥐고 있어 거부되면(10-22 r22): 세션을 멈추거나 잠금이 풀리기를 이만큼 기다리며 다시(초)
STUDIO_WAIT_S = float(os.getenv("STUDIO_WAIT_S", "15"))
STUDIO_POLL_S = float(os.getenv("STUDIO_POLL_S", "1.0"))

# 받아쓰기 환청 거르기(10-22 r16, agent/hearing.py): 녹음 정보가 오면 말소리 길이로도 거른다
# 10-22 실측: 무음·잡음 0.00~0.12초 / 합성 "멈춰" 0.24초·초당 글자 최대 15.6 / 실물 명령 0.78초~, 초당 7.9자 이하
STT_MIN_SPEECH_S = float(os.getenv("STT_MIN_SPEECH_S", "0.2"))    # 말소리가 이보다 짧으면 못 들은 것
STT_MAX_CPS = float(os.getenv("STT_MAX_CPS", "20"))               # 글자 수 ÷ 말소리 초가 이보다 크면(말보다 글이 길면) 환청

# RAG
STANDARD_PATH = ROOT / "docs" / "standard.md"
STANDARD_FALLBACK = FIXTURES_DIR / "standard_sample.md"
# 리랭크 지연 = 후보 수 × 조각당 시간(10-21 낮 CPU 약 1초, 밤 GPU 0.07초). 값은 eval --rag --sweep 로 골랐다(9/9 유지):
# 4개·생략 0.65/0.04 = 9문장 중 6개만 리랭크, GPU 평균 0.24초. 리랭커가 CPU 로 돌아가도 최악 약 4초에서 멈춘다
# 10-22 세 문서(99조각): 합친 순위 앞 6개 + 임베딩 코사인 앞 3개(rag.VEC_TOP)를 원문 질의로 한 번 리랭크.
# 4개만이면 기준서 6.1("손은 어디에", 코사인 2위·글자 36위)이 리랭크에 못 들었다. 51문항 상위 3개 51/51, 검색 최대 1.9초
RAG_RERANK_N = int(os.getenv("RAG_RERANK_N", "6"))          # 리랭크할 후보 수(0 = 리랭크 안 함)
RAG_SKIP_SCORE = float(os.getenv("RAG_SKIP_SCORE", "0.65"))   # 임베딩 1위 코사인이 이 이상이고
RAG_SKIP_MARGIN = float(os.getenv("RAG_SKIP_MARGIN", "0.04")) # 2위와 이만큼 벌어지면 리랭크 생략
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
# 리랭크 최고점이 이보다 낮으면 기준서 밖 질문으로 본다(10-21: "점심 메뉴" 0.06, 기준서 질문 0.79~0.95)
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.3"))
RAG_KEEP_SCORE = float(os.getenv("RAG_KEEP_SCORE", "0.1"))   # 이보다 낮은 조각은 LLM 근거에서 뺀다
# 임베딩 1위 코사인이 이보다 낮으면 기준서 밖 질문(10-21: 밖 4문장 0.31~0.46, 기준서 9문장 0.54~0.76).
# 리랭크를 건너뛴 경우에는 1위에서 RAG_KEEP_COS_GAP 안쪽 조각만 근거로
# 10-22 r10 문서 고친 뒤 0.50 → 0.45: "받아쓰기는 어느 장치에서 돌아?" 의 1위 코사인이 0.506 → 0.488 로 내려가 문서 밖이
# 됐다(리랭크 0.87, 정답 안내서 3.3). 세 문서 기준 문서 밖 9문장은 0.43 이하이거나 0.50 이상이라 판정이 그대로다
RAG_OOD_COS = float(os.getenv("RAG_OOD_COS", "0.45"))
RAG_KEEP_COS_GAP = float(os.getenv("RAG_KEEP_COS_GAP", "0.05"))

# 질문 답(10-22 r10): 답 LLM 최대 토큰, 질문으로 보이는 말이면 판별과 동시에 검색 시작(판별이 question 이 아니면 버림)
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "80"))
# 미리 검색은 기본으로 끈다(10-22 03:12 UI 끝난 뒤 47문항 비교): 질문꼴 지름길로 판별이 0.07초라 겹칠 틈이 거의 없고,
# LLM 으로 판별하는 질문은 리랭크와 GPU 를 나눠 써 첫마디 최대가 1.22 → 2.39초로 늘었다. 정확도는 같음(42/42·5/5)
QA_PREFETCH = os.getenv("QA_PREFETCH", "0") == "1"

CHECKPOINT_DB = RUNS_DIR / "agent_checkpoints.sqlite"
# 가짜 작업대(BALL_MOCK=1: selftest·eval·연속 시험)의 대화 상태는 따로 둔다. 10-22 soak 정리 뒤에도 시험 체크포인트가
# 실물 DB 에 25MB 쌓였다(작업 기록 runs/<날짜>_mock/ 과 같은 이유)
CHECKPOINT_DB_MOCK = RUNS_DIR / "agent_checkpoints_mock.sqlite"


def checkpoint_db() -> Path:
    return CHECKPOINT_DB_MOCK if os.getenv("BALL_MOCK") == "1" else CHECKPOINT_DB


def standard_source() -> Path:
    """RAG 원본: docs/standard.md(UI·문서 세션), 없으면 시험용 임시 파일."""
    return STANDARD_PATH if STANDARD_PATH.exists() else STANDARD_FALLBACK


# RAG 문서(10-22: 기준서 하나 → 세 문서). (키, 답에 쓰는 문서 이름, 경로). 조항 번호가 문서마다 따로라 인용에 이름을 붙인다
RAG_DOCS = [
    ("standard", "작업 기준서", None),                          # 경로 = standard_source()
    ("robot_manual", "로봇 운용 설명서", ROOT / "docs" / "robot_manual.md"),
    ("system_guide", "시스템 안내서", ROOT / "docs" / "system_guide.md"),
]
DOC_NAMES = {k: n for k, n, _ in RAG_DOCS}


def rag_sources() -> list[tuple[str, str, Path]]:
    """있는 문서만(기준서는 늘 — 없으면 시험용 임시본)."""
    out = []
    for key, name, path in RAG_DOCS:
        path = standard_source() if path is None else path
        if path.exists():
            out.append((key, name, path))
    return out
