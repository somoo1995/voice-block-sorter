"""의도 판별: ① "멈춰" 규칙 → ② 임베딩 라우터(예문 코사인) → ③ 애매하면 LLM with_structured_output."""
import hashlib
import json
import re
import threading
import time
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from . import config, llm
from .intent_examples import DESCRIPTIONS, EXAMPLES, INTENTS

# 물체 이름을 넣지 않는다. "block-sorting" 으로 바꾸자 "결과 보고에는 뭐가 들어가요?" 가 report 로 임베딩 확정됐다
# (차 0.023 → 0.046, 10-21 밤). 중립 문구는 76문장에서 확정 오답 0
ROUTER_INSTRUCT = "Classify the intent of a spoken command to a sorting robot"
NO_COUNT = {"report", "question", "scene", "plan", "stop", "none", "home"}   # plan 의 개수는 단계마다(graph.make_plan)
MOVING = ("defect", "red", "blue", "handoff", "plan")          # 로봇을 움직이는 의도(plan 은 단계·숫자 배치)

# ---------------------------------------------------------------- ① 멈춰 규칙
_STOP_WORDS = ("멈춰", "멈춰요", "멈춰라", "멈추", "정지", "그만", "스톱", "중단", "중지", "stop")
_QUESTION_CUES = ("?", "뭐", "왜", "어디", "어떻게", "언제", "무슨", "버튼")


def is_stop(text: str) -> bool:
    """LLM 을 거치지 않는 즉시 중단. 질문("비상 정지 버튼은 어디야?")은 빼고, 애매하면 멈추는 쪽으로."""
    t = text.strip().lower().replace(" ", "")
    if not t or not any(w in t for w in _STOP_WORDS):
        return False
    return not any(c in t for c in _QUESTION_CUES)


# ---------------------------------------------------------------- 홈 자세 명령(10-23)
# 10:12~10:14 실물: "홈자세로 고쳐줘" → stop("멈췄습니다"), "홈자세로 변경"·"그리퍼를 기본 시작 자세로 되돌려줘" → question.
# 의도 목록·임베딩 예문·LLM 은 그대로 두고 규칙으로만 잡는다(GPU 가 멈추면 LLM 판별이 5초 제한에 걸린다). '멈춰'·'정지'는 is_stop 이 먼저
_HOME_RE = re.compile(r"홈\s*(?:자세|위치|포지션|자리)?\s*(?:으로|로)|홈\s*(?:자세\s*)?(?:복귀|이동)|홈\s*자세\s*(?:를|을)?\s*(?:잡|취|만들)"
                      r"|(?:처음|기본|시작|원래|초기)\s*(?:시작\s*)?(?:자세|위치|자리)\s*(?:으로|로)|제자리\s*로|원위치")
_HOME_OBJ = re.compile(r"블[록럭]|볼(?![게까래수])|분홍|핑크|초록|녹색|연두|불량|그릇|숫자|모양")   # 다른 동작이 섞인 말은 판별로
_HOME_NOT = re.compile(r"(?:가지|보내지|돌리지|하지)\s*마|말고|려면|(?:갔|됐|왔|돌아왔)(?:어|나|니|냐|어요|습니까)\s*\??\s*$")


def is_home(text: str) -> bool:
    """팔을 홈 자세로 보내라는 말인가("홈으로"·"홈 자세로 돌려 줘"·"처음 자세로"·"기본 자세로 되돌려"·"제자리로").
    묻는 말("홈 자세는 어떻게 맞춰?"·"홈으로 갔어?")·하지 말라는 말·블록·색이 섞인 말은 아니다."""
    t = text or ""
    return bool(_HOME_RE.search(t)) and not (_HOME_NOT.search(t) or _HOME_OBJ.search(t) or _WH_ANY.search(t))


# ---------------------------------------------------------------- 그릇에서 꺼내기(10-23, 리허설 초기화)
# "그릇에 있는 거 다시 투입 구역으로 옮겨 줘"·"블록 다시 꺼내 줘"·"원래대로 돌려놔"·"초기화해 줘"·"다 꺼내 놔"·"초록 그릇 거 꺼내 줘".
# 홈 명령처럼 규칙으로만 잡는다(의도 목록·임베딩 예문·LLM 은 그대로). '꺼내'는 그릇·다시·도로·다(전부)·투입이 같이 있을 때만
# ("불량 꺼내 줘"는 골라내기일 수 있어 판별로). 홈 규칙과 겹치지 않는다(홈은 블록·그릇이 섞이면 아님, 여기는 홈·자세·팔이면 아님)
_UNSORT_RE = re.compile(r"원래\s*(?:대로|상태로)|처음\s*(?:대로|상태로)|초기화|리셋|그릇\S*\s*(?:\S+\s*)?비[워우]")
_UNSORT_TAKE = re.compile(r"꺼[내낼낸냈]")
_UNSORT_FROM = re.compile(r"그릇|다시|도로|투입|(?<![가-힣])(?:다|전부|모두|싹|몽땅)(?=\s|꺼)")
_UNSORT_TO = re.compile(r"투입\s*(?:구역|칸|자리|쪽)?\s*(?:으로|로)")
_UNSORT_VERB = re.compile(r"옮|돌려|돌리|갖다|가져다|놓|놔|넣|보내|꺼내")
_UNSORT_NOT = re.compile(
    r"려면|수\s*(?:있|없)|(?:꺼내|돌리|옮기|비우|하)지\s*(?:마|말)|말고|안\s*돼"            # 묻는 말·하지 말라는 말
    r"|(?:꺼냈|비웠|됐|했|놨|옮겼)(?:어|나|니|냐|어요|습니까|지)\s*\??\s*$"              # "다 꺼냈어?"
    r"|그릇\s*(?:으로|로|에|안에|안으로)\s*(?:다시\s*)?(?:넣|옮|담|놓|놔|보내|갖다)"      # 그릇으로 넣는 말 = 선별
    r"|(?<![말됐하])고\s+\S|그리고|(?:초기화|리셋)\s*하고|(?:은|한|낸|운)\s*(?:다음|뒤|후)"  # 여러 단계 → 판별(계획)
    r"|홈|자세|(?<![가-힣])팔(?:을|은|이|도)?(?![가-힣])|그리퍼|집게|속도|모드|방식|설정|기록|로그|대화|카메라|화면|소리|볼륨|음량"
    r"|로봇\s*(?:을|를)?\s*(?:초기화|리셋)|떨어진|구역\s*밖|바깥")


def is_unsort(text: str) -> bool:
    """그릇에 있는 블록을 투입 구역으로 되돌리라는 말인가(리허설 초기화). 묻는 말·하지 말라는 말·그릇으로 넣는 말·
    여러 단계·팔·홈·설정·기록 말("속도 원래대로"·"기록 초기화")은 아니다."""
    t = text or ""
    hit = _UNSORT_RE.search(t) or (_UNSORT_TAKE.search(t) and _UNSORT_FROM.search(t)) \
        or (_UNSORT_TO.search(t) and _UNSORT_VERB.search(t))
    return bool(hit) and not (_UNSORT_NOT.search(t) or _WH_ANY.search(t))


def unsort_colors(text: str) -> list[str] | None:
    """꺼낼 그릇: 말에 나온 색·불량(red·blue·defect — sort_blocks 의 color 와 같은 이름, 말한 차례). 없으면 None = 그릇 전부."""
    found = sorted((m.start(), k) for k, p in WORD_INTENT for m in [re.search(p, text or "")] if m)
    return [k for _, k in found] or None


# ---------------------------------------------------------------- 명령 아님(10-22 밤)
# 23:08 "아, 아, 마이크 테스트, 마이크 테스트" 가 red 로 정책 2회를 돌렸다(LLM 시간 초과 → 자신 없는 임베딩 1위).
# 움직이는 의도는 물체·색·불량·손·옮기기 낱말이 있을 때만 받는다. docs/intents.csv 동작 60문장·라우터 동작 예문 전부에 있다
_GROUND = re.compile(r"블[록럭]|볼(?![게까래수])|분홍|핑크|초록|녹색|연두|불량|엑스|하자|[xX]|손|그릇|숫자|모양|하트|동그라미|세모"
                     r"|글자|옮|담[아자고기]|넣|치워|치우|골라|분리|정리|건네|전달|쥐여|놔|놓|만들|쌓|세워|배치|빼")
_HEAR_CHECK = re.compile(r"마이크|테스트|체크|들려|들리|들립|듣고\s*있|알아\s*듣|목소리|소리\s*(?:가\s*)?(?:잘|나와|나오|커|작)|녹음")
_AUDIENCE = re.compile(r"여러분|다들|모두들")
_CHAT_CUE = re.compile(r"안녕|반갑|감사|고맙|수고|잠깐|잠시|라운드|여러분|다들|그러니까|저기요?|시작하겠|넘어가")
_CHAT_ONLY = re.compile(r"(?:(?:음+|어+|아+|에+|흠+|으+|그+|자|네|예|응|오케이|좋아요?|좋습니다|그럼|이제|다음)[\s,.!?~]*)+")
# 의문사·작업 낱말이 있으면 문서 질문·보고일 수 있다("마이크는 어느 장치에서 돌아?", "다들 수고했어 결과 알려 줘")
_WH_ANY = re.compile(r"얼마|몇|왜|어떻게|어떤|무엇|뭐|뭘|언제|어디|무슨|어느|누구|누가")
_TOPIC = re.compile(r"결과|보고|현황|개수|성공|실패|기준|규칙|안전|설명서|안내서|기준서|문서|카메라|작업대|구역|정책|로봇|시연")
NONE_WHY = {"hear": "듣고 있는지 묻는 말·마이크 시험", "chat": "잡담·인사·청중에게 하는 말",
            "unsure": "LLM 판별 실패 + 임베딩 자신 없음", "vague": "움직일 낱말(블록·색·불량·손·옮기기) 없음"}


def small_talk(text: str) -> str | None:
    """명령 아닌 말이면 "hear"(마이크 시험·잘 들려?) | "chat"(인사·잡담·청중에게), 아니면 None. LLM 없이 규칙으로."""
    t = (text or "").strip()
    if not t or _GROUND.search(t) or _WH_ANY.search(t) or _TOPIC.search(t) or SELF_Q.search(t):
        return None
    if _HEAR_CHECK.search(t):
        return "chat" if _AUDIENCE.search(t) else "hear"
    if _CHAT_CUE.search(t) or _CHAT_ONLY.fullmatch(t):
        return "chat"
    return None


def _llm_down(e: Exception) -> bool:
    """서버가 답하지 않은 것(시간 초과·연결 실패)인가. 구조화 출력 파싱 실패 등은 False."""
    return any(w in type(e).__name__ for w in ("Timeout", "Connect"))


# ---------------------------------------------------------------- 집는 방식 단서(10-22 r21)
# 명령마다 ACT 정책 / 좌표 집기를 고른다. 단서는 조사(으로·로·써서·해서·대로)가 붙은 꼴만 본다 → "ACT 정책은 몇 개야?" 같은
# 질문은 건드리지 않는다. 판별에는 단서를 뺀 말을 쓴다: 단서가 붙은 7문장이 임베딩 확정을 못 해 LLM 으로 가 약 1초 늦었다
_M_ACT = r"(?:a\.?\s?c\.?\s?t|액트|에이씨티)\s*(?:정책)?|(?:학습|배운|익힌)(?:한|된)?\s*(?:동작|정책)|정책|학습(?=\s*(?:모드|방식|방법))"
_M_COORD = r"(?:좌표\s*)?계산|좌표(?:\s*집기)?|yolo|욜로|역기구학"
_METHOD_CUE = re.compile(rf"(?:(?P<act>{_M_ACT})|(?P<coord>{_M_COORD}))"
                         r"(?:\s*(?:모드|방식|방법|버전))?"  # "좌표모드로"·"ACT 방식으로"(10-23 실물: 모드가 붙으면 단서를 놓쳐 기본값 ACT 로 갔다)
                         r"\s*(?:으로\s*해서|로\s*해서|으로|로|을\s*써서|를\s*써서|써서|해서|대로)"
                         r"(?P<neg>\s*(?:하지\s*말고|말고|대신|빼고|없이|아니라|아니고))?", re.I)


def method_cue(text: str) -> tuple[str | None, str | None, str]:
    """말 속의 집는 방식 → ("act"|"coord"|None, 단서 원문, 단서를 뺀 말). 여럿이면 뒤의 것, "~로 말고"면 반대쪽."""
    method, cues = None, []
    for m in _METHOD_CUE.finditer(text or ""):
        kind = "act" if m.group("act") else "coord"
        method = ({"act": "coord", "coord": "act"}[kind]) if m.group("neg") else kind
        cues.append(m.group(0).strip())
    clean = re.sub(r"\s{2,}", " ", _METHOD_CUE.sub(" ", text or "")).strip() if cues else (text or "")
    return method, (" · ".join(cues) or None), clean


# ---------------------------------------------------------------- 위치 단서(10-23)
# "초록 블록 왼쪽에 있는 거 옮겨 줘" → which="left". 같은 색 후보 가운데 위 카메라 사진 u 순서로 고르는 일은 로봇 함수
# (sort_blocks which=)가 한다. 09:50 실물: "초록블록 왼쪽에 있는거 좌표로 옮겨줘" 가 초록 2개를 다 옮기려 했다
# 10-23 13:30: 위·아래(top·bottom = 위 카메라 v, top = 로봇에서 먼 쪽)·가까운·먼(near·far = 로봇 베이스까지 거리)·
# "분홍 옆"(next_to)도 읽는다(로봇 sort_blocks which= 가 받는 값, robot/coord_pick/plan.py WHICH·NEXT_TO).
# 규칙이 못 읽은 위치 낱말("두 번째"·"구석" 등)이 남으면 classify 가 LLM 에 target 을 묻는다(rule_target by "need_llm")
WHICH_KO = {"left": "왼쪽", "right": "오른쪽", "middle": "가운데", "top": "맨 위", "bottom": "맨 아래",
            "near": "로봇에 가장 가까운", "far": "로봇에서 가장 먼"}
NEXT_KO = {"red": "분홍", "blue": "초록", "red_defect": "분홍 불량", "blue_defect": "초록 불량", "defect": "불량"}
_WHICH = re.compile(
    r"(?P<left>왼\s*(?:쪽|편)|좌측|좌편)|(?P<right>오른\s*(?:쪽|편)|우측)"
    r"|(?P<middle>(?:한\s*)?가운데|중앙|중간(?=\s*(?:거|것|꺼|건|블록|쪽|에\s*있)))"
    r"|(?P<top>(?:맨|제일|가장|젤)\s*위(?![치해험])|윗\s*(?:쪽|줄)|위\s*(?:쪽|줄)|(?<![가-힣])위(?=에|\s|$))"
    r"|(?P<bottom>(?:맨|제일|가장|젤)\s*(?:아래|밑)|아랫\s*(?:쪽|줄)|아래\s*(?:쪽|줄)|(?<![가-힣])(?:아래|밑)(?=에|\s|$))"
    r"|(?P<near>(?:로봇|팔|베이스)\S*\s*(?:(?:제일|가장|젤)\s*)?가까[운이]|로봇\s*쪽|(?:(?:제일|가장|젤)\s*)?가까[운이])"
    r"|(?P<far>(?:로봇|팔|베이스)\S*\s*(?:(?:제일|가장|젤)\s*)?(?:먼(?![지저])|멀리)|(?:제일|가장|젤)\s*먼(?![지저])"
    r"|(?<![가-힣])먼(?=\s*(?:거|것|꺼|쪽|블[록럭]|분홍|핑크|초록|녹색|연두|불량))|멀리\s*있)")
_WHICH_NEG = re.compile(r"\S*\s*(?:(?:에\s*)?있는\s*)?(?:(?:거|것|꺼|건|블록)\S*\s*)?(?:말고|빼고|아니고|아니라|대신)")
# 기준 블록 뒤에는 조사만(“초록 블록 중에 가까운 거”는 초록 가운데 로봇에 가까운 것 = near)
_REF_WORD = r"(?P<ref>(?:분홍|핑크|초록|녹색|연두)색?\s*(?:블[록럭]\s*)?(?:불량품?)?|불량품?|엑스)(?:이랑|랑|하고|과|와|의|이|가)?\s*"
_NEXT_TO = re.compile(_REF_WORD + r"(?:거\s*|것\s*)?(?:바로\s*)?(?:옆|근처|주변|가까이|가까운|곁|붙)\S*"
                      r"(?:\s*(?:있는|있던|놓인|붙은))?")
_REF_DIR = re.compile(_REF_WORD + r"(?:위|아래|밑|앞|뒤|사이|왼\s*쪽|오른\s*쪽)")
_KIND = [("defect", r"불량|엑스"), ("red", r"분홍|핑크"), ("blue", r"초록|녹색|연두")]
_BOWL_AT = re.compile(r"그릇\S*\s*(?:위|안|속)")
# 규칙이 위치로 못 읽은 낱말(남으면 LLM). '제일·가장·맨'만으로는 아니다("제일 먼저 분홍 옮겨")
_POS_NONE = re.compile(r"번째|사이")      # 순서·사이 — 로봇이 못 고르고 LLM 도 엉뚱한 값을 골랐다(13:57) → 거부
_POS_LEFT = re.compile(r"옆|사이|앞(?:쪽|에|줄)|뒤(?:쪽|에\s*있|줄)|끝(?:에|쪽|자리|\s*거)|구석|근처|주변|가장자리|모서리|번째"
                       r"|안쪽|바깥쪽|위(?:쪽|에|$)|아래|밑(?:에|쪽)|가까|먼\s*(?:거|것|쪽)|멀리|왼|오른|가운데|중앙")


def _kind(word: str) -> str:
    """기준 낱말 → next_to 이름: 분홍 red · 초록 blue · 분홍 불량 red_defect · 초록 불량 blue_defect · 색 없는 불량 defect."""
    bad, pink, green = re.search(r"불량|엑스", word), re.search(r"분홍|핑크", word), re.search(r"초록|녹색|연두", word)
    if bad:
        return "red_defect" if pink else "blue_defect" if green else "defect"
    return "red" if pink else "blue"


def strip_ref(text: str) -> str:
    """기준 블록 구절("분홍 옆에 있는"·"불량 근처")을 뺀 말 — 옮길 블록의 색·불량 판단과 임베딩·LLM 판별에 쓴다
    (13:51 실물: "분홍 옆에 있는 초록 옮겨줘" 가 임베딩 plan 0.92(차 0.013) → LLM 으로 갔다. 빼면 "초록 옮겨줘")."""
    return re.sub(r"\s{2,}", " ", _NEXT_TO.sub(" ", text or "")).strip()


_MOVE_VERB = re.compile(r"옮|담|넣|골라|치우|치워|빼|가져|집어|놔|놓")


def route_text(text: str) -> str:
    """판별(임베딩·LLM)에 쓸 말: 기준 구절·위치 낱말을 빼고 색 뒤에 '블록'을 붙인다(위치 문장만, 13:59 실측 —
    "분홍 옆에 있는 초록 옮겨줘" → "초록 옮겨줘" 는 plan 0.91 로 자신 없음, "초록 블록 옮겨줘" 는 blue 0.99 색 규칙 확정)."""
    # 규칙으로 위치를 읽은, 옮기라는 말만. "분홍 위에 있는 초록"(ref_rel)은 '위'를 빼면 옮길 색이 흐려지고, 묻는 말은 그대로
    if rule_target(text)["by"] != "rule" or "?" in (text or "") or not _MOVE_VERB.search(text or ""):
        return text
    t = _WHICH.sub(" ", strip_ref(text))
    t = re.sub(r"\s{2,}", " ", t).strip()
    if not t or t == (text or "").strip():
        return text
    return re.sub(r"(분홍|핑크|초록|녹색|연두)(?!\s*(?:색\s*)?(?:블[록럭]|불량))", r"\1 블록", t)


def rule_target(text: str) -> dict:
    """말 속 위치·관계 → {which, next_to, by, pos}. which: left·right·middle·top·bottom·near·far·mixed(둘 이상·뺀 쪽만,
    거부) · next_to: red·blue·red_defect·blue_defect·defect(색 없는 불량 — 로봇 이름이 없어 거부).
    by: rule(규칙으로 읽음) · need_llm(규칙이 못 읽은 위치 낱말이 남음 → classify 가 LLM 에 물음) · unsupported("두 번째"·"사이",
    거부) · ref_rel("분홍 위에 있는 초록"
    처럼 다른 블록 기준 위·아래 — 로봇이 못 고름, 거부) · None(위치 없음)."""
    t = _BOWL_AT.sub(" ", text or "")
    for m in _REF_DIR.finditer(t):          # "분홍 위에 있는 초록": 기준과 다른 색·불량이 뒤에 나오면 다른 블록 기준 방향
        base = {k for k, p in _KIND if re.search(p, m.group("ref"))}
        later = {k for k, p in _KIND if re.search(p, t[m.end():])}
        if later - base:                    # "초록블록 왼쪽에 있는거"(뒤에 다른 색 없음)는 초록 가운데 왼쪽 = which
            return {"which": None, "next_to": None, "by": "ref_rel", "pos": [m.group(0).strip()]}
    nxt = [(m.span(), _kind(m.group("ref"))) for m in _NEXT_TO.finditer(t)]
    for (a, b), _ in nxt:
        t = t[:a] + " " * (b - a) + t[b:]
    keep, neg, spans = [], False, []
    for m in _WHICH.finditer(t):
        spans.append(m.span())
        if _WHICH_NEG.match(t, m.end()):
            neg = True
        else:
            keep.append(m.lastgroup)
    for a, b in spans:
        t = t[:a] + " " * (b - a) + t[b:]
    left = [m.group(0) for m in _POS_LEFT.finditer(t)]
    refs = {k for _, k in nxt}
    which = "mixed" if len(set(keep)) > 1 or (neg and not keep) or (keep and refs) or len(refs) > 1 else (
        keep[-1] if keep else None)
    next_to = None if which == "mixed" or not refs else refs.pop()
    by = "unsupported" if _POS_NONE.search(t) else "need_llm" if left else "rule" if which or next_to else None
    return {"which": which, "next_to": next_to, "by": by, "pos": left}


def parse_which(text: str) -> str | None:
    """말 속 위치(규칙) → left·right·middle·top·bottom·near·far. 없으면 None. "왼쪽 말고 오른쪽 거" → right(말고 붙은 쪽은 뺀다).
    둘 이상("왼쪽이랑 오른쪽 거")이거나 뺀 쪽만 있으면("왼쪽 거 말고") "mixed" — 한 번에 하나만 고를 수 있어 거부한다."""
    return rule_target(text)["which"]


# ---------------------------------------------------------------- 불량 색(10-23)
# 11:35 실물 "초록 불량 옮겨 줘" → defect → 분홍 불량까지 2개를 다 옮겼다. defect 의도에 분홍·초록 낱말이 있으면 그 색 불량만
# (graph 가 sort_blocks(["red_defect"|"blue_defect"]) 로 부른다). "분홍 그릇"처럼 그릇 이름의 색은 대상이 아니다
DEFECT_KO = {"red_defect": "분홍 불량", "blue_defect": "초록 불량"}
_DCOLOR = re.compile(r"(?P<red_defect>분홍|핑크)|(?P<blue_defect>초록|녹색|연두)")
_DCOLOR_NEG = re.compile(r"\S*\s*(?:(?:거|것|꺼|건|블[록럭]|불량)\S*\s*)?(?:말고|빼고|아니고|아니라|대신)")
_DCOLOR_IN = re.compile(r"(?:분홍|핑크|초록|녹색|연두)\S*\s*(?:블[록럭]\S*\s*)?불량"
                        r"|불량\S*\s*(?:중에?서?\s*|가운데\s*)?(?:분홍|핑크|초록|녹색|연두)")
_BOWL_NAME = re.compile(r"(?:분홍|핑크|초록|녹색|연두)색?\s*그릇")


def defect_color(text: str) -> str | None:
    """defect 의도의 불량 색 → "red_defect"(분홍)|"blue_defect"(초록). 없거나 둘 다("분홍이랑 초록 불량")면 None(불량 전부).
    "분홍 불량 말고 초록 불량" → blue_defect(말고 붙은 색은 뺀다). 뺀 색만 있으면("초록 말고 불량") "mixed" — 그래프가 거부한다."""
    t = _BOWL_NAME.sub(" ", text or "")
    keep, neg = set(), False
    for m in _DCOLOR.finditer(t):
        if _DCOLOR_NEG.match(t, m.end()):
            neg = True
        else:
            keep.add(m.lastgroup)
    if len(keep) == 1:
        return keep.pop()
    return "mixed" if neg and not keep else None


def colored_defect(text: str) -> bool:
    """색이 붙은 불량("초록 불량"·"불량 중에 분홍")이 있나 — 여러 단계 계획은 단계마다 불량 색을 못 넘겨 거부한다(graph.planner_node)."""
    return bool(_DCOLOR_IN.search(_BOWL_NAME.sub(" ", text or "")))


# 13:05 실물: 받아쓰기가 붙여 쓴 "초록불량 옮겨줘" 를 임베딩이 blue(0.94)로 골라 정상 초록을 옮겼다("분홍불량" → red 0.93).
# 불량·엑스·X 표시 낱말이 있으면 classify 가 red·blue 를 defect 로 바꾼다. '불량 그릇'은 그릇 이름이라 아님,
# 뒤에 말고·빼고·아닌·없는이 붙으면 불량을 빼는 말("불량 말고 초록 옮겨 줘" → blue 그대로)
_DEFECT_WORD = re.compile(r"불량(?!\s*(?:그릇|칸|통))|엑스|[xX]\s*(?:표\s*시|자|있|붙|그려|친|쳐|블[록럭])")
_DEFECT_NOT = re.compile(r"[가-힣]{0,2}\s*(?:(?:블[록럭]|거|것|꺼|건|표\s*시)[가-힣]{0,2}\s*)?(?:(?:그대로|놔|남겨)\s*)?"
                         r"(?:두고|둬|말고|빼고|제외|아니|아닌|대신|없는|없이)")


def defect_word(text: str) -> bool:
    """불량을 시키는 말인가: 불량·엑스·X 표시 낱말 중 하나라도 뒤에 빼는 말(말고·빼고·아닌·없는…)이 안 붙었다."""
    t = text or ""
    return any(not _DEFECT_NOT.match(t, m.end()) for m in _DEFECT_WORD.finditer(t))


# ---------------------------------------------------------------- 개수 규칙
_NATIVE = {"하나": 1, "둘": 2, "셋": 3, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9}
_PRENOUN = {"한": 1, "두": 2, "세": 3, "석": 3, "네": 4, "넉": 4, "열": 10}
_UNIT = r"\s*(?:개|알|덩이)"
# LLM 이 낸 개수를 받아도 되는지 보는 느슨한 신호. 띄어쓰기 없는 STT 문장("빨간볼하나내손에줘")도 잡는다
_NUM_HINT = re.compile(r"\d|" + "|".join(_NATIVE) + r"|(?:" + "|".join(_PRENOUN) + r")\s*(?:개|알)")


def parse_count(text: str) -> int | None:
    """'두 개만'·'3개'·'하나' → 정수. '하나씩'(한 번에 하나)·'하나도'는 개수가 아니다."""
    m = re.search(r"(?<!\d)(\d+)(?!\d)(?:" + _UNIT + r")?+(?!\s*(?:번|초|분|시|씩))", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?<![가-힣])(" + "|".join(_PRENOUN) + r")" + _UNIT + r"(?!\s*씩)", text)
    if m:
        return _PRENOUN[m.group(1)]
    m = re.search(r"(?<![가-힣])(" + "|".join(_NATIVE) + r")(?=$|\s|만|를|을|[.!?])", text)
    if m:
        return _NATIVE[m.group(1)]
    return None


# ---------------------------------------------------------------- ② 임베딩 라우터
class EmbeddingRouter:
    def __init__(self):
        self.emb = llm.QwenEmbeddings(ROUTER_INSTRUCT)
        self.labels: list[str] = []
        texts: list[str] = []
        for intent in INTENTS:
            for t in EXAMPLES[intent]:
                self.labels.append(intent)
                texts.append(t)
        self.matrix = self._load_or_embed(texts)

    def _load_or_embed(self, texts: list[str]) -> np.ndarray:
        key = hashlib.sha1(json.dumps([config.EMB_MODEL, ROUTER_INSTRUCT, texts],
                                      ensure_ascii=False).encode()).hexdigest()[:12]
        path = config.CACHE_DIR / f"router_{key}.npy"
        if path.exists():
            return np.load(path)
        # 예문도 질의와 같은 지시문으로 넣는다(짧은 명령끼리 비교라 대칭이 더 잘 갈린다). 한 번에 묶어 보낸다
        vecs = np.array(self.emb.embed_documents(
            [f"Instruct: {ROUTER_INSTRUCT}\nQuery: {t}" for t in texts]), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs

    def route(self, text: str) -> dict:
        v = np.array(self.emb.embed_query(text), dtype=np.float32)
        v /= np.linalg.norm(v)
        sims = self.matrix @ v
        best: dict[str, float] = {}
        for label, s in zip(self.labels, sims):
            best[label] = max(best.get(label, -1.0), float(s))
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        (top, s1), (_, s2) = ranked[0], ranked[1]
        return {"intent": top, "score": round(s1, 4), "margin": round(s1 - s2, 4),
                "top3": [(k, round(s, 4)) for k, s in ranked[:3]]}


_router: EmbeddingRouter | None = None
_router_lock = threading.Lock()


def get_router() -> EmbeddingRouter:
    global _router
    with _router_lock:
        if _router is None:
            _router = EmbeddingRouter()
        return _router


# ---------------------------------------------------------------- ③ LLM 구조화 출력
class IntentOut(BaseModel):
    intent: Literal["defect", "red", "blue", "handoff", "report", "question", "scene", "plan", "stop", "none"] = Field(
        description="명령의 의도")
    count: Optional[int] = Field(
        default=None, description="'두 개만'처럼 개수를 말했으면 정수, 말하지 않았으면 null")


# 위치·관계(10-23 13:30) — 의도와 따로 묻는다. 13:51 실물: IntentOut 에 which·next_to 를 더하자 strict 스키마(필드 모두 필수)에서
# qwen3-4b 가 intent·count 뒤 다음 키를 안 쓰고 공백만 내다 max_tokens 에 걸렸다(LengthFinishReasonError, 96·400 둘 다).
# 값 하나짜리 스키마면 그 값을 쓰고 바로 닫는다
# 값은 한국어 위치 이름만(13:55 실측: 영어 이름 next_to_red 등이나 "초록 불량 옆"처럼 색으로 시작하는 값이 있으면 4B 가 말 속
# 색 낱말에 끌려 "두 번째"·"저 끝"까지 그 값을 골랐다). "◯ 옆"은 규칙(rule_target)이 읽는다
# 값 하나만 고르게 하면 한쪽("오른쪽")으로 쏠려, 위치 낱말을 먼저 그대로 옮겨 쓰게 한 뒤 고른다(13:57 실측: 저 끝·구석·끝 → 먼).
# 그래도 "두 번째"·"사이"를 가까운으로 골라 그 둘은 규칙이 LLM 없이 거부한다(_POS_NONE)
TARGET_KO = {"없음": "none", "왼쪽": "left", "오른쪽": "right", "가운데": "middle", "맨 위": "top", "맨 아래": "bottom",
             "로봇에 가까운": "near", "로봇에서 먼": "far"}
TARGETS = tuple(TARGET_KO)


class TargetOut(BaseModel):
    words: str = Field(default="", description="말에서 위치를 나타내는 낱말만 그대로(없으면 빈 문자열)")
    target: Literal[TARGETS] = Field(description="그 낱말이 뜻하는 위치")


_SYSTEM = (
    "너는 스펀지 블록 선별 로봇의 명령 분류기다. 사용자의 말을 아래 의도 중 정확히 하나로 고르고 개수를 뽑는다.\n"
    + "\n".join(f"- {k}: {v}" for k, v in DESCRIPTIONS.items())
    + "\n규칙: count 는 사용자가 개수를 말했을 때만 정수, 아니면 null. report·question·scene·plan·stop·none 은 항상 null."
    "\n블록·색·불량·손·그릇이나 옮기기·담기·건네기 같은 낱말이 없으면 동작 의도(defect·red·blue·handoff·plan)가 아니다."
    " 시키는 것도 묻는 것도 없거나 내 말을 듣고 있는지만 물으면 none."
    "\n색: 분홍=핑크, 초록=녹색=연두."
    "\n분홍 블록을 '손에'·'나한테'·'건네' 달라고 하면 handoff 다. 불량·X 표시가 나오면 색이 있어도 defect 다."
    "\n동작을 두 가지 이상 엮거나('~하고 ~해 줘', '~한 다음에') 예외·나머지를 말하면('~는 두고 나머지', '~ 빼고 다') plan."
    " 동작이 한 가지면 '두 개만'처럼 개수가 있어도 그 의도(red 등)이고 개수는 count 에 넣는다."
    " 'A 말고 B 부터' 처럼 결국 한 가지만 시키면 그 한 가지 의도다. 모양 만들기·쌓기·줄 세우기·다른 색 그릇에 넣기처럼"
    " 배운 동작이 아닌 것을 시키면 plan(거절은 다음 단계가 한다)."
    "\nscene 과 report 구분: 한 구역·한 색·한 그릇이 지금 몇 개인지, 어디(어느 칸)에 있는지, 남은 게 뭔지, 손이 보이는지"
    " 물으면 scene. 이번 작업에서 옮긴 개수·성공 횟수·결과, 그릇마다 결과를 물으면 report."
    "\nscene 과 question 구분: 지금 카메라에 보이는 블록·손이 어떤지 물으면 scene, '~하면 돼?'·'~해야 해?'·'순서가 어떻게 돼?'"
    " 처럼 어떻게 해야 하는지(규칙·방법) 물으면 question."
    # 시험 문장과 다른 말로 든 구분 예시(10-21 밤: 의도가 9개로 늘자 위 규칙만으로는 4B 가 report/question·scene 을 헷갈림)
    "\n구분 예시(정답): '보고서 항목이 뭐야?'→question, '방금 몇 개 옮겼어?'→report, '불량 그릇에 지금 뭐 있어?'→scene, "
    "'로봇 옆에 손을 둬도 돼?'→question, '핑크 ㄱㄱ'→red, '연두 고'→blue, '불량 치우고 분홍 옮겨'→plan, "
    "'분홍 세 개만 담아'→red(count 3), '블록 쌓아 줘'→plan, '소리 괜찮게 나가?'→none, '자 이제 다음 순서로 갈게요'→none"
    "\n'~해 줘' 처럼 시키면 동작 의도, '~야?'·'어떻게 해?' 처럼 기준을 물으면 question 이다."
    "\nreport 와 question 구분: 지금 작업대·그릇에 있는 개수, 이번 작업에서 옮긴 개수·성공 횟수·결과를 물으면 report."
    " 규칙·방법·기준·정의를 물으면(예: '보고서 양식은 어떻게 돼?', '불량은 어디에 담아?') question."
    # 10-21 4B: "결과 보고에는 뭐가 들어가요?"·"결과 보고는 언제 해?" 를 report 로 골랐다("결과 보고" 낱말에 끌림)
    "\n보고도 구분한다: 보고를 시키거나('보고해 줘'·'결과 알려 줘') 이번 작업의 개수·횟수를 물으면 report,"
    " 보고의 항목·시점·방법을 물으면('보고서 항목이 뭐야?'·'보고는 언제 해?') question."
)

# 위 카메라 사진: 위쪽 = 로봇에서 먼 쪽, 아래쪽 = 로봇 쪽(robot/coord_pick/plan.py WHICH)
_TARGET_SYSTEM = (
    "너는 스펀지 블록 선별 로봇에게 한 말에서, 옮길 블록이 같은 색 블록 가운데 어느 위치의 것인지 아래 중 하나만 고른다.\n"
    "왼쪽 · 오른쪽 · 가운데 · 맨 위(작업대 사진 위쪽 = 로봇에서 먼 쪽) · 맨 아래(사진 아래쪽 = 로봇 쪽) · 로봇에 가까운 ·"
    " 로봇에서 먼(저 끝·멀리) · 없음\n"
    "순서('두 번째'), 크기, 두 위치를 함께('오른쪽 아래'), 다른 블록 기준('분홍 위에 있는 초록'·'분홍 사이')처럼"
    " 위 목록으로 나타낼 수 없으면 없음.\n"
    "예: '초록 블록 저 끝에 있는 거' → 로봇에서 먼, '두 번째 초록' → 없음, '초록 중에 로봇 쪽에 있는 거' → 로봇에 가까운"
)


def _llm_classify(text: str, hint: dict | None) -> IntentOut:
    from langchain_core.messages import HumanMessage, SystemMessage

    model = llm.chat(max_tokens=64, timeout=config.LLM_CLASSIFY_TIMEOUT_S).with_structured_output(
        IntentOut, method=config.LLM_SO_METHOD)
    user = f"말: {text}"
    if hint:
        user += "\n(참고: 예문 유사도 후보 " + ", ".join(f"{k} {s:.2f}" for k, s in hint["top3"]) + ")"
    return model.invoke([SystemMessage(_SYSTEM), HumanMessage(user)])


def _llm_target_call(text: str) -> TargetOut:
    """위치·관계 하나(TargetOut.target)를 LLM 구조화 출력으로. 판별과 같은 시간 제한(LLM_CLASSIFY_TIMEOUT_S)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    model = llm.chat(max_tokens=48, timeout=config.LLM_CLASSIFY_TIMEOUT_S).with_structured_output(
        TargetOut, method=config.LLM_SO_METHOD)
    return model.invoke([SystemMessage(_TARGET_SYSTEM), HumanMessage(f"말: {text}")])


_COLOR_WORDS = {"red": ("분홍", "핑크"), "blue": ("초록", "녹색", "연두")}    # 10-21 밤 빨강·파랑 → 분홍·초록
# 두 동작을 엮거나 예외를 다는 말(plan 후보) — 색 규칙으로 red·blue 를 확정하지 않는다
# "A 말고 B 부터" 는 한 가지 동작이라 '말고'는 넣지 않는다(docs/intents.csv: "초록색 거 말고 분홍색 거부터 옮겨" → red)
_MULTI_RE = re.compile(r"(?:(?<![말됐])고\s|하고|다음|나머지|빼고|두고|그대로|먼저|이랑|랑\s|,|그리고|둘 다"
                       r"|(?:다|전부|모두|싹)\s*(?:정리|치워))")
_ALL = re.compile(r"(?:다|전부|모두|싹)\s*(?:그릇|옮겨|담아|넣어)")
_ROBOT_WITH = re.compile(r"(?:로봇|팔|베이스)\s*(?:이랑|하고|랑|과|와)")


class _Multi:
    """두 동작을 엮거나 예외를 다는 말인지. "전부 담아" 는 색이 하나면("남은 녹색 블록 전부 담아 줘") 한 동작이다."""
    @staticmethod
    def search(text: str) -> bool:
        text = strip_ref(_ROBOT_WITH.sub(" ", text))       # "로봇이랑 제일 가까운"·"분홍이랑 가까운"은 두 동작이 아님
        if _MULTI_RE.search(text):
            return True
        one_color = bool(re.search(PINK, text)) != bool(re.search(GREEN, text))
        return bool(_ALL.search(text)) and not one_color


_MULTI = _Multi


def _color_sure(text: str, r: dict) -> bool:
    """red·blue 예문은 색 낱말만 다르다. 볼 → 블록으로 바꾼 뒤 "빨간 블록 옮겨 줘" 의 red·blue 차가 0.023 으로
    줄어(낱말 '블록'이 색보다 크게 잡힘, 10-21 밤) 시연 명령이 LLM 으로 갔다. 1위 색 낱말만 있고 다른 색 낱말이 없으면
    다른 색 의도를 경쟁에서 빼고, 나머지(handoff 등)와의 차가 ROUTER_MARGIN 이상일 때만 확정한다."""
    top = r["intent"]
    if top not in _COLOR_WORDS or r["score"] < config.ROUTER_THRESHOLD:
        return False
    other = "blue" if top == "red" else "red"
    if _MULTI.search(text):
        return False
    t = strip_ref(text).replace(" ", "")      # "분홍 옆에 있는 초록" → 옮길 것은 초록(10-23 13:30)
    if not any(w in t for w in _COLOR_WORDS[top]) or any(w in t for w in _COLOR_WORDS[other]):
        return False
    rest = [s for k, s in r["top3"][1:] if k != other]
    return bool(rest) and r["score"] - rest[0] >= config.ROUTER_MARGIN


# 코드 검증 ① 할 수 없는 요청(LLM 보다 먼저, 로봇 안전망). 10-21 밤 4B 가 "숫자 3 모양으로 놓아 줘" 를
# [불량·분홍·초록 전부] 로 계획했다. 구절(…하고 / , / 다음에)마다 본다
# "그려" 는 넣지 않는다: "엑스 그려진 거 치워 줘"(defect) 가 걸렸다
_CANT_WORDS = re.compile(r"모양|쌓|탑처럼|탑을|줄로|일렬|한 줄|세워|섞어|뒤집|던져|굴려|색칠|숫자")
PINK, GREEN = r"(?:분홍|핑크)", r"(?:초록|녹색|연두)"
_CLAUSE = re.compile(r"고\s|,|그리고|다음에?|한\s*뒤에?")


def cant_do(text: str) -> str | None:
    if _CANT_WORDS.search(text):
        return "배운 적 없는 동작"
    for c in _CLAUSE.split(text):
        obj = re.sub(r"(?:" + PINK + "|" + GREEN + r")색?\s*그릇", "", c)    # 그릇 이름을 뺀 나머지 = 옮길 물체
        pink_bowl, green_bowl = re.search(PINK + r"색?\s*그릇", c), re.search(GREEN + r"색?\s*그릇", c)
        if re.search(GREEN, obj) and pink_bowl or re.search(PINK, obj) and green_bowl:
            return "다른 색 그릇에 넣기"
        if re.search(GREEN, obj) and "손" in obj and not re.search(PINK, obj):
            return "초록 블록을 손에 주기"
    return None


_RULE_Q = re.compile(r"하면|해야|언제\s*해|어떻게\s*해|(?:뭐가|무엇이)\s*들어가|항목")
_PAST = re.compile(r"[었았했됐]|옮긴|한\s*거")
_RECORD_WHEN = re.compile(r"오늘|방금|아까|지금까지|이번\s*작업")
_RECORD_WHAT = re.compile(r"몇\s*(?:번|개|회)|실패|성공|옮겼|옮긴|건넸|골라냈")
_EXCL = re.compile(r"(불량|엑스|분홍|핑크|초록|녹색|연두)\S*\s*(?:블[록럭]\S*\s*|거\s*|것\s*)?"
                   r"(?:그대로\s*두고|놔\s*두고|남겨\s*두고|두고|말고|제외하고|빼고)")
WORD_INTENT = [("defect", r"불량|엑스"), ("red", r"(?:분홍|핑크)"), ("blue", r"(?:초록|녹색|연두)")]


def excluded(text: str, strict: bool = False) -> set[str]:
    """문장에서 뺀 동작(…는 그대로 두고·말고·제외하고·놔두고·빼고). strict 면 '불량 빼고'(불량을 빼내고로도 읽힘)는 뺀다."""
    out = set()
    for m in _EXCL.finditer(text):
        word, verb = m.group(1), m.group(0)
        if strict and word in ("불량", "엑스") and verb.endswith("빼고"):
            continue
        out |= {it for it, pat in WORD_INTENT if re.fullmatch(pat, word)}
    return out


# 좌표 집기(arrange, plan 의 한 갈래): 블록으로 숫자·모양을 만들라는 말. 모양 낱말 + 만들기 동사가 함께 있어야 한다
# ("모양이 망가진 블록은?" 같은 질문은 아님). 숫자 아닌 모양·블록 부족·반경 밖은 arrange 노드가 이유를 말하고 거절
# 개수 단위가 안 붙은 숫자는 여러 자리도 모양으로 본다("블록으로 12 만들어 줘" 가 손 전달 12개로 확정됐다, 10-22 0시)
_SHAPE = re.compile(r"숫자|모양|하트|동그라미|세모|글자|별\s*모양|(?<!\d)\d+(?!\d)(?!\s*(?:개|번|초|분|칸|단계))")
_MAKE = re.compile(r"만들|놓아|놔\s*줘|놔줘|배치|그려\s*줘|꾸며|세워\s*줘|모양으로")
_SINO = {"영": "0", "공": "0", "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5", "육": "6", "륙": "6", "칠": "7",
         "팔": "8", "구": "9"}


def is_arrange(text: str) -> bool:
    return bool(_SHAPE.search(text) and _MAKE.search(text) and "?" not in text)


def arrange_target(text: str) -> tuple[str | None, str, str | None]:
    """(숫자 '0'~'9' 또는 None, 색 red|blue|any, 거절 이유 또는 None)."""
    pink, green = re.search(PINK, text), re.search(GREEN, text)
    color = "red" if pink and not green else "blue" if green and not pink else "any"
    nums = re.findall(r"(?<!\d)\d+(?!\d)(?!\s*(?:개|번|초|분|칸|단계))", text)
    if len(nums) > 1 or nums and len(nums[0]) > 1:
        return None, color, "한 자리 숫자(0~9)만 만들 수 있어요."
    if nums:
        return nums[0], color, None
    m = re.search(r"숫자\s*(" + "|".join(_SINO) + r")(?![가-힣])", text)
    if m:
        return _SINO[m.group(1)], color, None
    return None, color, "숫자 모양(0~9)만 만들 수 있어요. 예를 들어 '블록으로 숫자 7 만들어 줘'처럼 말해 주세요."


def _fits_color(intent: str, text: str) -> bool:
    text = strip_ref(text).split("말고")[-1]   # "초록색 거 말고 분홍색 거부터" → 뒤쪽이 시키는 것. "분홍 옆 초록" → 초록
    pink, green, bad = re.search(PINK, text), re.search(GREEN, text), re.search(r"불량|엑스|X", text)
    return {"defect": bool(bad), "red": bool(pink) and not green, "blue": bool(green) and not pink,
            "handoff": bool(pink) and "손" in text}[intent]


# 묻는 말 검사(10-22 세 문서 질문 47문항): 임베딩·4B 가 설명서·안내서 질문 8개를 동작·장면·보고·멈춤으로 골랐다
# ("숫자 7 만드는 데 얼마나 걸려?" → plan, "인터넷 없이 돌아가?" → stop, "불량 그릇에는 뭘 담아?" → scene,
# "시연 날 메모리는 얼마나 남아?" → report). docs/intents.csv 의 동작 문장 중 '?' 가 붙은 7개는 의문사가 없다("빼 줄래?")
_WH = re.compile(r"얼마나|몇\s*까지|몇\s*대|왜|어떻게|무엇|뭐|뭘|언제|어디|무슨|어느")
_SCENE_OBJ = re.compile(r"블[록럭]|볼|그릇|작업대|투입|구역|칸|자리|분홍|핑크|초록|녹색|연두|불량|엑스|정상|떨어진|지금|보여|보이|배치")
_SCENE_STATE = re.compile(r"있|없|보여|보이|남|어때|상태|뭐야|어디|몇\s*개야|놓여|배치")
_GENERAL = re.compile(r"(?:할|하는|될|일)\s*때|시연|하면|해야|원래|보통|기준|규칙")
_REPORT_CUE = re.compile(r"결과|보고|옮겼|옮긴|지금까지|아까|방금|오늘|성공|실패|그릇\s*마다|그릇별|몇\s*개씩|건넸|골라냈|했")


# 나(안내 담당) 자신을 묻는 말(graph 가 검색 전에 거절, 10-22)
SELF_Q = re.compile(r"(?:^|\s)(?:너|넌|니|당신)(?:는|의)?\s*(?:이름|나이|기분|누구|몇\s*살)|(?:너|넌|당신)\s*(?:는\s*)?누구")
_Q_END = re.compile(r"(?:나요|까요|가요|어요|아요|죠|니|냐)\s*[?.]?$")
_REQUEST = re.compile(r"줘|주세요|줄래|주시겠|주실래|해\s*봐")


def looks_question(text: str) -> bool:
    """문서 질문꼴인가: 의문사·'?'·묻는 어미가 있고, 부탁("~줄래?")·멈춤·자기 질문·작업 기록 질문("오늘 몇 번 실패했어?")·
    지금 작업대 질문(작업대 낱말 + 상태 낱말)이 아니다. 판별 지름길과 질문 미리 검색이 쓴다(10-22 r10)."""
    if is_stop(text) or SELF_Q.search(text) or _REQUEST.search(text):
        return False
    if not ("?" in text or _WH.search(text) or _Q_END.search(text.strip())):
        return False
    if _RECORD_WHEN.search(text) and _RECORD_WHAT.search(text):
        return False
    if _SCENE_OBJ.search(text) and _SCENE_STATE.search(text) and not _GENERAL.search(text):
        return False
    return True


def _ask_check(text: str, intent: str) -> str | None:
    """question 으로 돌릴 이유(없으면 None). 멈춤 규칙(is_stop)에 걸린 말은 여기 오지 않는다."""
    q = "?" in text
    if intent in ("defect", "red", "blue", "handoff", "plan") and q and _WH.search(text):
        return "의문사가 있는 묻는 말"
    if intent == "stop" and q and (_WH.search(text) or not any(w in text.replace(" ", "") for w in _STOP_WORDS)):
        return "멈추라는 말이 아닌 묻는 말"
    if intent == "scene":
        if not _SCENE_STATE.search(text) or not _SCENE_OBJ.search(text):
            return "작업대 상태를 묻지 않음"
        if _GENERAL.search(text) and "지금" not in text:
            return "일반 규칙을 묻는 말"
    if intent == "report" and q and not _REPORT_CUE.search(text):
        return "작업 기록을 묻지 않음"
    return None


def classify(text: str, use_llm: bool = True) -> dict:
    """{intent, count, method, score, margin, top3, latency_ms[, llm_ms, error, llm_down, none_kind, none_check]}
    intent none = 명령 아님(로봇 안 움직임). none_kind: hear·chat(규칙·LLM) · unsure(LLM 실패 + 자신 없음) · vague(낱말 근거 없음)."""
    t0 = time.perf_counter()
    if is_stop(text):
        return {"intent": "stop", "count": None, "method": "rule",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    if is_home(text):
        return {"intent": "home", "count": None, "method": "rule", "home_check": "홈 자세 규칙",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    if is_unsort(text):
        return {"intent": "unsort", "count": parse_count(text), "method": "rule", "unsort_check": "그릇에서 꺼내기 규칙",
                "colors": unsort_colors(text), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    talk = small_talk(text)
    if talk:
        return {"intent": "none", "count": None, "method": "rule", "none_kind": talk, "none_check": NONE_WHY[talk],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
    count = parse_count(text)
    out: dict = {}
    res = None                 # LLM 판별 결과(불렀으면)
    rtext = route_text(text)       # 임베딩·LLM 판별은 기준 구절·위치 낱말을 뺀 말로(13:51 실물 "분홍 옆에 있는 초록")
    if rtext != text:
        out["route_text"] = rtext
    try:
        r = get_router().route(rtext)
        out.update(r)
        confident = r["score"] >= config.ROUTER_THRESHOLD and r["margin"] >= config.ROUTER_MARGIN
        # question 은 로봇을 움직이지 않아 잘못 골라도 위험이 없다 → 기준을 낮춰 LLM 호출(약 0.8초)을 줄인다
        if r["intent"] == "question" and r["score"] >= config.ROUTER_Q_THRESHOLD and r["margin"] >= config.ROUTER_Q_MARGIN:
            confident = True
        if not confident and _color_sure(text, r):
            confident = True
            out["color_rule"] = True
        # 질문꼴 지름길(10-22 r10): 문서 질문꼴이고 임베딩 1·2위에 question 이 있으면 LLM 없이 question.
        # 질문 첫마디("자료를 찾아볼게요.")가 판별 뒤에 나가서, LLM 판별(약 0.85초)이 곧 첫마디 지연이었다.
        # docs/intents.csv 99 + 세 문서 질문 47 + 기준서 13 + 받아쓰기 변형 15 = 174문장에서 잘못 question 이 된 문장 0개
        if not confident and looks_question(text) and "question" in [k for k, _ in r["top3"][:2]]:
            confident = True
            out["intent"], out["question_form"] = "question", True
    except Exception as e:     # 임베딩 서버가 죽었으면 LLM 으로 바로
        r, confident = None, False
        out["error"] = f"router: {type(e).__name__}: {e}"
    if confident:
        out["method"] = "embedding"
    elif use_llm:
        t1 = time.perf_counter()
        try:
            res = _llm_classify(rtext, r)
            out["intent"] = res.intent
            out["method"] = "llm"
            # 4B 가 문서 질문 "음성 합성은 어디서 돌아?"·문서 밖 부탁 "점심 메뉴 추천해 줘" 를 none 으로 골랐다(10-22 밤) → 답(거절)하게
            if res.intent == "none" and (_WH_ANY.search(text) or _REQUEST.search(text)):
                out["intent"], out["none_check"] = "question", "묻는 말·부탁이라 none 대신 question"
            # Qwen3-8B 는 개수를 말하지 않아도 1·0 을 채운다(10-21 eval 10건) → 숫자 표현이 있을 때만 받는다
            if count is None and res.count and res.count > 0 and _NUM_HINT.search(text):
                count = res.count
        except Exception as e:
            out["error"] = (out.get("error", "") + f" llm: {type(e).__name__}: {e}").strip()
            out["method"] = "embedding-fallback" if r else "failed"
            # 자신 없는 임베딩 1위는 쓰되 움직이는 의도면 아래에서 none(unsure)로. 23:08 "마이크 테스트" red 0.721·차 0.014
            out["unsure"], out["llm_down"] = True, _llm_down(e)
        out["llm_ms"] = round((time.perf_counter() - t1) * 1000, 1)
    else:
        out["method"] = "embedding-low" if r else "failed"
    # 라우터·LLM 모두 실패하면 명령 아님(unsure, 로봇을 움직이지 않고 다시 말해 달라고 함)
    out.setdefault("intent", "none" if out.get("unsure") else "question")
    # plan 검사: 두 동작을 엮는 표지·예외도 없고 할 수 없는 요청도 아니면 한 가지 동작이다. 4B 가 프롬프트를 고쳐도
    # "분홍 블록 두 개만 옮겨 줘" 를 plan 으로 골랐다(10-21 밤) → 라우터 후보 중 1위 동작 의도로
    if out["intent"] == "plan" and r and not _MULTI.search(text) and not cant_do(text):
        acts = [k for k, _ in r["top3"] if k in ("defect", "red", "blue", "handoff")]
        fit = [k for k in acts if _fits_color(k, text)]      # 후보 1위가 blue 인데 "분홍색 거 두 개만" 이었다
        other = [k for k, _ in r["top3"] if k not in ("plan", "stop")]   # "작업 순서가 어떻게 돼요?" → question
        pick = fit or (acts if r["top3"][0][0] in acts else other) or acts
        if pick:
            out["intent"], out["plan_check"] = pick[0], "한 동작이라 plan 아님"
    # 보고/질문 검사: 4B 가 "결과 보고는 언제 해?"·"실패하면 몇 번까지 다시 해?" 를 report 로 골랐다(10-21 밤).
    # 규칙을 묻는 말(~하면·~해야·언제 해·뭐가 들어가)이고 이번 작업을 가리키는 과거형이 없으면 question
    if out["intent"] == "report" and _RULE_Q.search(text) and not _PAST.search(text):
        out["intent"], out["report_check"] = "question", "규칙을 묻는 말"
    # 그릇마다·그릇별 개수 = 결과 보고(기준서 7.2 "그릇별 개수"). 4B 가 "그릇마다 몇 개씩 들어 있어?" 를 scene 으로 골랐다
    if out["intent"] == "scene" and re.search(r"그릇\s*마다|그릇별|그릇\s*전부", text):
        out["intent"], out["scene_check"] = "report", "그릇마다 = 결과 보고"
    # 작업 기록 질문("오늘 몇 번 실패했어?")은 문서(RAG)가 아니라 runs/ 기록에서 → report(10-22 문서 3개로 넓히며 확인)
    if out["intent"] == "question" and _RECORD_WHEN.search(text) and _RECORD_WHAT.search(text) \
            and _PAST.search(text) and not re.search(r"기준서|설명서|안내서|문서", text):
        out["intent"], out["report_check"] = "report", "작업 기록 질문"
    # 뺀 동작 검사: "불량 말고 전부 그릇에 담아 줘" 를 defect 로 골랐다 → 문장에서 뺀 동작이면 plan(계획 단계가 제외를 처리)
    if out["intent"] in ("defect", "red", "blue") and out["intent"] in excluded(text):
        out["intent"], out["plan_check"] = "plan", "문장에서 뺀 동작"
    # 색 검사: LLM 이 red·blue 를 골랐는데 색 낱말이 반대면 바로잡는다("분홍 ㄱㄱ" → blue, 10-21 밤 4B)
    if out["intent"] in ("red", "blue") and not _fits_color(out["intent"], text):
        swap = "blue" if out["intent"] == "red" else "red"
        if _fits_color(swap, text):
            out["intent"], out["color_fix"] = swap, f"색 낱말이 {swap} 라 바꿈"
    # 불량 검사(10-23 13:05 실물 "초록불량 옮겨줘" → blue 0.94 → 정상 초록을 옮김): red·blue 인데 불량을 시키는 말이면 defect.
    # 색 낱말은 graph 가 불량 색(defect_cls)으로 넘긴다. 두 동작 표지가 있고 불량에 안 붙은 색이 따로 있으면
    # ("분홍 블록 옮기고 초록불량 골라내 줘") 한쪽만 하지 않게 plan
    if out["intent"] in ("red", "blue") and defect_word(strip_ref(text)):     # "불량 옆에 있는 분홍"은 분홍(기준 구절 뺌)
        two = _MULTI.search(text) and _DCOLOR.search(_DCOLOR_IN.sub(" ", _BOWL_NAME.sub(" ", text)))
        out["intent"], out["defect_check"] = ("plan", "불량과 색 블록 두 동작") if two else (
            "defect", f"불량 낱말이 있어 {out['intent']} 대신 defect")
    # 숫자·모양 만들기 → plan(좌표 집기 갈래). 위 plan 검사보다 뒤에 둔다(두 동작 표지가 없어 되돌려지지 않게)
    if is_arrange(text) and out["intent"] not in ("question", "stop", "scene", "report"):
        out["intent"], out["plan_check"] = "plan", "숫자·모양 배치(좌표 집기)"
    # 동작 의도인데 배운 적 없는 요청("초록 하나 내 손에 줘" → handoff 면 분홍을 건넨다) → plan 으로 보내 거절
    if out["intent"] in ("defect", "red", "blue", "handoff") and cant_do(text):
        out["intent"], out["plan_check"] = "plan", f"할 수 없는 요청({cant_do(text)})"
    ask = _ask_check(text, out["intent"])
    if ask:
        out["intent"], out["ask_check"] = "question", ask
    # 명령 아님(10-22 밤): LLM 이 실패했는데 자신 없는 임베딩이 움직이는 의도면 막는다. 자신 있으면(위 confident) 그대로
    if out["intent"] in MOVING and (out.get("unsure") or not _GROUND.search(text)):
        kind = "unsure" if out.get("unsure") else "vague"
        out["none_from"], out["intent"], out["none_kind"], out["none_check"] = out["intent"], "none", kind, NONE_WHY[kind]
    elif out["intent"] == "none" or out.get("unsure") and r is None:
        out["intent"] = "none"
        out.setdefault("none_kind", "unsure" if out.get("unsure") else "chat")
        out.setdefault("none_check", NONE_WHY[out["none_kind"]] if out.get("unsure") else "LLM 이 명령 아님으로 고름")
    # 위치·관계(10-23 13:30 "초록 블록 중에서 맨 위에 있는 거"): 규칙(rule_target)으로 먼저 읽고, 못 읽은 위치 낱말이 남으면
    # 의도가 임베딩으로 정해졌어도 LLM 에 target 을 묻는다. LLM 이 none·멈춤이면 graph 가 거부한다(위치 말을 버리지 않는다)
    if out["intent"] in ("defect", "red", "blue"):
        tg = rule_target(text)
        if tg["by"] == "need_llm" and use_llm:
            tg = _llm_target(text, r, res, tg)
        out["target"] = tg
    if out["intent"] in NO_COUNT:
        count = None
    out["count"] = count
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out


def _llm_target(text: str, r: dict | None, res, tg: dict) -> dict:
    """규칙이 못 읽은 위치 낱말 → LLM 구조화 출력 TargetOut.target 하나(판별과 따로 부른다, 13:51).
    by: llm(읽음) · llm_none(none — 목록으로 못 나타냄) · llm_down(시간 초과·오류)."""
    t1 = time.perf_counter()
    try:
        got = TARGET_KO[_llm_target_call(text).target]
    except Exception as e:
        return dict(tg, which=None, next_to=None, by="llm_down", error=f"{type(e).__name__}: {e}",
                    llm_ms=round((time.perf_counter() - t1) * 1000, 1))
    nt = got[len("next_to:"):] if got.startswith("next_to:") else None
    w = None if got == "none" or nt else got
    return dict(tg, which=w, next_to=nt, by="llm" if w or nt else "llm_none", llm_target=got,
                llm_ms=round((time.perf_counter() - t1) * 1000, 1))
