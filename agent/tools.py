"""LangChain 도구 3개(계약서 3절): detect_objects · run_policy · search_manual.

그래프가 순서를 정해 직접 부른다(소형 LLM 의 도구 호출 실수를 피하려고 LLM 에 도구를 맡기지 않는다).
"""
from typing import Literal, Optional

from langchain_core.tools import tool

from . import backends, config


@tool
def detect_objects(image_path: Optional[str] = None) -> dict:
    """위 카메라로 작업대를 한 장 찍어 구역별(input, bowl_red, bowl_blue, bowl_defect) 블록 개수와 손 유무를 센다.
    objects 에는 블록마다 cls·zone(구역 밖이면 outside)·cell(구역 3×3 칸 이름)·px·conf 가 있다(계약서 3절, 선택 필드)."""
    return backends.get()["detect_objects"](image_path)


@tool
def run_policy(name: Literal["handoff"]) -> dict:
    """로봇 정책을 한 번 실행한다(분홍 블록 1개를 손에 건넴). 시간은 넘기지 않는다(robot/policies.json 정책별 값, 기본 21초).
    10-23 13:10 사용자 결정: 학습한 동작은 손 전달만 — pick_red·pick_blue·pick_defect 는 부르지 않는다(옮기기는 좌표 집기)."""
    if name != "handoff":
        return {"name": name, "ok": False, "returncode": None, "reason": "refused", "error": "act_only_handoff"}
    return backends.get()["run_policy"](name)


@tool
def search_manual(query: str, k: int = config.RAG_TOP_K) -> list[dict]:
    """작업 기준서·로봇 운용 설명서·시스템 안내서에서 질문과 관련된 조각을 찾아 리랭크 순서로 돌려준다
    (조각마다 doc·doc_name·clause·title, LangChain 리트리버 rag.Manual.retriever)."""
    from .rag import get_manual
    return get_manual().search(query, k=k)


def stop_robot() -> None:
    backends.get()["stop"]()


TOOLS = [detect_objects, run_policy, search_manual]
