"""run_command 이벤트(계약서 3절): {"type", "node", "data", "ts"}."""
from datetime import datetime

TYPES = ("node", "tool", "say", "report", "error")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def make(type_: str, node: str, data: dict | None = None) -> dict:
    assert type_ in TYPES, type_
    return {"type": type_, "node": node, "data": data or {}, "ts": now_iso()}


def emit(type_: str, node: str, **data) -> None:
    """그래프 노드 안에서 호출. stream_mode="custom" 으로 run_command 에 흘러간다."""
    from langgraph.config import get_stream_writer
    get_stream_writer()(make(type_, node, data))
