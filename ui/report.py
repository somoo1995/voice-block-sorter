"""결과 보고서 HTML 만들기 — ui/report_template.html(Jinja2) 에 데이터를 채운다.

    render_report(data: dict, out_path: str | None = None) -> str   # HTML 문자열(out_path 가 있으면 파일로도)

사진은 data URI 로 넣어 HTML 한 파일만으로 열린다(제출·메일용). data 형식은 ui/README.md "보고서 데이터".
CLI: python -m ui.report sample.json out.html
"""
from __future__ import annotations

import base64
import json
import mimetypes
import sys
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
_env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=select_autoescape(["html"]))

POLICY_LABELS = {"pick_defect": "불량 골라내기", "pick_red": "분홍 옮기기", "pick_blue": "초록 옮기기", "handoff": "손 전달"}  # red = 분홍


def _img_src(p: str | None) -> str | None:
    if not p:
        return None
    if p.startswith("data:"):
        return p
    path = Path(p) if Path(p).is_absolute() else ROOT / p
    if not path.exists():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _ms_text(ms) -> str:
    if ms is None:
        return "-"
    return f"{ms / 1000:.1f}초" if ms >= 1000 else f"{ms:.1f} ms" if ms < 100 else f"{ms:.0f} ms"


def _rate(success: int, attempts: int) -> int:
    return round(100 * success / attempts) if attempts else 0


def build_context(data: dict) -> dict:
    counts = data.get("counts") or []
    by_key = {c.get("key"): c for c in counts}

    def kpi_value(key):
        c = by_key.get(key) or {}
        v = c.get("camera") if c.get("camera") is not None else c.get("record")
        return "-" if v is None else v

    policies = []
    for p in data.get("policies") or []:
        attempts, success, total_s = int(p.get("attempts", 0)), int(p.get("success", 0)), float(p.get("total_s", 0.0))
        policies.append(dict(p, label=p.get("label") or POLICY_LABELS.get(p.get("name"), p.get("name")),
                             attempts=attempts, success=success, total_s=total_s,
                             rate=_rate(success, attempts), avg_s=total_s / attempts if attempts else 0.0))
    total = dict(attempts=sum(p["attempts"] for p in policies), success=sum(p["success"] for p in policies),
                 total_s=sum(p["total_s"] for p in policies))
    total["rate"] = _rate(total["success"], total["attempts"])

    devices = [dict(d, avg_text=_ms_text(d.get("avg_ms")), max_text=_ms_text(d.get("max_ms"))) for d in data.get("devices") or []]
    duration_s = data.get("duration_s")
    kpis = [
        dict(kind="defect", label="불량", value=kpi_value("defect")),
        dict(kind="red", label="분홍", value=kpi_value("red")),
        dict(kind="blue", label="초록", value=kpi_value("blue")),
        dict(kind="hand", label="손 전달", value=kpi_value("hand")),
        dict(kind="", label=f"성공 {total['success']}/{total['attempts']}", value=f"{total['rate']}%"),
        dict(kind="", label="전체 소요", value=f"{duration_s:.0f}초" if duration_s is not None else "-"),
    ]
    return dict(
        title=data.get("title") or "스펀지 블록 선별 작업 결과 보고서",
        job_id=data.get("job_id"), started=data.get("started", ""), ended=data.get("ended"),
        summary=data.get("summary"), offline=data.get("offline", True), net_note=data.get("net_note"),
        kpis=kpis,
        photos=[dict(label="작업 전", src=_img_src(data.get("before_image")), ts=data.get("before_ts")),
                dict(label="작업 후", src=_img_src(data.get("after_image")), ts=data.get("after_ts"))],
        counts=counts, policies=policies, total=total, commands=data.get("commands") or [],
        duration_s=duration_s, devices=devices, human_checks=data.get("human_checks") or [],
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )


def render_report(data: dict, out_path: str | None = None) -> str:
    html = _env.get_template("report_template.html").render(**build_context(data))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(html, encoding="utf-8")
    return html


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("사용법: python -m ui.report <데이터.json> <출력.html>")
    render_report(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), sys.argv[2])
    print(sys.argv[2])
