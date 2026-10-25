"""명령 5 "결과 알려 줘": 작업 기록(log.jsonl) 집계 + 위 카메라 개수 대조 → 보고 문장·HTML.

보고 항목은 기준서 7.2(그릇별 개수·전달·남은 개수·기록 대조·정책별 시도·소요 시간·사람 확인 요청·장치별 추론 시간).
"""
import base64
import html
import re
from datetime import datetime

from . import config, llm, runlog

CHECK_STATUS = {"gave_up": "두 번 실패", "no_hand": "손이 보이지 않음", "limit": "시도 한도", "stopped": "멈춤",
                "error": "정책 실행 불가"}

BOWLS = {  # 그릇 → (정상으로 들어가야 할 클래스, 이름)
    "bowl_defect": (("red_defect", "blue_defect"), "불량 그릇"),
    "bowl_red": (("red_ok",), "분홍 그릇"),
    "bowl_blue": (("blue_ok",), "초록 그릇"),
}
# red_* = 분홍, blue_* = 초록(10-21 밤 색 변경, 코드 식별자는 그대로)
ENDED_KO = {"home_reached": "홈 복귀 감지(일찍 끝남)", "duration_elapsed": "시간 경과", "timeout": "벽시계 상한(timeout)",
            "stopped": "멈춤", "error": "오류", "refused": "거부"}
CLASS_KO = {"red_ok": "정상 분홍", "blue_ok": "정상 초록", "red_defect": "분홍 불량", "blue_defect": "초록 불량"}


def outside_counts(obs: dict) -> dict:
    """구역 밖(outside)에 떨어진 블록 수(클래스별). detect 의 objects 가 없으면 빈 dict."""
    out: dict = {}
    for o in obs.get("objects") or []:
        if o.get("zone") == "outside":
            out[o["cls"]] = out.get(o["cls"], 0) + 1
    return out


def facts(thread_id: str, obs: dict, history: list[dict]) -> dict:
    rows = runlog.read(thread_id=thread_id)
    s = runlog.summarize(rows)
    P = s["policies"]

    def moved(p):
        return P.get(p, {}).get("moved", 0)

    z = obs["zones"]
    # coord_* = 좌표 집기 그릇 옮기기(BALL_COORD=1, graph.coord_node 가 블록마다 한 줄). 정책 시도와 같은 그릇에 더한다
    expected = {"bowl_defect": moved("pick_defect") + moved("coord_defect"),
                "bowl_red": moved("pick_red") + moved("coord_red") - moved("handoff"),
                "bowl_blue": moved("pick_blue") + moved("coord_blue")}
    detected = {b: sum(z[b][k] for k in keys) for b, (keys, _) in BOWLS.items()}
    wrong = {b: {k: n for k, n in z[b].items() if k not in keys and n}
             for b, (keys, _) in BOWLS.items()}
    issues = []
    for b, (_, name) in BOWLS.items():
        if expected[b] != detected[b]:
            issues.append(f"{name}: 기록 {expected[b]}개, 카메라 {detected[b]}개")
        for k, n in wrong[b].items():
            issues.append(f"{name}에 {CLASS_KO[k]} {n}개가 섞여 있음")
    left = {k: n for k, n in z["input"].items() if n}
    outside = outside_counts(obs)
    commands = [{k: h.get(k) for k in ("text", "intent", "status", "done", "attempts", "elapsed_s", "route")}
                for h in history if h.get("intent") not in ("report", "question", "scene", "plan")]
    # plan 은 단계마다 finish 가 따로 남긴다 → plan 요약 줄은 빼야 두 번 세지 않는다
    checks = [{"text": h["text"], "why": CHECK_STATUS[h["status"]], "say": h.get("say")}
              for h in history if h.get("status") in CHECK_STATUS and h.get("intent") != "plan"]
    route_ms = [c["route"]["latency_ms"] for c in commands if (c.get("route") or {}).get("latency_ms") is not None]
    devices = {
        "vision": f"{obs.get('device')} {obs.get('latency_ms')}ms (마지막 1장)",
        "intent": f"임베딩(CPU) 라우터 평균 {sum(route_ms) / len(route_ms):.0f}ms" if route_ms else "기록 없음",
        "llm": f"{llm.model_name()} @ {config.LLM_BASE_URL}",
    }
    return {
        "attempts": s["attempts"], "ok": s["ok"], "policies": P, "ended": s.get("ended", {}),
        "handed": moved("handoff"), "expected": expected, "detected": detected, "wrong": wrong,
        "defect_red": z["bowl_defect"]["red_defect"], "defect_blue": z["bowl_defect"]["blue_defect"],
        "input_left": left, "outside": outside, "match": not issues, "issues": issues, "log_rows": len(rows),
        "commands": commands, "total_s": round(sum(c["elapsed_s"] or 0 for c in commands), 1),
        "checks": checks, "devices": devices,
    }


def template_sentence(f: dict) -> str:
    d = f["detected"]
    head = (f"불량 {d['bowl_defect']}개(분홍 {f['defect_red']}, 초록 {f['defect_blue']}), "
            f"분홍 {d['bowl_red']}개" + (f"({f['handed']}개 전달)" if f["handed"] else "")
            + f", 초록 {d['bowl_blue']}개입니다.")
    tail = f" 시도 {f['attempts']}회 중 {f['ok']}회 성공, "
    # 기준서 7.4: 다르면 차이를 그대로 말하고 확인을 요청한다(어느 한쪽 숫자로 맞추지 않는다)
    tail += ("기록과 카메라 개수가 일치합니다." if f["match"]
             else f"기록과 카메라가 다릅니다: {'; '.join(f['issues'])}. 확인해 주세요.")
    if f["input_left"]:
        tail += " 투입 구역에 " + ", ".join(f"{CLASS_KO[k]} {n}개" for k, n in f["input_left"].items()) + "가 남아 있습니다."
    if f.get("outside"):     # 기준서 5.4·5.5: 떨어뜨린 블록은 사람이 집어 와야 한다
        tail += (" 구역 밖에 " + ", ".join(f"{CLASS_KO[k]} {n}개" for k, n in f["outside"].items())
                 + "가 떨어져 있습니다. 확인해 주세요.")
    if f["checks"]:
        tail += f" 사람 확인 요청 {len(f['checks'])}건(" + ", ".join(c["why"] for c in f["checks"]) + ")."
    return head + tail


_TIMES = {1: "한", 2: "두", 3: "세", 4: "네", 5: "다섯", 6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열"}


def _times(n: int) -> str:
    return f"{_TIMES[n]} 번" if n in _TIMES else f"{n}번"


def spoken_sentence(f: dict) -> str:
    """말로 읽는 짧은 보고(10-22 r11: 발표 실연 ⑤ 가 한 문장 10.9초라 3분 예산을 먹었다). 5초 안팎을 목표로
    중요한 것 먼저 — 사람이 할 일(기록 불일치·구역 밖·확인 요청) → 그릇·전달 개수 → 시도 → 남은 블록.
    기록과 카메라가 같으면 말하지 않는다(다르면 맨 앞에서 차이를 말함, 기준서 7.4). 불량 색 구분·정책별 시도·
    장치별 시간 같은 자세한 숫자는 화면(report 이벤트 text·facts·HTML)에만 둔다.
    UI 측정의 말 속도는 약 6자/초(① "불량 1개(분홍 1)를 분리했습니다." 19자 3.0초)라 실연 ⑤ 는 41자 ≈ 7초다."""
    d, n, ok = f["detected"], f["attempts"], f["ok"]
    out = []
    if not f["match"]:      # 기준서 7.4: 다르면 차이를 그대로 말하고 확인을 요청한다
        out.append("기록과 카메라 개수가 다릅니다: " + "; ".join(f["issues"][:2]) + ". 확인해 주세요.")
    if f.get("outside"):
        out.append(f"구역 밖에 {sum(f['outside'].values())}개가 떨어져 있습니다. 집어 와 주세요.")
    if f["checks"]:
        out.append(f"사람 확인 요청이 {len(f['checks'])}건 있습니다.")
    out.append(f"불량 {d['bowl_defect']}개, 분홍 {d['bowl_red']}개, 초록 {d['bowl_blue']}개"
               + (f", 손 전달 {f['handed']}개" if f["handed"] else "") + "입니다.")
    if n:
        out.append("모두 성공했습니다." if ok == n else f"{_times(n)} 중 {_times(ok)} 성공했습니다.")
    if f["input_left"]:
        out.append(f"투입 구역에 {sum(f['input_left'].values())}개가 남아 있습니다.")
    return " ".join(out)


def fact_lines(f: dict) -> str:
    """LLM 에 주는 사실 목록. '빨강 3개(1개 전달)' 처럼 소형 모델이 잘못 읽는 줄임 표현을 풀어 쓴다."""
    d = f["detected"]
    lines = [
        f"불량 그릇에 불량 {d['bowl_defect']}개(분홍 불량 {f['defect_red']}개, 초록 불량 {f['defect_blue']}개)",
        f"분홍 그릇에 정상 분홍 {d['bowl_red']}개",
        f"초록 그릇에 정상 초록 {d['bowl_blue']}개",
        f"사람 손에 건넨 분홍 {f['handed']}개(분홍 그릇에서 꺼내 건넴, 분홍 그릇 개수에 포함 안 됨)",
        f"로봇 시도 {f['attempts']}회 중 성공 {f['ok']}회",
        "작업 기록과 카메라 개수: " + ("일치" if f["match"] else "다름 — " + "; ".join(f["issues"])),
    ]
    if f["input_left"]:
        lines.append("투입 구역에 남은 블록: " + ", ".join(f"{CLASS_KO[k]} {n}개" for k, n in f["input_left"].items()))
    if f.get("outside"):
        lines.append("구역 밖에 떨어진 블록: " + ", ".join(f"{CLASS_KO[k]} {n}개" for k, n in f["outside"].items()))
    if f["checks"]:
        lines.append("사람 확인 요청: " + "; ".join(f"'{c['text']}' {c['why']}" for c in f["checks"]))
    return "\n".join(f"- {x}" for x in lines)


def sentence_ok(sentence: str, f: dict) -> bool:
    """LLM 문장 검사: 사실에 없는 숫자가 없고, 템플릿의 숫자를 빠뜨리지 않고, 너무 길지 않아야 한다."""
    base = template_sentence(f)
    want = set(re.findall(r"\d+", base))
    got = set(re.findall(r"\d+", sentence))
    allowed = want | set(re.findall(r"\d+", fact_lines(f)))
    return got <= allowed and want <= got and len(sentence) <= len(base) * 1.6 + 60


def html_report(f: dict, sentence: str, obs: dict, sources: list[dict], thread_id: str) -> str:
    e = html.escape
    img = ""
    if obs.get("image_path"):
        p = config.ROOT / obs["image_path"]
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            img = f'<img alt="위 카메라" src="data:image/jpeg;base64,{b64}">'
    bowl_rows = "".join(
        f"<tr><td>{e(name)}</td><td>{f['expected'][b]}</td><td>{f['detected'][b]}</td>"
        f"<td class={'ok' if f['expected'][b] == f['detected'][b] and not f['wrong'][b] else 'bad'}>"
        f"{'일치' if f['expected'][b] == f['detected'][b] and not f['wrong'][b] else '다름'}</td></tr>"
        for b, (_, name) in BOWLS.items())
    pol_rows = "".join(
        f"<tr><td>{e(p)}</td><td>{v['attempts']}</td><td>{v['ok']}</td><td>{v['moved']}</td></tr>"
        for p, v in f["policies"].items()) or "<tr><td colspan=4>기록 없음</td></tr>"
    src_rows = "".join(
        f"<li><b>{e(s.get('section') or '')}</b> {e(s['text'].split(chr(10), 1)[-1][:160])}</li>"
        for s in sources)
    issues = "".join(f"<li>{e(i)}</li>" for i in f["issues"]) or "<li>없음</li>"
    checks = "".join(f"<li>{e(c['text'])} — {e(c['why'])}: {e(c['say'] or '')}</li>" for c in f["checks"]) \
        or "<li>없음</li>"
    cmd_rows = "".join(
        f"<tr><td>{e(c['text'])}</td><td>{e(c['intent'] or '')}</td><td>{e(c['status'] or '')}</td>"
        f"<td>{c['done'] or 0}</td><td>{c['attempts'] or 0}</td><td>{c['elapsed_s'] or ''}</td></tr>"
        for c in f["commands"]) or "<tr><td colspan=6>기록 없음</td></tr>"
    left = ", ".join(f"{CLASS_KO[k]} {n}개" for k, n in f["input_left"].items()) or "없음"
    out = ", ".join(f"{CLASS_KO[k]} {n}개" for k, n in (f.get("outside") or {}).items()) or "없음"
    dev = "".join(f"<li>{e(k)}: {e(v)}</li>" for k, v in f["devices"].items())
    ended = " · ".join(f"{ENDED_KO.get(k, k)} {n}회" for k, n in (f.get("ended") or {}).items()) or "기록 없음"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>선별 작업 보고서</title>
<style>
body{{font-family:system-ui,'Noto Sans KR',sans-serif;margin:16px;color:#1f2328;background:#fff;max-width:880px}}
h1{{font-size:1.3rem;margin:0 0 4px}} .meta{{color:#656d76;font-size:.85rem;margin-bottom:12px}}
.say{{font-size:1.1rem;background:#f6f8fa;border-left:4px solid #0969da;padding:10px 12px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;margin:8px 0 16px}} td,th{{border:1px solid #d0d7de;padding:6px 8px;text-align:left}}
th{{background:#f6f8fa}} .ok{{color:#1a7f37;font-weight:600}} .bad{{color:#cf222e;font-weight:600}}
img{{max-width:100%;border:1px solid #d0d7de;border-radius:6px}} h2{{font-size:1.05rem;margin:18px 0 6px}}
</style></head><body>
<h1>선별 작업 보고서</h1>
<div class="meta">{now} · 대화 {e(thread_id)} · 기록 {f['log_rows']}줄 · 카메라 {e(str(obs.get('device', '')))}</div>
<div class="say">{e(sentence)}</div>
<h2>그릇별 개수 — 기록 대 카메라</h2>
<table><tr><th>그릇</th><th>기록</th><th>카메라</th><th>대조</th></tr>{bowl_rows}</table>
<p>손 전달 {f['handed']}개 · 투입 구역에 남은 블록: {e(left)} · 구역 밖에 떨어진 블록: {e(out)}</p>
<h2>정책별 시도</h2>
<table><tr><th>정책</th><th>시도</th><th>성공</th><th>옮긴 개수</th></tr>{pol_rows}
<tr><th>합계</th><th>{f['attempts']}</th><th>{f['ok']}</th><th></th></tr></table>
<p>정책이 끝난 방식: {e(ended)}</p>
<h2>명령별 소요 시간 (합계 {f['total_s']}초)</h2>
<table><tr><th>말</th><th>의도</th><th>결과</th><th>옮김</th><th>시도</th><th>초</th></tr>{cmd_rows}</table>
<h2>기록과 다른 점</h2><ul>{issues}</ul>
<h2>사람 확인 요청</h2><ul>{checks}</ul>
<h2>장치별 추론</h2><ul>{dev}</ul>
<h2>기준서 근거</h2><ul>{src_rows or '<li>없음</li>'}</ul>
{img}
</body></html>"""
