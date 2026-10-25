"""가짜 작업대로 그래프 경로를 확인하는 회귀 시험(로봇·카메라 안 씀).

  python -m agent.selftest            # 의도 판별도 실제(OVMS 임베딩 + LLM)
  python -m agent.selftest --offline  # 의도 판별을 정답표로 바꿔 그래프만 시험(OVMS 없어도 됨)
"""
import argparse
import os
import sys
import threading
import time

os.environ["BALL_MOCK"] = "1"
os.environ["BALL_MOCK_SCENE"] = "full"          # 아래 기대 개수는 11개 장면 기준(발표 실연 demo4 는 14번에서)
os.environ.setdefault("BALL_MOCK_SNAP", "0")

from . import config, graph, mocks  # noqa: E402
from .report import template_sentence as report_template  # noqa: E402

STUB = {  # --offline 일 때 쓰는 의도 정답
    "불량품 골라내 줘": ("defect", None), "분홍 블록 옮겨 줘": ("red", None), "초록 블록 옮겨 줘": ("blue", None),
    "분홍 블록 하나 내 손에 줘": ("handoff", 1), "결과 알려 줘": ("report", None),
    "분홍 블록 두 개만 옮겨 줘": ("red", 2), "분홍 블록 다섯 개 옮겨 줘": ("red", 5), "초록 블록 두 개만 옮겨 줘": ("blue", 2),
    "분홍블록 하나를 옮겨줘": ("red", 1),     # 방식 단서("act정책으로"·"좌표로")를 뺀 말(10-22 r21)
    "분홍 블록 몇 개 남았어?": ("scene", None), "초록 그릇에 몇 개야?": ("scene", None),
    "분홍 두 개만 옮기고 불량 골라내 줘": ("plan", None), "분홍 하나는 내 손에, 나머지 분홍은 그릇에": ("plan", None),
    "초록은 그대로 두고 나머지 정리해 줘": ("plan", None), "숫자 3 모양으로 놓아 줘": ("plan", None),
    "불량 빼고 다 정리해 줘": ("plan", None), "분홍이랑 초록 두 개씩 옮겨 줘": ("plan", None),
    "지금 배치 어때?": ("scene", None), "불량 빼고 나머지 다 정리해 줘": ("plan", None), "블록 탑처럼 쌓아 줘": ("plan", None),
    "블록으로 7 만들어 줘": ("plan", None), "블록으로 하트 만들어 줘": ("plan", None),
    "분홍으로 숫자 7 만들어 줘": ("plan", None),
    "불량 기준이 뭐예요?": ("question", None), "너 이름이 뭐야?": ("question", None), "오늘 몇 번 실패했어?": ("report", None),
    # 위치 단서(10-23). 방식 단서를 뺀 말: "초록블록 왼쪽에 있는거 좌표로 옮겨줘"(09:50 실물) → "초록블록 왼쪽에 있는거 옮겨줘"
    "초록블록 왼쪽에 있는거 옮겨줘": ("blue", None), "오른쪽 분홍 하나 옮겨 줘": ("red", 1),
    "초록 블록 가운데 거 옮겨 줘": ("blue", None), "왼쪽 분홍 블록 옮겨 줘": ("red", None),
    "왼쪽이랑 오른쪽 초록 옮겨 줘": ("blue", None), "왼쪽 분홍 옮기고 불량 골라내 줘": ("plan", None),
    "분홍블록 옮겨줘봐": ("red", None), "분홍블록 옮겨줘": ("red", None),   # "좌표모드로"·"학습모드로"를 뺀 말(10-23 총괄)
    # 불량 색(10-23 11:35 실물 "초록 불량 옮겨 줘" → 분홍 불량까지 2개). "초록 불량 학습한 동작으로 골라내 줘" → 방식 단서를 뺀 말
    "초록 불량 옮겨 줘": ("defect", None), "분홍 불량 하나 골라내 줘": ("defect", 1), "왼쪽 분홍 불량 옮겨 줘": ("defect", None),
    "초록 불량 골라내 줘": ("defect", None), "초록 말고 불량 옮겨 줘": ("defect", None),
    "분홍이랑 초록 불량 옮겨 줘": ("defect", None), "초록 불량 골라내고 분홍 옮겨 줘": ("plan", None),
}
# 10-23 13:10 사용자 결정: 학습한 동작(ACT)은 손 전달만, 옮기기(defect·red·blue·plan 단계)는 늘 좌표 집기.
# 아래 1~12절의 재시도·놓침·떨어뜨림·Studio 거부 주입(BALL_MOCK_FAIL_AT 등, 가짜 ACT 정책)은 손 전달로 시험한다
PLAN_STUB = {  # --offline 일 때 쓰는 계획(LLM 대신). 손 전달을 앞에 둬 '그릇에 먼저 옮기기' 자동 보충도 시험한다
    "분홍 두 개만 옮기고 불량 골라내 줘": (True, [("red", 2), ("defect", None)]),
    "분홍 하나는 내 손에, 나머지 분홍은 그릇에": (True, [("handoff", 1), ("red", None)]),
    "초록은 그대로 두고 나머지 정리해 줘": (True, [("defect", None), ("red", None)]),
    "숫자 3 모양으로 놓아 줘": (False, []),
    "분홍이랑 초록 두 개씩 옮겨 줘": (True, [("red", 2), ("blue", 2)]),   # "~씩" 은 규칙이 안 읽어 LLM 경로
}


def run(text: str, tid: str) -> tuple[dict, list[dict]]:
    evs = list(graph.run_command(text, tid))
    return evs[-1]["data"], evs


def says(evs):
    return [e["data"]["text"] for e in evs if e["type"] == "say"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    if a.offline:
        graph.classify = lambda t: {"intent": STUB[t][0], "count": STUB[t][1], "method": "stub", "latency_ms": 0}
        PlanOut = graph._plan_models()
        graph.make_plan = lambda t, obs: PlanOut(feasible=PLAN_STUB[t][0],
                                                 steps=[{"intent": i, "count": c} for i, c in PLAN_STUB[t][1]])
    os.environ["BALL_MOCK_POLICY_S"] = "0.05"
    config.HAND_POLL_S = 0.2
    base = f"selftest-{time.strftime('%H%M%S')}"
    fails = []

    def check(name, cond, detail=""):
        print(f"{'ok ' if cond else 'XX '} {name}" + (f"  ({detail})" if detail and not cond else ""))
        if not cond:
            fails.append(name)

    # 1. 시연 1→5: 불량·분홍·초록은 좌표 집기(블록마다 1회), 손 전달만 ACT → 12회 중 12회, 기록·카메라 일치
    os.environ["BALL_MOCK_FAIL_AT"] = "3"
    os.environ["BALL_MOCK_HAND_DELAY"] = "0.5"
    mocks.workbench.reset()
    tid = base + "-demo"
    ends = [run(t, tid) for t in ["불량품 골라내 줘", "분홍 블록 옮겨 줘", "초록 블록 옮겨 줘", "분홍 블록 하나 내 손에 줘", "결과 알려 줘"]]
    check("1 불량 3개 분리(좌표 집기)", ends[0][0]["status"] == "done" and ends[0][0]["done"] == 3
          and says(ends[0][1])[:1] == ["불량품 3개(분홍 2, 초록 1)를 좌표로 골라낼게요."], says(ends[0][1]))
    check("1 불량 보고 문장", "불량 3개(분홍 2, 초록 1)를 분리했습니다." in says(ends[0][1]), says(ends[0][1]))
    check("2 분홍 4개", ends[1][0]["done"] == 4)
    check("3 초록 4개", ends[2][0]["done"] == 4)
    check("4 손 대기 후 전달", "손을 내밀어 주세요." in says(ends[3][1]) and ends[3][0]["done"] == 1, says(ends[3][1]))
    rep = [e for e in ends[4][1] if e["type"] == "report"]
    f = rep[0]["data"]["facts"] if rep else {}
    check("5 보고: 12회 중 12회·일치", f.get("attempts") == 12 and f.get("ok") == 12 and f.get("match"),
          {k: f.get(k) for k in ("attempts", "ok", "match", "issues")})
    check("5 보고서 HTML", bool(rep) and (config.ROOT / rep[0]["data"]["html_path"]).exists())
    check("5 보고: 종료 방식 기록(좌표 집기 done 11 · 손 전달 정책 = 홈 복귀 감지 1)",
          f.get("ended") == {"done": 11, "home_reached": 1}, f.get("ended"))
    # 시간 경과(duration_elapsed)로 끝나도 성공, 따로 센다(정책은 손 전달만)
    os.environ["BALL_MOCK_END"] = "duration_elapsed"
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    mocks.workbench.reset()
    mocks.workbench.zones["bowl_red"]["red_ok"] = 2
    end, evs = run("분홍 블록 하나 내 손에 줘", base + "-dur")
    os.environ["BALL_MOCK_END"] = "home_reached"
    end2, evs2 = run("결과 알려 줘", base + "-dur")
    f2 = next((e["data"]["facts"] for e in evs2 if e["type"] == "report"), {})
    check("정책 시간 경과 종료도 성공·따로 기록", end["done"] == 1 and f2.get("ended") == {"duration_elapsed": 1}
          and f2.get("ok") == 1, f2.get("ended"))

    # 2. 개수 지정 / 남은 것보다 많이 지정
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    mocks.workbench.reset()
    end, evs = run("분홍 블록 두 개만 옮겨 줘", base + "-count")
    check("개수 지정 2개", end["done"] == 2 and mocks.workbench.zones["input"]["red_ok"] == 2)
    end, evs = run("분홍 블록 다섯 개 옮겨 줘", base + "-count")
    check("남은 2개뿐 → 2개만", end["done"] == 2 and any("2개뿐" in s for s in says(evs)), says(evs))
    end, evs = run("분홍 블록 옮겨 줘", base + "-count")
    check("대상 없음", end["status"] == "nothing", end)

    # 3. 두 번 연속 실패 → 멈추고 사람 확인 요청(ACT = 손 전달)
    os.environ["BALL_MOCK_FAIL_AT"] = "1,2"
    mocks.workbench.reset()
    mocks.workbench.zones["bowl_red"]["red_ok"] = 2
    end, evs = run("분홍 블록 하나 내 손에 줘", base + "-giveup")
    check("두 번 실패 → gave_up", end["status"] == "gave_up" and end["attempts"] == 2
          and any("확인해 주세요" in s for s in says(evs)), says(evs))

    # 4. 손이 안 보임 → 대기 시간 초과
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    os.environ["BALL_MOCK_HAND"] = "never"
    old = config.HAND_WAIT_S
    config.HAND_WAIT_S = 1.0
    mocks.workbench.reset()
    mocks.workbench.zones["bowl_red"]["red_ok"] = 2
    end, evs = run("분홍 블록 하나 내 손에 줘", base + "-nohand")
    check("손 없음 → no_hand, 로봇 0회", end["status"] == "no_hand" and end["attempts"] == 0, end)
    config.HAND_WAIT_S = old
    os.environ["BALL_MOCK_HAND"] = "auto"

    # 5. 실행 중 "멈춰"
    os.environ["BALL_MOCK_POLICY_S"] = "1.0"
    mocks.workbench.reset()
    box = {}
    th = threading.Thread(target=lambda: box.update(r=run("초록 블록 옮겨 줘", base + "-stop")))
    th.start()
    time.sleep(1.5)
    t0 = time.perf_counter()
    stop_end, stop_evs = run("멈춰", base + "-stop")
    th.join(10)
    end = box["r"][0]
    check("멈춰 → 규칙 처리·즉시", stop_end["method"] == "rule" and time.perf_counter() - t0 < 0.5)
    check("실행 중 명령이 stopped 로 끝남", end["status"] == "stopped" and end["done"] <= 2, end)

    # 6. 실행과 실행 사이(다시 세는 중)에 "멈춰" → 다음 정책을 시작하지 않는다
    os.environ["BALL_MOCK_POLICY_S"] = "0.05"
    os.environ["BALL_MOCK_DETECT_S"] = "0.6"
    mocks.workbench.reset()
    box = {}
    th = threading.Thread(target=lambda: box.update(r=run("초록 블록 옮겨 줘", base + "-stop2")))
    th.start()
    time.sleep(1.0)                        # plan 0.6s → 정책 0.05s → verify 의 detect 0.6s 도중
    run("멈춰", base + "-stop2")
    at_stop = mocks.workbench.attempts
    th.join(10)
    end = box["r"][0]
    check("실행 사이 멈춰 → 다음 정책 안 함", end["status"] == "stopped" and mocks.workbench.attempts == at_stop,
          {"status": end["status"], "at_stop": at_stop, "after": mocks.workbench.attempts})
    os.environ["BALL_MOCK_DETECT_S"] = "0.03"

    # 7. refused(Studio 세션) → 재시도 없이 이유를 말하고 끝
    os.environ["BALL_MOCK_REFUSE"] = "studio_session"
    mocks.workbench.reset()
    mocks.workbench.zones["bowl_red"]["red_ok"] = 2
    end, evs = run("분홍 블록 하나 내 손에 줘", base + "-refused")
    os.environ["BALL_MOCK_REFUSE"] = ""
    check("refused → 1회로 끝·Studio 안내", end["status"] == "error" and end["attempts"] == 1
          and any("Studio 화면" in x for x in says(evs)), says(evs))

    # 8. 실패 주입: 집었다가 구역 밖에 떨어뜨림 → 1회 재시도, 끝 말과 보고에 구역 밖 1개·확인 요청(ACT = 손 전달)
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    os.environ["BALL_MOCK_DROP_AT"] = "1"
    mocks.workbench.reset()
    mocks.workbench.zones["bowl_red"]["red_ok"] = 2
    end, evs = run("분홍 블록 하나 내 손에 줘", base + "-drop")
    sv = says(evs)
    # 손 전달 확인은 분홍 그릇이 줄면 건넨 것으로 센다 → 떨어진 블록도 전달 1로 세고 재시도하지 않는다(예전부터, 손 전달은 그대로)
    check("떨어뜨림(손 전달) → 끝 말에 구역 밖 1개·확인 요청", end["status"] == "done"
          and any("1개는 구역 밖에 떨어져 있습니다. 확인해 주세요." in x for x in sv), (end, sv))
    end, evs = run("결과 알려 줘", base + "-drop")
    f = next((e["data"]["facts"] for e in evs if e["type"] == "report"), {})
    check("떨어뜨림 → 보고 사실·문장", f.get("outside") == {"red_ok": 1}
          and "구역 밖에 정상 분홍 1개가 떨어져 있습니다" in report_template(f), f.get("outside"))
    os.environ["BALL_MOCK_DROP_AT"] = ""

    # 9. 실패 주입: 두 번 연속 못 옮김 → 포기 → 보고에 사람 확인 요청 1건(ACT = 손 전달)
    os.environ["BALL_MOCK_FAIL_AT"] = "1,2"
    mocks.workbench.reset()
    mocks.workbench.zones["bowl_red"]["red_ok"] = 2
    end, evs = run("분홍 블록 하나 내 손에 줘", base + "-giveup2")
    check("못 옮김 2회 → gave_up·재시도 1회", end["status"] == "gave_up" and end["attempts"] == 2
          and "놓쳤어요. 한 번 더 해 볼게요." in says(evs)
          and any("분홍 블록을 두 번 옮기지 못했습니다. 확인해 주세요." in x for x in says(evs)), says(evs))
    end, evs = run("결과 알려 줘", base + "-giveup2")
    f = next((e["data"]["facts"] for e in evs if e["type"] == "report"), {})
    check("포기 → 보고에 확인 요청 1건", len(f.get("checks") or []) == 1 and f.get("attempts") == 2 and f.get("ok") == 0
          and "사람 확인 요청 1건(두 번 실패)" in report_template(f), f.get("checks"))

    # 10. 실패 주입: 손이 안 옴 → 로봇 0회 → 보고에 확인 요청(손이 보이지 않음)
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    os.environ["BALL_MOCK_HAND"] = "never"
    config.HAND_WAIT_S = 1.0
    mocks.workbench.reset()
    mocks.workbench.zones["bowl_red"]["red_ok"] = 1
    end, evs = run("분홍 블록 하나 내 손에 줘", base + "-nohand2")
    end2, evs2 = run("결과 알려 줘", base + "-nohand2")
    f = next((e["data"]["facts"] for e in evs2 if e["type"] == "report"), {})
    check("손 안 옴 → 0회·보고에 손이 보이지 않음", end["status"] == "no_hand" and end["attempts"] == 0
          and [c["why"] for c in f.get("checks") or []] == ["손이 보이지 않음"]
          and "손이 보이지 않아 건네지 않았습니다." in says(evs), (says(evs), f.get("checks")))
    config.HAND_WAIT_S = old
    os.environ["BALL_MOCK_HAND"] = "auto"

    # 11. scene: 지금 작업대 질문 → 로봇 안 움직임, 답의 숫자·칸이 detect JSON 에 있음
    mocks.workbench.reset()
    end, evs = run("분홍 블록 몇 개 남았어?", base + "-scene")
    sc = next((e["data"] for e in evs if e["type"] == "say" and e["node"] == "scene"), {})
    ok_sc, why = graph.scene_answer_ok(sc.get("text", ""), sc.get("facts") or {"블록 위치": []})
    check("scene 분홍 몇 개 → 4 포함·검사 통과·로봇 0회", end["intent"] == "scene" and "4" in sc.get("text", "") and ok_sc
          and mocks.workbench.attempts == 0, (sc.get("text"), why))
    mocks.workbench.zones["input"]["blue_ok"] = 1
    mocks.workbench.zones["bowl_blue"]["blue_ok"] = 3
    end, evs = run("초록 그릇에 몇 개야?", base + "-scene")
    sc = next((e["data"] for e in evs if e["type"] == "say" and e["node"] == "scene"), {})
    check("scene 초록 그릇 → 3", "3" in sc.get("text", "") and graph.scene_answer_ok(sc.get("text", ""), sc["facts"])[0],
          sc.get("text"))

    # 12. plan: 두 단계(분홍 2개 → 불량 전부), 실행 전에 계획을 말하고 끝에 계획 대비 결과
    os.environ["BALL_MOCK_HAND_DELAY"] = "0.3"
    mocks.workbench.reset()
    end, evs = run("분홍 두 개만 옮기고 불량 골라내 줘", base + "-plan1")
    w, sv = mocks.workbench, says(evs)
    check("plan 분홍 2 → 불량 3", end["intent"] == "plan" and w.zones["bowl_red"]["red_ok"] == 2
          and w.zones["bowl_defect"]["red_defect"] + w.zones["bowl_defect"]["blue_defect"] == 3
          and w.zones["input"]["red_ok"] == 2, w.zones)
    check("plan 실행 전 계획 알림·끝 보고", sv and sv[0].startswith(graph.HOW["coord"] + " 순서대로 할게요.") and "분홍 블록 2개" in sv[0]
          and any(x.startswith("계획 2단계를 모두 마쳤습니다") for x in sv), sv)
    # 손 전달 + 나머지: 분홍 그릇이 비어 있으면 먼저 옮기는 단계를 넣는다 → 손 1·그릇 3
    mocks.workbench.reset()
    end, evs = run("분홍 하나는 내 손에, 나머지 분홍은 그릇에", base + "-plan2")
    w = mocks.workbench
    check("plan 손 1·분홍 그릇 3", w.handed == 1 and w.zones["bowl_red"]["red_ok"] == 3 and w.zones["input"]["red_ok"] == 0,
          (w.handed, w.zones["bowl_red"], says(evs)))
    mocks.workbench.reset()
    end, evs = run("초록은 그대로 두고 나머지 정리해 줘", base + "-plan3")
    w = mocks.workbench
    check("plan 초록 그대로·불량·분홍 정리", w.zones["input"]["blue_ok"] == 4 and w.zones["bowl_red"]["red_ok"] == 4
          and w.zones["bowl_defect"]["red_defect"] + w.zones["bowl_defect"]["blue_defect"] == 3, w.zones)
    mocks.workbench.reset()
    end, evs = run("숫자 3 모양으로 놓아 줘", base + "-plan4")       # 10-22: 숫자 모양은 좌표 집기(arrange)로 → 블록 부족 거절
    check("숫자 3 모양 → arrange·블록 부족 이유 말하고 거절·로봇 0회", mocks.workbench.attempts == 0
          and end["status"] == "refused" and end["intent"] == "arrange" and not mocks.workbench.arranged
          and any("블록 11개가 필요한데" in x for x in says(evs)), says(evs))
    end, evs = run("블록 탑처럼 쌓아 줘", base + "-plan4b")
    check("plan 배운 적 없는 동작(쌓기) → 거절·로봇 0회", mocks.workbench.attempts == 0 and end["status"] == "refused"
          and any(x == graph.REFUSE_MSG for x in says(evs)), says(evs))
    # 규칙 계획이 못 읽는 말 → 첫 말은 바로 "네, 계획을 세울게요." → LLM 계획 알림 → 실행
    mocks.workbench.reset()
    end, evs = run("분홍이랑 초록 두 개씩 옮겨 줘", base + "-plan6")
    sv, w = says(evs), mocks.workbench
    check("plan LLM 경로 → 수락 멘트 먼저·계획 알림·실행", sv[:1] == [graph.ACCEPT_MSGS["coord"]] and len(sv) > 1
          and sv[1].startswith(graph.HOW["coord"] + " 순서대로 할게요.") and end["intent"] == "plan"
          and w.zones["bowl_red"]["red_ok"] == 2 and w.zones["bowl_blue"]["blue_ok"] == 2, sv[:2])
    # "불량 빼고 다 정리해 줘" = 불량 빼내고 나머지(규칙, LLM 없이) → 첫 말이 곧 계획 알림
    mocks.workbench.reset()
    end, evs = run("불량 빼고 다 정리해 줘", base + "-plan7")
    sv, w = says(evs), mocks.workbench
    check("plan '불량 빼고 다 정리' → 규칙 계획(불량→분홍→초록)", sv[:1] and sv[0].startswith(graph.HOW["coord"] + " 순서대로 할게요." + " 1) 불량")
          and w.zones["bowl_defect"]["red_defect"] + w.zones["bowl_defect"]["blue_defect"] == 3
          and w.zones["bowl_red"]["red_ok"] == 4 and w.zones["bowl_blue"]["blue_ok"] == 4, sv[:1])
    # 단계 중간 실패 → 남은 단계 안 함(ACT = 손 전달 단계. 앞에 '분홍 그릇에 먼저 옮기기'가 붙는다)
    os.environ["BALL_MOCK_FAIL_AT"] = "1,2"
    mocks.workbench.reset()
    end, evs = run("분홍 하나는 내 손에, 나머지 분홍은 그릇에", base + "-plan5")
    w, sv = mocks.workbench, says(evs)
    check("plan 손 전달 단계 포기 → 다음 단계 안 함·보고", w.attempts == 2 and w.handed == 0 and w.zones["input"]["red_ok"] > 0
          and any("남은 단계는 하지 않았습니다" in x for x in sv), (sv, w.zones["input"]))
    os.environ["BALL_MOCK_FAIL_AT"] = ""

    # 13. 좌표 집기(arrange, plan 의 한 갈래): dry-run 계획 → "할까요?"(interrupt) → 응/아니 → 가짜 실행(fake.py 모의 + 흉내)
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    mocks.workbench.reset()
    end, evs = run("블록으로 7 만들어 줘", base + "-arr1")
    sv = says(evs)
    lay = next((e["data"].get("layout") for e in evs if e["type"] == "tool" and e["data"].get("name") == "arrange_digit"), {})
    check("arrange 수락 멘트 먼저·계획 말하고 할까요?", sv[:1] == [graph.ACCEPT_ARRANGE.format(d="7")]
          and sv[-1].endswith("할까요?") and end["status"] == "waiting_confirm" and end["intent"] == "arrange"
          and len((lay or {}).get("cells", [])) == 7, (sv, end["status"]))
    end, evs = run("응", base + "-arr1")
    w = mocks.workbench
    check("arrange 응 → 가짜 실행 끝까지·블록 7개 놓음", end["status"] == "done" and len(w.arranged) == 7
          and w.zones["input"]["red_ok"] + w.zones["input"]["blue_ok"] == 1 and w.attempts == 0, (says(evs), end["status"]))
    mocks.workbench.reset()
    run("블록으로 7 만들어 줘", base + "-arr2")
    end, evs = run("아니", base + "-arr2")
    check("arrange 아니 → 취소·안 움직임", end["status"] == "cancelled" and not mocks.workbench.arranged
          and mocks.workbench.zones["input"]["red_ok"] == 4, (says(evs), end["status"]))
    end, evs = run("블록으로 하트 만들어 줘", base + "-arr3")
    check("arrange 숫자 아닌 모양 → 이유 말하고 거절", end["status"] == "refused"
          and any("숫자 모양(0~9)만" in x for x in says(evs)), says(evs))
    end, evs = run("분홍으로 숫자 7 만들어 줘", base + "-arr4")
    check("arrange 블록 부족(분홍 4개) → 이유 말하고 거절", end["status"] == "refused"
          and any("블록 7개가 필요한데" in x for x in says(evs)) and not mocks.workbench.arranged, says(evs))
    # 실행 중 멈춰 → robot.stop() 과 같은 길(가짜는 mocks.stop) → 다 놓기 전에 멈춤
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.05"
    mocks.workbench.reset()
    run("블록으로 7 만들어 줘", base + "-arr5")
    box = {}
    th = threading.Thread(target=lambda: box.update(r=run("응", base + "-arr5")))
    th.start()
    time.sleep(2.0)
    run("멈춰", base + "-arr5")
    th.join(30)
    end = box["r"][0]
    check("arrange 실행 중 멈춰 → stopped·7개 못 채움", end["status"] == "stopped" and len(mocks.workbench.arranged) < 7,
          (end["status"], len(mocks.workbench.arranged)))
    # 보정 전(가짜가 아닌 경우의 문) → 확인 없이 계획(칸 배치)만
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    orig_gates, graph.arrange_gates = graph.arrange_gates, (lambda r: ["보정 전"])
    try:
        mocks.workbench.reset()
        end, evs = run("블록으로 7 만들어 줘", base + "-arr6")
    finally:
        graph.arrange_gates = orig_gates
    tool = next((e["data"] for e in evs if e["type"] == "tool" and e["data"].get("name") == "arrange_digit"), {})
    check("arrange 보정 전 → 계획만·칸 배치 이벤트", end["status"] == "plan_only" and tool.get("plan_only")
          and len(tool["layout"]["cells"]) == 7 and any(x.startswith("보정 전이라 계획만 보여 드릴게요") for x in says(evs)),
          says(evs))
    # 기다리는 중 다른 명령 → 기다림을 끝내고 그 명령을 한다
    mocks.workbench.reset()
    run("블록으로 7 만들어 줘", base + "-arr7")
    end, evs = run("분홍 블록 옮겨 줘", base + "-arr7")
    check("arrange 기다리는 중 다른 명령 → 취소하고 새 명령", end["intent"] == "red" and end["done"] == 4
          and not mocks.workbench.arranged and any("하지 않을게요" in x for x in says(evs)), says(evs))

    # 14. 발표 실연(계약서 2절, BALL_MOCK_SCENE=demo4): 불량 1 → 배치 질문 → 불량 빼고 나머지(3회) → 손 1 → 보고
    os.environ["BALL_MOCK_SCENE"] = "demo4"
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    os.environ["BALL_MOCK_HAND_DELAY"] = "0.3"
    mocks.workbench.reset()
    tid = base + "-demo4"
    seq = [run(t, tid) for t in ("불량품 골라내 줘", "지금 배치 어때?", "불량 빼고 나머지 다 정리해 줘",
                                 "분홍 블록 하나 내 손에 줘", "결과 알려 줘")]
    w = mocks.workbench
    f = next((e["data"]["facts"] for e in seq[4][1] if e["type"] == "report"), {})
    s3 = says(seq[2][1])
    # 발표 실연 ②·⑤ 말을 짧게(10-22 r11, 각 5초 안팎). 자세한 숫자는 say 의 facts·report 이벤트에
    check("실연 demo4 ② 요약 말 짧게", says(seq[1][1]) == ["투입 구역에 분홍 2개, 초록 1개가 있고, 불량 그릇에 1개가 있어요."],
          says(seq[1][1]))
    check("실연 demo4 ⑤ 보고 말 짧게", says(seq[4][1]) == ["불량 1개, 분홍 1개, 초록 1개, 손 전달 1개입니다. 모두 성공했습니다."],
          says(seq[4][1]))
    check("실연 demo4 ① 불량 1개 ② scene", seq[0][0]["done"] == 1 and seq[1][0]["intent"] == "scene")
    check("실연 demo4 ③ 규칙 계획 분홍 2·초록 1(불량 단계 건너뜀)", seq[2][0]["intent"] == "plan"
          and s3[:1] and s3[0].startswith(graph.HOW["coord"] + " 순서대로 할게요.") and "불량품이 없어서 그 단계는 건너뜁니다" in s3[0]
          and any(x.startswith("계획 2단계를 모두 마쳤습니다") for x in s3), s3)
    check("실연 demo4 ④ 손 1 ⑤ 보고 5회 중 5회·일치", w.handed == 1 and w.zones["bowl_red"]["red_ok"] == 1
          and w.zones["bowl_blue"]["blue_ok"] == 1 and sum(w.zones["input"].values()) == 0
          and f.get("attempts") == 5 and f.get("ok") == 5 and f.get("match"), (w.zones, f.get("attempts")))
    os.environ["BALL_MOCK_SCENE"] = "full"

    # 질문 첫마디(10-22 r10): question 으로 판별된 뒤 "자료를 찾아볼게요." 를 먼저, 답은 그다음. 자기 질문·작업 기록 질문엔 없음
    end, evs = run("불량 기준이 뭐예요?", base + "-ack")
    sv = says(evs)
    check("질문 → 첫마디 먼저·답 다음", end["intent"] == "question" and sv[:1] == [graph.ASK_ACK] and len(sv) == 2
          and sv[1] != graph.ASK_ACK, sv)
    end2, evs2 = run("너 이름이 뭐야?", base + "-self")
    end3, evs3 = run("오늘 몇 번 실패했어?", base + "-rec")
    check("자기 질문·작업 기록 질문엔 첫마디 없음", graph.ASK_ACK not in says(evs2) + says(evs3)
          and says(evs2)[-1:] == [graph.NO_ANSWER] and end3["intent"] == "report", (says(evs2), end3.get("intent")))

    # 15. 좌표 집기(BALL_COORD=1, 10-22 r16): defect·red·blue 와 plan 의 그 단계는 ACT 대신 sort_blocks(dry-run → 말 → 바로 실행
    #     → 다시 세기). handoff 는 ACT. 가짜 sort_blocks = mocks.Workbench.sort_blocks(블록당 40초 × BALL_MOCK_ARRANGE_SCALE)
    def nodes(evs):
        return [e["node"] for e in evs if e["type"] == "node"]

    graph.set_method("coord")
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    os.environ["BALL_MOCK_SCENE"] = "demo4"
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    mocks.workbench.reset()
    tid = base + "-coord"
    seq = [run(t, tid) for t in ("불량품 골라내 줘", "불량 빼고 나머지 다 정리해 줘", "분홍 블록 하나 내 손에 줘", "결과 알려 줘")]
    w = mocks.workbench
    f = next((e["data"]["facts"] for e in seq[3][1] if e["type"] == "report"), {})
    check("coord ① 불량 → plan·coord·finish(act 없음)·말 3개", "coord" in nodes(seq[0][1]) and "act" not in nodes(seq[0][1])
          and says(seq[0][1]) == ["불량품 1개(분홍 1)를 좌표로 골라낼게요.", "블록 1개, 약 41초 걸려요.",
                                  "불량 1개(분홍 1)를 분리했습니다."] and w.zones["bowl_defect"]["red_defect"] == 1,
          says(seq[0][1]))
    check("coord ③ 계획 분홍 2·초록 1 을 좌표 집기로", w.zones["bowl_red"]["red_ok"] + w.handed == 2   # ④ 가 분홍 그릇에서 1개
          and w.zones["bowl_blue"]["blue_ok"] == 1
          and nodes(seq[1][1]).count("coord") >= 4 and "act" not in nodes(seq[1][1])
          and any(x.startswith("계획 2단계를 모두 마쳤습니다") for x in says(seq[1][1])), says(seq[1][1]))
    check("coord ④ 손 전달은 ACT 그대로", w.handed == 1 and w.attempts == 1 and "act" in nodes(seq[2][1]))
    check("coord ⑤ 보고: 블록마다 시도 1회·5회 중 5회·기록 일치·짧은 말 같음", f.get("attempts") == 5 and f.get("ok") == 5
          and f.get("match") and set(f.get("policies", {})) == {"coord_defect", "coord_red", "coord_blue", "handoff"}
          and says(seq[3][1]) == ["불량 1개, 분홍 1개, 초록 1개, 손 전달 1개입니다. 모두 성공했습니다."],
          (f.get("attempts"), f.get("ok"), f.get("issues"), says(seq[3][1])))
    os.environ["BALL_MOCK_SCENE"] = "full"
    mocks.workbench.reset()
    end, evs = run("분홍 블록 두 개만 옮겨 줘", base + "-coord2")
    dry = next((e["data"]["result"] for e in evs if e["type"] == "tool" and e["data"].get("name") == "sort_blocks"), {})
    check("coord 개수 지정 2개 → max_blocks 2", end["done"] == 2 and dry.get("blocks") == 2
          and mocks.workbench.zones["input"]["red_ok"] == 2, (end, dry.get("blocks")))
    for fail, want, moved in (("unreachable", "팔이 닿지 않는 자리에 블록이 있어 좌표 집기 계획을 세우지 못했어요. 확인해 주세요.", 0),
                              ("miss", "분홍 블록 3개를 담았습니다. 좌표 집기로 4개 중 3개만 옮겨졌어요. 확인해 주세요.", 3),
                              ("tracking", "좌표 집기가 팔이 목표를 따라가지 못해 멈췄어요. 확인해 주세요.", None)):
        os.environ["BALL_MOCK_COORD_FAIL"] = fail
        mocks.workbench.reset()
        end, evs = run("분홍 블록 옮겨 줘", base + "-coordf-" + fail)
        sv = says(evs)
        got = mocks.workbench.zones["bowl_red"]["red_ok"]
        check(f"coord {fail} → error·이유 말함", end["status"] == "error" and sv[-1:] and sv[-1].endswith(want)
              and (moved is None or got == moved) and mocks.workbench.attempts == 0, (sv[-1:], got))
    os.environ["BALL_MOCK_COORD_FAIL"] = ""
    # 실행 중 멈춰 → mocks.stop(실제는 robot.run_policy.stop) → 다 옮기기 전에 멈춤
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.05"          # 4개 × 40초 × 0.05 = 8초
    mocks.workbench.reset()
    box = {}
    th = threading.Thread(target=lambda: box.update(r=run("분홍 블록 옮겨 줘", base + "-coordstop")))
    th.start()
    time.sleep(1.5)
    run("멈춰", base + "-coordstop")
    th.join(15)
    end = box["r"][0]
    check("coord 실행 중 멈춰 → stopped·4개 못 채움", end["status"] == "stopped"
          and mocks.workbench.zones["bowl_red"]["red_ok"] < 4 and says(box["r"][1])[-1].startswith("멈췄습니다."),
          (end["status"], mocks.workbench.zones["bowl_red"], says(box["r"][1])[-1:]))
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    # 실제 로봇 함수인 척 + 송신 문 닫힘(BALL_ROBOT 없음, 보정 전이면 그것도) → 보내지 않고 계획만
    info = graph.backends.info()
    info["coord"] = "real"
    mocks.workbench.reset()
    end, evs = run("분홍 블록 옮겨 줘", base + "-coordgate")
    info["coord"] = "mock"
    check("coord 송신 문 닫힘 → 계획만·안 움직임", end["status"] == "error" and mocks.workbench.zones["bowl_red"]["red_ok"] == 0
          and any("이라 좌표 집기는 계획만 세웠어요(블록 4개" in x for x in says(evs)), says(evs))
    # sort_blocks 를 못 불러왔으면 거부(10-23 13:10: ACT 로 대신하지 않음, 로그 = tool 이벤트 unavailable). 손 전달은 그대로
    be = graph.backends.get()
    real_sort, be["sort"] = be["sort"], None
    mocks.workbench.reset()
    end, evs = run("분홍 블록 옮겨 줘", base + "-coordoff")
    e2, v2 = run("분홍 두 개만 옮기고 불량 골라내 줘", base + "-coordoff")
    mocks.workbench.zones["bowl_red"]["red_ok"] = 1
    e3, v3 = run("분홍 블록 하나 내 손에 줘", base + "-coordoff")
    be["sort"] = real_sort
    check("좌표 집기 함수 없음 → 옮기기·계획 거부(정책 0·카메라 0)·손 전달은 ACT", says(evs) == [graph.COORD_OFF_MSG]
          and end["status"] == "refused" and "act" not in nodes(evs) and "coord" not in nodes(evs)
          and any(e["type"] == "tool" and e["data"].get("status") == "unavailable" for e in evs)
          and e2["status"] == "refused" and says(v2)[-1:] == [graph.COORD_OFF_MSG] and "act" not in nodes(v2)
          and e3["done"] == 1 and mocks.workbench.attempts == 1 and mocks.workbench.zones["input"]["red_ok"] == 4,
          (says(evs), says(v2), e3.get("done"), mocks.workbench.attempts))
    try:
        graph.set_method("act")
        refused = False
    except ValueError as e:
        refused = str(e).startswith(graph.ACT_ONLY_MSG)
    m = graph.get_method()
    check("set_method('act') 거부·기본 coord·act_for = 손 전달", refused and m["default"] == "coord"
          and m["act_for"] == ["handoff"], m)
    pol = {n: graph.run_policy.invoke({"name": "handoff"})["reason"] for n in ("handoff",)}
    try:
        bad = graph.run_policy.func("pick_red")
    except Exception as e:
        bad = {"reason": type(e).__name__}
    check("run_policy 는 handoff 만(pick_red → refused·로봇 0)", bad.get("reason") == "refused"
          and bad.get("error") == "act_only_handoff" and pol == {"handoff": "home_reached"}, (bad, pol))
    os.environ["BALL_MOCK_FAIL_AT"] = "3"

    # 16. 받아쓰기 환청(10-22 r16): 그래프 앞에서 거르고 "잘 못 들었어요"(실물 15:57 "감사합니다." → question 으로 갔다)
    from . import hearing
    end, evs = run("감사합니다.", base + "-hear")
    check("환청 '감사합니다.' → 잘 못 들었어요·그래프 안 감", end["intent"] == "unheard"
          and says(evs) == [hearing.UNHEARD_MSG] and "classify" not in nodes(evs), (end.get("intent"), says(evs)))
    junk = ["시청해 주셔서 감사합니다.", "MBC 뉴스 이덕영입니다.", "구독과 좋아요 부탁드립니다", "감사합니다. 감사합니다.", "음...", "♪♪"]
    keep = ["분홍 블록 옮겨 줘 감사합니다", "네", "그렇게 해", "결과 알려 줘", "감사 기준이 뭐예요?", "아니"]
    check("환청 문구·말 없는 소리는 거르고 명령·대답은 둔다", all(hearing.check(t) for t in junk)
          and not any(hearing.check(t) for t in keep), [(t, hearing.check(t)) for t in junk + keep])
    check("녹음 정보: 말소리 0.05초 → 거름, 0.8초 → 둠, 0.36초에 12자 → 거름",
          hearing.check("분홍 블록 옮겨 줘", {"speech_s": 0.05}) and not hearing.check("분홍 블록 옮겨 줘", {"speech_s": 0.8})
          and hearing.check("오늘 영상은 여기까지입니다", {"speech_s": 0.36}) and not hearing.check("네", {"speech_s": 0.3}))
    # "할까요?" 기다림은 숫자 배치만 쓴다. 로봇 계획(설정이 바뀌는 중)과 떼려고 arrange 함수를 고정 결과로 바꿔 시험
    be = graph.backends.get()
    real_arrange = be["arrange"]
    calls = []

    def fixed_arrange(d, color="any", dry_run=True, **kw):
        calls.append(dry_run)
        plan = [{"order": 1, "color": "분홍", "to_cell": [0, 0]}]
        return {"ok": True, "plan": plan, "pattern": ["#"], "blocks": 1, "time_s": 30, "warnings": [],
                "reason": "planned" if dry_run else "done", "mode": "dry_run" if dry_run else "real"}
    be["arrange"] = fixed_arrange
    mocks.workbench.reset()
    e0, _ = run("블록으로 7 만들어 줘", base + "-hear2")
    e1, ev1 = run("감사합니다.", base + "-hear2")
    e2, ev2 = run("응", base + "-hear2")
    be["arrange"] = real_arrange
    check("'할까요?' 기다리는 중 환청 → 기다림 유지·다음 '응'으로 실행", e0["status"] == "waiting_confirm"
          and e1["intent"] == "unheard" and e2["status"] == "done" and calls == [True, False],
          (e0.get("status"), e1.get("intent"), e2.get("status"), calls))

    # 17. 홈 맞춤(10-22 r20): 정책·좌표 집기 직전 go_home(가짜 = mocks.Workbench.go_home). '멈춤'으로 끊기면 팔이 홈 밖에 남고,
    #     다음 명령은 먼저 홈으로 옮긴 뒤 정책을 돌린다(16:53 실물 handoff 가 홈 밖에서 시작해 조기 종료가 안 걸림)
    def homes(evs):      # 정책·좌표 집기 직전 홈 맞춤만(동작 뒤 집게 열고 홈 복귀 r22 는 posts)
        return [e["data"]["result"]["reason"] for e in evs if e["type"] == "tool" and e["data"].get("name") == "go_home"
                and e["data"].get("status") == "end" and not e["data"].get("open_gripper")]

    def posts(evs):
        return [e["data"]["result"]["reason"] for e in evs if e["type"] == "tool" and e["data"].get("name") == "go_home"
                and e["data"].get("status") == "end" and e["data"].get("open_gripper")]

    os.environ["BALL_MOCK_FAIL_AT"] = ""
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.05"           # 좌표 집기 4개 × 40초 × 0.05 = 8초 — 그 사이에 멈춤
    mocks.workbench.reset()
    graph._HOME.update(ok=False, fresh=None, moved=False)
    box = {}
    th = threading.Thread(target=lambda: box.update(r=run("초록 블록 옮겨 줘", base + "-home")))
    th.start()
    time.sleep(1.5)
    run("멈춰", base + "-home")
    th.join(15)
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    mocks.workbench.zones["bowl_red"]["red_ok"] = 2
    end, evs = run("분홍 블록 하나 내 손에 줘", base + "-home")
    check("멈춤 뒤 다음 명령(손 전달 정책) → 정책 전에 홈으로 옮김", box["r"][0]["status"] == "stopped"
          and homes(box["r"][1])[:1] == ["already_home"] and homes(evs) == ["done"]
          and mocks.workbench.homed == 1 and end["done"] == 1, (homes(box["r"][1]), homes(evs), end["status"]))
    os.environ["BALL_MOCK_HOME"] = "too_far"
    mocks.workbench.reset()
    graph._HOME.update(ok=False, fresh=None, moved=False)
    end, evs = run("분홍 블록 옮겨 줘", base + "-home2")
    os.environ["BALL_MOCK_HOME"] = ""
    check("홈에서 60° 넘게 멂 → '홈 근처로 옮겨 주세요'·정책 0회", end["status"] == "error" and mocks.workbench.attempts == 0
          and says(evs)[-1:] and says(evs)[-1].endswith(graph.HOME_FAR_MSG), says(evs))
    config.HOME_ALIGN = "after_stop"
    mocks.workbench.reset()
    graph._HOME.update(ok=False, fresh=None, moved=False)
    _, ev1 = run("분홍 블록 두 개만 옮겨 줘", base + "-home3")
    run("멈춰", base + "-home3")
    _, ev2 = run("초록 블록 두 개만 옮겨 줘", base + "-home3")
    config.HOME_ALIGN = "always"
    check("HOME_ALIGN=after_stop → 정상 끝 다음은 건너뜀·멈춰 뒤엔 다시", homes(ev1) == ["already_home"]
          and homes(ev2) == ["already_home"], (homes(ev1), homes(ev2)))
    graph.set_method("coord")
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.05"
    mocks.workbench.reset()
    graph._HOME.update(ok=False, fresh=None, moved=False)
    box = {}
    th = threading.Thread(target=lambda: box.update(r=run("분홍 블록 옮겨 줘", base + "-home4")))
    th.start()
    time.sleep(1.5)
    run("멈춰", base + "-home4")
    th.join(15)
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    end, evs = run("분홍 블록 옮겨 줘", base + "-home4")
    check("좌표 집기 멈춤 뒤 다음 좌표 집기 → 실행 전에 홈으로", box["r"][0]["status"] == "stopped" and homes(evs) == ["done"]
          and end["status"] == "done" and mocks.workbench.zones["input"]["red_ok"] == 0, (homes(evs), end["status"]))
    os.environ["BALL_MOCK_FAIL_AT"] = "3"

    # 18. 명령마다 방식(10-22 r21 → 10-23 13:10): 옮기기는 늘 좌표, ACT 단서는 첫 말 "학습한 동작은 손 전달에만 써요."로 알림
    def picks(evs):
        return next((e["data"]["result"] for e in evs
                     if e["type"] == "tool" and e["data"].get("name") == "method_pick"), {})

    os.environ["BALL_MOCK_FAIL_AT"] = ""
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    graph.set_method("coord")
    mocks.workbench.reset()
    graph._HOME.update(ok=False, fresh=None, moved=False)
    end, evs = run("분홍블록 하나를 act정책으로 옮겨줘", base + "-m1")
    check("'act정책으로' 옮기기 → 좌표 집기·첫 말 '학습한 동작은 손 전달에만 써요. …좌표로 옮길게요.'·정책 0",
          "coord" in nodes(evs) and "act" not in nodes(evs) and mocks.workbench.attempts == 0
          and says(evs)[:1] == ["학습한 동작은 손 전달에만 써요. 분홍 블록 1개를 좌표로 옮길게요."]
          and end["robot_method"] == "coord" and picks(evs).get("asked") == "act" and end["done"] == 1,
          (says(evs)[:1], picks(evs)))
    end, evs = run("초록 블록 옮겨 줘", base + "-m1")
    check("기본 coord + 단서 없음 → 좌표 집기·첫 말 '좌표로 옮길게요'", "coord" in nodes(evs) and end["robot_method"] == "coord"
          and says(evs)[:2] == ["초록 블록 4개를 좌표로 옮길게요.", "블록 4개, 약 2분 41초 걸려요."], says(evs)[:2])
    end, evs = run("분홍블록 하나를 좌표로 옮겨줘", base + "-m1")
    check("'좌표로' → 좌표 집기", "coord" in nodes(evs) and end["done"] == 1
          and says(evs)[:1] == ["분홍 블록 1개를 좌표로 옮길게요."], says(evs)[:1])
    end, evs = run("분홍 블록 하나 좌표로 내 손에 줘", base + "-m1")
    check("손 전달 + '좌표로' → ACT 로만·첫 말에 이유", end["intent"] == "handoff" and "coord" not in nodes(evs)
          and says(evs)[:1] and says(evs)[0].startswith("손에 건네기는 학습한 동작으로만 해요.") and mocks.workbench.handed == 1,
          says(evs)[:1])
    mocks.workbench.reset()
    end, evs = run("불량 빼고 나머지 좌표로 다 정리해 줘", base + "-m2")
    sv = says(evs)
    check("plan + '좌표로' → 규칙 계획 '좌표로 순서대로'·단계 모두 좌표 집기", sv[:1]
          and sv[0].startswith("좌표로 순서대로 할게요. 1) 불량") and "act" not in nodes(evs)
          and nodes(evs).count("coord") >= 6 and sum(mocks.workbench.zones["input"].values()) == 0,
          (sv[:1], nodes(evs).count("coord")))
    mocks.workbench.reset()
    end, evs = run("불량 빼고 나머지 학습한 동작으로 다 정리해 줘", base + "-m3")
    e2, v2 = run("분홍이랑 초록 두 개씩 학습한 동작으로 옮겨 줘", base + "-m3")
    mocks.workbench.zones["bowl_red"]["red_ok"] = 1
    e3, v3 = run("학습한 동작으로 분홍 블록 하나 내 손에 줘", base + "-m3")
    check("plan + ACT 단서 → 좌표로 순서대로(규칙 계획·LLM 수락 멘트 앞에 알림)·손 전달 + ACT 단서 → 알림 없이 ACT",
          says(evs)[:1] and says(evs)[0].startswith("학습한 동작은 손 전달에만 써요. 좌표로 순서대로 할게요. 1) 불량")
          and "act" not in nodes(evs) and says(v2)[:1] == [graph.ACCEPT_ACT_CUE] and "act" not in nodes(v2)
          and says(v3)[:1] == ["분홍 블록 1개를 학습한 동작으로 건네 드릴게요."] and e3["done"] == 1,
          (says(evs)[:1], says(v2)[:1], says(v3)[:1]))
    m = graph.get_method()
    be = graph.backends.get()
    real_sort, be["sort"] = be["sort"], None
    try:
        graph.set_method("coord")
        refused = False
    except ValueError:
        refused = True
    be["sort"] = real_sort
    check("get_method 표시·좌표 집기 없으면 set_method('coord') 거부", m["default"] == "coord" and m["coord_available"]
          and refused and graph.get_method()["default"] == "coord", m)
    cue = {t: graph.method_cue(t)[0] for t in ("ACT로 말고 좌표로 옮겨 줘", "좌표로 하지 말고 정책으로 분홍 옮겨 줘",
                                             "분홍 블록 계산해서 옮겨 줘", "ACT 정책은 몇 개야?", "좌표 집기는 어떻게 해?")}
    check("단서: 뒤의 것·'말고'는 반대·질문의 낱말은 단서 아님·질문꼴은 원문으로 판별",
          cue == {"ACT로 말고 좌표로 옮겨 줘": "coord", "좌표로 하지 말고 정책으로 분홍 옮겨 줘": "act",
                  "분홍 블록 계산해서 옮겨 줘": "coord", "ACT 정책은 몇 개야?": None, "좌표 집기는 어떻게 해?": None}
          and graph.looks_question("정책으로 하면 몇 초 걸려?"), cue)
    os.environ["BALL_MOCK_FAIL_AT"] = "3"

    # 19. Studio 세션 정리 + 동작 뒤 집게 열고 홈(10-22 r22). 가짜 Studio = mocks.Workbench.studio_list·studio_stop
    def bench(studio="", lock_s="2"):
        os.environ["BALL_MOCK_STUDIO"], os.environ["BALL_MOCK_STUDIO_LOCK_S"] = studio, lock_s
        mocks.workbench.reset()
        graph._HOME.update(ok=False, fresh=None, moved=False)

    os.environ["BALL_MOCK_FAIL_AT"] = ""
    config.STUDIO_POLL_S = 0.2
    bench("teleop")
    end, evs = run("분홍 블록 두 개만 옮겨 줘", base + "-st1")
    check("Studio 원격조작 세션 → '정리하고 바로 할게요'·세션 멈춤·잠금 풀리면 실행", graph.STUDIO_CLEAR_MSG in says(evs)
          and len(mocks.workbench.studio_stops) == 1 and end["done"] == 2, (says(evs), mocks.workbench.studio_stops))
    bench("recording")
    end, evs = run("분홍 블록 두 개만 옮겨 줘", base + "-st2")
    check("Studio 녹화 세션 → 멈추지 않고 거부·정책 0회", end["status"] == "error" and not mocks.workbench.studio_stops
          and mocks.workbench.attempts == 0 and says(evs)[-1:] and says(evs)[-1].endswith(graph.STUDIO_REC_MSG), says(evs))
    bench("unknown")
    end, evs = run("분홍 블록 두 개만 옮겨 줘", base + "-st3")
    check("Studio 세션 상태를 모름 → '←로 나와 주세요'만", end["status"] == "error" and not mocks.workbench.studio_stops
          and says(evs)[-1:] and says(evs)[-1].endswith(graph.STUDIO_OPEN_MSG), says(evs))
    bench("lock", "1.0")
    end, evs = run("분홍 블록 두 개만 옮겨 줘", base + "-st4")
    check("세션 없이 잠금만 → 기다렸다가 실행", graph.STUDIO_WAIT_MSG in says(evs) and end["done"] == 2, says(evs))
    config.STUDIO_WAIT_S = 1.0
    bench("lock", "5")
    end, evs = run("분홍 블록 두 개만 옮겨 줘", base + "-st5")
    config.STUDIO_WAIT_S = 15.0
    check("잠금이 안 풀리면 지금처럼 거부 문구", end["status"] == "error" and mocks.workbench.attempts == 0
          and any(graph._REFUSED["robot_busy"] in x for x in says(evs)), says(evs))
    bench()
    end, evs = run("분홍 블록 두 개만 옮겨 줘", base + "-st6")
    end2, evs2 = run("초록 블록 두 개만 옮겨 줘", base + "-st6")
    skip = [e for e in evs2 if e["type"] == "tool" and e["data"].get("name") == "go_home" and e["data"].get("status") == "skip"]
    check("동작 뒤 집게 열고 홈 1회·다음 명령 첫 홈 맞춤은 건너뜀", posts(evs) == ["done"] and posts(evs2) == ["done"]
          and mocks.workbench.opened == 2 and len(skip) == 1 and end2["done"] == 2, (posts(evs), posts(evs2), len(skip)))
    bench()
    end, evs = run("분홍 두 개만 옮기고 불량 골라내 줘", base + "-st7")
    check("계획 2단계 → 단계마다 집게 열고 홈", end["intent"] == "plan" and posts(evs) == ["done", "done"], posts(evs))
    bench()
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.05"           # 좌표 집기 8초 — 그 사이에 멈춤
    box = {}
    th = threading.Thread(target=lambda: box.update(r=run("초록 블록 옮겨 줘", base + "-st8")))
    th.start()
    time.sleep(1.5)
    run("멈춰", base + "-st8")
    th.join(15)
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    end, evs = run("초록 블록 두 개만 옮겨 줘", base + "-st8")
    check("멈춰로 끝나면 그 자리 유지(홈 복귀 없음)·다음 명령 전 홈 맞춤", box["r"][0]["status"] == "stopped"
          and posts(box["r"][1]) == [] and homes(evs)[:1] == ["done"], (posts(box["r"][1]), homes(evs)))
    graph.set_method("coord")
    bench()
    end, evs = run("분홍 블록 옮겨 줘", base + "-st9")
    check("좌표 집기 뒤에도 집게 열고 홈", "coord" in nodes(evs) and posts(evs) == ["done"] and end["status"] == "done",
          posts(evs))
    os.environ["BALL_MOCK_STUDIO"] = ""
    os.environ["BALL_MOCK_FAIL_AT"] = "3"

    # 20. 실제 로봇 함수면 좌표 집기 dry-run·실행 모두 detection 없이(로봇이 위 카메라 원본을 직접 찍어 블록 방향까지 잼,
    #     10-22 17:50 실물 첫 성공). 가짜는 plan 의 detection 을 넘긴다. 홈 맞춤은 찍기 전에(팔이 사진에 안 걸리게)
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    be, info = graph.backends.get(), graph.backends.info()
    real_sort, real_gates, calls = be["sort"], graph.arrange_gates, []

    def spy(color, dry_run=True, **kw):
        calls.append((dry_run, "detection" in kw))
        return real_sort(color, dry_run, **kw)
    be["sort"], graph.arrange_gates = spy, (lambda r, which="arrange": [])
    graph.set_method("coord")
    info["coord"] = "real"
    bench()
    end, evs = run("분홍 블록 옮겨 줘", base + "-live")
    live_calls, calls[:] = list(calls), []
    info["coord"] = "mock"
    bench()
    end2, evs2 = run("분홍 블록 옮겨 줘", base + "-live2")
    be["sort"], graph.arrange_gates = real_sort, real_gates
    idx = [i for i, e in enumerate(evs) if e["type"] == "tool" and e["data"].get("name") in ("go_home", "sort_blocks")
           and e["data"].get("status") == "end" and not e["data"].get("open_gripper")]
    order = [evs[i]["data"]["name"] for i in idx][:2]
    check("실제 로봇 함수면 detection 없이(dry-run·실행)·가짜는 넘김·홈 맞춤이 먼저", live_calls == [(True, False), (False, False)]
          and calls == [(True, True), (False, True)] and end["done"] == 4 and end2["done"] == 4
          and order == ["go_home", "sort_blocks"], (live_calls, calls, order))
    os.environ["BALL_MOCK_FAIL_AT"] = "3"

    # 21. 명령 아님(10-22 밤): 23:08 "아, 아, 마이크 테스트, 마이크 테스트" 가 LLM 시간 초과 → 자신 없는 임베딩 1위 red 로
    #     정책 2회를 돌렸다. 규칙(small_talk)·낱말 근거·LLM 실패 + 자신 없음 → none(로봇 안 움직임). 라우터·LLM 은 가짜로
    rt = graph._rt
    chatter = ["아, 아, 마이크 테스트, 마이크 테스트", "마이크 테스트", "안녕하세요 여러분", "다들 잘 들리세요?", "잠깐만요",
               "3라운드", "음 그러니까", "지금 내 말은 잘 들리고 있냐", "내 말 잘 들려?", "마이크 되나?", "지금 듣고 있어?"]
    talk = {t: rt.small_talk(t) for t in chatter}
    check("잡담·마이크 시험은 규칙으로 명령 아님(듣는지 묻는 말 = hear)", all(talk.values())
          and [t for t, k in talk.items() if k == "hear"] == ["아, 아, 마이크 테스트, 마이크 테스트", "마이크 테스트",
                                                             "지금 내 말은 잘 들리고 있냐", "내 말 잘 들려?", "마이크 되나?",
                                                             "지금 듣고 있어?"], talk)
    check("명령·질문 문장은 규칙에 안 걸림", not [t for t in STUB if rt.small_talk(t)], [t for t in STUB if rt.small_talk(t)])

    class FakeRouter:
        def __init__(self, top3):
            self.top3 = top3

        def route(self, text):
            (k1, s1), (_, s2) = self.top3[:2]
            return {"intent": k1, "score": s1, "margin": round(s1 - s2, 4), "top3": self.top3}

    class APITimeoutError(Exception):
        pass

    def llm_dead(text, hint):
        raise APITimeoutError("Request timed out.")

    real_get, real_llm = rt.get_router, rt._llm_classify
    incident = FakeRouter([("red", 0.721), ("plan", 0.7073), ("scene", 0.70)])      # 23:08 실측 top3
    rt.get_router, rt._llm_classify = (lambda: incident), llm_dead
    o1 = rt.classify("그게 말이지 이번에는")                          # 규칙에 안 걸리는 말로 LLM 실패 경로를 탄다
    rt.get_router = lambda: FakeRouter([("red", 0.95), ("handoff", 0.90), ("blue", 0.89)])
    o2 = rt.classify("분홍 블록 옮겨 줘")                                 # 임베딩 자신 있음 → LLM 안 부름 → 그대로
    rt.get_router = lambda: FakeRouter([("red", 0.79), ("blue", 0.78), ("plan", 0.70)])
    o3 = rt.classify("분홍 블록 옮겨 줘")                                 # 자신 없음 + LLM 실패 → 움직이지 않음
    rt._llm_classify = lambda text, hint: rt.IntentOut(intent="handoff")
    rt.get_router = lambda: FakeRouter([("plan", 0.84), ("handoff", 0.83), ("red", 0.80)])
    o4 = rt.classify("자 그럼 시작해 볼게요")                             # LLM 이 움직이는 의도 + 낱말 근거 없음
    rt._llm_classify = lambda text, hint: rt.IntentOut(intent="none")
    rt.get_router = lambda: FakeRouter([("scene", 0.75), ("report", 0.74), ("question", 0.70)])
    o5 = rt.classify("음성 합성은 어디서 돌아?")                          # LLM none 이어도 묻는 말이면 question
    rt.get_router, rt._llm_classify = real_get, real_llm
    check("LLM 시간 초과 + 자신 없는 red → none(unsure)·llm_down", (o1["intent"], o1.get("none_kind"), o1.get("llm_down"),
                                                               o1.get("none_from")) == ("none", "unsure", True, "red"), o1)
    check("임베딩 자신 있으면 LLM 없이 그대로 red", o2["intent"] == "red" and o2["method"] == "embedding", o2)
    check("색 낱말 있어도 자신 없음 + LLM 실패면 안 움직임", o3["intent"] == "none" and o3["none_kind"] == "unsure", o3)
    check("LLM 이 고른 동작도 낱말 근거 없으면 none(vague)", (o4["intent"], o4.get("none_kind")) == ("none", "vague"), o4)
    check("LLM none 이어도 의문사 있으면 question", o5["intent"] == "question", o5)

    moves = ("run_policy", "sort_blocks", "go_home", "arrange_digit", "detect_objects")
    graph.classify = lambda t: {"intent": "none", "count": None, "method": "embedding-fallback", "unsure": True,
                                "llm_down": True, "none_kind": "unsure", "none_from": "red", "error": "llm: APITimeoutError"}
    bench()
    end, evs = run("아, 아, 마이크 테스트, 마이크 테스트", base + "-none")
    errs = [e["data"]["message"] for e in evs if e["type"] == "error"]
    tools = [e["data"].get("name") for e in evs if e["type"] == "tool" and e["data"].get("name") in moves]
    check("LLM 끊김: 로봇·카메라 0·'잘 못 알아들었어요'·화면에 LLM 응답 없음", not tools and says(evs) == [graph.UNSURE_MSG]
          and end["intent"] == "none" and end["status"] == "ignored" and errs and errs[0].startswith("LLM 서버 응답 없음"),
          (tools, says(evs), errs, end))
    graph.classify = rt.classify             # 규칙 경로는 서버 없이 돈다
    e1, v1 = run("지금 내 말은 잘 들리고 있냐", base + "-none")
    e2, v2 = run("안녕하세요 여러분", base + "-none")
    t2 = [e["data"].get("name") for ev in (v1, v2) for e in ev if e["type"] == "tool" and e["data"].get("name") in moves]
    check("잘 들리냐 → '네, 잘 들려요.'·인사 → 말 없이 넘어감·로봇 0", says(v1) == ["네, 잘 들려요."] and says(v2) == []
          and e1["intent"] == e2["intent"] == "none" and not t2, (says(v1), says(v2), t2))
    if a.offline:
        graph.classify = lambda t: {"intent": STUB[t][0], "count": STUB[t][1], "method": "stub", "latency_ms": 0}

    # 22. 위치 단서(10-23): 09:50 실물 "초록블록 왼쪽에 있는거 좌표로 옮겨줘" 가 초록 2개를 다 옮기려 했다 → which 로 1개만.
    #     로봇 함수가 which 를 안 받으면·ACT 면·여러 단계면 거부(로봇·카메라 0). 방식 단서 '모드·방식·방법·버전'(10-23 총괄)
    def sorts(evs, mode="dry_run"):
        return [e["data"] for e in evs if e["type"] == "tool" and e["data"].get("name") == "sort_blocks"
                and e["data"].get("mode") == mode]

    def us_of(cls):             # 투입 구역 그 클래스 블록의 가짜 사진 u(가로) — 작은 것 = 왼쪽
        return sorted(o["px"][0] for o in mocks.workbench.detect_objects()["objects"]
                      if o["zone"] == "input" and o["cls"] == cls)

    which = {t: graph.parse_which(t) for t in ("초록블록 왼쪽에 있는거 옮겨줘", "오른쪽 분홍 하나 옮겨 줘", "초록 블록 가운데 거 옮겨 줘",
                                               "왼쪽 말고 오른쪽 거 옮겨 줘", "왼쪽이랑 오른쪽 거 옮겨 줘", "분홍 블록 좌표로 옮겨 줘",
                                               "분홍 다 옮기고 중간에 결과 알려 줘", "우측 초록 하나")}
    check("위치 단서 읽기: 왼쪽·오른쪽·가운데·'말고'는 뺌·둘이면 mixed·'좌표'·'중간에'는 아님", which == {
        "초록블록 왼쪽에 있는거 옮겨줘": "left", "오른쪽 분홍 하나 옮겨 줘": "right", "초록 블록 가운데 거 옮겨 줘": "middle",
        "왼쪽 말고 오른쪽 거 옮겨 줘": "right", "왼쪽이랑 오른쪽 거 옮겨 줘": "mixed", "분홍 블록 좌표로 옮겨 줘": None,
        "분홍 다 옮기고 중간에 결과 알려 줘": None, "우측 초록 하나": "right"}, which)
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    graph.set_method("coord")
    info = graph.backends.info()
    bench()
    us = us_of("blue_ok")
    end, evs = run("초록블록 왼쪽에 있는거 좌표로 옮겨줘", base + "-w1")
    d = sorts(evs)
    check("'초록블록 왼쪽에 있는거 좌표로' → which=left·1개·첫 말·가장 왼쪽(u 최소)만", end["which"] == "left"
          and says(evs)[:1] == ["왼쪽 초록 블록 1개를 좌표로 옮길게요."] and d and d[0]["which"] == "left"
          and d[0]["result"]["plan"][0]["from_px"][0] == us[0] and len(d[0]["result"]["plan"]) == 1
          and end["done"] == 1 and mocks.workbench.zones["input"]["blue_ok"] == 3, (says(evs)[:2], end.get("which"), end["done"], us))
    bench()
    us = us_of("red_ok")
    end, evs = run("오른쪽 분홍 하나 옮겨 줘", base + "-w1")
    d = sorts(evs)
    check("'오른쪽 분홍 하나' → which=right·첫 말·가장 오른쪽(u 최대)", says(evs)[:1] == ["오른쪽 분홍 블록 1개를 좌표로 옮길게요."]
          and d and d[0]["which"] == "right" and d[0]["result"]["plan"][0]["from_px"][0] == us[-1] and end["done"] == 1,
          (says(evs)[:1], d and d[0].get("which"), us))
    bench()
    us = us_of("blue_ok")
    end, evs = run("초록 블록 가운데 거 옮겨 줘", base + "-w1")
    d = sorts(evs)
    check("'초록 블록 가운데 거' → which=middle·u 순서 가운데", says(evs)[:1] == ["가운데 초록 블록 1개를 좌표로 옮길게요."]
          and d and d[0]["result"]["plan"][0]["from_px"][0] in us[1:3] and end["done"] == 1, (says(evs)[:1], us))
    moved_tools = ("run_policy", "sort_blocks", "go_home", "detect_objects")

    def touched(evs):
        return [e["data"].get("name") for e in evs if e["type"] == "tool" and e["data"].get("name") in moved_tools
                and e["data"].get("status") != "unavailable"]

    info["which"] = False                     # 로봇 함수에 which 인자가 아직 없을 때
    bench()
    end, evs = run("초록블록 왼쪽에 있는거 좌표로 옮겨줘", base + "-w2")
    info["which"] = True
    check("로봇 함수가 which 를 안 받으면 '아직 위치로 고르기는 못 해요'·로봇·카메라 0", says(evs) == [graph.WHICH_MSG]
          and not touched(evs) and end["status"] == "refused" and mocks.workbench.zones["input"]["blue_ok"] == 4,
          (says(evs), touched(evs), end["status"]))
    end, evs = run("왼쪽 분홍 블록 학습한 동작으로 옮겨 줘", base + "-w2")
    check("ACT 단서 + 위치 단서 → 좌표 집기 which=left·첫 말에 알림·정책 0", end["robot_method"] == "coord"
          and end["done"] == 1 and (sorts(evs) or [{}])[0].get("which") == "left" and mocks.workbench.attempts == 0
          and says(evs)[:1] == ["학습한 동작은 손 전달에만 써요. 왼쪽 분홍 블록 1개를 좌표로 옮길게요."], says(evs)[:1])
    end, evs = run("왼쪽이랑 오른쪽 초록 옮겨 줘", base + "-w2")
    check("위치 둘(mixed) → 하나만 말해 달라·로봇 0", end["status"] == "refused" and not touched(evs)
          and says(evs)[0].startswith("위치는 한 번에 하나만"), says(evs))
    end, evs = run("왼쪽 분홍 옮기고 불량 골라내 줘", base + "-w2")
    check("여러 단계 + 위치 단서 → 거부·로봇·카메라 0", end["status"] == "refused" and not touched(evs)
          and says(evs)[0].startswith("위치로 고르기는 한 가지 명령에서만"), (says(evs), touched(evs)))
    end, evs = run("초록 블록 옮겨 줘", base + "-w2")
    check("위치 단서 없으면 그대로(초록 4개·which 없음)", end["which"] is None and end["done"] == 4
          and says(evs)[:1] == ["초록 블록 4개를 좌표로 옮길게요."] and (sorts(evs) or [{}])[0].get("which") is None,
          (says(evs)[:1], end["done"]))
    cue = {t: graph.method_cue(t) for t in ("분홍블록 좌표모드로 옮겨줘봐", "분홍블록을 좌표모드로 옮겨줘", "분홍블록 학습모드로 옮겨줘",
                                          "ACT 방식으로 분홍 옮겨 줘", "YOLO 버전으로 초록 옮겨 줘", "좌표 방법으로 해서 옮겨 줘")}
    check("방식 단서 + '모드·방식·방법·버전'(10-23): 좌표모드로 → coord·학습모드로·ACT 방식으로 → act", {
        t: c[0] for t, c in cue.items()} == {"분홍블록 좌표모드로 옮겨줘봐": "coord", "분홍블록을 좌표모드로 옮겨줘": "coord",
                                           "분홍블록 학습모드로 옮겨줘": "act", "ACT 방식으로 분홍 옮겨 줘": "act",
                                           "YOLO 버전으로 초록 옮겨 줘": "coord", "좌표 방법으로 해서 옮겨 줘": "coord"}
          and cue["분홍블록 좌표모드로 옮겨줘봐"][2] == "분홍블록 옮겨줘봐", cue)
    bench()
    end, evs = run("분홍블록 좌표모드로 옮겨줘봐", base + "-w3")
    bench()
    e2, v2 = run("분홍블록 학습모드로 옮겨줘", base + "-w3")
    check("'좌표모드로'(09:40 실물 문장) → 좌표 집기 / '학습모드로' → 좌표 집기 + 'ACT 는 손 전달만' 알림",
          "coord" in nodes(evs) and says(evs)[:1] == ["분홍 블록 4개를 좌표로 옮길게요."] and e2["robot_method"] == "coord"
          and "coord" in nodes(v2) and "act" not in nodes(v2)
          and says(v2)[:1] == ["학습한 동작은 손 전달에만 써요. 분홍 블록 4개를 좌표로 옮길게요."], (says(evs)[:1], says(v2)[:1]))
    os.environ["BALL_MOCK_FAIL_AT"] = "3"
    if a.offline:
        graph.classify = lambda t: {"intent": STUB[t][0], "count": STUB[t][1], "method": "stub", "latency_ms": 0}

    # 23. 장면 답 다듬기(AGENT_SCENE_POLISH, 10-23 기본 끔): 요약이 아닌 장면 질문도 LLM 을 부르지 않고 템플릿을 말한다
    calls = []
    real_chat = graph.llm.chat
    graph.llm.chat = lambda *a_, **k: calls.append(k.get("max_tokens")) or real_chat(*a_, **k)
    bench()
    end, evs = run("초록 그릇에 몇 개야?", base + "-sc")
    graph.llm.chat = real_chat
    sm = [e["data"].get("method") for e in evs if e["type"] == "say"]
    check("장면 질문(요약 아님) → 다듬기 끔이면 LLM 0회·method template", not config.SCENE_POLISH and 80 not in calls
          and sm == ["template"] and end["status"] == "done", (calls, sm))

    # 24. 홈 자세 명령(10-23): 10:12~10:14 실물 "홈자세로 고쳐줘" → stop("멈췄습니다"), "홈자세로 변경"·"그리퍼를 기본 시작
    #     자세로 되돌려줘" → question. 규칙(router.is_home)으로 LLM·임베딩 없이 home → go_home(open_gripper=True)
    def homes_of(evs):
        return [e["data"] for e in evs if e["type"] == "tool" and e["data"].get("name") == "go_home"
                and e["data"].get("status") == "end"]

    def others(evs):
        return [e["data"].get("name") for e in evs if e["type"] == "tool"
                and e["data"].get("name") in ("run_policy", "sort_blocks", "arrange_digit", "detect_objects")]

    home_texts = ["홈자세로 고쳐줘", "홈자세로 변경", "그리퍼를 기본 시작 자세로 되돌려줘", "홈으로", "홈 자세로 보내 줘",
                  "홈 자세로 돌려 줘", "홈자세로 가", "처음 자세로", "기본 자세로 되돌려", "제자리로"]
    got = {t: (rt.classify(t)["intent"], rt.classify(t)["method"]) for t in home_texts}
    check("홈 문장 10개(실물 3문장 포함) → home·규칙(LLM·임베딩 안 부름)", set(got.values()) == {("home", "rule")}, got)
    not_home = {t: rt.is_home(t) for t in ("멈춰", "정지", "홈으로 가지 말고 멈춰", "홈 자세가 뭐야?", "홈 자세는 어떻게 맞춰?",
                                           "홈으로 갔어?", "홈으로 가지 마", "분홍 블록 옮기고 홈으로", "블록을 제자리로 돌려놔")}
    stops = [rt.classify(t)["intent"] for t in ("멈춰", "정지", "홈으로 가지 말고 멈춰")]
    check("'멈춰'·'정지'·'홈으로 가지 말고 멈춰' → stop 그대로·묻는 말·하지 말라는 말·블록 섞인 말은 home 아님",
          not any(not_home.values()) and stops == ["stop"] * 3, (not_home, stops))
    graph.classify = rt.classify             # 홈 문장은 규칙 경로라 서버 없이 돈다
    os.environ["BALL_MOCK_HOME"], os.environ["BALL_MOCK_HOME_S"] = "", "0.05"
    bench()
    mocks.workbench.at_home, mocks.workbench.gripper_open = False, False    # 멈춤으로 끊겨 홈 밖·집게 닫힘
    end, evs = run("홈자세로 고쳐줘", base + "-home")
    h = homes_of(evs)
    check("홈 밖 → '팔을 홈 자세로 돌릴게요.'·go_home(open_gripper) 1회·'홈 자세로 돌렸어요.'·다른 도구 0",
          says(evs) == [graph.HOME_CMD_MSG, "홈 자세로 돌렸어요."] and len(h) == 1 and h[0]["open_gripper"]
          and "max_deg" not in h[0] and end["intent"] == "home" and end["status"] == "done" and not others(evs)
          and mocks.workbench.at_home and mocks.workbench.gripper_open, (says(evs), h, others(evs)))
    end, evs = run("홈으로", base + "-home")
    check("이미 홈 → '이미 홈이에요.'", says(evs) == [graph.HOME_CMD_MSG, "이미 홈이에요."] and end["status"] == "done",
          says(evs))
    os.environ["BALL_MOCK_HOME"] = "too_far"                              # 가짜 팔이 홈에서 90° 떨어짐
    mocks.workbench.at_home = False
    end, evs = run("그리퍼를 기본 시작 자세로 되돌려줘", base + "-home")
    h = homes_of(evs)
    check("홈에서 너무 멂 → '천천히 돌릴게요' 뒤 max_deg 150 으로 한 번 더 → 돌림", says(evs) == [
        graph.HOME_CMD_MSG, graph.HOME_SLOW_MSG, "홈 자세로 돌렸어요."] and [x["result"]["reason"] for x in h] == [
        "too_far", "done"] and h[1].get("max_deg") == 150 and end["status"] == "done", (says(evs), h))
    os.environ["BALL_MOCK_HOME_FAR_DEG"] = "200"                         # 150° 로도 모자람
    mocks.workbench.at_home = False
    end, evs = run("홈자세로 변경", base + "-home")
    check("150° 로도 멀면 '팔을 홈 근처로 옮겨 주세요'·error·두 번만", says(evs)[-1] == graph.HOME_FAR_MSG
          and len(homes_of(evs)) == 2 and end["status"] == "error", says(evs))
    os.environ["BALL_MOCK_HOME"] = ""
    del os.environ["BALL_MOCK_HOME_FAR_DEG"]
    be = graph.backends.get()
    real_home, be["home"] = be["home"], None                             # BALL_ROBOT=dry·go_home 못 불러옴
    end, evs = run("홈으로", base + "-home")
    be["home"] = real_home
    check("홈 이동을 못 불러오면 거부·go_home 0", not homes_of(evs) and end["status"] == "error"
          and says(evs)[0].startswith("지금은 팔을 홈으로 돌릴 수 없어요"), says(evs))
    os.environ["BALL_MOCK_HOME_S"] = "1.5"
    mocks.workbench.at_home = False
    box = {}
    th = threading.Thread(target=lambda: box.update(r=run("홈으로", base + "-home2")))
    th.start()
    time.sleep(0.5)
    _, sv = run("멈춰", base + "-home2")
    th.join(15)
    os.environ["BALL_MOCK_HOME_S"] = "0.05"
    check("홈으로 가는 중 '멈춰' → stopped·끝 말 없음(멈췄습니다는 멈춤 쪽이 함)", box["r"][0]["status"] == "stopped"
          and says(box["r"][1]) == [graph.HOME_CMD_MSG] and says(sv) == ["멈췄습니다."],
          (box["r"][0]["status"], says(box["r"][1]), says(sv)))
    if a.offline:
        graph.classify = lambda t: {"intent": STUB[t][0], "count": STUB[t][1], "method": "stub", "latency_ms": 0}

    # 25. 그릇에서 꺼내기(10-23 intent unsort, 리허설 초기화): 규칙(router.is_unsort)으로 LLM·임베딩 없이 → unsort_blocks
    #     (가짜 = mocks.Workbench.unsort_blocks). 작업 기록(runlog)·대화 기록(history)에는 안 남는다(결과 보고에 안 섞임)
    unsort_texts = {"그릇에 있는 거 다시 투입 구역으로 옮겨 줘": None, "블록 다시 꺼내 줘": None, "원래대로 돌려놔": None,
                    "초기화해 줘": None, "다 꺼내 놔": None, "초록 그릇 거 꺼내 줘": ["blue"], "다꺼내놔": None,
                    "분홍 그릇에 있는 블록 꺼내 줘": ["red"], "불량 그릇 거 다시 꺼내 줘": ["defect"], "처음 상태로 돌려 줘": None,
                    "리셋해 줘": None, "그릇 다 비워 줘": None, "하나만 다시 꺼내 줘": None}
    got = {t: (r["intent"], r["method"], r.get("colors")) for t in unsort_texts for r in [rt.classify(t)]}
    check("꺼내기 문장 13개(총괄 6문장 포함) → unsort·규칙(LLM·임베딩 안 부름)·색을 말하면 그 그릇",
          got == {t: ("unsort", "rule", c) for t, c in unsort_texts.items()}, got)
    not_unsort = {t: rt.is_unsort(t) for t in (
        "불량 꺼내 줘", "분홍 블록 꺼내서 분홍 그릇에 넣어 줘", "다 꺼내고 분홍 옮겨 줘", "분홍 옮기고 다 꺼내 줘",
        "그릇에서 꺼내려면 어떻게 해?", "다 꺼냈어?", "다 꺼내지 마", "팔 원래대로 돌려놔", "속도 원래대로 해 줘",
        "기록 초기화해 줘", "로봇 초기화해 줘", "떨어진 블록 투입 구역으로 옮겨 줘", "투입 구역에 있는 분홍 블록 옮겨 줘",
        "홈으로", "제자리로", "블록을 제자리로 돌려놔")}
    homes = {t: rt.is_home(t) for t in unsort_texts}
    check("꺼내기 아님: 그릇으로 넣는 말·여러 단계·묻는 말·하지 말라는 말·팔·설정·기록·홈 말 16개 / 꺼내기 문장은 홈 아님·"
          "'하나만' → 1개·'꺼내는 거 멈춰' → stop", not any(not_unsort.values()) and not any(homes.values())
          and rt.classify("하나만 다시 꺼내 줘")["count"] == 1 and rt.classify("꺼내는 거 멈춰")["intent"] == "stop",
          ({t: v for t, v in not_unsort.items() if v}, {t: v for t, v in homes.items() if v}))

    def unsorts(evs, mode=None):
        return [e["data"] for e in evs if e["type"] == "tool" and e["data"].get("name") == "unsort_blocks"
                and e["data"].get("status") == "end" and mode in (None, e["data"].get("mode"))]

    def fill_bowls(**kw):          # 가짜 그릇에 블록을 더 넣어 둔다: fill_bowls(bowl_red={"red_ok": 2}, …)
        for zone, cnt in kw.items():
            for k, n in cnt.items():
                mocks.workbench.zones[zone][k] += n

    def in_bowls():
        return sum(mocks.workbench.zones[b][k] for b in ("bowl_red", "bowl_blue", "bowl_defect") for k in mocks.KEYS)

    graph.classify = rt.classify             # 꺼내기 문장은 규칙 경로라 서버 없이 돈다
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    bench()
    mocks.workbench.reset(dict.fromkeys(mocks.KEYS, 0))
    fill_bowls(bowl_red={"red_ok": 2}, bowl_blue={"blue_ok": 1}, bowl_defect={"red_defect": 1})
    z, tid = mocks.workbench.zones, base + "-unsort"
    end, evs = run("다 꺼내 놔", tid)
    d = unsorts(evs)
    hist = graph.get_graph().get_state({"configurable": {"thread_id": tid}}).values.get("history") or []
    check("'다 꺼내 놔' → '꺼낼게요'·dry-run → 실행·'4개를 꺼냈어요.'·그릇이 비고 투입 구역에 4개·집게 열고 홈",
          says(evs) == [graph.UNSORT_MSG, "4개를 꺼냈어요."] and [x["mode"] for x in d] == ["dry_run", "real"]
          and d[0]["colors"] is None and end["intent"] == "unsort" and end["status"] == "done" and end["done"] == 4
          and in_bowls() == 0 and z["input"] == {"red_ok": 2, "blue_ok": 1, "red_defect": 1, "blue_defect": 0}
          and mocks.workbench.gripper_open and mocks.workbench.at_home, (says(evs), [x.get("mode") for x in d], z))
    check("꺼내기는 작업 기록(runlog)·대화 기록(history)에 안 남는다", not graph.runlog.read(thread_id=tid) and not hist,
          (len(graph.runlog.read(thread_id=tid)), hist))
    fill_bowls(bowl_red={"red_ok": 1}, bowl_blue={"blue_ok": 2})
    end, evs = run("초록 그릇 거 꺼내 줘", tid)
    d = unsorts(evs)
    check("'초록 그릇 거 꺼내 줘' → colors ['blue']·초록 그릇만·'2개를 꺼냈어요.'", d and d[0]["colors"] == ["blue"]
          and says(evs)[-1:] == ["2개를 꺼냈어요."] and z["bowl_blue"]["blue_ok"] == 0 and z["bowl_red"]["red_ok"] == 1,
          (says(evs), d and d[0].get("colors")))
    end, evs = run("하나만 다시 꺼내 줘", tid)
    check("'하나만 다시 꺼내 줘' → max_blocks 1·'1개를 꺼냈어요.'", (unsorts(evs) or [{}])[0].get("max_blocks") == 1
          and says(evs)[-1:] == ["1개를 꺼냈어요."] and in_bowls() == 0, says(evs))
    end, evs = run("초기화해 줘", tid)
    check("그릇이 비었으면 '그릇에 꺼낼 블록이 없어요.'·실행 0", says(evs) == [graph.UNSORT_MSG, graph.UNSORT_EMPTY_MSG]
          and not unsorts(evs, "real") and end["status"] == "nothing", says(evs))
    be = graph.backends.get()
    real_unsort, be["unsort"] = be["unsort"], None                       # robot/coord_pick unsort_blocks 가 아직 없을 때
    fill_bowls(bowl_red={"red_ok": 1})
    end, evs = run("다 꺼내 놔", tid)
    be["unsort"] = real_unsort
    check("unsort_blocks 가 없으면 '아직 그릇에서 꺼내기는 못 해요'만·로봇·카메라 0", says(evs) == [graph.UNSORT_NO_MSG]
          and not touched(evs) and not unsorts(evs) and end["status"] == "refused" and in_bowls() == 1,
          (says(evs), touched(evs)))
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.05"                      # 블록 3개 × 40초 × 0.05 ≈ 6초
    fill_bowls(bowl_blue={"blue_ok": 2})
    box = {}
    th = threading.Thread(target=lambda: box.update(r=run("다 꺼내 놔", tid + "2")))
    th.start()
    time.sleep(1.0)
    _, sv = run("멈춰", tid + "2")
    th.join(15)
    os.environ["BALL_MOCK_ARRANGE_SCALE"] = "0.005"
    check("꺼내는 중 '멈춰' → stopped·끝 말 없음·그릇에 블록이 남음", box["r"][0]["status"] == "stopped"
          and says(box["r"][1]) == [graph.UNSORT_MSG] and says(sv) == ["멈췄습니다."] and in_bowls() > 0,
          (box["r"][0]["status"], says(box["r"][1]), says(sv), in_bowls()))
    if a.offline:
        graph.classify = lambda t: {"intent": STUB[t][0], "count": STUB[t][1], "method": "stub", "latency_ms": 0}

    # 26. 불량 색(10-23): 11:35 실물 "초록 불량 옮겨 줘" → defect → 분홍 불량까지 2개를 옮겼다. 색이 있으면 그 색 불량만
    #     sort_blocks(["blue_defect"]). 로봇이 그 이름을 모르면·ACT 면·여러 단계면 거부(로봇·카메라 0)
    dc = {t: graph.defect_color(t) for t in ("초록 불량 옮겨 줘", "분홍 불량 하나 골라내 줘", "핑크색 불량", "연두 불량 빼 줘",
                                             "불량 골라내 줘", "분홍이랑 초록 불량 옮겨 줘", "분홍 불량 말고 초록 불량 옮겨 줘",
                                             "초록 말고 불량 옮겨 줘", "불량을 분홍 그릇 옆 불량 그릇에 넣어 줘")}
    check("불량 색 읽기: 분홍·핑크 → red_defect·초록·연두 → blue_defect·둘 다·없음 → None·'말고' 뺌·뺀 색만 → mixed·그릇 이름은 아님",
          dc == {"초록 불량 옮겨 줘": "blue_defect", "분홍 불량 하나 골라내 줘": "red_defect", "핑크색 불량": "red_defect",
                 "연두 불량 빼 줘": "blue_defect", "불량 골라내 줘": None, "분홍이랑 초록 불량 옮겨 줘": None,
                 "분홍 불량 말고 초록 불량 옮겨 줘": "blue_defect", "초록 말고 불량 옮겨 줘": "mixed",
                 "불량을 분홍 그릇 옆 불량 그릇에 넣어 줘": None}, dc)
    graph.set_method("coord")
    os.environ["BALL_MOCK_FAIL_AT"] = ""

    def defects():                 # 투입 구역 분홍 불량 2·초록 불량 1 만
        bench()
        mocks.workbench.reset({"red_ok": 0, "blue_ok": 0, "red_defect": 2, "blue_defect": 1})
        return mocks.workbench.zones

    z = defects()
    end, evs = run("초록 불량 옮겨 줘", base + "-dc")
    d = sorts(evs)
    check("'초록 불량 옮겨 줘' → '초록 불량 1개를 좌표로 골라낼게요.'·sort_blocks(['blue_defect'])·분홍 불량 2개는 그대로",
          says(evs)[:1] == ["초록 불량 1개를 좌표로 골라낼게요."] and d and d[0]["result"]["colors"] == ["blue_defect"]
          and end["defect_cls"] == "blue_defect" and end["done"] == 1 and z["input"]["red_defect"] == 2
          and z["input"]["blue_defect"] == 0 and z["bowl_defect"]["blue_defect"] == 1
          and says(evs)[-1:] == ["초록 불량 1개를 분리했습니다."], (says(evs), end.get("defect_cls"), z["input"]))
    end, evs = run("초록 불량 옮겨 줘", base + "-dc")
    check("초록 불량이 없으면 '투입 구역에 초록 불량이 없습니다.'", says(evs) == ["투입 구역에 초록 불량이 없습니다."], says(evs))
    end, evs = run("분홍 불량 하나 골라내 줘", base + "-dc")
    check("'분홍 불량 하나' → '분홍 불량 1개를 좌표로 골라낼게요.'·1개만", says(evs)[:1] == ["분홍 불량 1개를 좌표로 골라낼게요."]
          and end["done"] == 1 and z["input"]["red_defect"] == 1, (says(evs), z["input"]))
    z = defects()
    us = us_of("red_defect")
    end, evs = run("왼쪽 분홍 불량 옮겨 줘", base + "-dc")
    d = sorts(evs)
    check("'왼쪽 분홍 불량' → which=left·1개·'왼쪽 분홍 불량 1개를 좌표로 골라낼게요.'·u 최소", says(evs)[:1] == [
        "왼쪽 분홍 불량 1개를 좌표로 골라낼게요."] and d and d[0]["which"] == "left" and d[0]["result"]["colors"] == [
        "red_defect"] and d[0]["result"]["plan"][0]["from_px"][0] == us[0] and end["done"] == 1, (says(evs)[:1], us))
    z = defects()
    info["defect_color"] = False             # 로봇 SORT_CLASSES 에 red_defect·blue_defect 가 아직 없을 때
    end, evs = run("초록 불량 옮겨 줘", base + "-dc2")
    info["defect_color"] = True
    check("로봇이 불량 색 이름을 모르면 '아직 불량을 색으로 고르지는 못 해요'·로봇·카메라 0", says(evs) == [graph.DCOLOR_MSG]
          and not touched(evs) and end["status"] == "refused" and z["input"]["blue_defect"] == 1, (says(evs), touched(evs)))
    end, evs = run("초록 불량 학습한 동작으로 골라내 줘", base + "-dc2")
    check("ACT 단서 + 불량 색 → 좌표 집기 blue_defect·첫 말에 알림·정책 0", end["robot_method"] == "coord"
          and end["done"] == 1 and sorts(evs)[0]["result"]["colors"] == ["blue_defect"] and mocks.workbench.attempts == 0
          and says(evs)[:1] == ["학습한 동작은 손 전달에만 써요. 초록 불량 1개를 좌표로 골라낼게요."], says(evs)[:1])
    z = defects()
    end, evs = run("초록 말고 불량 옮겨 줘", base + "-dc2")
    check("뺀 색만(mixed) → 하나만 말해 달라·로봇 0", end["status"] == "refused" and not touched(evs)
          and says(evs)[:1] and says(evs)[0].startswith("불량 색은 하나만"), says(evs))
    end, evs = run("초록 불량 골라내고 분홍 옮겨 줘", base + "-dc2")
    check("여러 단계 + 불량 색 → 거부·로봇·카메라 0", end["status"] == "refused" and not touched(evs)
          and says(evs)[:1] and says(evs)[0].startswith("불량을 색으로 고르기는 한 가지 명령에서만"), (says(evs), touched(evs)))
    end, evs = run("분홍이랑 초록 불량 옮겨 줘", base + "-dc2")
    check("두 색 다 → 불량 전부(색 없이, 지금처럼 3개)", end.get("defect_cls") is None and end["done"] == 3
          and says(evs)[:1] == ["불량품 3개(분홍 2, 초록 1)를 좌표로 골라낼게요."] and sorts(evs)[0]["result"]["colors"] == [
              "defect"], (says(evs)[:1], end["done"]))

    # 27. 불량 낱말이 이긴다(10-23 13:05 실물): 붙여 쓴 "초록불량 옮겨줘" → 임베딩 blue 0.94 → "초록 블록 1개를 좌표로
    #     옮길게요" → 정상 초록을 옮겼다. red·blue 인데 불량·엑스·X 표시 낱말이 있으면 defect(색은 defect_cls),
    #     불량을 빼는 말은 그대로. 라우터는 실물 점수를 흉내 낸 가짜(임베딩 서버 안 부름)
    real_get, real_llm = rt.get_router, rt._llm_classify
    rt._llm_classify = llm_dead
    words = {   # 문장: (가짜 임베딩 top3, 기대 의도, 기대 불량 색)
        "초록불량 옮겨줘": ([("blue", 0.9397), ("handoff", 0.89), ("defect", 0.88)], "defect", "blue_defect"),
        "분홍불량 옮겨줘": ([("red", 0.9262), ("defect", 0.92), ("blue", 0.90)], "defect", "red_defect"),
        "초록 불량 옮겨줘": ([("defect", 0.932), ("blue", 0.925), ("red", 0.90)], "defect", "blue_defect"),
        "불량 말고 초록 옮겨 줘": ([("blue", 0.93), ("defect", 0.92), ("red", 0.90)], "blue", None),
        "초록불량품 옮겨줘": ([("blue", 0.93), ("defect", 0.92), ("red", 0.90)], "defect", "blue_defect"),
        "분홍불량품 빼줘": ([("red", 0.92), ("defect", 0.91), ("blue", 0.89)], "defect", "red_defect"),
        "핑크불량 골라내줘": ([("red", 0.92), ("defect", 0.91), ("blue", 0.89)], "defect", "red_defect"),
        "연두불량 하나 옮겨줘": ([("blue", 0.92), ("defect", 0.91), ("red", 0.89)], "defect", "blue_defect"),
        "엑스 있는 초록 블록 옮겨 줘": ([("blue", 0.92), ("defect", 0.91), ("red", 0.89)], "defect", "blue_defect"),
        "X 표시 있는 분홍 옮겨 줘": ([("red", 0.92), ("defect", 0.91), ("blue", 0.89)], "defect", "red_defect"),
        "불량 아닌 초록 옮겨 줘": ([("blue", 0.93), ("defect", 0.92), ("red", 0.90)], "blue", None),
        "불량품은 빼고 분홍 옮겨 줘": ([("red", 0.93), ("defect", 0.92), ("blue", 0.90)], "red", None),
        "엑스 없는 초록 옮겨 줘": ([("blue", 0.93), ("defect", 0.92), ("red", 0.90)], "blue", None),
        "분홍 블록 옮기고 초록불량 골라내 줘": ([("red", 0.92), ("defect", 0.91), ("plan", 0.90)], "plan", None),
        "분홍불량이랑 초록불량 옮겨 줘": ([("red", 0.92), ("defect", 0.91), ("blue", 0.90)], "defect", None),
    }
    got = {}
    for t, (top3, _, _) in words.items():
        rt.get_router = lambda r=FakeRouter(top3): r
        o = rt.classify(t, use_llm=False)
        got[t] = (o["intent"], graph.defect_color(t) if o["intent"] == "defect" else None)
    check("붙은 꼴('초록불량'·'분홍불량'·'불량품')·엑스·X 표시 → defect·색은 불량 색 / 불량 빼는 말은 색 그대로 / 두 동작 → plan",
          got == {t: (i, c) for t, (_, i, c) in words.items()}, {t: g for t, g in got.items() if g != words[t][1:]})
    o = rt.classify("연두불량 하나 옮겨줘", use_llm=False)
    check("바꾼 까닭(defect_check)·개수는 그대로", o.get("defect_check") == "불량 낱말이 있어 blue 대신 defect"
          and o["count"] == 1, o)
    graph.classify = rt.classify
    rt.get_router = lambda: FakeRouter(words["초록불량 옮겨줘"][0])
    graph.set_method("coord")
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    bench()
    mocks.workbench.reset({"red_ok": 0, "blue_ok": 2, "red_defect": 2, "blue_defect": 1})
    z = mocks.workbench.zones
    end, evs = run("초록불량 옮겨줘", base + "-dw")
    d = sorts(evs)
    check("'초록불량 옮겨줘'(임베딩 blue 0.94) → '초록 불량 1개를 좌표로 골라낼게요.'·sort_blocks(['blue_defect'])·정상 초록 2개 그대로",
          says(evs)[:1] == ["초록 불량 1개를 좌표로 골라낼게요."] and d and d[0]["result"]["colors"] == ["blue_defect"]
          and end["intent"] == "defect" and end["defect_cls"] == "blue_defect" and z["input"]["blue_ok"] == 2
          and z["input"]["blue_defect"] == 0 and z["input"]["red_defect"] == 2, (says(evs)[:1], end.get("intent"), z["input"]))
    rt.get_router = lambda: FakeRouter([("blue", 0.93), ("handoff", 0.88), ("defect", 0.87)])   # 자신 있음 → LLM 안 부름
    end, evs = run("불량 말고 초록 옮겨 줘", base + "-dw")
    check("'불량 말고 초록 옮겨 줘' → blue 그대로·정상 초록 2개만·불량 그대로", end["intent"] == "blue"
          and z["input"]["blue_ok"] == 0 and z["input"]["red_defect"] == 2, (says(evs)[:1], end.get("intent"), z["input"]))
    rt.get_router, rt._llm_classify = real_get, real_llm
    os.environ["BALL_MOCK_FAIL_AT"] = "3"
    if a.offline:
        graph.classify = lambda t: {"intent": STUB[t][0], "count": STUB[t][1], "method": "stub", "latency_ms": 0}

    # 28. 위치·관계를 LLM 이 읽는다(10-23 13:30 사용자 지적): "초록블록중에서 맨 위에 잇는거 옮겨줘" → blue·which 없음 →
    #     아무 초록이나 옮길 뻔했다. 규칙(맨 위·맨 아래·가까운·먼·◯ 옆)이 먼저, 못 읽은 위치 낱말은 의도가 임베딩으로
    #     정해졌어도 LLM target(which·next_to). 못 읽음·LLM 멈춤·로봇이 모르는 값이면 거부. 라우터·LLM 은 가짜
    real_get, real_llm, real_tgt = rt.get_router, rt._llm_classify, rt._llm_target_call
    llm_calls = []

    def llm_says(target="없음"):              # 위치 LLM(TargetOut, 판별과 따로 — 13:51 실물 LengthFinishReasonError 뒤)
        def f(text):
            llm_calls.append(text)
            return rt.TargetOut(words=text, target=target)
        return f

    def llm_dead_t(text):
        raise APITimeoutError("Request timed out.")

    rt._llm_classify = lambda text, hint: (llm_calls.append(text), rt.IntentOut(intent="blue"))[1]

    top3 = {"blue": [("blue", 0.93), ("handoff", 0.88), ("defect", 0.87)],
            "red": [("red", 0.93), ("handoff", 0.88), ("defect", 0.87)],
            "plan": [("plan", 0.86), ("blue", 0.85), ("red", 0.80)]}
    rule_cases = {   # 문장: (가짜 임베딩 1위, 기대 의도, which, next_to, by) — 규칙만으로 읽혀 LLM 을 안 부른다
        "초록블록중에서 맨 위에 잇는거 옮겨줘": ("blue", "blue", "top", None, "rule"),
        "분홍 옆에 있는 초록 옮겨줘": ("blue", "blue", None, "red", "rule"),
        "로봇이랑 제일 가까운 초록 옮겨줘": ("plan", "blue", "near", None, "rule"),
        "제일 오른쪽 아래 분홍 옮겨줘": ("red", "red", "mixed", None, "rule"),
        "불량 옆에 있는 분홍 옮겨줘": ("red", "red", None, "defect", "rule"),
        "가운데 있는 거 말고 왼쪽 초록 옮겨줘": ("blue", "blue", "left", None, "rule"),
        "로봇에서 제일 먼 분홍 옮겨 줘": ("red", "red", "far", None, "rule"),
        "초록 불량 옆에 있는 분홍 옮겨 줘": ("red", "red", None, "blue_defect", "rule"),
        "분홍 위에 있는 초록 옮겨 줘": ("blue", "blue", None, None, "ref_rel"),
    }
    rt._llm_target_call = llm_says()
    got = {}
    for t, (k, _, _, _, _) in rule_cases.items():
        rt.get_router = lambda r=FakeRouter(top3[k]): r
        o = rt.classify(t, use_llm=k != "plan")         # plan 1위는 총괄 재현(use_llm=False)과 같게
        tg = o.get("target") or {}
        got[t] = (o["intent"], tg.get("which"), tg.get("next_to"), tg.get("by"))
    want = {t: v[1:] for t, v in rule_cases.items()}
    check("위치 규칙: 맨 위·로봇이랑 제일 가까운(plan 아님)·분홍 옆·제일 먼·오른쪽 아래 = mixed·불량 옆(분홍 그대로)·'가운데 말고 왼쪽'"
          "·분홍 위에 있는 초록 = 다른 블록 기준 · LLM 안 부름", got == want and not llm_calls,
          ({t: g for t, g in got.items() if g != want[t]}, llm_calls))
    seen = []

    class SeenRouter(FakeRouter):
        def route(self, text):
            seen.append(text)
            return super().route(text)
    rt.get_router = lambda: SeenRouter(top3["blue"])
    o = rt.classify("분홍 옆에 있는 초록 옮겨줘")
    check("13:51 실물 '분홍 옆에 있는 초록 옮겨줘' → 임베딩은 기준 구절을 뺀 '초록 블록 옮겨줘' 로(LLM 안 감)",
          seen == ["초록 블록 옮겨줘"] and o["route_text"] == "초록 블록 옮겨줘" and o["method"] == "embedding" and not llm_calls
          and (o.get("target") or {}).get("next_to") == "red", (seen, o.get("method"), o.get("route_text")))
    check("'가운데 있는 거 말고 왼쪽 거'(색 없는 말) → 규칙 left", rt.rule_target("가운데 있는 거 말고 왼쪽 거")["which"] == "left",
          rt.rule_target("가운데 있는 거 말고 왼쪽 거"))
    rt.get_router = lambda: FakeRouter(top3["blue"])
    llm_calls.clear()
    outs = {}
    for name, fake, t in (("none", llm_says(), "초록 블록 앞쪽 거 옮겨 줘"),
                          ("far", llm_says("로봇에서 먼"), "초록 블록 저 끝 구석에 있는 거 옮겨 줘"),
                          ("near", llm_says("로봇에 가까운"), "초록 블록 가장자리에 있는 거 옮겨 줘"),
                          ("top", llm_says("맨 위"), "초록 블록 모서리에 있는 거 옮겨 줘"),
                          ("down", llm_dead_t, "초록 블록 뒤쪽 거 옮겨 줘")):
        rt._llm_target_call = fake
        o = rt.classify(t)
        outs[name] = (o["intent"], o["method"], (o.get("target") or {}).get("which"), (o.get("target") or {}).get("next_to"),
                      (o.get("target") or {}).get("by"))
    check("규칙이 못 읽은 위치 낱말 → 임베딩 확정이어도 위치 LLM(TargetOut): none → llm_none·far·near·top·멈춤 → llm_down",
          outs == {"none": ("blue", "embedding", None, None, "llm_none"), "far": ("blue", "embedding", "far", None, "llm"),
                   "near": ("blue", "embedding", "near", None, "llm"),
                   "top": ("blue", "embedding", "top", None, "llm"), "down": ("blue", "embedding", None, None, "llm_down")}
          and len(llm_calls) == 4 and rt.rule_target("초록 블록 두 번째 거 옮겨 줘")["by"] == "unsupported"
          and rt.rule_target("초록 블록 사이에 낀 거 옮겨 줘")["by"] == "unsupported", (outs, len(llm_calls)))
    graph.classify = rt.classify
    graph.set_method("coord")
    os.environ["BALL_MOCK_FAIL_AT"] = ""
    rt._llm_target_call = llm_says()

    def go(text, k="blue", fake=None):
        rt.get_router = lambda: FakeRouter(top3[k])
        if fake:
            rt._llm_target_call = fake
        bench()
        return run(text, base + "-tg")

    bench()                                   # go() 와 같은 장면에서 초록 블록의 가짜 사진 v(세로)
    vs = sorted(o["px"][1] for o in mocks.workbench.detect_objects()["objects"]
                if o["zone"] == "input" and o["cls"] == "blue_ok")
    end, evs = go("초록블록중에서 맨 위에 잇는거 옮겨줘")
    d = sorts(evs)
    check("'맨 위' → '맨 위 초록 블록 1개를 좌표로 옮길게요.'·sort_blocks(which='top')·v 가장 작은 것 1개",
          says(evs)[:1] == ["맨 위 초록 블록 1개를 좌표로 옮길게요."] and d and d[0]["which"] == "top"
          and d[0]["result"]["plan"][0]["from_px"][1] == vs[0] and end["done"] == 1 and end["which"] == "top",
          (says(evs)[:1], end.get("done")))
    end, evs = go("분홍 옆에 있는 초록 옮겨줘")
    d = sorts(evs)
    check("'분홍 옆에 있는 초록' → '분홍 옆 초록 블록 1개를…'·which={'next_to': 'red'}·1개",
          says(evs)[:1] == ["분홍 옆 초록 블록 1개를 좌표로 옮길게요."] and d and d[0].get("next_to") == "red"
          and d[0]["result"]["which"] == {"next_to": "red"} and end["done"] == 1 and end["next_to"] == "red",
          (says(evs)[:1], d[:1] and d[0]["result"].get("which")))
    end, evs = go("로봇이랑 제일 가까운 초록 옮겨줘", "plan")
    check("'로봇이랑 제일 가까운 초록' → blue(plan 아님)·'로봇에 가장 가까운 초록 블록 1개를…'·which near",
          end["intent"] == "blue" and says(evs)[:1] == ["로봇에 가장 가까운 초록 블록 1개를 좌표로 옮길게요."]
          and (sorts(evs) or [{}])[0].get("which") == "near" and end["done"] == 1, (end.get("intent"), says(evs)[:1]))
    info = graph.backends.info()
    real_vals = info["which_values"]
    info["which_values"] = ["left", "right", "middle"]          # 로봇이 아직 top 을 모를 때
    end, evs = go("초록블록중에서 맨 위에 잇는거 옮겨줘")
    info["which_values"] = real_vals
    e2, v2 = go("초록 블록 두 번째 거 옮겨 줘", fake=llm_says())
    e3, v3 = go("초록 블록 뒤쪽 거 옮겨 줘", fake=llm_dead_t)
    e4, v4 = go("제일 오른쪽 아래 분홍 옮겨줘", "red")
    e5, v5 = go("불량 옆에 있는 분홍 옮겨줘", "red")
    e6, v6 = go("분홍 위에 있는 초록 옮겨 줘")
    refusals = [(says(v)[:1], e["status"], bool(touched(v))) for e, v in ((end, evs), (e2, v2), (e3, v3), (e4, v4), (e5, v5),
                                                                         (e6, v6))]
    check("거부(로봇·카메라 0): 로봇이 모르는 값·LLM none·LLM 멈춤·두 위치·색 없는 불량 옆·다른 블록 기준 위", refusals == [
        ([graph.POS_MSG], "refused", False), ([graph.POS_MSG], "refused", False), ([graph.POS_DOWN_MSG], "refused", False),
        (["위치는 한 번에 하나만 말해 주세요. 예를 들어 '왼쪽 초록 블록 옮겨 줘'처럼요."], "refused", False),
        ([graph.POS_DEFECT_MSG], "refused", False), ([graph.POS_MSG], "refused", False)]
          and e5["intent"] == "red", (refusals, e5.get("intent")))
    end, evs = go("초록 블록 저 끝 구석에 있는 거 옮겨 줘", fake=llm_says("로봇에서 먼"))
    check("LLM 이 읽은 위치(far) → '로봇에서 가장 먼 초록 블록 1개를…'·which far", says(evs)[:1] == [
        "로봇에서 가장 먼 초록 블록 1개를 좌표로 옮길게요."] and (sorts(evs) or [{}])[0].get("which") == "far"
          and end["target_by"] == "llm", (says(evs)[:1], end.get("target_by")))
    end, evs = go("맨 위 초록 옮기고 불량 골라내 줘", "plan")
    check("여러 단계 + 위치 → 거부·로봇 0", end["status"] == "refused" and not touched(evs)
          and says(evs)[:1] == ["위치로 고르기는 한 가지 명령에서만 해요. 한 가지씩 말해 주세요."], says(evs)[:1])
    rt.get_router, rt._llm_classify, rt._llm_target_call = real_get, real_llm, real_tgt
    os.environ["BALL_MOCK_FAIL_AT"] = "3"
    if a.offline:
        graph.classify = lambda t: {"intent": STUB[t][0], "count": STUB[t][1], "method": "stub", "latency_ms": 0}

    print(f"\n{'통과' if not fails else '실패 ' + str(len(fails)) + '건: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
