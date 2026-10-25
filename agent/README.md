# agent/ — 말 → 의도 → LangGraph → 카메라·로봇 → 보고

담당: 에이전트 세션. 다른 세션은 읽기만 한다(고칠 곳은 총괄에 [막힘] 으로). 인터페이스 정본은 `../CONTRACT.md` 3절.

> **색 변경(10-21 밤샘)**: 빨강·파랑 → **분홍·초록**(파스텔, 흰 무광 식탁보). red_* = 분홍, blue_* = 초록(계약서 머리말). 사람이 보고 듣는 말만 분홍·초록이고, 동의어 핑크(=분홍)·녹색·연두(=초록)를 받는다. 새 의도 **scene**(지금 작업대 질문)·**plan**(배운 정책 4개를 엮는 명령)도 이때 넣었다. 바꾸기 전 코드는 `runs/agent_backup_20261021_block/`.
>
> **물체 변경(10-21 밤)**: 부직포볼 → 스펀지 블록(약 3cm, 빨강·파랑, 불량 = 6면 전부 검정 X). 사람이 보고 듣는 말(라우터 예문·의도 설명·LLM 프롬프트·say 문장·보고서)만 "볼" → "블록" 으로 바꿨고 코드 식별자(`pick_red`·`handoff`·`red_ok` …)는 그대로다. 받아쓰기 표기 "블럭" 은 라우터 예문에 섞고, 기준서 검색에서는 "블럭"·버릇으로 남은 "볼·공"(명사)을 "블록" 으로 넓힌다. 바꾸기 전 버전은 `runs/agent_backup_20261021_kv1/`.

## 실행 (프로젝트 루트에서, 프로젝트 `.venv`)

| 하는 일 | 명령 |
|---|---|
| 가짜 작업대로 시연 1→5 | `BALL_MOCK=1 .venv/bin/python -m agent.cli --scenario` |
| 한 문장 / 여러 문장(가짜 작업대 상태 유지) | `BALL_MOCK=1 .venv/bin/python -m agent.cli "불량품 골라내 줘" "빨간 블록 옮겨 줘"` |
| 대화형 | `BALL_MOCK=1 .venv/bin/python -m agent.cli` |
| 이벤트 원문(JSON 한 줄씩) | `... -m agent.cli --json "결과 알려 줘"` |
| 의도 정답률·지연 | `.venv/bin/python -m agent.eval docs/intents.csv` (`--no-llm` 라우터만, `--sweep` 임계값 조합표) |
| 그래프 회귀 시험(가짜 작업대) | `.venv/bin/python -m agent.selftest` (`--offline` = 의도 판별을 정답표로, OVMS 없어도 됨) |
| 기준서 검색 9문항(세 문서 색인에서, 정답 = 문서·조항) | `.venv/bin/python -m agent.eval --rag` (`--sweep` 리랭크 설정 조합표) |
| 기준서 질문 끝까지 단계별 지연 | `.venv/bin/python -m agent.eval --qa` |
| 발표용 응답 시간표(가짜 작업대 1→5 + 질문 + 멈춰) | `.venv/bin/python -m agent.eval --timeline` (`--repeat 3`, `--policy-s 0.3`) |
| scene 8문항·plan 10문항 끝까지(가짜 작업대) | `.venv/bin/python -m agent.eval --sp` |
| 세 문서 질문 47문항(`docs/rag_questions.csv`): 검색·인용 적중·문서 밖 거절·지연·작업 기록 질문 → report | `.venv/bin/python -m agent.eval --docs` |

UI 에서는 이것만 쓴다:

```python
from agent import run_command, warmup
warmup()   # 선택: 시작할 때 한 번(라우터 예문·Chroma·보고 근거 캐시, 약 1초). 안 불러도 첫 명령 때 뒤에서 돈다
for ev in run_command("불량품 골라내 줘", thread_id="demo"):
    ...  # ev = {"type", "node", "data", "ts"}
```

- 같은 `thread_id` 는 체크포인터(`runs/agent_checkpoints.sqlite`)에 명령 요약이 쌓이고, "결과 알려 줘"는 **오늘 `log.jsonl` 중 같은 `thread_id` 의 시도만** 집계한다. 새 시연은 새 `thread_id` 로 시작한다(예: `demo-1030`).
- "멈춰"는 다른 스레드에서 `run_command("멈춰")` 를 부르면 된다. 그래프·LLM 을 거치지 않고 `robot.stop()` 을 바로 부르고, 돌고 있던 명령은 `stopped` 로 끝난다.
- 같은 `thread_id` 로 명령이 이미 돌고 있으면 새 명령(멈춰 제외)은 `error` 이벤트 하나로 거절한다.

## 이벤트 (계약서 3절 형식)

| type | node | data |
|---|---|---|
| `node` | `start` | `text`, `thread_id`, `backend`(vision/robot 이 real·mock·dry_run 인지와 이유), `llm`(모델 이름), `llm_fallback`(지정 모델이 서버 목록에 없어 대신 쓴 경우 그 사유, 아니면 null) |
| `node` | `classify` `plan` `hand_wait` `act` `verify` `finish` `report` `answer` `halt` `scene` `planner` `next_step` `plan_report` `coord` | 노드 진행. `next_step`: `step` `of` `result` / `act`: `policy` `attempt` `done` `goal` / `verify`·`coord`(끝): `ok` `moved` `done` `remaining` `status` `fail_reason` / `coord`(시작): `intent` `goal` |
| `tool` | 부른 노드 | `name`(`intent_router`·`detect_objects`·`run_policy`·`search_manual`·`make_plan`(planner: `result` LLM 계획·`steps` 검증한 단계·`notes`)·`sort_blocks`(coord: `mode` `dry_run`/`real`, `color`, `gates`)·`stt_guard`(unheard: `result.why`)), `status`(`start`·`end`·`unavailable` = BALL_COORD=1 인데 sort_blocks 를 못 불러와 ACT 로, `error` 에 이유), `args`, `result`, `latency_ms` |
| `say` | 말한 노드 | `text` — TTS 로 읽을 문장. `answer` 는 두 번: 첫마디 `{text: "자료를 찾아볼게요.", ack: true}`(question 판별 직후, 자기 질문엔 없음) → 답 `sources`(`doc`·`cite`·조항·리랭크 점수)·`finish`(LLM 종료 이유, `length` 면 잘림), `scene` 은 `facts`(카메라 JSON)·`method`(llm/template)·`image_path`, `planner`·`plan_report` 는 `steps`(`results`)도 |
| `report` | `report` | `text`(화면용 자세한 보고 문장), `spoken`(말로 한 짧은 보고 = `say` 와 같음), `method`(`llm`·`template…`), `html`(보고서 HTML 전체, 사진은 base64), `html_path`, `facts`(집계), `image_path`, `sources` |
| `error` | 난 곳 | `message` |
| `node` | `end` | `intent` `count` `method` `status` `answer` `attempts` `done` `elapsed_s` — 항상 마지막 1개. 받아쓰기 환청이면 `intent: "unheard"`, `status: "ignored"` |

`status`: `done` 끝까지 함 · `refused` 배운 적 없는 동작·숫자 배치 불가라 거절(plan·arrange) · `waiting_confirm` 숫자 배치 "할까요?" 대답 기다림 · `plan_only` 보정 전이라 계획만 · `cancelled` 숫자 배치 취소 · `nothing` 대상 없음 · `gave_up` 같은 블록을 두 번 실패 · `limit` 시도 한도 · `stopped` 멈춤 · `no_hand` 손 대기 시간 초과 · `error` 카메라·정책 실행 불가(좌표 집기: 계획 거부·송신 문 닫힘·덜 옮김·중간 끊김) · `ignored` 받아쓰기 환청을 거름.

## 그래프

```
"멈춰"(규칙) ──────────────────────────────────── robot.stop() 즉시
받아쓰기 환청(hearing.check) ──────────────────── "잘 못 들었어요. 다시 말씀해 주세요." (그래프 안 감)
classify ─┬─ defect/red/blue ─ plan ─ coord ─ finish   (늘 좌표 집기 sort_blocks, 10-23 13:10 사용자 결정)
          ├─ handoff ─ plan ─ hand_wait ─ act ─ verify ─┬─ hand_wait (count 가 2 이상이면)   (ACT 는 손 전달만)
          │                                              └─ finish
          ├─ plan ───── planner ─ [단계마다] plan ─ … ─ finish ─ next_step ─┬─ plan (다음 단계)
          │             (계획 LLM + 코드 검증, 할 수 없으면 거절)            └─ plan_report (계획 대비 결과)
          ├─ report ─── report   (log.jsonl 집계 + detect 대조 + search_manual + LLM 문장 → HTML)
          ├─ question ─ answer   (세 문서 search_manual → LLM, "[문서 이름 조항]" 인용)
          ├─ scene ──── scene    (detect JSON → 예시 답(템플릿)을 말함. LLM 다듬기는 AGENT_SCENE_POLISH=1 일 때만, 로봇 안 움직임)
          ├─ stop ───── halt     (LLM 이 stop 으로 판별한 말)
          ├─ home ───── home     ("홈으로"·"처음 자세로" 규칙 → go_home(open_gripper), 너무 멀면 150° 로 한 번 더, 10-23)
          ├─ unsort ─── unsort   ("다 꺼내 놔"·"초기화해 줘" 규칙 → unsort_blocks, 작업 기록·결과 보고에 안 넣음, 10-23)
          └─ none ───── none     (명령 아님: 마이크 시험 → "네, 잘 들려요." / 잡담 → 말 없음 /
                                  LLM 실패 + 자신 없음·낱말 근거 없음 → "잘 못 알아들었어요…", 로봇 안 움직임)
```

- **판별 뒤 코드 검사**(router.classify, 4B 가 실제로 틀린 것마다 하나씩, `plan_check`·`color_fix`·`report_check`·`scene_check` 로 결과에 남김):
  - 두 동작 표지(…하고·다음·나머지·두고·빼고·그대로·먼저·이랑·"다 정리")가 없는 plan 은 한 동작으로 되돌린다(문장 색에 맞는 의도로).
  - 할 수 없는 요청(`router.cant_do`: 모양·쌓기·줄 세우기·섞기, 다른 색 그릇에 넣기, 초록을 손에 주기)인 동작 의도는 plan 으로 보내 거절한다.
  - red·blue 가 문장 색 낱말과 반대면 바로잡는다.
  - 규칙을 묻는 말(~하면·~해야·언제 해·뭐가 들어가)이고 과거형이 없는 report 는 question 으로 본다.
  - "그릇마다·그릇별" 은 report 로 본다(기준서 7.2).
  - 문장에서 뺀 동작으로 판별되면 plan 으로 보낸다.
  - 작업 기록 질문("오늘 몇 번 실패했어?"·"방금 몇 개 옮겼어?": 오늘·방금·아까·지금까지 + 몇 번·실패·성공·옮겼 + 과거형)은 question 이어도 report 로 보낸다 → 문서(RAG)가 아니라 `runs/` 작업 기록으로 답한다(`report_check: "작업 기록 질문"`, 10-22).
  - 질문꼴 지름길(10-22 r10): 문서 질문꼴(`router.looks_question`: 의문사·"?"·묻는 어미가 있고, 부탁 "~줄래?"·멈춤·자기 질문·작업 기록 질문·작업대 낱말 + 상태 낱말이 아님)이고 임베딩 1·2위에 question 이 있으면 LLM 없이 question(`question_form: true`). 첫마디가 판별 뒤에 나가서 LLM 판별 0.85초가 곧 첫마디 지연이었다. 174문장(의도 99·세 문서 47·기준서 13·받아쓰기 변형 15)에서 잘못 question 이 된 문장 0개, 세 문서 질문의 판별 중앙 0.86 → 0.08초.
  - 묻는 말 검사(`ask_check`, 10-22 세 문서 질문 47문항에서 8개가 틀림): 의문사(얼마나·몇까지·왜·어떻게·뭐·언제·어디·무슨·어느) + "?" 인데 동작·plan 이면 question("숫자 7 만드는 데 얼마나 걸려?"). LLM·임베딩이 고른 stop 은 의문사가 있거나 멈춤 낱말이 없으면 question("인터넷 없이 돌아가?"). scene 은 작업대 낱말(블록·그릇·구역·칸·색·불량·지금·보여…)과 상태 낱말(있·남·보여·어때·몇 개야…)이 둘 다 있어야 하고, "~할 때·시연·하면·해야"(지금 없이)면 question. report 는 기록 낱말(결과·보고·옮겼·지금까지·성공·실패…) 없는 "?" 면 question. `docs/intents.csv` 동작 문장의 "?" 7개("빼 줄래?")는 의문사가 없어 그대로다.
- **plan**: 계획은 먼저 규칙으로 세운다(`graph.rule_plan`). 구절(…고 / , / 다음에)마다 대상 하나(불량·분홍·초록, 분홍+손 = 손 전달)와 개수를 읽고, 뺀 동작 구절은 지우고, 대상 없는 "나머지·다 정리"는 채우기 표지로 둔다.
  - 규칙이 못 읽으면 LLM 구조화 출력 `{feasible, steps:[{intent, count}], reason}` 을 쓴다. 못 읽는 경우: 모르는 구절, "불량 빼고"(빼내라는 건지 빼놓으라는 건지), "~씩", 여러 대상에 개수.
  - LLM 을 부를 때는 먼저 "네, 계획을 세울게요." 라고 말한다(첫 말 1초 안). 계획 프롬프트는 동작 이름과 예 두 개로 줄였다(`max_tokens` 120).
  - 규칙이든 LLM 이든 그 뒤 코드 검증(`graph.check_plan`·`resolve_plan`)은 같다.
  - 할 수 없으면 거절한다(LLM 전에 `cant_do`).
  - 뺀 동작은 지우고, 대상 없는 "나머지·다 정리"는 불량→분홍→초록으로 채운다. "나머지 분홍"은 분홍 전부 단계를 넣는다. 대상을 말했으면 말하지 않은 대상은 뺀다.
  - 개수는 감지 개수 이하로 맞춘다. 손 전달할 분홍이 그릇에 없으면 먼저 옮기는 단계를 넣는다.
  - 실행 전에 계획을 말하고, 단계마다 plan·act·verify·finish 를 돈다. 포기·멈춤·오류면 남은 단계는 하지 않는다. 끝에 계획 대비 결과를 말한다.
- **scene**: detect → `scene_facts`(구역마다 한 줄 + "투입 구역 왼쪽 위: 정상 분홍 1개" 같은 위치 줄) → 질문 낱말로 고른 예시 답(`scene_template`)을 그대로 말한다(`method: "template"`). `AGENT_SCENE_POLISH=1` 이면 LLM 이 한 문장으로 다듬고 `scene_answer_ok` 로 검사한다(10-23 기본 끔: 09:35 GPU HANG 이 다듬기 중에 났다, 총괄).
  - 요약 질문("지금 배치 어때?" — 구역·종류·위치·손을 묻지 않음, `graph._is_summary`)은 LLM 없이 짧은 코드 문장(`graph.scene_summary`, `method: "summary"`)을 바로 말한다(10-22 r11: 발표 실연 ② 가 빈 그릇까지 읽어 판단 2.4초 + 말 10.8초). 중요한 것 먼저: 손 보임 → 구역 밖 → 그릇에 섞인 블록 → 투입 구역 → 그릇이고, 빈 구역은 말하지 않는다. 예: "투입 구역에 분홍 2개, 초록 1개가 있고, 불량 그릇에 1개가 있어요."(40자, UI 말 속도 약 6자/초로 약 7초). 자세한 숫자는 `say` 의 `facts`. 같은 숫자·칸 검사를 통과하지 못하면 예전 전체 답으로 돌아간다.
  - 검사: 말한 구역의 숫자·칸만 쓸 수 있고, 예시 답 숫자를 빼면 안 되고, 숫자를 말로 쓰거나 어미가 깨지면 탈락이다.
  - 탈락하면 예시 답을 그대로 말한다.
- **의도 판별**: ① "멈춰·정지·그만·스톱·중단" 이 있고 질문 표지(?·뭐·왜·어디·어떻게·버튼)가 없으면 규칙으로 `stop` ② 임베딩 라우터 — `intent_examples.py` 예문과 코사인 유사도, 최고값 ≥ `ROUTER_THRESHOLD` 이고 2등 의도와 차 ≥ `ROUTER_MARGIN` 이면 확정. 1위가 `question` 이면 더 낮은 `ROUTER_Q_THRESHOLD`·`ROUTER_Q_MARGIN` 으로 확정(질문은 로봇을 안 움직여 잘못 골라도 위험이 없고, LLM 판별 약 0.8초를 아낀다). 1위가 red·blue 이고 문장에 그 색 낱말(빨간·빨강·레드 / 파란·파랑·블루)만 있으면 다른 색 의도는 빼고 나머지와의 차로 확정한다(볼 → 블록 뒤 "빨간 블록 옮겨 줘" 의 red·blue 차가 0.023 으로 줄었음, 결과에 `color_rule: true`) ③ 아니면 LLM `with_structured_output(IntentOut)` 으로 `{intent, count}`. 개수는 규칙(`parse_count`: "두 개만"·"3개"·"하나")이 먼저, 못 찾으면 LLM 값. `report`·`question`·`stop` 의 count 는 항상 null.
- **report 말**(10-22 r11): `say` 는 짧은 보고(`report.spoken_sentence`)를 집계 직후 바로 한다. 중요한 것 먼저: 기록·카메라 불일치(차이를 그대로, 기준서 7.4) → 구역 밖 → 사람 확인 요청 → 그릇·전달 개수 → 실패가 있으면 시도 수 → 남은 블록이고, 기록과 카메라가 같으면 말하지 않는다. 실연 ⑤ "불량 1개, 분홍 1개, 초록 1개, 손 전달 1개입니다. 모두 성공했습니다."(43자, 약 7초). 자세한 문장(`template_sentence` 또는 LLM 다듬기)은 그 뒤에 만들어 `report` 이벤트 `text`·HTML 에 넣는다(말을 기다리게 하지 않음).
- **성공 판정(기준서 5.2~5.4)**: 정책이 정상 종료(`ok=True` = `reason=duration_elapsed`)하고, 다시 찍었을 때 집는 구역이 1개 줄고 놓는 그릇이 1개 늘어야 성공. 2개를 함께 옮기면 개수는 세되 시도는 실패. 건네기는 빨강 그릇에서 1개 줄면 성공. 옮긴 개수는 정책 결과와 상관없이 카메라 기준으로 센다.
- **정책 반환 `reason`**(robot/README 1절): `home_reached`(홈 복귀 정지 감지로 일찍 끝남, 10-22 로봇 조기 종료)·`duration_elapsed`(시간 경과)는 정상 종료다. 작업 기록 `policy_reason`·`early_stop` 에 남기고, 보고서에 "정책이 끝난 방식"으로 따로 센다. 로봇이 docker 종료를 기다리지 않고 결과를 먼저 돌려주면, 그 사이 verify 의 detect·말이 먼저 돈다(카메라가 안 풀렸으면 `camera_released=False` → 1.5초 뒤, 실패하면 한 번 더). 그 밖에: `refused`(Studio 세션·장치 잠금·모델 없음, 팔은 안 움직임) → 재시도하지 않고 이유를 말하고 끝(`status=error`, 예 "Studio 화면이 로봇을 쓰고 있어 실행하지 않았습니다…"). `error`·`timeout` → 실패한 시도로 보고 다시 세어 1회 재시도. `stopped` → 멈춤. `camera_released=False` 면 `CAMERA_WAIT_S` 기다렸다 찍고, 카메라 열기에 실패하면 한 번 더 기다렸다 찍는다.
- **반복**: 개수 지정이 없으면 대상이 0개가 될 때까지, 있으면 그 개수만큼(남은 것보다 많으면 남은 만큼). 같은 블록에서 연속 실패가 `RETRY_PER_ITEM`(1)을 넘으면 멈추고 "확인해 주세요"(기준서 5.5). 시도 상한 = min(`MAX_ATTEMPTS`, 목표 × 2).
- **멈춤**: `run_command("멈춰")` 가 멈춤 플래그를 세우고 `robot.stop()` 을 부른다(돌고 있는 실행만 끊긴다). ACT 는 스스로 끝나지 않고 `timeout_s` 만큼 돌기 때문에, 실행과 실행 사이에 온 멈춰는 `act`·`verify`·`hand_wait` 가 플래그를 보고 다음 실행을 시작하지 않는다(selftest 6).
- **말이 팔을 막지 않게**(10-22, 발표 실연 3분 예산):
  - `run_command` 는 그래프를 뒤 스레드에서 돌리고 이벤트는 큐로만 넘긴다(`graph._detached`).
  - LangGraph `stream` 은 받는 쪽이 당겨 갈 때만 진행한다. 그래서 받는 쪽이 말(TTS)을 끝까지 하고 다음 이벤트를 가져가면 그동안 다음 `run_policy` 가 출발하지 않았다(측정: say 마다 1.5초 멈추는 소비자 → 실연 ③ 16.6초 → 고친 뒤 6.1초, 빠른 소비자와 같음).
  - 받는 쪽이 중간에 그만 읽어도 명령은 끝까지 돈다(멈추려면 "멈춰").
  - plan 단계 사이에는 앞 단계 verify(또는 planner)가 3초 안에 찍은 결과를 다시 쓴다(`FRESH_OBS_S`). 실제 카메라 한 장 0.75초(vision README) → 실연 ③ 감지 6 → 4회.
  - plan 단계 안의 말은 한 문장("분홍 블록을 옮기는 중이에요.")이다. 단계 끝 말은 문제가 있을 때만 하고, 끝 보고(`plan_report`)가 한꺼번에 말한다.
- **손 대기**: 손이 안 보이면 "손을 내밀어 주세요" 한 번 말하고 `HAND_POLL_S` 마다 다시 찍는다. `HAND_WAIT_S`(15초) 안에 안 보이면 로봇을 움직이지 않고 끝낸다.

## 좌표 집기(arrange) — 블록으로 숫자 만들기

로봇 담당 `robot/coord_pick` 의 `arrange_digit`(robot/COORD_PICK.md 9절)을 부른다. **plan 의도의 한 갈래**다. 판별은 plan 이고, "숫자·모양 + 만들기 동사"(`router.is_arrange`)면 그래프가 `arrange` 로 보낸다. 끝 이벤트의 `intent` 는 `arrange` 다.

```
classify(plan) ─ arrange ─┬─ (거절·계획만) 끝
                          └─ arrange_confirm (LangGraph interrupt "할까요?") ─┬─ 응 → arrange_run ─ 끝
                                                                           └─ 아니·다른 말·멈춰 → 끝
```

1. `arrange`
   - 말에서 숫자(0~9, "숫자 칠" 같은 한자어 숫자도)와 색(분홍 → `red`, 초록 → `blue`, 없으면 `any`)을 읽는다.
   - 바로 "네, 숫자 7 배치를 계획해 볼게요." 라고 말한다(첫 말).
   - `detect_objects` → `arrange_digit(d, color, dry_run=True, detection=det)`.
   - 계획을 말한다: "숫자 7: 블록 7개, 약 3분 0초 걸립니다. 경고 4건은 화면에 띄웠어요. 할까요?"
2. `arrange_confirm`: `interrupt()` 로 그래프가 멈춘다(끝 이벤트 `status=waiting_confirm`). 같은 `thread_id` 의 다음 `run_command` 가 답이다.
   - "응·네·해 줘·시작해 줘·그렇게 해" → 실행. 짧은 대답 전체가 맞을 때만 수락으로 본다.
   - "아니·취소·하지 마" → "알겠어요, 숫자 배치는 하지 않을게요."
   - 다른 말 → 기다림을 취소하고 그 말을 새 명령으로 한다.
   - "멈춰" → 기다림을 취소한다.
3. `arrange_run`: 같은 detection 으로 `dry_run=False`.
   - 몇 분 걸린다. 그동안 "멈춰"는 다른 스레드의 `run_command` 가 `robot.stop()` 으로 멈춘다(정책과 같은 길).
   - 끝나면 다시 찍고 "숫자 7을 다 놓았습니다(블록 7개)." 또는 "멈췄습니다. 블록 3개까지 놓았어요."

**거절**(이유를 말하고 끝, `status=refused`):
- 숫자가 아닌 모양(하트·별·동그라미), 두 자리 숫자.
- dry-run `ok=False`: 블록 부족("숫자 3은 블록 11개가 필요한데 지금 블록으로는 7개만 놓을 수 있어요"), 작업 반경 밖, 공간 위반.
- 거절 말 끝에는 "지금은 7·1처럼 블록이 적은 숫자가 현실적이에요." 를 붙인다.

**계획만**(`status=plan_only`, 확인을 묻지 않음): 가짜 작업대가 아닌데 보정 전(`robot/coord_pick/config.json` `calibrated=false`)·금지 구역 자료 없음·`BALL_ROBOT=1` 아님 중 하나면, "보정 전이라 계획만 보여 드릴게요. 숫자 7: 블록 7개, 약 3분 0초 걸립니다. 칸 배치는 화면에 띄웠어요."

**대시보드용 이벤트**(칸 배치를 그릴 때):

```json
{"type": "tool", "node": "arrange", "data": {
  "name": "arrange_digit", "status": "end", "mode": "dry_run", "digit": "7", "color": "any",
  "plan_only": false, "gates": [],
  "layout": {"rows": 5, "cols": 3, "pattern": ["###", "..#", "..#", "..#", "..#"],
             "cells": [{"order": 1, "row": 0, "col": 2, "color": "분홍", "from_mm": [79.9, 174.4], "to_mm": [295.0, 225.0]}]},
  "result": {"ok": true, "reason": "planned", "summary": "...", "blocks": 7, "time_s": 180.0,
             "warnings": [], "failures": [], "violations": [], "sim": {}, "plan": []},
  "latency_ms": 1600}}
```

- `layout.cells`: 놓는 순서대로 칸 하나씩이다. `row` 0 = 위, `col` 0 = 왼쪽(사람이 숫자를 읽는 방향). `pattern` 의 `#` = 블록 자리다.
- `color` 는 그 칸에 놓을 블록 색이다. `from_mm`·`to_mm` 은 작업대 mm(로봇 좌표, 보정 전에는 예시값)이다.
- `plan_only: true` 면 실행하지 않은 계획이다. `gates` 는 막힌 이유("보정 전", "금지 구역 자료(vision/zones.json) 없음", "실제 구동 꺼짐(BALL_ROBOT)")다.
- 실행 뒤에는 `node: "arrange_run"`, `mode: "real"` 인 같은 꼴의 이벤트가 하나 더 온다. `result.run.steps_done` = 놓은 블록 수다.
- `result` 는 `arrange_digit` 반환값 그대로다(COORD_PICK.md 9절 표).

**가짜 작업대(`BALL_MOCK=1`)**:
- 계획은 로봇 함수 dry-run 그대로다. 그 안의 `sim` 이 `robot/coord_pick/fake.py` 가짜 로봇으로 끝까지 돈다.
- 실행은 `mocks.Workbench.arrange_digit` 가 예상 시간 × `BALL_MOCK_ARRANGE_SCALE`(기본 0.02) 동안 기다리며 흉내 낸다. 멈춤을 받고, 놓은 블록은 투입 구역에서 빠져 사진 아래쪽 "digit" 자리에 칸대로 그려진다.
- 기록: dry-run·실행 모두 `runs/<날짜>/coord_pick/*.jsonl`(robot/ 에 쓰지 않음).

## 학습한 동작(ACT)은 손 전달만 — 10-23 13:10 사용자 결정

- 옮기기(`defect`·`red`·`blue`·plan 의 그 단계)는 늘 좌표 집기(`sort_blocks`). ACT 는 손 전달(`handoff`, ④ "분홍 블록 하나 내 손에 줘")만.
- `pick_red`·`pick_blue`·`pick_defect` 를 부르는 길을 없앴다: `after_plan` 은 옮기기를 `act` 로 보내지 않고, `act_node` 는 handoff 가 아니면 정책을 부르지 않고 오류, `tools.run_policy` 도 `handoff` 가 아니면 `reason: "refused"`·`error: "act_only_handoff"`.
- 옮기기에 ACT 단서("학습한 동작으로·ACT로·정책으로·학습모드로")가 붙으면 좌표로 하고 첫 말 앞에 "학습한 동작은 손 전달에만 써요."를 붙인다.
  - "학습한 동작은 손 전달에만 써요. 분홍 블록 1개를 좌표로 옮길게요." · "…초록 불량 1개를 좌표로 골라낼게요." · "…왼쪽 분홍 블록 1개를 좌표로 옮길게요."
  - 계획: "학습한 동작은 손 전달에만 써요. 좌표로 순서대로 할게요. 1) …" · LLM 계획이면 수락 멘트가 "학습한 동작은 손 전달에만 써요. 좌표로 할 계획을 세울게요."(`graph.ACCEPT_ACT_CUE`)
  - 손 전달에 ACT 단서는 알림 없이 그대로("분홍 블록 1개를 학습한 동작으로 건네 드릴게요."). 손 전달에 "좌표로"면 지금처럼 "손에 건네기는 학습한 동작으로만 해요."
- 좌표 집기를 못 불러왔으면(`backends.info()` `coord: "off"`) 옮기기·옮기는 단계가 있는 계획은 카메라·로봇을 거치지 않고 거부: "좌표 집기를 불러오지 못해 블록을 옮길 수 없어요."(`graph.COORD_OFF_MSG`, `status: "refused"`). ACT 로 대신하지 않는다. 손 전달은 된다.
- `set_method("act")` 는 ValueError("학습한 동작은 손 전달에만 써요. 옮기기는 늘 좌표로 해요."). `get_method()` `default` 는 늘 `coord`, `act_for: ["handoff"]`. `BALL_COORD` 는 더 이상 방식을 정하지 않는다.
- 이벤트: `method_pick` `{method: "coord", by, cue, asked(act|coord|None), fallback: false, …}` · `end.robot_method` 늘 `coord`(손 전달도 이 값과 상관없이 ACT).
- 아래 r21 절의 단서 읽기(`router.method_cue`)는 그대로 쓰고, 단서는 첫 말을 고르는 데만 쓴다.

## 집는 방식 고르기(명령마다 ACT / 좌표 집기) — 10-22 r21 (10-23 13:10 부터 옮기기는 늘 좌표, 위 절)

17:08 실물: `BALL_COORD=1` 대시보드에서 "분홍블록 하나를 act정책으로 옮겨줘" 가 좌표 집기로 갔다. 이제 `BALL_COORD` 는 시작 기본값일 뿐이고, 명령마다 고른다.

- 말 속 단서(`router.method_cue`, 조사 으로·로·써서·해서·대로가 붙은 꼴만):
  - ACT 단서: ACT·액트·에이씨티(정책)·학습한/배운 동작·정책 → `act`
  - 좌표 단서: 좌표·좌표 계산·계산·YOLO·욜로·역기구학 → `coord`
  - 단서 뒤에 모드·방식·방법·버전이 붙어도 된다("좌표모드로"·"ACT 방식으로"·"학습모드로", 10-23 총괄. 09:40 실물 "분홍블록 좌표모드로 옮겨줘봐" 가 단서를 놓쳐 기본값 ACT 로 갔다).
  - 여럿이면 뒤의 것, "~로 말고·대신·빼고·하지 말고" 면 반대쪽.
  - "ACT 정책은 몇 개야?" 처럼 조사가 다른 질문은 단서가 아니다.
- 단서가 없으면 기본값(`graph.METHOD`)을 쓴다. 시작값은 `BALL_COORD=1` → coord, 없으면 act. 대시보드 토글은 `agent.set_method("act"|"coord")`(다음 명령부터), 표시는 `agent.get_method()` → `{default, coord_available, coord_backend(real·mock·off), robot, home, which, labels}`(`which` = 위치로 고르기를 할 수 있나, 아래 위치 절). 좌표 집기를 못 불러왔으면 `set_method("coord")` 는 ValueError.
- 판별(`classify`)·계획(`planner`)에는 단서를 뺀 말(`state.cmd`)을 쓴다. 질문꼴 말은 원문 그대로다.
  - 단서가 붙은 7문장이 임베딩 확정을 못 해 LLM 으로 가 약 1초 늦었다. 뺀 뒤에는 14문장 중 12문장이 임베딩(약 0.06초)이다(`runs/20261022_mock/eval_r21.cue_route.log`).
  - 남은 2개("분홍 블록 하나 옮겨 줘" 꼴)는 단서가 없어도 원래 LLM 으로 간다.
- 첫 말에 방식(`graph.HOW`):
  - "분홍 블록 1개를 학습한 동작으로 옮길게요." / "분홍 블록 1개를 좌표로 옮길게요."
  - "불량품 1개(분홍 1)를 좌표로 골라낼게요."
  - plan: "좌표로 순서대로 할게요. 1) …", LLM 계획이면 "네, 좌표로 할 계획을 세울게요."
  - 좌표 집기의 dry-run 뒤 말은 "블록 N개, 약 M분 S초 걸려요." 다(방식은 첫 말에서 함).
  - handoff 는 늘 ACT. "좌표로" 라고 하면 "손에 건네기는 학습한 동작으로만 해요." 를 앞에 붙인다.
  - (10-23 13:10 전) coord 를 원했는데 sort_blocks 를 못 불러오면 ACT 로 했다 → 이제 거부(위 절).
- 이벤트:
  - `start.robot_method_default`
  - `tool` `classify` `method_pick` `{method, by: cue|default, cue, default, fallback, cmd}`
  - `end.robot_method`
  - 작업 기록 finish `by`
- 실제 송신은 coord 도 `BALL_ROBOT=1`·보정·금지 구역 문이 다 열려야 한다(아래 coord 절).

## 좌표 집기 그릇 옮기기(coord) — 10-22 r16

실물 시험에서 ACT `pick_red` 가 "분홍 1 + 초록 1" 남은 장면을 3/3 놓쳤다(색보다 위치를 배움). 방식이 coord(위 절)면 intent `defect`·`red`·`blue` 와 plan 의 그 단계를 ACT 대신 로봇 담당 `robot.coord_pick.executor.sort_blocks`(COORD_PICK.md 9절)로 한다. `handoff` 는 그대로 ACT.

```
plan(detect·개수, "분홍 블록 2개를 좌표로 옮길게요.") ─ coord ─ finish
  coord: 홈 맞춤(r20) → sort_blocks(색, dry_run=True) → "블록 2개, 약 1분 21초 걸려요."
         → sort_blocks(색, dry_run=False)   (숫자 배치와 달리 "할까요?" 를 묻지 않음 — 시연 흐름)
         → detect 다시 → 투입 구역에서 준 개수·그릇에 는 개수로 옮긴 개수(카메라 기준) → finish 뒤 집게 열고 홈(r22)
```

- **detection 을 넘기지 않는다(실제 로봇 함수, 10-22 17:50 실물 첫 성공 뒤).**
  - `sort_blocks` 가 위 카메라 원본을 직접 찍어(`detect_live`, 원본은 `robot/logs/<날짜>/front_*.jpg`) 블록 긴 축 방향(yaw)까지 잰다. 집게는 긴 축에 수직(짧은 변을 끼움)이 된다.
  - 비전 `detect_objects()` 의 `image_path` 는 구역·상자를 그린 사진이라 `front_image` 로 쓸 수 없다.
  - dry-run 도 직접 찍게 한다. 방향 없는 계획은 "집게 두 끝 높이 차가 큼"으로 거부되는 블록이 있었다(r16 실연 흐름의 초록).
  - 그래서 홈 맞춤을 dry-run 앞에 둔다(팔이 위 카메라 사진에 안 걸리게).
  - 카메라를 dry-run·실행에서 한 번씩 더 연다(각 약 1초).
  - 두 번 찍은 장면이 같다는 보장은 없어서, 실행이 말한 개수와 다를 수 있다. 옮긴 개수는 늘 마지막 detect 로 센다.
  - dry-run 결과의 `block_yaws` 는 `state.coord.block_yaws` 에 남긴다.
  - 가짜 로봇(`BALL_MOCK=1`)은 plan 의 detection 을 그대로 넘긴다(같은 가짜 장면).
  - tool 이벤트 `sort_blocks` 에 `detect: "live"|"plan"` 이 붙는다.

- 개수를 말하면(`"분홍 하나만"`) `max_blocks` = 그 개수(로봇에 가까운 것부터). 없으면 그 색 전부.
- 말: plan 의 첫 말(방식 포함) → dry-run 뒤 "블록 N개, 약 M분 S초 걸려요."(경고가 있으면 "경고 N건은 화면에 띄웠어요." 를 붙임, 로봇 `summary` 원문은 `say` 의 `summary`) → finish 의 끝 말(ACT 와 같은 문장).
- 실행은 그래프 스레드에서 몇 분 걸린다(`run_policy` 와 같음). "멈춰"는 다른 스레드의 `run_command` 가 `robot.run_policy.stop()`(가짜는 `mocks.stop`)으로 끊는다. 실행 전에 멈춤 플래그를 보면 보내지 않는다.
- 송신 문: 실제 로봇 함수일 때 `arrange_gates(r, "coord")` — 보정 전(`robot/coord_pick/config.json` `calibrated`)·금지 구역 자료 없음·`BALL_ROBOT=1` 아님 중 하나면 보내지 않고 "…이라 좌표 집기는 계획만 세웠어요(블록 4개, 약 2분 41초). 계획은 화면에 띄웠어요. 확인해 주세요." 로봇 함수 자체에도 같은 문이 있다.
- 끝(`status`): `done` 계획한 개수만큼 옮김 / `error` dry-run 거부("팔이 닿지 않는 자리에 블록이 있어 좌표 집기 계획을 세우지 못했어요."·"작업 공간 검사에 걸려 좌표 집기를 하지 않았어요.")·송신 문 닫힘·덜 옮김("좌표 집기로 4개 중 3개만 옮겨졌어요.")·중간 끊김(`tracking` → "좌표 집기가 팔이 목표를 따라가지 못해 멈췄어요.")·실행 거부(`already_running`) / `stopped` 멈춤. finish 가 "확인해 주세요." 를 붙인다.
- 작업 기록: 블록마다 한 줄(시도 = 블록 1개, ACT 와 같은 단위), `policy: coord_defect|coord_red|coord_blue`, `method: "coord"`, `coord: {run_id, blocks, time_s, log_path}`. 계획 순서대로 카메라가 센 개수만큼 성공으로 친다. 보고의 기록 대조(`report.facts.expected`)는 `coord_*` 를 같은 그릇에 더한다.
- `sort_blocks` 를 못 불러오면(로봇 모듈 import 실패) 옮기기는 거부한다(10-23 13:10, ACT 로 돌아가지 않음): `backends.info()` `coord: "off"`·이유, 시작 때 stderr(ui.log) 한 줄, 명령마다 `tool` 이벤트 `name: "sort_blocks", status: "unavailable"` + "좌표 집기를 불러오지 못해 블록을 옮길 수 없어요."
- 대시보드 그래프 칸 이름: `agent.NODE_LABELS = {"coord": "좌표 집기", "unheard": "못 알아들음"}` → UI(`ui/app.py` `_load_agent`)가 덧붙이고 칸 아래 작은 글씨로 노드 id(`coord`)를 함께 보인다.
- 가짜(`BALL_MOCK=1`): `mocks.Workbench.sort_blocks`. 계획 흉내(투입 구역의 그 색 블록, 블록당 `BALL_MOCK_COORD_BLOCK_S`=40초 — 로봇 dry-run 10-22 실측 38~48초) 또는 `BALL_MOCK_COORD=robot` 이면 로봇 함수 dry-run(가짜 사진 자리는 `mocks.robot_detection` 으로 작업대 투입 구역 픽셀로 옮김). 실행은 예상 시간 × `BALL_MOCK_ARRANGE_SCALE` 동안 기다리며 멈춤을 받는다. `BALL_MOCK_COORD_FAIL=unreachable|violation|tracking|miss` 로 실패를 흉내 낸다.

## 그릇에서 꺼내기(`unsort`, 리허설 초기화) — 10-23

리허설을 다시 하려고 그릇에 들어간 블록을 투입 구역으로 되돌린다(총괄 요청).

- 판별: 규칙 `router.is_unsort` 만 쓴다(`classify` 에서 `is_home` 다음, `method: "rule"`). 의도 목록·임베딩 예문·LLM 은 그대로라 다른 문장 판별은 안 바뀐다.
  - 잡는 꼴: 원래대로·원래 상태로·처음대로·처음 상태로 · 초기화 · 리셋 · 그릇 (다) 비워 · '꺼내'(그릇·다시·도로·다·전부·모두·투입이 같이 있을 때: "블록 다시 꺼내 줘"·"다 꺼내 놔"·"초록 그릇 거 꺼내 줘") · 투입 구역으로 + 옮겨·돌려·놓아·넣어.
  - 아닌 것: "불량 꺼내 줘"(골라내기일 수 있어 판별로) · 그릇으로 넣는 말("꺼내서 분홍 그릇에 넣어") · 여러 단계("다 꺼내고 분홍 옮겨 줘"·"분홍 옮기고 다 꺼내 줘" → 판별, 계획에는 꺼내기 단계가 없다) · 묻는 말·"~려면"·"~수 있어"·"꺼냈어?" · "꺼내지 마"·"말고" · 홈·자세·팔·그리퍼·속도·모드·설정·기록·대화·카메라·화면 말("속도 원래대로"·"기록 초기화"·"로봇 초기화") · 떨어진 블록·구역 밖.
  - 홈 규칙과 겹치지 않는다(홈은 블록·그릇이 섞이면 아님, 꺼내기는 홈·자세·팔이면 아님). '멈춰'·'정지'는 stop 이 먼저.
  - 색·불량을 말하면 그 그릇만(`route.colors`: `red`·`blue`·`defect`, sort_blocks 의 color 이름), 개수를 말하면 그만큼(`max_blocks`).
- 실행(`graph.unsort_node`): `unsort_blocks(dry_run, colors=…, max_blocks=…)`(robot/coord_pick, sort_blocks 와 같은 꼴의 dict).
  - 첫 말 "그릇에 있는 블록을 투입 구역으로 꺼낼게요." → 홈 맞춤 → dry-run(계획) → 송신 문(`arrange_gates`: 보정 전·BALL_ROBOT 등이면 계획만) → 실행(Studio 세션 정리 포함, `_robot_call`) → 집게 열고 홈(`_home_after`).
  - 끝 말: "N개를 꺼냈어요." · 그릇이 비었으면 "그릇에 꺼낼 블록이 없어요."(`status: "nothing"`) · 덜 꺼냈으면 "N개를 꺼냈어요. M개는 못 꺼냈어요." · 중간에 끊기면 "그릇에서 꺼내기가 …로 멈췄어요."
  - N = 로봇 결과로 센다(손목 확인 실행이면 `wrist[].ok` 개수, 아니면 `reason: "done"` 일 때 계획 개수). 카메라로 다시 세지 않는다.
  - 로봇 함수가 없으면(`backends.info()["unsort"]` = `off`) 로봇·카메라를 거치지 않고 "아직 그릇에서 꺼내기는 못 해요."(`status: "refused"`). '멈춰'면 그 자리(끝 말 없음).
- 작업 기록(runlog)·대화 기록(history)·결과 보고에는 넣지 않는다(시연 기록을 흐리지 않게). 그래서 같은 대화에서 꺼낸 뒤 "결과 알려 줘" 를 하면 기록과 카메라 개수가 달라진다 — 리허설을 다시 할 때는 대시보드 초기화(새 thread)로 시작한다.
- 이벤트: `node unsort`(colors·count) · `tool unsort_blocks`(dry_run → start → real, `colors`·`max_blocks`) · `end.intent: "unsort"`·`end.done` = 꺼낸 개수.
- 가짜: `mocks.Workbench.unsort_blocks` 가 그릇 개수를 투입 구역으로 옮긴다(블록당 `BALL_MOCK_COORD_BLOCK_S` × `BALL_MOCK_ARRANGE_SCALE`, 멈춤 받음).

## 불량 색 고르기(`defect_cls`) — 10-23

11:35 실물: "초록 불량 옮겨 줘" → defect → 분홍 불량까지 2개를 다 옮겼다.

- 판별 — 불량 낱말이 이긴다(13:05 실물: 받아쓰기가 붙여 쓴 "초록불량 옮겨줘" → 임베딩 blue 0.94 → 정상 초록을 옮겼다): `classify` 가 red·blue 를 골랐는데 불량을 시키는 말(`router.defect_word`)이면 defect 로 바꾼다(`defect_check`). 색 낱말은 아래 읽기로 `defect_cls` 가 된다. 임베딩·LLM·color_fix 뒤, 할 수 없는 요청·묻는 말·명령 아님 검사 앞.
  - 불량을 시키는 말: '불량'(붙은 꼴 "초록불량"·"분홍불량품" 포함) · '엑스' · 'X 표시'·'X 있는'·'X 그려진'. '불량 그릇'은 그릇 이름이라 아님.
  - 빼는 말이 붙으면 아님: "불량 말고 초록 옮겨 줘"·"불량 아닌"·"불량품은 빼고"·"엑스 없는"·"불량은 그대로 두고" → 색 그대로. 불량 낱말이 둘이면 하나라도 시키는 쪽이면 defect("분홍 불량 말고 초록 불량").
  - 두 동작 표지가 있고 불량에 안 붙은 색이 따로 있으면("분홍 블록 옮기고 초록불량 골라내 줘") 한쪽만 하지 않게 plan(색 붙은 불량이라 계획 단계가 거부한다).
- 읽기(`router.defect_color`, 판별과 따로): defect 의도에 분홍·핑크 → `red_defect`, 초록·녹색·연두 → `blue_defect`. "분홍 그릇"처럼 그릇 이름의 색은 뺀다. "분홍 불량 말고 초록 불량" → `blue_defect`. 두 색 다("분홍이랑 초록 불량")거나 색이 없으면 None = 불량 전부(지금과 같음). 뺀 색만("초록 말고 불량")이면 `mixed` → "불량 색은 하나만 말해 주세요. …"
- 좌표 집기: `sort_blocks(["blue_defect"], …)`(색 없으면 지금처럼 `"defect"`). 개수(`max_blocks`)·위치(`which`)와 같이 쓴다.
- 첫 말 "초록 불량 1개를 좌표로 골라낼게요." · "왼쪽 분홍 불량 1개를 좌표로 골라낼게요." · 끝 말 "초록 불량 1개를 분리했습니다." · 없으면 "투입 구역에 초록 불량이 없습니다." 세기(투입 구역·불량 그릇)도 그 색만(`graph._task`).
- 할 수 없으면 카메라·로봇을 거치지 않고 거부(`status: "refused"`). 색 없이 부르면 다른 색 불량까지 옮기기 때문이다.
  - 로봇이 이름을 모름(`backends.info()["defect_color"]` = False, 로봇 `SORT_CLASSES` 에 `red_defect`·`blue_defect` 가 있나): "아직 불량을 색으로 고르지는 못 해요."
  - (10-23 13:10 부터 없음) 방식이 ACT 면 거부했다 → 이제 ACT 단서가 있어도 좌표로 하고 첫 말에 "학습한 동작은 손 전달에만 써요."(위 ACT 절). 좌표 집기를 못 불러오면 "좌표 집기를 불러오지 못해 블록을 옮길 수 없어요."
  - 여러 단계(plan)에 색 붙은 불량("초록 불량 골라내고 분홍 옮겨 줘"): "불량을 색으로 고르기는 한 가지 명령에서만 해요. 한 가지씩 말해 주세요."
- 이벤트: `intent_router` 결과 `defect_cls` · `node plan`·`node coord` `defect_cls` · `tool sort_blocks` `defect_cls`(거부면 `status: "unavailable"`) · `end.defect_cls` · 작업 기록 `defect_cls` · `get_method()["defect_color"]`.
- 가짜: `BALL_MOCK_DEFECT_COLOR=0` 이면 로봇이 이름을 모르는 것처럼 거부. `BALL_MOCK_COORD=robot` 이면 로봇 `SORT_CLASSES` 를 본다. 로봇 쪽에 이름이 생기면 대시보드 재시작으로 반영된다(시작 때 한 번 본다).

## 홈 자세 명령(`home`) — 10-23

10:12~10:14 실물: "홈자세로 고쳐줘" → stop("멈췄습니다"), "홈자세로 변경"·"그리퍼를 기본 시작 자세로 되돌려줘" → question(설명서 답). 사용자는 팔을 홈으로 보내려 했다.

- 판별: 규칙 `router.is_home` 만 쓴다(`classify` 에서 `is_stop` 바로 다음, `method: "rule"`). 의도 목록·임베딩 예문·LLM 은 그대로라 다른 문장 판별은 안 바뀐다. GPU 가 멈춰도 LLM 5초를 기다리지 않는다.
  - 잡는 꼴: 홈(자세·위치·자리)으로/로 · 홈 복귀·이동 · 홈 자세 잡아 · 처음·기본·시작·원래·초기 (시작) 자세·위치·자리로 · 제자리로 · 원위치.
  - 아닌 것: '멈춰'·'정지'가 있으면 stop(먼저 봄) · 의문사("홈 자세가 뭐야?"·"어떻게 맞춰?") · "~려면" · "홈으로 갔어?" · "가지 마"·"말고" · 블록·색·불량·그릇이 섞인 말("분홍 블록 옮기고 홈으로"·"블록을 제자리로 돌려놔"는 판별로).
- 실행(`graph.home_node`): "팔을 홈 자세로 돌릴게요." → `go_home(dry_run=False, open_gripper=True)`(홈 맞춤과 같은 길, Studio 세션 정리 포함).
  - 홈에서 너무 멀면(`too_far`, 로봇 기본 `safety.home_max_deg` 60°): "홈에서 너무 멀어 천천히 돌릴게요." 뒤 `max_deg=HOME_CMD_MAX_DEG`(150°)로 한 번 더. 사람이 옆에서 말로 시키는 명령이라 허용(총괄). 그래도 멀면 "팔이 홈 자세에서 너무 멀어요. 팔을 홈 근처로 옮겨 주세요."
  - 끝 말: "홈 자세로 돌렸어요."(옮김) · "이미 홈이에요."(`already_home`). 그 밖의 거부는 홈 맞춤과 같은 문구.
  - '멈춰'면 그 자리(끝 말 없음, "멈췄습니다."는 멈춤 쪽이 함). `go_home` 을 못 불러오면(`BALL_ROBOT=dry` 등) "지금은 팔을 홈으로 돌릴 수 없어요(…)".
  - 성공하면 홈 맞춤 상태(`_HOME`)를 "방금 홈" 으로 둬서 다음 정책 전 홈 맞춤을 건너뛴다(r22 와 같음).
- 작업 기록(history)·결과 보고에는 넣지 않는다(선별 작업이 아님). 이벤트: `node home`, `tool go_home`(두 번째는 `max_deg: 150`), `end.intent: "home"`.
- 가짜: `BALL_MOCK_HOME=too_far` 는 팔이 홈에서 `BALL_MOCK_HOME_FAR_DEG`(90°) 떨어진 것 — `max_deg` 가 그보다 작으면 거부.

## 위치·관계를 LLM 도 읽는다(`target`: which·next_to) — 10-23 13:30

사용자 지적: "초록 블록 중에서 맨 위에 있는 거 옮겨 줘"는 LLM 이 판단해야 한다. 재현(use_llm=False): "초록블록중에서 맨 위에 잇는거 옮겨줘" → blue·which 없음 → 아무 초록이나 옮긴다(위험), "분홍 옆에 있는 초록 옮겨줘" → which 없음, "로봇이랑 제일 가까운 초록 옮겨줘" → plan.

- 규칙(`router.rule_target`, 빠른 길): 아래 절의 왼쪽·오른쪽·가운데에 더해
  - 맨/제일/가장 위·위쪽·위에 → `top`(위 카메라 사진 위쪽 = 로봇에서 먼 쪽) · 맨 아래·아래쪽·밑에 → `bottom`(로봇 쪽)
  - (로봇이랑) 제일 가까운·가까이 → `near` · 로봇에서 (제일) 먼·멀리 있는 → `far`("먼저"는 아님)
  - "분홍 옆에 있는"·"분홍이랑 가까운"·"초록 불량 근처" → `next_to`(red·blue·red_defect·blue_defect). 로봇에는 `which={"next_to": …}` 로 넘긴다.
  - 기준 구절은 옮길 블록의 색·불량 판단에서 뺀다(`router.strip_ref`): "불량 옆에 있는 분홍"은 defect 가 아니라 red, "분홍 옆 초록"은 blue. "로봇이랑"·"분홍이랑 가까운"은 두 동작(plan)이 아니다.
  - "가운데 있는 거 말고 왼쪽 거" → left('있는 거 말고'도 뺀다). "그릇 위에"는 위치가 아니다.
- 판별(임베딩·LLM)은 기준 구절·위치 낱말을 뺀 말로 한다(`router.route_text`, 13:59 — 13:51 실물 "분홍 옆에 있는 초록 옮겨줘" 가 임베딩 plan 0.92 로 자신 없어 LLM 으로 갔고 LengthFinishReasonError). "분홍 옆에 있는 초록 옮겨줘" → "초록 블록 옮겨줘"(blue 색 규칙 확정, 0.05초). 규칙으로 위치를 읽은 옮기라는 말만(묻는 말·다른 블록 기준은 원문), `intent_router` 결과 `route_text`.
- **(13:59 바뀜) 위치 LLM 은 의도와 따로**: `IntentOut` 은 intent·count 두 필드로 되돌렸다. 넷으로 늘리자 strict 스키마(모두 필수)에서 qwen3-4b 가 intent·count 뒤 다음 키를 안 쓰고 공백만 내다 max_tokens 에 걸렸다(96·400 둘 다 LengthFinishReasonError — 판별 LLM 경로 전체가 위험). 위치는 `TargetOut{words, target}`(위치 낱말을 먼저 그대로 옮겨 쓴 뒤 한국어 이름 8개 — 없음·왼쪽·오른쪽·가운데·맨 위·맨 아래·로봇에 가까운·로봇에서 먼 — 에서 고름, `max_tokens` 48)을 따로 부른다. 값 하나만 고르게 하면 "오른쪽"으로 쏠렸고, 색으로 시작하는 값("초록 불량 옆")이 있으면 말 속 색에 끌려 그 값을 골랐다 → "◯ 옆"은 규칙만. "두 번째"·"사이"는 LLM 도 "가까운"을 골라 규칙이 LLM 없이 거부(`target.by` = `unsupported`).
- (13:30 처음) LLM(`IntentOut.which`·`IntentOut.next_to`, 모두 기본 none — 13:59 에 위처럼 바꿈): 규칙이 못 읽은 위치 낱말("두 번째"·"구석"·"끝"·"사이"·"앞쪽"·"뒤쪽"·"모서리" 등)이 남으면 **의도가 임베딩으로 정해졌어도** LLM 을 불러 target 만 받는다(의도는 그대로, `target.llm_intent` 에 LLM 의도를 남김). 판별에서 이미 LLM 을 불렀으면 그 결과를 쓴다. 위치 낱말이 없으면 LLM 의 target 은 버린다(지어낸 위치를 쓰지 않게).
  - 프롬프트: 위·맨 위 = 사진 위쪽 = 로봇에서 먼 쪽(top), 아래 = 로봇 쪽(bottom), 로봇에 가장 가까운 near·먼 far, 다른 블록 옆 next_to. 두 위치를 함께("오른쪽 아래")·다른 블록 기준 위·아래·순서("두 번째")는 none.
  - LLM `max_tokens` 64 → 96(필드가 둘 늘어남).
- 거부(카메라·로봇 0, `status: "refused"`): 위치 말을 조용히 버리지 않는다.
  - LLM 이 none(`target.by` = `llm_none`)·LLM 을 못 부름(`need_llm`)·다른 블록 기준 위·아래("분홍 위에 있는 초록", `ref_rel`)·로봇이 그 값을 모름(`backends.info()["which_values"]`·`["next_to"]`, 시작 때 `robot/coord_pick/plan.WHICH`·`NEXT_TO` 를 본다): "그 위치로는 아직 고르지 못해요."
  - 색 없는 불량 옆("불량 옆에 있는 분홍", 로봇 기준 이름이 없음): "그 위치로는 아직 고르지 못해요. 불량 옆이면 '초록 불량 옆 분홍'처럼 불량 색을 말해 주세요."
  - 두 위치("제일 오른쪽 아래 분홍", 위치 + 옆): "위치는 한 번에 하나만 말해 주세요. …"
  - LLM 이 멈춤(5초, `llm_down`): "위치를 알아듣지 못해 옮기지 않았어요. 다시 말해 주세요."
  - 여러 단계(plan)에 위치: "위치로 고르기는 한 가지 명령에서만 해요. 한 가지씩 말해 주세요."
- 첫 말: "맨 위 초록 블록 1개를 좌표로 옮길게요." · "맨 아래 …" · "로봇에 가장 가까운 초록 블록 1개를…" · "로봇에서 가장 먼 …" · "분홍 옆 초록 블록 1개를 좌표로 옮길게요." · "초록 불량 옆 분홍 블록 1개를…". 개수를 말하지 않았으면 1개.
- 이벤트: `intent_router` 결과 `target{which, next_to, by, pos[, llm_intent, llm_ms, error]}`·`which`·`next_to` · `node plan`·`node coord`·`tool sort_blocks` `next_to` · 거부면 `tool sort_blocks` `status: "unavailable"`·`target_by` · `end.next_to`·`end.target_by` · 작업 기록 `next_to` · `get_method()` `which_values`·`next_to`.
- 가짜: `mocks.Workbench.sort_blocks(which=)` 가 top·far = 작은 v 부터, bottom·near = 큰 v 부터(가짜 사진 아래쪽 = 로봇 쪽), `{"next_to": 색}` = 그 색 블록에 가장 가까운 것부터(없으면 거부). 가짜 로봇은 일곱 값·네 기준을 다 안다.

## 위치로 고르기(`which`, 왼쪽·오른쪽·가운데) — 10-23

09:50 실물: "초록블록 왼쪽에 있는거 좌표로 옮겨줘" 가 초록 2개를 다 옮기려 했다. 위치 낱말을 읽어 로봇 함수에 넘긴다.

- 읽기(`router.parse_which`, 판별과 따로):
  - 왼쪽·왼편·좌측 → `left`, 오른쪽·오른편·우측 → `right`, 가운데·한가운데·중앙·중간(+거·것·블록·에 있는) → `middle`.
  - "왼쪽 말고 오른쪽 거" → `right`. 둘 이상("왼쪽이랑 오른쪽 거")이거나 뺀 쪽만 있으면 `mixed` → "위치는 한 번에 하나만 말해 주세요. …" 로 거부.
  - "좌표로"의 '좌'·"중간에 결과 알려 줘"의 '중간에'는 위치가 아니다.
- 의도가 defect·red·blue 일 때만 본다(`classify_node`, `state.which`). 개수를 말하지 않았으면 1개. handoff 는 그릇에서 건네므로 위치를 보지 않는다.
- 첫 말: "왼쪽 초록 블록 1개를 좌표로 옮길게요." · "가운데 불량 1개를 좌표로 골라낼게요."
- 좌표 집기: `sort_blocks(색, dry_run, which=…, max_blocks=개수)`. 같은 색 후보를 위 카메라 사진 u 순서로 고르는 일은 로봇 함수가 한다(로봇 세션이 만드는 중, 10-23).
- 할 수 없으면 카메라·로봇을 거치지 않고 거부한다(`status: "refused"`, finish 가 그 말만 함). which 를 빼고 부르면 엉뚱한 블록을 집기 때문이다.
  - 로봇 함수에 `which` 인자가 없음(`backends.info()["which"]` = False, `inspect.signature` 로 봄): "아직 위치로 고르기는 못 해요."
  - (10-23 13:10 부터 없음) 방식이 ACT 면 거부했다 → 이제 ACT 단서가 있어도 좌표로 하고 첫 말에 알림(위 ACT 절). 좌표 집기를 못 불러오면 "좌표 집기를 불러오지 못해 블록을 옮길 수 없어요."
  - 여러 단계(plan): "위치로 고르기는 한 가지 명령에서만 해요. 한 가지씩 말해 주세요."(13:30 전 "위치(왼쪽·오른쪽·가운데)로 …")
- 이벤트: `intent_router` 결과 `which` · `node plan` `which` · `tool sort_blocks` `which`(거부면 `status: "unavailable"`, `error` = 거부 말) · `end.which` · 작업 기록 `which`.
- 가짜: `mocks.Workbench.sort_blocks(which=)` 가 가짜 사진 u 로 고른다(left = 작은 u 부터, right = 큰 u 부터, middle = u 순서 가운데부터). `BALL_MOCK_WHICH=0` 이면 로봇 함수에 인자가 없는 것처럼 거부한다. `BALL_MOCK_COORD=robot` 이면 로봇 함수 인자를 본다.
- 로봇 함수에 인자가 생기면 대시보드 재시작으로 반영된다(시작 때 한 번 본다).

## 홈 맞춤(정책·좌표 집기 직전, `HOME_ALIGN`) — 10-22 r20

16:53 실물 handoff: 앞 명령 pick_defect 가 '멈춤'으로 끝나 팔이 어깨 −10 근처에 남았다. 그 자세에서 시작하자 분홍 하나를 손에 잘 건넨 뒤에도 조기 종료(시작 자세로 돌아와 멈춤)가 진짜 홈 정지를 못 알아채, 남은 시간에 다음 블록을 집으러 갔다. ACT 도 홈이 아닌 데서 시작하면 학습 분포 밖이다.

- `act`(run_policy 직전, 시도마다)·`coord`(sort_blocks 실행 직전)·`arrange_run`(arrange_digit 실행 직전)에서 `robot.coord_pick.executor.go_home(dry_run=False)` 를 먼저 부른다(`graph._home_align`). `BALL_ROBOT=1` 일 때만 실제다. `BALL_ROBOT=dry`·import 실패면 하지 않는다.
- go_home 이 컨테이너에서 연결해 측정 자세를 읽는다.
  - 몸통 5관절이 홈(config `home_n`) 1.8° 안이면 `already_home`, 보내지 않는다.
  - 아니면 관절 공간 직선으로 8°/s 로 옮긴다(`done`).
  - 어느 관절이든 60° 넘게 멀면 하나도 안 보내고 `too_far` → "팔이 홈 자세에서 너무 멀어요. 팔을 홈 근처로 옮겨 주세요." 라고 말하고 그 명령을 멈춘다(정책 0회).
  - 그 밖의 거부(`studio_session` 등)는 정책 거부와 같은 말이다.
- '멈춰' 뒤 다음 명령도 같은 홈 맞춤이 먼저 돈다.
- `tool` 이벤트 `name: "go_home"`(`start`/`end`, `result.reason` = `already_home`·`done`·`too_far`…)가 남는다. 말은 하지 않는다.
- 시간(robot/logs/20261022 go_home 실측): 이미 홈 4.0~4.2초(연결·읽기), 옮기면 5.3~6.6초. 정책이 끝난 자세는 조기 종료가 걸려도 시작 자세에서 1.03~3.13(정규화, 약 1.9~5.6°)이라, `always` 면 거의 매번 옮긴다. 실연 ①~④ 정책 5회면 약 +20~30초다.
- `HOME_ALIGN`:
  - `always`(기본, 요청대로 매번)
  - `after_stop`: 이 프로세스 첫 동작·멈춤·오류·시간 초과 뒤에만. 정상으로 끝난 정책·좌표 집기 다음은 건너뛴다. 16:53 같은 경우만 막고 시간을 아낀다.
  - `off`
- 가짜: `mocks.Workbench.go_home`. 멈춤으로 끊긴 정책·좌표 집기 뒤에는 팔이 홈 밖으로 남는다. `BALL_MOCK_HOME=too_far`, `BALL_MOCK_HOME_S`(기본 0.05).

## 동작 뒤 집게 열고 홈 복귀(`HOME_AFTER`) — 10-22 r22

집게에 블록이 끼어 남거나 홈이 아닌 자세로 끝나는 일이 실물에서 여러 번 있었다.

- 명령(plan 이면 단계)이 끝나면 `finish` 가 끝 말을 낸 뒤 `go_home(dry_run=False, open_gripper=True)` 를 부른다(`graph._home_after`). 숫자 배치 `arrange_run` 끝도 같다. `BALL_ROBOT=1` 일 때만 실제다.
- 대상은 ACT 정책(불량·분홍·초록·손 전달)·좌표 집기·숫자 배치가 이번에 팔을 움직였을 때다(`_HOME.moved`). 거부로 안 움직였거나 할 일이 없었으면 하지 않는다.
- '멈춰'로 끝나면 움직이지 않고 그 자리 유지. 다음 명령 전에 홈 맞춤(r20)이 돈다.
- 실패해도 말하지 않는다(`error` 이벤트만). 다음 명령 전 홈 맞춤이 다시 본다.
- 방금 홈 복귀가 됐고 그 뒤 팔이 안 움직였으면(`HOME_FRESH_S` 120초 안) 다음 정책 전 홈 맞춤은 다시 재지 않는다(`tool go_home status=skip`). 이미 홈 측정 4초를 아낀다.
- 이벤트: `tool go_home` 에 `open_gripper: true`.
- `HOME_AFTER=open`(기본) | `off`.
- 시간(실물 go_home): 이미 홈 측정 4.0~4.2초, 옮기면 5.3~6.6초, 집게 열기만 1.7초(17:24 로봇 시험 1회). 명령·단계마다 한 번씩 붙는다.

## Studio 세션 정리(`studio.py`, `graph._robot_call`) — 10-22 r22

Studio 원격조작 탭을 닫거나 옮겨도 Studio 는 45초 idle + 10초 뒤에야 팔을 놓는다(16:56 로그). 그동안 명령이 "로봇을 다른 곳에서 쓰고 있어" 로 거부됐다. 세션 멈춤은 사용자 허락(17:17).

- 정책(`run_policy`)·좌표 집기 실행·숫자 배치 실행·홈 맞춤이 `studio_session`·`robot_busy` 로 거부되면 `GET {Studio}/api/runtime/sessions` 를 본다. 주소: `STUDIO_URL` → `robot/policies.json studio_url` → `:7860`.
  - 세션마다 `activity.is_recording` 또는 `dataset_loaded` 면 녹화(화면) → 멈추지 않는다. "Studio 에서 녹화 중이라 그대로 뒀어요. 녹화를 끝낸 뒤 다시 말해 주세요." 로 거부.
  - `activity` 가 없으면(starting·unreachable) 상태를 모른다 → 멈추지 않는다. "Studio 화면이 열려 있어요. ←로 나와 주세요."
  - 그 밖에 세션이 있으면 "Studio 연결을 정리하고 바로 할게요." → `POST …/sessions/{session_name}/stop`(204, Studio 가 SIGTERM 으로 정리).
    - 세션이 목록에서 사라질 때까지 `STUDIO_POLL_S`(1초)마다 목록을 본다.
    - 사라지면 원래 호출을 다시 한다. `robot_busy` 면 잠금이 풀릴 때까지 다시. 모두 `STUDIO_WAIT_S`(15초) 안이다.
  - 세션 없이 잠금만이면 "로봇이 풀리기를 잠깐 기다릴게요." → 15초 안에서 다시.
  - 그래도 안 되면 지금 거부 문구("로봇을 다른 곳에서 쓰고 있어 실행하지 않았습니다." 등).
- 이벤트(부른 노드): `tool studio_sessions`(`result` = `[{session, status, follower, source, recording}]`, `refused`) · `tool studio_stop`(`session`, `result` = HTTP 코드) · `tool studio_wait`(`ok`).
- `BALL_ROBOT=dry` 면 하지 않는다.
- 가짜: `mocks.Workbench.studio_list·studio_stop`. `BALL_MOCK_STUDIO=teleop|recording|unknown|lock` 이면 run_policy·go_home·좌표 집기·숫자 배치 실행이 실물처럼 사전 점검에서 거부된다.

## 명령 아님(`none`) — 10-22 밤

23:08 mock 대시보드에서 "아, 아, 마이크 테스트, 마이크 테스트"(받아쓰기 정확)가 red 로 판별돼 pick_red 를 2회 돌렸다.
- OVMS 가 멎어 LLM 이 30초 뒤 시간 초과로 끝났고, 자신 없는 임베딩 1위(red 0.721·차 0.014)를 그대로 썼다.
- 의도 목록에 '명령 아님'이 없어 LLM 이 살아 있어도 잡담을 9개 중 하나로 골랐다(23:1x "지금 내 말은 잘 들리고 있냐" → scene).

**고친 점**
- 의도 `none` 을 더했다(LLM `IntentOut` 목록·설명). 그래프 `none` 노드는 로봇·카메라를 쓰지 않고, 기록(history)에 남기지 않으며, `status: ignored` 로 끝난다.
- 판별 순서(`router.classify`):
  1. 멈춤 규칙.
  2. 규칙 `small_talk`. LLM 을 부르지 않는다. 움직일 낱말·의문사·작업 낱말(결과·보고·기준·카메라·로봇·시연 등)이 하나라도 있으면 걸지 않는다.
     - `hear`: 마이크·테스트·들려·듣고 있·목소리·소리 잘·녹음.
     - `chat`: 안녕·감사·잠깐·잠시·라운드·여러분·다들·그러니까·넘어가, 또는 군말만(음·어·자·네·오케이…). 청중에게 묻는 "다들 잘 들리세요?" 도 chat 이다.
  3. 임베딩, 자신 없으면 LLM. LLM 은 none 을 고를 수 있다. LLM 이 none 인데 의문사나 부탁("~줘")이 있으면 question 으로 보낸다(문서 질문·문서 밖 거절).
  4. 마지막 검사. 움직이는 의도(defect·red·blue·handoff·plan)인데:
     - LLM 이 실패(시간 초과·연결 실패·결과 파싱 실패)했고 임베딩이 자신 없으면 → none `unsure`.
     - 움직일 낱말이 없으면 → none `vague`. 낱말은 블록·색·불량·엑스·손·그릇·숫자·모양, 옮기·담·넣·치워·골라·분리·정리·건네·놓·만들·쌓·빼 등이다(`router._GROUND`). `docs/intents.csv` 동작 60문장과 라우터 동작 예문에는 모두 있다.
  - 임베딩이 자신 있으면(기존 기준 0.80·0.04, 색 규칙) 지금처럼 LLM 없이 그대로 간다.
- 말:
  - `hear` → "네, 잘 들려요."
  - `chat` → 말 없음.
  - `unsure`·`vague` → "잘 못 알아들었어요. 다시 말해 주세요."
- LLM 서버가 답하지 않으면(시간 초과·연결 실패, `route.llm_down`) `error` 이벤트 "LLM 서버 응답 없음(5초 안에 답 없음). 판별이 불확실해 로봇을 움직이지 않았어요." 를 낸다. 대시보드 대화창에 ⚠ 로 뜬다.
- 판별 LLM 시간 제한은 `LLM_CLASSIFY_TIMEOUT_S`(5초, 정상 0.8초대)다. 답·계획·장면·보고 LLM 은 `LLM_TIMEOUT_S`(30초) 그대로다.
- 결과 필드: `none_kind`(hear·chat·unsure·vague) · `none_check`(이유) · `none_from`(막기 전 의도) · `unsure` · `llm_down`.

**대가**
- LLM 이 죽으면 임베딩이 확정하지 못하는 명령은 움직이지 않고 다시 말해 달라고 한다. 모의(잘못된 주소)에서 `docs/intents.csv` 동작 60문장 중 28개만 임베딩 확정으로 움직였다.
- "하나 더 줘"처럼 물체 낱말이 없는 명령은 이제 움직이지 않는다(vague).

## 받아쓰기 환청 거르기(`hearing.py`) — 10-22 r16

실물 15:57: 마이크를 허용하는 순간의 1.26초 녹음(말소리 약 0.36초)이 "감사합니다." 로 받아적혀 question 으로 갔다. `run_command` 가 "멈춰" 규칙 다음, 그래프 전에 거른다. 걸리면 `say` "잘 못 들었어요. 다시 말씀해 주세요." 하나와 끝(`intent: unheard`, `status: ignored`)만 내고, 숫자 배치 "할까요?" 기다림·실행 중 명령은 건드리지 않는다.

- 글 기준(늘): 띄어쓰기·문장부호를 뺀 말 전체가 대표 환청 문구(여럿 이어 붙은 것 포함)·말 없는 소리·문장부호뿐이면 거른다. 문구: 감사합니다·고맙습니다·시청해 주셔서 감사합니다·구독과 좋아요·MBC/KBS/SBS/YTN 뉴스(…입니다)·자막 제공·다음 영상에서 만나요·안녕히 계세요·음·아·흠·에. 명령 속에 섞인 "감사합니다"는 거르지 않는다. "어"는 "할까요?" 대답이라 넣지 않았다. M1 목소리 "응"이 "음"으로 받아써지면 거절(취소) 대신 "잘 못 들었어요"가 되어 기다림이 이어진다.
- 소리 기준(UI 가 녹음 정보를 넘길 때만): `run_command(text, thread_id, stt={"speech_s": 초})` 또는 `stt={"wav_path": 경로}`(에이전트가 `agent.speech_seconds` 로 잰다). 말소리 < `STT_MIN_SPEECH_S`(0.2초) 또는 글자 수 ÷ 말소리 초 > `STT_MAX_CPS`(20) 면 거른다.
- `speech_seconds`: 30ms 틀 세기가 문턱(잡음 바닥 p10 × 2.5, 큰 소리 p95 × 0.3, 0.02 중 큰 값)을 넘는 틀 수, 첫 0.15초(버튼 소리) 뺌. 10-22 실측:

| 녹음 | 말소리 | 초당 글자 |
|---|---|---|
| 무음·잡음(rms 0.01·0.05) / 잡음 + 딸깍 | 0.00초 / 0.12초 | — |
| 합성 음성 50개(`ui/samples` F1·M1, 글을 아는 24개만 초당 글자) | 0.24("멈춰")~1.65초 | 5.6~15.6 |
| 실물 마이크 명령 8개(`/tmp/gradio` 15:57~16:11) | 0.78~2.10초 | 3.6~7.9 |
| 실물 환청 "감사합니다."(15:57) | 0.36초 | 13.9 → 글 기준으로 거름 |

## 실제 장치 연결 (`backends.py`)

| 조건 | vision | robot |
|---|---|---|
| `BALL_MOCK=1` | 가짜 | 가짜 |
| `vision.detect` import 성공 | **실제**(카메라 한 장 찍고 닫음) | `BALL_ROBOT=1` 일 때만 **실제 구동**, `BALL_ROBOT=dry` 면 `run_policy(..., dry_run=True)`, 그 밖에는 가짜 |
| `vision.detect` import 실패 | 가짜 | 가짜(가짜 개수로 진짜 로봇을 움직이지 않게) |
| 좌표 집기 `arrange` | — | robot 이 가짜면 `mocks.Workbench.arrange_digit`, 아니면 `robot.coord_pick.executor.arrange_digit`(자체 송신 문: BALL_ROBOT=1·보정·금지 구역) |
| 그릇 옮기기 `sort`(옮기기는 늘 이것, 10-23 13:10) | — | robot 이 가짜면 `mocks.Workbench.sort_blocks`, 아니면 `robot.coord_pick.executor.sort_blocks`(같은 송신 문). import 실패면 None → 옮기기 거부 |
| 홈 맞춤 `home` | — | `BALL_ROBOT=1` 이면 `robot.coord_pick.executor.go_home`, 가짜 로봇이면 `mocks.Workbench.go_home`, `BALL_ROBOT=dry`·import 실패면 None(안 함) |

⚠ `robot.run_policy` 는 부르면 바로 팔을 움직인다. 에이전트는 `BALL_ROBOT=1` 이 없으면 실제 로봇을 부르지 않는다(계약서 5절: 실제 구동은 사용자 입회 때만). 시연 때 UI 를 `BALL_ROBOT=1` 로 띄운다.
정책 1회 시간(`timeout_s`)은 **넘기지 않는다** → robot 모듈의 정책별 값(`robot/policies.json`, 10-22 15초 녹화 기준 기본 21초). ACT 는 그 시간만큼 돌기 때문에 40 같은 큰 값을 넘기면 남는 시간에 블록을 하나 더 집으러 간다(로봇 담당 발견, 계약서 3절 `timeout_s: float | None = None`). 가짜 정책도 같은 서명이고, 실제라면 돌 시간을 결과 `policy_s` 에 남긴다.

## 환경변수

| 이름 | 기본 | 뜻 |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8100/v3` | OpenAI 호환 주소(OVMS). 임시 Gemma: `http://<외부-LLM-서버>:8000/v1` |
| `LLM_MODEL` | `qwen3-4b` | 8B 가 다시 뜨면 `LLM_MODEL=qwen3-8b`. 지정한 모델이 서버 목록에 없으면 목록에 있는 `qwen3-4b`·`qwen3-8b` 중 하나로 대신 쓰고 `start` 이벤트 `llm_fallback` 에 남김(목록은 60초마다 다시 봄) |
| `LLM_THINKING` | `0` | 0 이면 `chat_template_kwargs.enable_thinking=false` 로 Qwen3 생각 끔 |
| `LLM_SO_METHOD` | `json_schema` | `with_structured_output` 방식(`function_calling`·`json_mode`) |
| `LLM_CLASSIFY_TIMEOUT_S` / `LLM_TIMEOUT_S` | `5` / `30` | 의도 판별 LLM 시간 제한(넘기면 움직이는 의도를 막음, 위 명령 아님 절) / 답·계획·장면·보고 LLM |
| `EMB_BASE_URL` | LLM 주소가 `/v3` 면 같은 곳, 아니면 로컬 OVMS | 임베딩·리랭크 주소 |
| `EMB_MODEL` / `RERANK_MODEL` | `qwen3-embedding` / `qwen3-reranker` | |
| `ROUTER_THRESHOLD` / `ROUTER_MARGIN` | `0.80` / `0.04` | 라우터 확정 조건 |
| `ROUTER_Q_THRESHOLD` / `ROUTER_Q_MARGIN` | `0.78` / `0.02` | 1위가 question 일 때의 확정 조건 |
| `RAG_RERANK_N` | `6` | 리랭크할 후보 수(+ 임베딩 코사인 앞 3개, 0 = 리랭크 안 함) |
| `RAG_SKIP_SCORE` / `RAG_SKIP_MARGIN` | `0.65` / `0.04` | 임베딩 1위 코사인 ≥ 이 값이고 2위와 차 ≥ 이 값이면 리랭크 생략 |
| `RAG_TOP_K` | `3` | 검색 결과 수 |
| `RAG_MIN_SCORE` / `RAG_OOD_COS` | `0.3` / `0.45` | 리랭크 최고점 또는 임베딩 1위 코사인이 이보다 낮으면 문서 밖 질문 |
| `ANSWER_MAX_TOKENS` | `80` | 질문 답 LLM 최대 토큰. 60 은 나열 답 2개가 잘렸다(10-22 r10) |
| `AGENT_SCENE_POLISH` | `0` | 1 이면 장면 답(요약이 아닌 질문)을 LLM 이 다듬는다. 기본은 끔: 템플릿 답을 바로 말한다(10-23, GPU 호출 줄이기) |
| `QA_PREFETCH` | `0` | 1 이면 질문꼴 말(`router.looks_question`)은 판별과 동시에 검색을 시작하고, question 이 아니면 결과를 버린다. 기본은 끔: 질문꼴 지름길로 판별이 0.07초라 겹칠 틈이 거의 없고, LLM 판별 질문은 리랭크와 GPU 를 나눠 써 첫마디 최대가 1.22 → 2.39초로 늘었다(10-22 r10, 정확도는 같음) |
| `RAG_KEEP_SCORE` / `RAG_KEEP_COS_GAP` | `0.1` / `0.05` | LLM 근거로 남길 조각(리랭크 점수 하한 / 리랭크 생략 때 1위 코사인과의 차) |
| `RETRY_PER_ITEM` / `MAX_ATTEMPTS` | `1` / `20` | |
| `HAND_WAIT_S` / `HAND_POLL_S` | `15` / `0.7` | |
| `CAMERA_WAIT_S` | `1.5` | 정책 카메라가 안 풀렸을 때 다시 찍기 전 대기 |
| `BALL_COORD` | (비움) | 10-23 13:10 부터 방식을 정하지 않는다(옮기기는 늘 좌표, ACT 는 손 전달만). 시작 스크립트 호환으로만 남음. 예전: `1` 이면 coord 시작 기본값 |
| `HOME_ALIGN` | `always` | 정책·좌표 집기 직전 홈 맞춤: `always` · `after_stop`(첫 동작·멈춤·오류 뒤에만) · `off` (위 홈 맞춤 절) |
| `HOME_CMD_MAX_DEG` | `150` | 홈 자세 명령에서 홈에서 너무 멀 때 한 번 더 보내는 각도(위 홈 자세 명령 절). 홈 맞춤·동작 뒤 홈은 로봇 기본값(60°) |
| `HOME_AFTER` / `HOME_FRESH_S` | `open` / `120` | 동작 뒤 집게 열고 홈 복귀(`off` 로 끔) / 그 뒤 이 초 안이면 다음 홈 맞춤을 건너뜀 |
| `STUDIO_URL` `STUDIO_WAIT_S` `STUDIO_POLL_S` | (policies.json) `15` `1.0` | Studio 세션 정리: 주소 / 세션이 사라지고 잠금이 풀리기를 기다리는 최대 초 / 확인 간격 |
| `STT_MIN_SPEECH_S` / `STT_MAX_CPS` | `0.2` / `20` | 받아쓰기 환청 소리 기준(녹음 정보가 올 때만, 위 환청 절) |
| `BALL_MOCK` `BALL_ROBOT` | | 위 표 |
| `BALL_MOCK_ARRANGE_SCALE` | `0.02` | 가짜 좌표 집기(숫자·그릇 옮기기·그릇에서 꺼내기) 실행 시간 = 예상 시간 × 이 값 |
| `BALL_MOCK_COORD` `BALL_MOCK_COORD_BLOCK_S` `BALL_MOCK_COORD_FAIL` | (비움) `40` (비움) | 가짜 그릇 옮기기: `robot` = 로봇 함수 dry-run 으로 계획 / 블록당 예상 초 / `unreachable`·`violation`·`tracking`·`miss` |
| `BALL_MOCK_WHICH` | (비움) | `0` 이면 가짜 로봇이 위치(which)를 못 받는 것처럼(거부 시험, 위 위치 절) |
| `BALL_MOCK_DEFECT_COLOR` | (비움) | `0` 이면 가짜 로봇이 불량 색 이름(red_defect·blue_defect)을 모르는 것처럼(거부 시험, 위 불량 색 절) |
| `BALL_MOCK_HOME` `BALL_MOCK_HOME_S` `BALL_MOCK_HOME_FAR_DEG` | (비움) `0.05` `90` | 가짜 홈 맞춤: `too_far` = 홈에서 `FAR_DEG` 떨어짐(`max_deg` 가 더 작으면 거부) / 옮기는 시간 |
| `BALL_MOCK_STUDIO` `BALL_MOCK_STUDIO_LOCK_S` `BALL_MOCK_STUDIO_RELEASE_S` | (비움) `2` `0.5` | 가짜 Studio 세션: `teleop`·`recording`·`unknown`·`lock` / 잠금만 남는 초 / 세션을 멈춘 뒤 잠금이 풀리는 초 |
| `BALL_MOCK_END` | `home_reached` | 가짜 정책이 정상으로 끝날 때의 reason(`duration_elapsed` 로 바꿔 시험) |
| `BALL_MOCK_SCENE` | `demo4` | 가짜 작업대 처음 장면: `demo4` = 발표 실연(계약서 2절, 투입 분홍 2·초록 1·불량 1 — 불량 색은 계약서에 없어 분홍으로 둠), `full` = 시연 전체 흐름 11개(selftest·eval 기대 개수 기준) |
| `BALL_MOCK_POLICY_S` `BALL_MOCK_FAIL_AT` `BALL_MOCK_DROP_AT` `BALL_MOCK_HAND` `BALL_MOCK_HAND_DELAY` `BALL_MOCK_SNAP` `BALL_MOCK_REFUSE` `BALL_MOCK_DETECT_S` | `0.3` `3` (비움) `auto` `2` `1` (비움) `0.03` | 가짜 작업대(`mocks.py` 머리말). `DROP_AT` = 그 시도에서 집었다가 구역 밖에 떨어뜨림 |

프로젝트 루트에 `.env` 가 있으면 읽는다(이미 설정된 환경변수가 우선). Chroma·LangSmith 원격 전송은 끈다(시연 중 외부 통신 0건).

## RAG — 세 문서 (10-22)

- 원본 3개(`config.RAG_DOCS`): `docs/standard.md` **작업 기준서**(없으면 `fixtures/standard_sample.md`)·`docs/robot_manual.md` **로봇 운용 설명서**·`docs/system_guide.md` **시스템 안내서**. 제목(`##`·`###`) 단위로 잘라 99조각(28·32·39).
- 조각 메타데이터: `doc`(standard·robot_manual·system_guide)·`doc_name`·`clause`(예 "4.3")·`title`·`section`. 조항 번호가 문서마다 따로라 **인용은 문서 이름까지**: "[로봇 운용 설명서 4.3]에 따르면 …". `say` 이벤트의 `sources` 와 `search_manual` 도구 결과에도 `doc`·`cite` 가 있다.
- 색인: Chroma 컬렉션 `docs`(`.cache/chroma/docs.json` 에 원본 경로·해시·조각 수, 세 파일 중 하나라도 바뀌면 다시 만든다). 조각 본문 앞에 `[문서 이름 > 장 > 조항]`. 조각 임베딩은 numpy 행렬로 들고 질의마다 코사인을 직접 잰다. 색인할 때 임베딩은 16조각씩·제한 120초(`llm.QwenEmbeddings.docs`). OVMS 임베딩은 CPU 라 99조각을 한 번에 보내니 20초 제한을 넘었다(쉬고 난 첫 요청 18초).

### LangChain 구성 (발표용)

```
질문 "그리퍼가 뜨거우면 어떻게 해?"
  │
  │  ① 검색       rag.HybridRetriever ················ LangChain BaseRetriever
  │                 임베딩 코사인(llm.QwenEmbeddings ··· LangChain Embeddings → OVMS qwen3-embedding)
  │               + 글자 2-gram BM25 → RRF 로 합친 순위 → 앞 6개 + 코사인 앞 3개 (1장·머리말 뺌)
  ▼
  │  ② 리랭크     rag.QwenRerankCompressor ··········· LangChain BaseDocumentCompressor
  │                 OVMS qwen3-reranker(/v3/rerank)로 다시 줄 세워 3개
  │                 (임베딩 1위가 확실하고 합친 1위와 같으면 건너뜀)
  │     ①+②  = rag.RerankRetriever (ContextualCompressionRetriever 와 같은 일)
  │            = tools.search_manual ················· LangChain @tool
  ▼
  │  ③ 근거 고르기 graph.select_evidence (코드) ─ 문서 밖이면 LLM 없이 "자료에는 그 내용이 없습니다"
  ▼
  │  ④ 프롬프트   graph.answer_chain: ChatPromptTemplate ··· {question, context="[근거 1] 로봇 운용 설명서 4.3 …", history}
  ▼
  │  ⑤ LLM        ChatOpenAI → OVMS qwen3-4b (max_tokens 80) ─ 첫 줄 근거 번호, 둘째 줄 한 문장
  ▼
  │  ⑥ 출력 파서  graph.cited_answer_parser ·········· LangChain BaseOutputParser ─ 인용은 코드가 붙인다
  ▼
"[로봇 운용 설명서 4.3]에 따르면 …"
```

- **구조화 출력**(`with_structured_output(method="json_schema")`)은 의도 판별 `IntentOut{intent, count}` 와 계획 `PlanOut{feasible, steps}` 에 쓴다. 답은 두 줄 글 + 출력 파서다. 4B 가 json_schema 의 문자열 칸에서 `max_tokens` 까지 같은 말을 되풀이했다(10-22 0시 planner 의 reason 칸).
- **LangGraph 가 흐름, LangChain 이 부품.** LangGraph(`StateGraph`)는 "무엇을 어떤 순서로 하는가"를 맡는다. 판별 결과로 갈래를 고르고, act → verify 반복·재시도·멈춤·손 대기, 좌표 집기의 "할까요?" 기다림(`interrupt`), 대화 상태 저장(`SqliteSaver`)을 노드와 간선으로 그린다. LangChain 은 그 노드 안에서 쓰는 부품이다: 리트리버, 문서 압축기(리랭크), 프롬프트 템플릿, `ChatOpenAI`, 구조화 출력, 출력 파서, 도구(`@tool` detect_objects·run_policy·search_manual). 그래서 question 갈래의 `answer` 노드는 위 LangChain 체인 하나를 부르는 한 칸이다. 흐름을 바꿀 때는 그래프를, 검색·답 품질을 바꿀 때는 부품을 고친다.

### 검색·답 규칙

- **질문 첫마디**(10-22 r10): question 으로 판별되면 `answer` 노드가 먼저 "자료를 찾아볼게요."(`graph.ASK_ACK`, `say` 의 `ack: true`)를 말하고 검색·답을 이어 한다. 판별 전에는 말하지 않는다(명령을 질문으로 잘못 알아들었을 때 더 나쁨). 자기 질문 거절에는 붙이지 않고, 작업 기록 질문은 report 로 가서 붙지 않는다.

- 검색할 때만 낱말을 넓힌다(`rag.SYNONYMS`). "로봇" → "로봇 팔"("로봇 팔"·"로봇 운용"은 그대로), "블럭"·"볼"·"공"(명사) → "블록", "핑크" → "분홍", "녹색·연두" → "초록", "작업 순서" → "작업 처리 순서"(기준서 5.1 제목). LLM 에는 원래 질문을 준다.
- **리랭크는 원문 질의로 한 번**(`rag.rerank_scores`). 10-21 기준서 하나일 때는 원문·넓힌 질의 두 번의 평균이 필요했다. 세 문서에서 아래 세 가지를 더하자 원문 한 번으로 51문항 상위 3개 51/51 이 같고 리랭크는 반으로 줄었다. GPU 리랭크는 조각당 약 0.19초다(조각이 길어짐, 평균 352자). 두 번을 동시에 보내도 GPU 에서 차례로 돌아 이득이 없었다.
  - 1장(목적과 적용 범위)·머리말은 후보에서 뺀다(질문에 목적·범위·적용·기준서·설명서·안내서·문서가 없을 때). 세 문서의 6조각이 리랭크 자리를 차지했다.
  - 리랭크 후보 = 합친 순위 앞 `RAG_RERANK_N`(6)개 + 임베딩 코사인 앞 3개(`rag.VEC_TOP`). 기준서 6.1("손은 어디에")은 코사인 2위인데 글자 순위가 36위라 합친 순위 8위 밖이었다.
  - 합친 순위 1위(anchor)는 리랭크 점수가 1위의 절반 이상일 때만 남긴다(설명서 2.1, 0.45 가 기준서 6.1, 0.95 를 밀어냈음).
- 리랭크 생략 = 임베딩 1위 코사인 ≥ `RAG_SKIP_SCORE`(0.65), 2위와 차 ≥ `RAG_SKIP_MARGIN`(0.04), **그리고 합친 순위 1위가 그 조각일 때**. "불량을 분량으로 알아듣는 건 왜 그래?" 는 코사인 1위(기준서 3.4)와 합친 1위(3.2)가 달랐다. 이때 건너뛰어 정답(안내서 6.5)이 근거에서 빠졌고, 4B 가 이유를 지어냈다.
- LLM 근거(`graph.select_evidence`): 리랭크 최고점의 60% 이상 + anchor(건너뛰었으면 코사인 1위에서 `RAG_KEEP_COS_GAP` 0.05 안쪽 + anchor). 리랭크가 1위의 80% 이상인 안전 수칙(6장) 조항은 맨 앞에 둔다. 근거는 `[근거 1] 로봇 운용 설명서 4.3 그리퍼가 뜨거울 때` 처럼 번호·문서·조항으로 준다.
- **인용 고르기**(`graph._pick_hit`). 4B 는 근거로 답하고도 첫 줄을 0 으로 쓴 게 51문항 중 23건, 다른 번호를 쓴 게 4건이었다.
  - 첫 줄 번호(0 이면 첫 근거)에서 시작한다. 답 문장의 글자 2-gram 이 `CITE_SWITCH`(0.1) 이상 더 겹치는 근거가 있으면 그쪽을 인용한다(겹침은 조항 제목 + 본문 기준). 이렇게 51문항 인용이 40 → 48로 올랐다.
  - 안전 질문(손·안전·위험·다치·끼이·부딪)이고 안전 조항이 맨 앞이면 그 조항을 인용하고, LLM 문장 대신 원문을 읽는다(4B 가 6.1 금지 문장을 뒤집어 말했음).
  - 번호가 0 이고 작업 기록이 있고 겹침이 0.2 미만이면 기록으로 답한 것으로 보고 인용하지 않는다.
  - 한자·가나가 섞이면("자动") 근거 첫 문장을 읽는다. 반말 줄기 + "입니다" 어미는 고친다("돌아입니다" → "돌아갑니다", "해입니다" → "합니다").
- **문서 밖 질문**: 리랭크 최고점 < `RAG_MIN_SCORE`(0.3) 또는 임베딩 1위 코사인 < `RAG_OOD_COS`(0.45, 10-22 r10 문서 수정 뒤 0.50 → 0.45: "받아쓰기는 어느 장치에서 돌아?" 가 0.488 로 걸렸음)이고 작업 기록도 없으면, LLM 없이 "자료(작업 기준서·로봇 운용 설명서·시스템 안내서)에는 그 내용이 없습니다. …"(`graph.NO_ANSWER`).
  - 4B 가 "…은 자료에 없습니다" 처럼 다르게 말해도 이 문장으로 맞춘다.
  - 나 자신을 묻는 말("너 이름이 뭐야?"·"너는 누구야?")은 검색 점수와 상관없이 거절한다(`graph._SELF_Q`). 세 문서에서 리랭커가 기준서 5.2 에 0.94 를 줬고, 4B 가 프롬프트 첫 줄을 보고 이름을 지어냈다. 프롬프트에 "자신·잡담은 0" 을 넣는 방법은 문서 안 질문까지 거절해 인용이 40 → 26/42 로 떨어져 버렸다.
  - 리랭커 점수만으로는 문서 밖 질문을 가르지 못한다(문서 밖 9문장 중 6개가 0.5 이상, "오늘 날씨 어때?" 0.95). 코사인도 겹친다(문서 안 최저 0.506, 문서 밖 0.37~0.60). 그래서 마지막은 LLM 의 "자료에는 그 내용이 없습니다" 판단에 기댄다(9문장 모두 거절).
- 작업 기록 질문("오늘 몇 번 실패했어?")은 위 판별 검사로 report 로 가서 `runs/` 기록을 센다(RAG 를 거치지 않음).
- 리랭크는 Qwen3-Reranker 틀(`<Instruct>/<Query>/<Document>` + yes/no 시스템 문장)을 씌워 보낸다. 틀 없이 보내면 점수가 갈리지 않는다(10-21 실측: 맞는 조각 0.02 → 틀 씌우면 0.83 으로 1위).
- **보고 근거는 미리 계산**(작업 기준서만): `rag.CITATION_QUERIES`(보고 항목·기록 불일치·확인 요청·멈춘 뒤·의도별 처리)마다 리랭크까지 해서 2조항씩 골라 `.cache/citations_<기준서+검색어 해시>.json` 에 둔다. 기준서나 검색어가 바뀌면 키가 바뀌어 다시 만든다. 보고는 이 캐시만 읽는다.

## 작업 기록

`runs/<YYYYMMDD>/log.jsonl`(가짜 작업대 `BALL_MOCK=1` 이면 `runs/<YYYYMMDD>_mock/` — 연속 시험·selftest·eval 이 실물 기록과 섞이지 않게, 10-22. 대화 상태 체크포인트도 가짜면 `runs/agent_checkpoints_mock.sqlite`, 실물은 `runs/agent_checkpoints.sqlite`) — 정책 시도마다 한 줄: `ts intent policy before after ok`(계약서) + `thread_id text attempt moved moved_by_class fail_reason policy_ok returncode duration_s backend`. `before`/`after` 는 detect 의 `zones`. 보고서 HTML 은 같은 폴더 `report_<시각>.html`, 가짜 사진은 `mock/`.

## 파일

| 파일 | 내용 |
|---|---|
| `graph.py` | StateGraph 노드·분기, `run_command()` |
| `router.py` | 멈춤 규칙·개수 규칙·임베딩 라우터·LLM 구조화 출력 |
| `intent_examples.py` | 라우터 예문·의도 설명 |
| `tools.py` | LangChain 도구 3개(`detect_objects`·`run_policy`·`search_manual`) |
| `rag.py` / `llm.py` | 세 문서 RAG(LangChain 리트리버·압축기) / ChatOpenAI·임베딩·리랭크 |
| `report.py` / `runlog.py` | 보고 집계·문장·HTML / log.jsonl |
| `backends.py` / `mocks.py` | 실제·가짜 연결 / 가짜 작업대(정책·숫자 배치·그릇 옮기기 흉내) |
| `hearing.py` | 받아쓰기 환청 거르기(글 문구·말소리 길이) |
| `studio.py` | Studio 런타임 세션 목록·멈춤(거부될 때만) |
| `cli.py` `eval.py` `selftest.py` | 시험 도구 |
| `fixtures/` | `docs/` 가 생기기 전 시험용 임시 기준서·문장, `intents_stt.csv`(받아쓰기 오류 꼴 "골라내죠" 등), `intents_none.csv`(잡담·마이크 시험 25문장, 정답 none) |
| `.cache/` | 라우터 예문 임베딩·Chroma(지워도 다시 만듦) |

## 설치한 패키지

`~/.local/bin/uv pip install --python .venv/bin/python langgraph-checkpoint-sqlite` (2026-10-21) → `langgraph-checkpoint-sqlite 3.1.2`, `aiosqlite 0.22.1`, `sqlite-vec 0.1.9`. 다른 패키지 버전은 바뀌지 않았다(dry-run 확인).

## 실측 기록

### 2026-10-21 젠북 · OVMS `127.0.0.1:8100` · vision·robot 가짜

| 항목 | 결과 |
|---|---|
| 의도 `docs/intents.csv` 63문장 | **qwen3-8b 63/63**(의도·개수 모두) · qwen3-4b 62/63 두 번(1회차: `빨간볼하나내손에줘` 개수 null, 2회차: `결과 보고에는 뭐가 들어가요?` → report). 기준 58/63 은 둘 다 넘음 |
| └ 경위 | 첫 측정 52/63(의도 62/63, Qwen3-8B 가 개수를 말하지 않은 문장에 1·0 을 채움) → LLM 개수는 숫자 표현이 있을 때만 받기 + 기준서 7절 질문 예문 2개 추가 후 63/63. 라우터 예문과 겹치지 않는 49문장만 세면 49/49. 시험용 `fixtures/intents_sample.csv` 32/32 |
| 판별 지연 | 규칙 9건 0ms · 임베딩 확정 31건 평균 51ms(최대 65) · LLM 23건 평균 1.5초(8B) / 1.8초(4B, 최대 2.7초) |
| `--sweep` | LLM 만 쓰면 의도 63/63. 임계값 0.80·여유 0.04 에서 LLM 호출 35%(여유 0.02 도 같은 정답률, 호출 25%) |
| 기준서 검색 `--rag` | 9/9(정답 조항이 상위 3개에). 검색 평균 4.3초(대부분 CPU 리랭크 5개) |
| 질문 1개 끝까지 | 약 10초 = 판별 0.05~2초 + 검색 4~5초 + 답 3~4초(qwen3-4b) |
| 시연 1→5 | 판별 약 45ms(전부 임베딩). 보고 7.2초(검색 3.7 + LLM 3.4). qwen3-4b 보고 문장이 검사 통과: "불량 3개(빨강 2, 파랑 1), 빨강 3개(1개 전달), 파랑 4개입니다. 시도 13회 중 12회 성공, 작업 기록과 카메라 개수가 일치합니다." |
| `selftest` | 16/16 |
| OVMS | 측정 중 두 번 내려감(16:50~17:10, 17:20:27 GPU `CL_EXEC_STATUS_ERROR` → SEGV). 17:22 재기동 뒤 목록에 qwen3-8b 가 없어 자동 선택이 qwen3-4b. 지연 수치는 OVMS 에 다른 부하가 있던 때의 값 |

### 2026-10-21 밤(20:57~21:05) 젠북 · qwen3-4b · 리랭커 GPU · vision·robot 가짜

20:22~20:33(xpu 학습과 겹쳐 GPU HANG) 값은 버렸다. 로그 `runs/20261021/eval_night.*.log`, 자세한 표는 `RESULT_20261021.md` 밤 절.

| 항목 | 결과 |
|---|---|
| 의도 63문장 | qwen3-4b **63/63**(판별 프롬프트에 보고 질문/보고 명령 구분 규칙 추가 뒤) · LLM 21건 평균 0.79초 |
| 받아쓰기 변형 `fixtures/intents_stt.csv` | 6/6("불량품 골라내죠" → defect, 임베딩 0.996) |
| `--rag` | 9/9 · 평균 0.23초(리랭크 조각당 0.07초, GPU) |
| `--qa` | 인용 9/9 + 기준서 밖 4/4 · 합계 중앙 1.84초, 최대 4.82초(LLM 판별 1.6 + 7항목 답 3.2) |
| `--timeline` | 명령 첫 응답 0.08~0.09초, 에이전트 몫 0.13~0.23초(정책·손 기다림 제외), 보고 3.2초, 질문 2.35초, 멈춰 0.00초 |
| `selftest` | 16/16 |

**KV 캐시 1GB 고정 뒤(21:14 인프라 재기동, SETUP.md §7.7) 21:18~21:20 재측정** — 로그 `runs/20261021/eval_kv1.*.log`

| 항목 | 결과 |
|---|---|
| 정확도 | 의도 63/63 · 받아쓰기 변형 6/6 · `--rag` 9/9 · `--qa` 13/13 · `selftest` 16/16 (변화 없음) |
| `--qa` | 합계 중앙 **1.59초**, 최대 4.06초 · 답 생성 뒤 LLM 판별 1.6 → **0.76초** |
| `--timeline` | 보고 3.21 → **2.53초**, 질문 2.35 → **1.83초**, 명령 1~4·멈춰는 그대로 |

**분홍·초록 + scene·plan 뒤(22:51~22:56)** — 로그 `runs/20261021/eval_color.*.log`, 자세한 것은 `RESULT_20261021.md` 5절

| 항목 | 결과 |
|---|---|
| 의도 87문장·9 intent | 87/87 · LLM 36건 평균 0.64초 |
| `--rag` / `--qa` | 9/9 / 13/13(중앙 1.58초, 최대 3.67초) |
| `--sp` | scene 8/8(중앙 1.60초) · plan 10/10(계획 알림 중앙 3.68초, 거절 0.7초) |
| `selftest` | 30/30(실패 주입·scene·plan 14건 추가) |

**블록 변경 뒤(22:13~22:16)** — 로그 `runs/20261021/eval_block.*.log`, 자세한 것은 `RESULT_20261021.md` 4절

| 항목 | 결과 |
|---|---|
| 정확도 | 의도 63/63(새 intents.csv) · 받아쓰기·블럭·볼 변형 13/13 · `--rag` 9/9 · `--qa` 13/13 · `selftest` 16/16 |
| `--qa` | 합계 중앙 1.68초, 최대 3.73초 |
| `--timeline` | 명령 첫 응답 0.08초, 보고 2.52초, 질문 1.86초(앞과 같음) |

## 남은 일

- 실제 카메라·로봇으로 반복 판정 확인(YOLO 개수 흔들림이 있으면 verify 에 두 번 찍어 다수결 추가 검토).
- `SETUP.md` 가 나오면 모델 이름 확인(지금은 `LLM_MODEL` 기본 qwen3-4b, 목록에 없으면 대체).
- 보고서 "장치별 추론 시간"은 지금 마지막 detect 지연·판별 평균·LLM 이름뿐. `bench/` 결과가 나오면 연결.
- UI 세션: `run_command` 이벤트 연결, `say` → TTS, `report` 의 `html` 표시, 시연 때 `BALL_ROBOT=1`.

## 재개 (2026-10-21 마감)

> 10-21 밤 재개에서 아래 "아직 안 한 것" ①~⑥ 을 마쳤다(결과는 위 실측 기록). 합치기 전 버전은 `runs/agent_backup_20261021_tested/`, `_wip_latency/` 는 합친 뒤 남은 사본(지워도 됨).

**끝난 것**
- `agent/` 본체(이 README 위 내용) — 시연 1→5 mock 통과, 의도 63/63(8B)·62/63(4B), `eval --rag` 9/9, `selftest` 16/16. 총괄에 [결과] 회신함(`RESULT_20261021.md`).
- 로봇 세션 추가 사항(reason·camera_released·실행 사이 멈춰) 반영·시험 완료.

**하다 멈춘 것 — 기준서 질문 10초 → 3초 (총괄 요청 17:4x)**
- 고친 코드는 **`agent/_wip_latency/`** 에 따로 두었다(시험 전이라 지금 돌아가는 `agent/` 에는 안 넣음). 바뀐 파일: `rag.py`(임베딩 행렬 numpy 코사인·리랭크 생략 조건 `RAG_SKIP_SCORE/MARGIN`·`RAG_RERANK_N`·보고 근거 캐시 `citations`/`quick_citations`/`warmup`), `graph.py`(보고가 캐시 조항 사용·질문 기준서 밖 판정에 `RAG_OOD_COS`·첫 명령 때 백그라운드 warmup), `eval.py`(`--rag --sweep` 조합표, `--qa` 질문 끝까지 단계별 지연), `config.py`(위 설정값·`LLM_MODEL` 기본 `qwen3-4b`), `llm.py`(`model_info`: 지정 모델이 목록에 없으면 qwen3-4b→8b 대체, 60초마다 목록 다시 봄), `__init__.py`(`warmup` 공개).
- 아직 안 한 것: ① `_wip_latency/*` 를 `agent/` 로 옮기기 ② `eval --rag --sweep` 으로 9/9 유지되는 가장 빠른 `RAG_RERANK_N`·`RAG_SKIP_SCORE`·`RAG_SKIP_MARGIN` 고르기, 기준서 밖 질문 cos 로 `RAG_OOD_COS` 정하기 ③ `eval --qa` 로 전후 지연 측정(이전: 질문 약 10초 = 판별 0.05~2 + 검색 4~5 + 답 3~4) ④ 3초가 안 되면 질문 쪽 라우터 판별 완화(question 은 로봇을 안 움직이는 안전한 의도)·답 길이 줄이기 검토 ⑤ `selftest`·`eval docs/intents.csv` 재확인 ⑥ README 의 RAG 절·환경변수 표 갱신 ⑦ 총괄에 [결과] 1회(전후 지연·정답률).
- 관찰: 17:44 OVMS 목록에는 qwen3-8b 만 있었다(총괄 메시지의 "4b 만"과 다름). 그래서 `LLM_MODEL` 기본 4b + 목록에 없으면 대체하는 방식으로 짰다.

**재개 첫 명령** (젠북, 프로젝트 루트)
```bash
cd ~/dev/projects/ball-sorter && curl -s http://127.0.0.1:8100/v3/models && cp agent/_wip_latency/*.py agent/ && .venv/bin/python -m py_compile agent/*.py && .venv/bin/python -m agent.eval --rag --sweep
```
