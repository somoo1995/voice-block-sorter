# ball-sorter 세션 공용 계약서

> 말로 시키는 스펀지 블록 선별 로봇 (인텔 OpenVINO 과정 M6 최종 프로젝트). 물체 = 스펀지 수세미를 깍둑썬 약 3cm 블록(**분홍·초록**, 파스텔), 불량 = 검정 X 6면(10-21 밤 볼→블록, 빨강·파랑→분홍·초록 변경). **코드 식별자는 그대로**: red_* = 분홍, blue_* = 초록(클래스 red_ok·blue_ok…, 구역 bowl_red·bowl_blue, 정책 pick_red·pick_blue, intent red·blue). 사람이 보고 듣는 말만 분홍·초록. 받아쓰기·말 동의어: 분홍=핑크, 초록=녹색=연두. 여러 Claude Code 세션(총괄·인프라·에이전트·로봇·비전·UI·문서)이 이 폴더를 나눠 맡았다.
> 이 파일은 총괄 세션만 고친다. 다른 세션은 바꿀 필요가 있으면 총괄에 요청한다.
>
> ⚠ 개발 중에 쓰던 계약서라 일부(정책 `pick_*` 로 집기 등)는 초기 설계 기준이다. 최종 구성(좌표 집기 + ACT 는 손 건네기만)은 루트 `README.md` 를 본다.

## 1. 폴더 소유

| 폴더/파일 | 담당 세션 |
|---|---|
| `.venv` `ovms/` `models/` `dl/` `bench/` `logs/` `config.json` `pull_models.sh` `start_ovms.sh` `ovms_watch.py` `camera_ctrls.conf` `start_demo.sh` `SETUP.md` | 인프라 |
| `agent/` | 에이전트 |
| `robot/` | 로봇 |
| `vision/` | 비전 |
| `ui/` `docs/` | UI·문서 |
| `runs/` | 실행 결과 (에이전트가 씀) |

남의 폴더는 읽기만 한다. 각 폴더에 `README.md`(실행법·설치한 패키지·남은 일)를 둔다.

## 2. 시나리오 (시연 순서)

| 순서 | 말 | intent | 정책 |
|---|---|---|---|
| 1 | 불량품 골라내 줘 | `defect` | `pick_defect` → 불량 그릇, 0개 될 때까지 반복 |
| 2 | 분홍 블록 옮겨 줘 | `red` | `pick_red` → 분홍 그릇, 반복 |
| 3 | 초록 블록 옮겨 줘 | `blue` | `pick_blue` → 초록 그릇, 반복 |
| 4 | 분홍 블록 하나 내 손에 줘 | `handoff` | 손 자리에 손이 보이면 `handoff` (분홍 그릇 → 손) |
| 5 | 결과 알려 줘 | `report` | 없음. 개수 세기 + 기록 대조 → 음성·HTML 보고 |
| - | (기준서 질문) | `question` | 없음. docs/standard.md RAG |
| - | 멈춰 | `stop` | 실행 중 정책 즉시 중단 (LLM 거치지 않음) |

**발표 실연(3분, 사용자 확정 10-21 밤)**: 투입에 분홍 2·초록 1·불량 1(블록 4개) → ① 불량품 골라내 줘(defect 1회) ② 지금 배치 어때?(scene) ③ 불량 빼고 나머지 다 정리해 줘(plan, 3회) ④ 분홍 블록 하나 내 손에 줘(handoff 1회) ⑤ 결과 알려 줘(report). 위 표의 전체 흐름(블록 11개)은 미리 찍은 성공 영상으로. 가짜 작업대 기본 장면도 이 4개(BALL_MOCK_SCENE=demo4).

`count`: "두 개만" 같은 개수 지정이 있으면 정수, 없으면 null(= 0개 될 때까지).

## 3. 인터페이스

### vision — `vision/detect.py`
```python
detect_objects(image_path: str | None = None) -> dict
# image_path 없으면 위 카메라에서 한 장 찍고 카메라를 바로 닫는다(정책이 같은 카메라를 씀)
{
  "ts": "2026-10-22T10:00:00+09:00",
  "zones": {
    "input":       {"red_ok": 4, "blue_ok": 4, "red_defect": 2, "blue_defect": 1},
    "bowl_red":    {...같은 4키},
    "bowl_blue":   {...},
    "bowl_defect": {...},
  },
  "hand": {"present": false, "conf": 0.0},
  "objects": [   # 10-21 밤 추가(선택 필드, 없으면 빈 목록). 블록 하나당 하나
    {"cls": "red_ok", "zone": "input", "cell": "왼쪽 위", "px": [412, 233], "conf": 0.93}
  ],             # zone = 위 5구역 또는 "outside"(구역 밖), cell = 그 구역 박스를 3×3 으로 나눈 칸(카메라 화면 기준 위/가운데/아래 × 왼쪽/가운데/오른쪽, 가운데 칸은 "가운데"), px = 위 카메라 픽셀 중심
  "image_path": "runs/.../snap_xxx.jpg",   # 박스 그린 사진
  "device": "NPU", "latency_ms": 12.3
}
```
YOLO 클래스: `red_ok` `blue_ok` `red_defect` `blue_defect` `hand`. 구역은 `vision/zones.json`(위 카메라 픽셀 다각형: input, bowl_red, bowl_blue, bowl_defect, hand_spot).

### robot — `robot/run_policy.py`
```python
run_policy(name: str, timeout_s: float | None = None) -> dict   # None = robot/policies.json 정책별 값(에피소드 90%+3초, 15초 녹화 기준 약 21초). 40 같은 큰 값을 넘기면 남는 시간에 블록을 하나 더 집으러 간다   # name: pick_defect | pick_red | pick_blue | handoff
{"name": "pick_red", "ok": True, "started": "...", "ended": "...", "duration_s": 18.2, "returncode": 0, "log_path": "..."}
stop() -> None   # 실행 중인 정책을 즉시 중단
```
CLI: `python -m robot.run_policy pick_red --dry-run` (로봇 안 움직이고 실행될 명령·설정만 출력).

### agent — `agent/`
```python
run_command(text: str, thread_id: str = "demo") -> Iterator[dict]
# event: {"type": "node"|"tool"|"say"|"report"|"error", "node": str, "data": dict, "ts": str}
```
- `BALL_MOCK=1` 이면 vision·robot 을 `agent/mocks.py` 가짜로 대체(작업대 상태를 메모리에서 흉내).
- 작업 기록: `runs/<YYYYMMDD>/log.jsonl`, 시도마다 `{ts, intent, policy, before, after, ok}`.
- 도구 3개: `detect_objects`, `run_policy`, `search_manual`.
- 10-21 밤 추가 intent: `scene`(지금 배치·개수·위치 질문 → detect_objects 로 보고 답함, 로봇 안 움직임), `plan`(배운 정책을 순서·개수·조건으로 엮는 처음 듣는 명령 → 단계 목록 [{intent, count}] 을 세우고 detect 로 확인하며 실행). 둘 다 기존 intent 7개와 같은 이벤트 형식.

### ui — `ui/`
- `ui/app.py`: Gradio. 버튼 누르고 말하기 → STT → `run_command` → 이벤트 스트리밍(노드 진행·로그), 위 카메라 스냅샷, 보고서 패널, 텍스트 입력 백업.
- `ui/tts.py`: `speak(text: str) -> str` (wav 경로, Supertonic 3 한국어, CPU).
- `ui/stt.py`: `transcribe(wav_path: str) -> str` (OVMS `/v3/audio/transcriptions`).

### docs — `docs/`
- `standard.md` 작업 기준서(불량 정의·처리 규칙·안전 수칙, RAG 원본), `intents.csv`(text,intent,count 25줄 이상), `recording_guide.md`(정책 4개 녹화 절차).

## 4. 로컬 AI 주소 (인프라가 SETUP.md 에 확정)

환경변수: `LLM_BASE_URL`(OVMS `/v3`), `LLM_MODEL`, `EMB_MODEL`, `RERANK_MODEL`, `STT_MODEL`. 값은 `SETUP.md` 를 따른다.
시연·최종 코드는 로컬 OVMS 만 쓴다.

## 5. 공통 금지

- Physical AI Studio(Docker·네이티브) 컨테이너·볼륨·DB 변경 금지(읽기·`--dry-run` 만).
- 로봇 실제 구동 금지(사용자 입회 때만). 카메라는 다른 프로세스가 안 쓸 때만 열고 바로 닫기.
- **젠북 GPU 는 한 번에 한 작업.** xpu 학습(YOLO 등)은 `systemctl --user stop ball-ovms` 로 OVMS 를 내린 뒤에만 돌리고, 끝나면 `start_ovms.sh` 로 다시 올린다. 측정·시연·리허설 중엔 xpu 학습 금지(10-21 20:22 학습과 LLM 이 겹쳐 GPU HANG, 20:33 복구).
- 프로젝트 `.venv` 에 패키지가 더 필요하면 `~/.local/bin/uv pip install --python .venv/bin/python <패키지>` 후 자기 README 에 기록.
- 비밀값을 파일·화면에 남기지 않기.

## 6. 회신

끝나면 총괄 세션에 `[결과] <결론 먼저, 경로>` 를 한 번만 보낸다. 막히면 `[막힘] <질문 1개>`. 중간 보고·확인 회신은 하지 않는다.
