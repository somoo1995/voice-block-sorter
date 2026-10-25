# 민어 — 말로 시키는 블록 선별 로봇

> 인텔 OpenVINO 과정 M6 최종 프로젝트 · 팀 **민어**(김민석, 어성권) · 2026-10

SO-ARM101 로봇 팔에게 말로 일을 시키면 위 카메라로 작업대를 보고 분홍·초록 스펀지 블록을 색깔별 그릇에 나눠 담는다. 검은 X 가 그려진 블록은 불량으로 따로 골라낸다. 언어 모델·받아쓰기·물체 검출은 모두 노트북 한 대(Intel Core Ultra 9 185H) 안에서 OpenVINO 로 돌고, 인터넷 없이 동작한다.

- 발표 사이트(덱·시연 영상): <https://robot.echokwon.org>

> 이 저장소는 **어떻게 구성했는지 보여 주기 위한 공개 사본**이다. 모델·학습 데이터·녹화 로그·기기별 보정값 일부는 빠져 있어서 그대로는 실행되지 않는다(아래 [실행하려면](#실행하려면)).

## 이렇게 말하면

| 말 | 하는 일 |
|---|---|
| 불량품 골라내 줘 | 검정 X 블록을 불량 그릇으로 옮긴다 |
| 분홍 블록 옮겨 줘 · 초록 블록 옮겨 줘 | 같은 색 그릇으로 옮긴다(좌표 집기) |
| 초록 블록 중에 맨 위에 있는 거 옮겨 줘 · 분홍 옆에 있는 거 | LLM 이 검출 결과(블록마다 구역·칸·픽셀 위치)를 보고 대상 블록을 고른다 |
| 불량 빼고 나머지 다 정리해 줘 | 여러 단계 계획을 세우고, 단계마다 카메라로 확인하며 실행한다 |
| 분홍 블록 하나 내 손에 줘 | 손 자리에 손이 보이면 분홍 그릇에서 꺼내 건넨다(ACT 정책) |
| 그릇에 있는 블록 다시 꺼내 줘 | 그릇 → 투입 구역으로 되돌린다 |
| 지금 배치 어때? · 결과 알려 줘 | 카메라로 세어 답한다 · 작업 기록과 대조해 음성·HTML 보고서를 만든다 |
| 초록 불량은 왜 불량이야? | 작업 기준서·로봇 설명서·시스템 안내서에서 조항을 찾아 인용해 답한다(RAG) |
| 홈 자세로 보내 줘 · 멈춰 | 팔을 홈 자세로 · LLM 을 거치지 않고 즉시 멈춘다 |

## 전체 구성

```text
마이크(브라우저)
   │
   ▼
Whisper (NPU) ── 받아쓰기
   │
   ▼
의도 판별 agent/router.py ── 규칙 → 임베딩 유사도 → LLM 구조화 출력(JSON 스키마)
   │
   ▼
LangGraph 상태 그래프 agent/graph.py
   ├─ 보기    vision/      YOLO (NPU) → 구역·3×3 칸 판정
   ├─ 집기    robot/coord_pick/
   │            위 카메라 픽셀 ─(호모그래피)→ 작업대 mm ─(IK)→ 관절각 → 30Hz 궤적
   │            → 손목 카메라로 맞춤·집었는지 확인
   │            → docker exec → Physical AI Studio 컨테이너의 SharedRobot.send_action → 서보
   ├─ 건네기  robot/run_policy.py  ACT 정책(physicalai run) — 분홍 블록 손에 건네기
   ├─ 질문    agent/rag.py  Chroma + 임베딩 + 리랭크 → 조항 인용 답
   └─ 보고    agent/report.py · ui/tts.py  HTML 보고서 + 한국어 음성(Supertonic 3)
   │
   ▼
대시보드 ui/app.py (Gradio) ── 노드 진행·로그·카메라·보고서를 실시간 표시
```

**LLM 은 도구 호출(tool calling)로 팔을 직접 움직이지 않는다.** 의도·대상 블록·단계 목록을 JSON 스키마 구조화 출력으로 정하는 데까지만 하고, 실제 동작은 그래프 노드가 파이썬 함수(`sort_blocks`, `unsort_blocks`, `go_home`, `run_policy`, …)를 불러 실행한다. 관절각 계산(IK/FK)과 안전 검사(작업 공간·금지 구역·속도 제한·집게 힘)는 전부 코드가 한다. 4B 모델이 각도를 직접 내면 틀릴 수 있고, 틀리면 팔이 부딪히기 때문이다.

## 장치 배치

| 모델 | 장치 | 이유 |
|---|---|---|
| Qwen3-4B int4 — 의도·대상·계획 판단 | GPU (Arc iGPU) | 길이가 바뀌는 KV 캐시와 메모리 대역폭이 중요하다 |
| Whisper large-v3-turbo int8 — 받아쓰기 | NPU | 입력 모양이 고정된 인식 모델이라 NPU 에 맞고 전력이 적게 든다 |
| YOLO int8 — 블록·손 검출 | NPU | 위와 같다 |
| Qwen3-Embedding 0.6B — 의도 판별·문서 검색 | CPU | 한 문장에 25~58ms 로 충분하다 |
| Qwen3-Reranker 0.6B — 문서 순위 | GPU | CPU 5.8초 → GPU 0.4초(문서 8개), 순위는 같다 |
| Supertonic 3 — 한국어 음성 합성 | CPU | OVMS 내장 TTS 는 한국어를 지원하지 않는다 |

LLM·임베딩·리랭크·받아쓰기는 OpenVINO Model Server(OVMS) 하나가 OpenAI 호환 주소로 제공한다(`config.json`, `start_ovms.sh`). LangChain 코드는 `ChatOpenAI` 그대로 쓰고 `base_url` 만 로컬로 바꿨다. 장치는 가장 빠른 곳이 아니라 **비어 있는 곳**에 나눠 둔다. 같은 iGPU 에 일을 겹치면 멈춘다(아래).

## 폴더

| 폴더 | 내용 |
|---|---|
| `agent/` | 의도 판별(`router.py`), LangGraph 그래프(`graph.py`), RAG(`rag.py`), 보고서(`report.py`), 가짜 작업대(`mocks.py`), 시험(`selftest.py`, `eval.py`) |
| `vision/` | 카메라(`camera.py`), YOLO 검출(`detect.py`), 구역·칸(`zones.py`, `zones.json`), 합성 데이터·자동 라벨·학습(`synth_dataset.py`, `autolabel.py`, `train.py`) |
| `robot/coord_pick/` | 좌표 집기: 기구학(`kinematics.py`, `geometry.py`), 계획(`plan.py`), 실행기(`executor.py`), 손목 카메라 확인(`wrist.py`), 궤적 송신(`stream.py`, `servo.py`), 설정(`config.json`) |
| `robot/` | ACT 정책 실행(`run_policy.py`, `policies.json`), 사전 점검(`preflight.py`), 정책 비교 실험(`compare/`) |
| `ui/` | Gradio 대시보드(`app.py`), 보기 전용 화면(`view.py`), 받아쓰기(`stt.py`), 음성 합성(`tts.py`), 보고서 템플릿, 외부 연결 차단(`netguard.py`) |
| `docs/` | RAG 원본 문서 3종(`standard.md`, `robot_manual.md`, `system_guide.md`), 녹화 가이드, 받아쓰기 실측, 대시보드 화면(`screens/`), ACT 비교 그림(`compare_20261022/`) |
| `bench/` | 응답 속도·메모리·NPU 동시 사용·KV 캐시·GPU 멈춤 실험 스크립트 |
| 루트 | OVMS 설정·기동·모델 받기(`config*.json`, `start_ovms.sh`, `pull_models.sh`), GPU 멈춤 감시(`ovms_watch.py`), 카메라 고정값(`camera_ctrls.conf`), 전체 기동(`start_demo.sh`) |

## 해 보고 배운 것

- **ACT 로 집기는 접었다.** 집기 동작만 39개 녹화해 ACT 를 학습했지만(검증 손실 2.51 → 0.51) 정책이 카메라 영상을 거의 보지 않았다. 손목 사진을 좌우로 뒤집어도, 블록을 지워도 동작이 그대로였고, 시작 관절 자세로 동작을 외운 것이었다. 사람이 녹화한 방식(시작 높이·들어 올리는 높이)을 정책이 그대로 닮는다는 것도 확인했다. 에피소드 수와 예외 장면(놓쳤을 때, 블록이 기울었을 때)이 부족하면 생기는 일이다. 그래서 집기는 좌표(IK)로 바꾸고, ACT 는 동작이 정해진 "분홍 블록 손에 건네기"에만 썼다. → `robot/GRASP_ACT.md`, `docs/compare_20261022/`
- **IK 집기: 블록 1개 46초 → 21초.** 5축 팔이라 높이마다 낼 수 있는 집게 기울기가 3~8° 달라서, 이동마다 가장 가까운 기울기로 다시 푼다. 공중 이동 속도를 따로 올리고(36°/s), 멈춤 시간을 줄이고, 손목 사진을 한 장으로 줄였다. 손목 카메라 노출을 고정하니 연한 초록이 절반만 잡혀서 색별 기준을 다시 쟀다. 최종 리허설 명령 5개는 모두 한 번에 성공했다. → `robot/COORD_PICK.md`
- **iGPU 가 가끔 멈춘다.** GPU 요청 500번에 2번꼴로 i915 드라이버가 OVMS 요청을 끊는다(`Fence expiration`, `GPU HANG`). 펌웨어(GuC)를 바꿔도 남는 드라이버 수준 문제라 없애지는 못했다. 대신 `ovms_watch.py` 가 커널 기록에서 그 줄을 보면 바로 OVMS 를 재시작해 복구 시간을 101초에서 13~30초로 줄였다. 시연 문장은 규칙·임베딩으로도 판별되게 해서 LLM 이 멈춰도 팔은 움직인다.
- **크롬이 카메라를 잠깐 쥔다.** 대시보드에서 마이크를 켜고 끌 때 크롬 도우미가 장치 목록을 만들면서 카메라를 약 0.3초 연다. 그 순간 블록을 찾으면 "카메라 사용 중"으로 실패해서, `vision/camera.py` 가 크롬 도우미만 잡고 있으면 1초 기다렸다 연다.

## 실행하려면

필요한 것

- Intel Core Ultra 노트북(Arc iGPU + NPU), Ubuntu 24.04, Python 3.12
- SO-ARM101 팔(팔로워) + 위 카메라 + 손목 카메라
- [Physical AI Studio](https://github.com/open-edge-platform/physical-ai-studio)(Docker) — 팔 연결·녹화·ACT 학습·정책 실행
- [OpenVINO Model Server](https://github.com/openvinotoolkit/model_server) 실행 파일을 `ovms/` 에 풀어 둔다

순서

1. `pip install -r requirements.txt` (torch 는 CPU 판: `--extra-index-url https://download.pytorch.org/whl/cpu`)
2. `./pull_models.sh` — OVMS 모델 약 10GB 를 `models/` 에 받는다
3. `./start_ovms.sh` → `./start_demo.sh` — 대시보드 <http://127.0.0.1:7870>

이 사본에 없는 것: 학습한 YOLO 가중치(`vision/models/`), ACT 손 건네기 정책(Studio 안), 녹화·학습 데이터, 실행 기록(`runs/`).

로봇 없이 흐름만 보기

```bash
BALL_MOCK=1 python -m agent.cli --scenario   # 가짜 작업대로 시연 순서 실행(OVMS 필요)
python -m agent.selftest --offline           # 그래프 회귀 시험(OVMS 없이)
```

## 문서

| 문서 | 내용 |
|---|---|
| `CONTRACT.md` | 모듈 사이 인터페이스(`detect_objects`, `run_policy`, `run_command` 이벤트 형식)와 폴더 분담 |
| `SETUP.md` | OVMS·드라이버·모델 설치와 운영 기록, GPU 멈춤 판독 |
| `agent/README.md` · `vision/README.md` · `robot/README.md` · `ui/README.md` | 모듈별 실행법·실측·작업 기록 |
| `robot/COORD_PICK.md` · `robot/GRASP_ACT.md` | 좌표 집기 설계·보정 · ACT 집기 시도와 분석 |
| `docs/system_guide.md` | 장치 배치 이유와 실측(응답 시간·정확도·메모리) |

## 개발 방식

폴더를 나눠 맡은 여러 Claude Code 세션(총괄·인프라·에이전트·로봇·비전·UI·문서)과 함께 개발했다. 폴더 소유와 인터페이스는 `CONTRACT.md` 에 있고, 각 폴더의 README 는 그 세션의 작업 기록이라 날짜별 메모가 많다. 코드 식별자의 `red_*`·`blue_*` 는 각각 분홍·초록이다(개발 중 색을 바꿨고 식별자는 그대로 뒀다).
