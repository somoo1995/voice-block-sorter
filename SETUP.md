# ball-sorter 로컬 AI 기반 — 설치·실측 기록

> 인프라 세션, 2026-10-21 16:20~17:50 실측. 장비 = 젠북(`robot-pc`, Core Ultra 9 185H · Arc iGPU · NPU 3720 · RAM 30GB · Ubuntu 24.04).
> 외부 API 없이 LLM·임베딩·리랭크·STT·TTS 전부 이 노트북 안에서 돈다. OVMS 는 `127.0.0.1` 에만 열려 있어 다른 PC 에서는 안 보인다.
> 10-21 밤(20:10~21:00) 추가: §7.6 NPU 동시 사용, §7.5 시연 당일 메모리 여유, §8-13 GPU HANG(xpu 학습과 겹침), §11 오프라인 점검.
> 10-21 21:15 추가: §7.7 KV 캐시 1GB 고정(의도 판별 1.6 → 0.6초), `config.json` 반영.
> 10-22 00:40 추가: §1 시연 날 한 줄 기동 `start_demo.sh`, §7.8 30분 연속 시험(실패 0·GPU HANG 0 → 시연 당일 안정).
> 10-22 밤(22:01 재부팅 뒤, 22:04~22:27 측정 구간) 추가: §7.9 자원표(대기·피크), §7.10 ACT 정책 장치별 추론 지연(로봇 없이), §7.11 더 올릴 모델 후보, §8-14 10-22 GPU HANG 재현 조건·절전 복귀 실패, §1 `start_demo.sh` 응답 확인·예열.
> 10-23 13:2x 추가: §8-14 오늘 네 사건표·펜스 줄 정정(i915 20초 요청 감시)·렌더 가설 철회, §1 무응답 감시(커널 펜스 줄 → 바로 KILL, 1토큰 60초)·StartLimitBurst 5.

## 0. 결론

| 항목 | 결정 | 근거 (§7 실측) |
|---|---|---|
| LLM | **`qwen3-4b` (GPU) 권장** | 혼자 쓸 때 첫 토큰 0.13초·초당 25.9토큰, 의도 JSON 5/5, tools 정상. 8B 는 초당 13.9토큰·의도 3/5·메모리 2배 |
| LLM 개수 | **GPU 에 LLM 은 하나만** | 8B+4B 를 한 서버에 같이 올린 상태에서 GPU HANG 2회 → OVMS 통째로 죽음. LLM 하나 구성은 동시 부하 90~120초 실패 0 |
| STT | **`whisper` = NPU** | 1.56초, CER 0.074. GPU 는 0.90초로 빠르지만 정확도 같고 GPU 를 LLM 과 나눠 쓴다 |
| 임베딩 | CPU | 질문 1문장 25~58ms |
| 리랭크 | **GPU 로 바꿈** (요청은 CPU) | CPU 는 문서 8개 5.8초 → GPU 0.36~0.44초, 순위 같음. CPU 로 되돌리기 = `config-cpu-rerank.json` |
| TTS | Supertonic-3 한국어, CPU | 동작함. 단 **`SUPERTONIC_INTRA_OP_THREADS=4`** 를 줘야 짧은 문장 0.86초(기본값은 2.1초) |
| KV 캐시 | **고정 1GB** (원본은 동적) | 보고·답 뒤 의도 판별 1.60 → 0.63초. 동적 캐시(31MB)에선 시스템 프롬프트의 prefix 캐시가 밀려나 매번 처음부터 계산했다 (§7.7) |
| 메모리 | 권장 구성 OVMS ≈ 7GB | 권장 구성을 올리고 부하를 준 뒤 `free -g` used 14 · available 16 (§7.5). KV 캐시 1GB 로 OVMS +1.3GB. 시연 당일 여유 약 11GB(추정) |
| NPU 동시 사용 | **장치 배치 그대로** | Whisper(NPU)+YOLO(NPU 10fps)+LLM(GPU) 동시 실패 0. STT 1.56→2.22초, YOLO 14→27ms, LLM 29→28 tok/s (§7.6). 비전 미리보기는 NPU 10fps 이하 |
| GPU 공유 | **xpu 학습 중엔 OVMS 를 내린다** | 비전 xpu 학습과 겹쳐 OVMS LLM 이 GPU HANG → SEGV (20:29, §8-13). OpenVINO 끼리(별도 프로세스 YOLO GPU 30fps + LLM, 3분)는 실패 0 |
| 연속 시험 | **시연 당일 안정** | 30분 동안 에이전트 명령 1,060개(1→5·scene·plan·질문) + STT 131 + TTS 104 + YOLO NPU 3,438회 실패 0, GPU HANG 0, 5분 단위 지연 증가 없음, OVMS 메모리 10분 뒤 평평 (§7.8) |
| 오프라인 | OVMS 쪽은 막음 | `start_ovms.sh` 에 `HF_ENDPOINT` 차단. 남은 것 = 비전의 Ultralytics `YOLO_OFFLINE` (§11) |
| 자원 여유 (10-22 밤) | **메모리는 넉넉, 막히는 건 GPU 시간** | 시연 피크(LLM+리랭크+STT+TTS+YOLO+ACT 동시)에도 가용 16.8GB·스왑 0. LLM 이 답을 만드는 동안 GPU 98~100%, STT 동안 NPU 74~100% (§7.9) |
| 정책 장치 (10-22 밤) | **CPU 로 옮기려면 AsyncExecution 과 같이** | ACT 추론 CPU 60ms(시연 피크 102ms) → 33ms 안에 못 듦. NPU 는 13~28ms 지만 Studio 컨테이너가 NPU 를 못 본다. GPU 실물 421회 중앙 33ms (§7.10) |
| GPU HANG (10-22) | **재현 조건 미확정 — LLM·ACT 동시 사용과 안 맞음** | GPU 사건 5건(10:18 GuC·16:11 HANG·펜스 만료 3회 시각) 때 LLM 생성 없음(1.6~10분 전 끝), 직전 ACT 추론 3.7~11.7분 전, Studio GPU 작업 없음. 같은 부팅 2.3일·절전 복귀 14회 뒤. 시연 날 아침 재부팅 권장 (§8-14) |
| GPU 멈춤 (10-23) | **막는 수 없음 → 빨리 되살리기** | 4건(GPU HANG 2·i915 20초 요청 감시 끊김 2). 요청 종류·렌더 동시·GPU 잠듦·부팅 나이와 무관, 기록 구간 GPU 요청 약 500번에 2번 → 렌더 동시 가설 철회. 감시가 커널 펜스 줄로 바로 KILL, 1토큰 물음 60초, StartLimitBurst 5 (§8-14, §1) |

## 1. 띄우기·끄기 (한 줄 명령)

**시연 날은 이것 하나** — `start_demo.sh`(10-21 밤): OVMS 확인(없으면 `start_ovms.sh`, 4모델 준비까지 최대 120초 대기) → 대시보드(`ui/start_ui.sh`) → 점검표(`robot.preflight`, 시연 기준) → 브라우저 주소.

```bash
~/dev/projects/ball-sorter/start_demo.sh                    # 실제 로봇 모드(BALL_ROBOT=1). 기동만으로는 팔이 안 움직인다 — 명령을 내려야 움직임(사용자 입회 때만)
~/dev/projects/ball-sorter/start_demo.sh --mock             # 가짜 작업대(BALL_MOCK=1), 하드웨어 없이 리허설. 밤엔 앞에 UI_TTS_OUT=silent
~/dev/projects/ball-sorter/start_demo.sh --restart          # 대시보드가 다른 모드로 떠 있을 때 고른 모드로 다시 띄움(OVMS 는 그대로)
```

- 이미 떠 있는 OVMS·대시보드는 **건드리지 않고 상태만** 보여 준다: OVMS 4모델·구성 파일·메모리·가용 메모리, 이번 부팅과 지금 OVMS 기동 뒤의 커널 GPU HANG 건수·자동 재시작 횟수, 대시보드 모드. 응답 확인 뒤 무응답 감시(`ball-ovms-watch`, 아래)가 안 떠 있으면 띄운다. 떠 있는 대시보드의 모드가 고른 것과 다르면 `--restart` 명령을 알려 주고 그대로 둔다.
- **응답 확인·예열(10-22 밤 추가)**: 4모델이 AVAILABLE 이어도 네 모델에 한 번씩 실제로 묻는다(LLM 1토큰·임베딩·리랭크·`ui/samples/tts_F1_01.wav` 받아쓰기). 하나라도 실패하면 거기서 멈추고 `systemctl --user stop ball-ovms; ./start_ovms.sh` 를 안내, 종료 코드 1. 10-22 16:33~17:00 OVMS 는 모델 상태가 AVAILABLE 인 채 27분 동안 답을 안 했다(§8-14). 재부팅 뒤 첫 임베딩 요청은 53초 걸렸다(§7.9) — 첫 기준서 질문 대신 여기서 치른다. 데워진 뒤엔 1.8초(받아쓰기 1.6초).
  - **구조화 출력 1번(10-23 추가)**: LLM 1토큰 다음에 의도 판별과 같은 꼴(`response_format` json_schema strict·생각 끔)의 작은 스키마 요청을 한 번 보내고 시간을 찍는다. OVMS 를 띄운 뒤 첫 json_schema 요청은 문법 준비로 약 9초 걸린다(10-23 09:57 재부팅 → 10:06:39 첫 의도 판별이 9초 뒤에야 스케줄돼 5초 제한에 걸림 → "명령 아님", 그다음 같은 경로 2.0초). 스키마마다가 아니라 프로세스마다 한 번이다(데워진 뒤 새 스키마 0.70초·같은 스키마 0.77초·또 다른 새 스키마 0.67초, 10-23 10:09 curl). 자동 재시작(Restart=on-failure)으로 새로 뜬 OVMS 는 이 예열을 안 거친다(10-23 10:13 재시작 뒤 첫 json_schema 5.96초) → **10-23 10:16 부터 감시(`ovms_watch.py`)가 새 프로세스에 처음 묻는 것을 같은 WarmIntent 요청으로 바꿔 메웠다**(그 뒤는 1토큰, 구조화 출력은 데우기에만 쓰고 "살았나" 판단엔 안 씀).
- **카메라 고정값(10-23 추가, 맨 먼저)**: 재부팅·USB 다시 꽂기 때마다 자동 노출·화이트밸런스·초점이 풀린다(10-23 09:32·09:39·09:58 손으로 넣음). `camera_ctrls.conf`(**값의 정본**, 아래 표는 옮겨 적은 것)를 읽어 `v4l2-ctl -c` 로 단계마다 따로 넣고(auto 를 먼저 꺼야 값이 먹음) 제어마다 `--get-ctrl` 로 다시 읽어 한 줄로 찍는다 — 같으면 "맞음", 다르면 "⚠ 주의 이름: 제어(바람 값·읽음 값)". 카메라가 없으면 건너뜀. Studio 가 카메라를 쥐고 있어도 `-c` 는 된다(10-23 10:25 총괄 확인). 안 맞아도 기동은 계속(종료 코드에 안 넣음).

  | 이름 | 장치(`/dev/v4l/by-id/`) | 단계1(auto 끄기) | 단계2(값) |
  |---|---|---|---|
  | 위 | `usb-ZVCAI_USB_Camera_zvcai231100001-video-index0` | auto_exposure=1, exposure_dynamic_framerate=0, white_balance_automatic=0, focus_automatic_continuous=0, power_line_frequency=2 | exposure_time_absolute=200, white_balance_temperature=4600, focus_absolute=130 |
  | 손목 | `usb-icSpring_icspring_camera_20240307110322-video-index0` | auto_exposure=1, white_balance_automatic=0, power_line_frequency=2 | white_balance_temperature=4600, exposure_time_absolute=250, gain=0 (10-23 10:08 총괄 9조합 비교, 오늘 집기 녹화도 이 값) |

  손으로 한 대만: `v4l2-ctl -d /dev/v4l/by-id/<장치> -c <단계1>` 뒤 `-c <단계2>`, 확인 `--get-ctrl=<이름,…>`.
- 셸의 `UI_* BALL_* TTS_*` 등은 `ui/start_ui.sh` 가 그대로 유닛에 넘긴다. 실제 모드는 `BALL_MOCK` 을, `--mock` 은 `BALL_ROBOT` 을 지우고 넘긴다.
- 종료 코드: OVMS·대시보드가 안 뜨면 1. 실제 모드는 점검에 "안됨"이 하나라도 있으면 1. `--mock` 은 팔·카메라·Studio·정책 항목(가짜 작업대에선 안 씀)이 안됨이어도 0.
- 대시보드는 `127.0.0.1:7870` 에만 열린다. 다른 PC 에서 보려면 `ssh -L 7870:127.0.0.1:7870 robot-pc` 뒤 `http://127.0.0.1:7870`.
- 시험(10-21 22:17~22:20, `--mock`): 둘 다 떠 있음 → 상태만 1초 · `--restart` → 대시보드 다시 띄워 응답까지 3.3초 · 대시보드 꺼짐 → 띄워 응답까지 3.4초 · 대시보드 가짜 모드에 실제 모드 요청 → 그대로 두고 경고 · (10-22 00:01) OVMS·대시보드 둘 다 꺼짐 → 4모델 준비·대시보드 응답·점검표까지 11.7초.

```bash
~/dev/projects/ball-sorter/start_ovms.sh                                              # 권장 구성(4B) 띄우기
systemctl --user stop ball-ovms                                                        # 끄기
~/dev/projects/ball-sorter/start_ovms.sh ~/dev/projects/ball-sorter/config-8b.json     # 8B 구성으로 (먼저 끄고)
curl -s http://127.0.0.1:8100/v3/models                                                # 떠 있는 모델 목록
tail -f ~/dev/projects/ball-sorter/logs/ball-ovms.log                                  # 로그
```

- `start_ovms.sh` 는 `systemd-run --user` 로 `ball-ovms` 유닛을 만든다. 이미 떠 있으면 아무것도 안 하고 끝난다.
- **재부팅하면 자동으로 안 뜬다**(일회성 유닛). 필요하면 위 첫 줄을 다시 실행.
- 메모리 보호: 상한(`MemoryMax`)은 기동 시점 가용 메모리 − 3GB 로 자동 계산, `OOMScoreAdjust=1000`. 모자라면 OVMS 만 죽고 Studio·학습·다른 작업은 안전하다(§8-1 사고 뒤 넣은 장치).
- 실행 중에 `config.json` 을 고치면 OVMS 가 1초 안에 다시 읽는다(모델 추가·제거). **LLM 을 두 개 넣지는 말 것**(§8-3).
- 첫 적재는 컴파일 때문에 느리다(Whisper NPU 약 3분 25초, 8B 36초). 이후엔 `.ov_cache` 덕에 전체 7~14초.
- 오프라인: 기동에 인터넷이 필요 없다. 서버 안의 HF 내려받기 모듈이 혹시 불려도 `HF_ENDPOINT=http://127.0.0.1:9`(닫힌 로컬 포트)로 가서 바로 실패한다(§11).
- **죽으면 3초 뒤 혼자 다시 뜬다**(10-23 09:38 부터 `Restart=on-failure`·`RestartSec=3`, 09:35 GPU HANG → SEGV 뒤 넣음). 다시 뜨는 동안(3초 + 적재 약 8초)은 `ActiveState=activating` — `start_ovms.sh`·`start_demo.sh` 는 이것을 "재시작 중 → 기다림"으로 본다. `systemctl --user stop` 으로 끈 것은 다시 안 뜬다.
- **5분 안에 기동이 5번을 넘으면 더 안 띄운다**(`StartLimitIntervalSec=300`·`StartLimitBurst=5`, GPU 가 망가진 채 끝없이 다시 뜨지 않게). 10-23 10시에 3번으로 넣었다가 13:2x 에 5번으로 올렸다 — 12:49·12:54 처럼 사건이 5분 남짓 사이에 몰려도 시연 중 "다시 안 뜸"이 되지 않게. 둘 다 `start_ovms.sh` 로 띄우는 다음 기동부터 먹는다(그 전에 떠 있던 유닛은 systemd 기본 10초에 5번). 시험(감시 단독 시험 ⑥): 기동 5번(처음 + 자동 4번) 뒤 6번째는 `Start request repeated too quickly` 로 거절. 이때는 `journalctl -k -b | grep -iE "gpu hang|fence"` 로 원인을 본 뒤 첫 줄을 다시 실행. 재시작 횟수 = `systemctl --user show ball-ovms -p NRestarts --value`(`start_demo.sh` 의 HANG 줄에도 나옴).
- **무응답 감시 `ovms_watch.py`**(유닛 `ball-ovms-watch`, `start_demo.sh` 가 띄움): 죽지 않고 답만 안 하는 꼴(10-22 16:33 펜스 만료 뒤 27분, 23:08 30초, 10-23 10:12·12:49)을 `systemctl --user kill -s KILL ball-ovms` 로 죽여 위 Restart 에 넘긴다. 10-23 13:19 부터 둘로 본다:
  - **커널 줄**: `journalctl -k -f` 에 `Fence expiration time out i915-…:ovms[스레드]:…!`(i915 요청 감시가 20초 넘은 OVMS GPU 요청을 끊음, §8-14)이 뜨고 그 스레드가 지금 OVMS 프로세스의 것이면 **바로 KILL**. 12:49 였다면 15초 일찍 KILL. 남의 프로세스 줄·이미 KILL 한 프로세스의 둘째 줄은 건드리지 않는다. `GPU HANG` 줄은 기록만(OVMS 가 스스로 SEGV 로 죽는다). user 는 `adm` 그룹이라 sudo 없이 읽힌다. 예외: 10-22 17:36 은 펜스 뒤에도 답했다 — 그런 꼴이면 필요 없는 재시작 1번.
  - **1토큰 물음(안전망)**: 60초마다 LLM 에 1토큰을 15초 제한으로 묻고, 한 번 답이 없으면 20초 뒤 다시, 2번 연속 답이 없으면 KILL. 20초 → 60초로 띄운 까닭 = 물음도 GPU 요청이라 멈춤을 부를 수 있다(10-23 12:48:40 에 멈춘 요청이 감시 물음, 그날 기록 구간 OVMS 연산 시작 501번 중 487번이 감시 차례).
  - 세지 않는 때: 유닛이 active 가 아닐 때(재시작 중·멈춤)와 새 프로세스가 처음 답할 때까지(최대 180초). KILL 뒤엔 간격을 안 기다리고 재시작을 2초마다 보고, 적재 중(연결 거절·404 — GPU 안 씀)엔 예열을 2초마다 다시 묻는다(전엔 20초 차례라 다시 답하기가 최대 20초 늦었다: 10-23 10:13 은 적재 7초인데 기동 22초 뒤에 첫 답, 12:54 는 적재 20초인데 24초 뒤).
  - OVMS 를 띄우지는 않는다 — stop 으로 내린 OVMS(xpu 학습 등)는 그대로. 기록 `logs/ball-ovms-watch.log`(사건마다 한 줄, 30분마다 요약). 끄기 `systemctl --user stop ball-ovms-watch`. 값 바꾸기 = 환경변수 `WATCH_INTERVAL_S`(60)·`WATCH_FAIL_INTERVAL_S`(20)·`WATCH_WARM_RETRY_S`(2)·`WATCH_KMSG_CMD`(빈 값이면 커널 줄 안 봄), 예 `WATCH_INTERVAL_S=30 ./start_demo.sh`(감시가 안 떠 있을 때). 단독 시험 `bench/ovms_watch_test.sh`(§10).

## 2. 주소·모델 이름 (CONTRACT §4 값 확정)

| 환경변수 | 값 |
|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:8100/v3` (`/v1` 도 같은 동작) |
| `LLM_MODEL` | `qwen3-4b` (8B 구성을 띄웠을 때만 `qwen3-8b`) |
| `EMB_MODEL` | `qwen3-embedding` |
| `RERANK_MODEL` | `qwen3-reranker` |
| `STT_MODEL` | `whisper` |
| API 키 | 없음(인증 꺼 둠). OpenAI·LangChain 클라이언트에는 아무 문자열(예: `unused`) |

| 포트 | 용도 |
|---|---|
| `127.0.0.1:8100` | OVMS REST (본 서버, 유닛 `ball-ovms`) |
| `127.0.0.1:8101` | 측정용 임시 서버(유닛 `ball-ovms-probe`). 평소엔 꺼져 있음 |
| gRPC | 끔 |

| 엔드포인트 | 쓰임 |
|---|---|
| `POST /v3/chat/completions` | LLM (스트리밍·tools·response_format) |
| `POST /v3/embeddings` | 임베딩 (1024차원, 정규화됨) |
| `POST /v3/rerank` | 리랭크 (Cohere 형식: `query`, `documents` → `results[].index/relevance_score`) |
| `POST /v3/audio/transcriptions` | STT (multipart: `file`, `model=whisper`, `language=ko`) |
| `GET /v3/models`, `GET /v2/health/ready` | 상태 확인 |

## 3. 호출 요령 — 꼭 지킬 것

1. **thinking 끄기**: 요청 본문에 `"chat_template_kwargs": {"enable_thinking": false}`. 안 끄면 `message.reasoning_content` 에 300~400토큰을 먼저 생각해서 답이 16~37초 걸린다(실측). LangChain `ChatOpenAI` 는 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
2. **도구 호출**: graph 에 `tool_parser: "hermes3"`(Qwen3 용)·`reasoning_parser: "qwen3"` 설정됨. 응답은 `finish_reason: "tool_calls"`, `message.tool_calls[0].function.arguments` 가 JSON 문자열, `content` 는 빈 문자열.
3. **구조화 출력**: `response_format: {"type": "json_schema", "json_schema": {...}}` 동작(10회 모두 파싱 성공). 8B 는 줄바꿈 넣은 JSON 을 낸다(파싱엔 문제 없음).
4. **임베딩**: 질문 쪽에만 `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" + 질문` 을 붙인다. 문서는 그대로. pooling(LAST)은 OVMS 가 자동 감지.
5. **리랭크는 템플릿 필수** — 원문 그대로 보내면 엉뚱한 문서(손 건네기 문서)가 0.91 로 1위가 된다. 모델 카드 형식:
   ```python
   PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
             'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
   SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
   query = f"{PREFIX}<Instruct>: {TASK}\n<Query>: {질문}\n"
   documents = [f"<Document>: {d}{SUFFIX}" for d in 문서들]
   ```
   `agent/llm.py` 는 이미 이렇게 하고 있다(읽어서 확인만 함).
6. **STT**: Supertonic 이 만든 44.1kHz wav 를 그대로 보내도 된다(서버가 변환). NPU 판은 음성 길이와 거의 무관하게 약 1.5초.
7. **여러 세션이 같이 쓰면 느려진다**: LLM 은 continuous batching 으로 요청을 섞어 처리한다. 다른 세션 평가와 겹쳤을 때 4B 가 초당 25.9 → 8.2토큰까지 떨어지는 것을 봤다. 시연 중에는 다른 평가를 돌리지 말 것.

## 4. 파일·모델 경로 (프로젝트 = `~/dev/projects/ball-sorter`)

| 경로 | 내용 | 크기 |
|---|---|---|
| `ovms/` | OVMS **2026.4.1** Ubuntu 24 바이너리(`python_off`, sha256 확인). OpenVINO 2026.4.1·GenAI 2026.4.1 내장 | 457MB |
| `models/OpenVINO/Qwen3-4B-int4-ov/` | LLM, GPU. `graph_kv1.pbtxt`=KV 캐시 고정 1GB(사용 중, §7.7), `graph.pbtxt`=동적 캐시(원본) | 2.2GB |
| `models/OpenVINO/Qwen3-8B-int4-ov/` | LLM, GPU (`config-8b.json`) | 4.6GB |
| `models/OpenVINO/Qwen3-Embedding-0.6B-int8-ov/` | 임베딩, CPU | 615MB |
| `models/OpenVINO/Qwen3-Reranker-0.6B-seq-cls-fp16-ov/` | 리랭크. `graph.pbtxt`=CPU, `graph_gpu.pbtxt`=GPU(사용 중) | 1.2GB |
| `models/OpenVINO/whisper-large-v3-turbo-int8-ov/` | STT. `graph.pbtxt`=NPU(사용 중), `graph_gpu.pbtxt`=GPU(비교용) | 792MB |
| `models/supertonic-3/` | TTS ONNX(Supertone/supertonic-3, OpenRAIL-M) | 386MB |
| `config.json` | **권장 구성**: 4B(KV 캐시 1GB) + 임베딩(CPU) + 리랭크(GPU) + Whisper(NPU) | |
| `config-8b.json` | 위와 같고 LLM 만 8B (KV 캐시는 동적 그대로, 미시험) | |
| `config-cpu-rerank.json` | 리랭크를 CPU 로 (요청 원안, KV 캐시 동적) | |
| `config-all.json` | 8B+4B+임베딩+리랭크(CPU)+Whisper — **기록용, 쓰지 말 것**(GPU HANG) | |
| `start_ovms.sh` | 기동 스크립트 (§1) | |
| `pull_models.sh` | 모델 다시 받기 + graph 생성(`ovms --pull`, HF 공식 변환본) | |
| `.ov_cache/` | 컴파일 캐시. 지워도 되지만 다음 적재가 다시 수 분 걸린다 | 9.9GB |
| `logs/` | `ball-ovms.log`(현재), `ovms-run*.log`(사고 기록), `pull.log` | |
| `bench/` | 측정 스크립트와 결과(`bench/out/*.json`, TTS wav) | |

## 5. 에이전트용 venv — `~/dev/projects/ball-sorter/.venv`

Python 3.12.3, `~/.local/bin/uv` 로 만듦. 추가 설치: `~/.local/bin/uv pip install --python .venv/bin/python <패키지>`.

| 패키지 | 버전 |
|---|---|
| openvino / openvino-genai / openvino-tokenizers | 2026.4.1 / 2026.4.1.0 / 2026.4.1.0 (`CPU·GPU·NPU` 보임) |
| langchain / langchain-core / langchain-openai / langchain-chroma / langgraph | 1.4.6 / 1.6.12 / 1.7.2 / 1.1.0 / 1.2.16 |
| chromadb | 1.5.9 |
| ultralytics | 8.4.174 |
| torch / torchvision | **2.14.1+cpu / 0.29.1+cpu** (CPU 판. 기본 PyPI 판은 CUDA 라 수 GB) |
| opencv-python | 5.0.0.93 |
| sounddevice / soundfile | 0.5.6 / 0.14.0 |
| gradio | 6.30.0 |
| onnxruntime | 1.31.0 (CPU) |
| supertonic | 1.3.1 (Supertonic-3 SDK) |
| huggingface-hub / requests / numpy | 2.2.0 / 2.34.2 / 2.5.2 |

- 시스템 패키지 **`libportaudio2`** (apt, 19.6.0) 를 설치했다 — 없으면 `import sounddevice` 가 `PortAudio library not found` 로 실패한다.
- 수업 venv `~/openvino-project/.venv` 는 건드리지 않았다.

## 6. TTS — Supertonic-3 (CPU, onnxruntime)

```python
import os
os.environ.setdefault("SUPERTONIC_INTRA_OP_THREADS", "4")   # 이 줄이 속도를 2.5배 바꾼다 (import 전에)
os.environ.setdefault("SUPERTONIC_INTER_OP_THREADS", "1")
from supertonic import TTS
tts = TTS(model="supertonic-3", model_dir="/path/to/ball-sorter/models/supertonic-3", auto_download=False)
wav, dur = tts.synthesize("빨간 공 세 개를 옮겼습니다.", voice_style=tts.get_voice_style("F1"), lang="ko")
tts.save_audio(wav, "out.wav")   # 44.1kHz
```

- 목소리 10종(`F1~F5`, `M1~M5`). 기본 스텝 8. 로드 약 2초(첫 실행은 다운로드 포함 19.5초), 프로세스 최대 RSS 571MB.
- 스레드를 안 주면 onnxruntime 이 전체 코어(P·E·LP 섞인 22스레드)를 써서 오히려 느리다. `ui/tts.py` 는 UI 세션 소관이라 고치지 않았다 — 환경변수만 주면 코드 수정 없이 적용된다.
- MeloTTS 대안은 필요 없어서 설치하지 않았다.

## 7. 실측표

조건: VMware VM 은 16:45경 사용자 지시로 끔(그 전 측정은 표시). Physical AI Studio 컨테이너·Docker Desktop·Chrome·VS Code 는 계속 켜져 있었다. 다른 세션(에이전트 평가·비전 학습)이 같은 시간에 OVMS·GPU 를 쓴 측정은 따로 표시했다. "혼자" = 측정 구간 동안 OVMS 로그에 동시 요청이 한 줄도 없던 회차만 골라 낸 값(`bench/bench_speed_clean.py`).

### 7.1 LLM (프롬프트 49토큰 → 생성 약 170~210토큰, temperature 0, thinking 끔)

| 항목 | qwen3-8b | qwen3-4b |
|---|---|---|
| 첫 토큰 시간(혼자, 3회 중앙값) | 0.232초 | **0.133초** (VM 켜져 있을 때 0.162초) |
| 초당 생성 토큰(혼자) | 13.9 | **25.9** (VM 켜져 있을 때 20.5) |
| 초당 생성 토큰(다른 세션과 겹침) | 9.7~10.0 | 8.2 |
| 의도 JSON 정답(5문장, json_schema) | **3/5** | **5/5** (3회 모두) |
| 의도 1건 지연 중앙값 | 1.95~2.24초 (겹친 상태) | 0.68~0.78초 (혼자) · 1.55초 (겹침) |
| tools 1건 (`pick_balls` 파랑 2개) | O, 4.1~4.4초 (겹침) | O, 1.8~2.2초 |
| thinking 기본값(안 끔) | 280~324토큰, 29~37초 | 377~400토큰, 16~32초 |
| 상주 메모리(단독 적재·추론 후) | GPU 5.46 + CPU 0.32 ≈ 5.8GB | GPU 2.80 + CPU 0.21 ≈ 3.0GB |

의도 문장별(시스템 프롬프트 = `bench/bench_llm.py` 의 `INTENT_SYSTEM`, 예시 없이):

| 문장 | 정답 | 4B | 8B |
|---|---|---|---|
| 불량품 골라내 줘 | defect, null | O | O |
| 빨간 볼 옮겨 줘 | red, null | O | **X** (count 1) |
| 파란 공 다 담아 | blue, null | O | **X** (count 2) |
| 빨간 거 하나 내 손에 줘 | handoff, 1 | O | O |
| 결과 알려 줘 | report, null | O | O |

8B 오답 2건은 모두 "개수를 말하지 않으면 null" 규칙 위반이다. 예시(few-shot)를 넣으면 달라질 수 있다 — 에이전트의 `intent_examples.py` 기준 평가는 에이전트 세션 몫.

### 7.2 임베딩·리랭크 (질문 "찢어진 공은 어떻게 처리해야 해?", 작업 기준 문서 8개, 정답 = 불량 판정 문서 #0)

| 항목 | 값 |
|---|---|
| 임베딩 질문 1문장 (CPU) | 25~58ms (템플릿 붙이면 54~86ms) |
| 임베딩 문서 8개 한 번에 (CPU) | 0.52초 (혼자) ~ 1.49초 (겹침) |
| 임베딩 1위 | #7 "오염된 공은 불량으로 본다"(cos 0.501), 정답 #0 은 0.472로 2위 — 둘 다 불량 문서 |
| 리랭크 CPU, 원문 | 5개 1.3초 · 8개 2.5초, **1위 오답**(#4, 0.91) |
| 리랭크 CPU, 템플릿 | 5개 3.7초 · 8개 5.8초, 1위 정답(#0, 0.958) |
| **리랭크 GPU, 템플릿 (사용 중)** | **5개 0.26~0.28초 · 8개 0.36~0.44초**, 1위 정답(#0, 0.961) |

### 7.3 STT — Supertonic 이 읽은 wav 6개를 Whisper 로 받아쓰기 (각 3회 중앙값)

| 항목 | NPU (`whisper`, 사용 중) | GPU (`whisper-gpu`, 비교용) |
|---|---|---|
| 지연 중앙값 | 1.56~1.58초 | 0.90초 |
| 평균 CER | 0.074 | 0.074 |
| 글자 완전 일치 | 4/6 | 4/6 |
| 상주 메모리 | NPU 버퍼 2.0GB + CPU 0.1GB | GPU 1.9GB + CPU 0.9GB |
| 적재(캐시 후 / 첫 컴파일) | 5초 / 약 3분 25초 | 10초 |

| 원문 | 받아쓰기 (NPU·GPU 같음) |
|---|---|
| 불량품 골라내 줘 | 불량품 골라내줘. |
| 빨간 볼 옮겨 줘 | 빨간 볼 옮겨줘. |
| 파란 공 다 담아 | **파랑궁** 다 담아. |
| 빨간 거 하나 내 손에 줘 | 빨간 거 하나 내 손에 줘. |
| 결과 알려 줘 | 결과 알려줘. |
| 빨간 공 세 개와 파란 공 두 개를 옮겼고, 불량품은 한 개 골라냈습니다. | 빨간 공 **3개**와 파란 공 **2개**를 옮겼고 불량품은 **1개** 골라냈습니다. |

실제 오류는 "파란 공 → 파랑궁" 1건(TTS 발음의 연음일 수 있음). 마지막 줄은 숫자 표기만 다르다 — 의도 분류 전에 띄어쓰기·숫자를 정규화하면 된다.

### 7.4 TTS — Supertonic-3, 목소리 F1, 짧은 명령 5문장 + 보고 1문장(6.3초 분량)

| 스레드 | 스텝 | 짧은 문장 지연 중앙값 | 긴 문장 | RTF 중앙값 |
|---|---|---|---|---|
| 기본값(자동) | 8 | 2.15초 | 6.0초 | 1.11 |
| **4** | **8** | **0.86초** | 2.6초 | **0.46** |
| 4 | 5 | 0.60초 | 3.1초 | 0.33 |
| 6 | 8 | 1.03초 | 2.2초 | 0.53 |
| 8 | 8 | 1.30초 | 3.7초 | 0.70 |

다른 프로세스 부하에 따라 ±30% 흔들린다. 권장 = 스레드 4, 스텝 8.

### 7.5 메모리 — `free -g`

| 상태 | used | free | shared | buff/cache | available |
|---|---|---|---|---|---|
| OVMS 끔 · VM 끔 (17:10) | 6 | 20 | 1 | 6 | 24 |
| **요청 모델 전부** (8B+4B+임베딩+리랭크 CPU+Whisper NPU) 적재 직후 | 16 | 6 | 11 | 20 | 14 |
| **권장 구성** (4B+임베딩+리랭크 GPU+Whisper NPU) 동시 부하 120초 뒤 | 14 | 8 | 8 | 17 | 16 |
| 8B 구성 동시 부하 90초 뒤 | 16 | 3 | 10 | 22 | 14 |
| 참고: VM 켜진 상태에서 4B+임베딩+리랭크 CPU+Whisper NPU | (MemAvailable 7.8GB) | | | | |

Arc iGPU·NPU 가 쓰는 메모리는 시스템 RAM 의 `shared`(shmem)로 잡힌다. GPU 가 쓸 수 있는 전체 = 28.4GiB(OpenVINO `GPU_DEVICE_TOTAL_MEM_SIZE`).

**시연 당일 여유(추정)**: 10-21 20:56 지금 상태(권장 구성 OVMS cgroup 11.7GB(GPU·NPU 버퍼 포함)·대시보드 0.8GB·Studio 컨테이너 대기 0.3GB, 학습·VM 없음)에서 available 19.0GB. 여기서 Studio 가동분(컨테이너 cgroup 최대 실측 6.2GB, 페이지 캐시 포함이라 상한)과 에이전트+비전 프로세스 약 1GB<!-- TODO: 확인 필요 — 에이전트 프로세스 실측 -->를 빼면 **약 12GB 여유**. 21:15 KV 캐시를 1GB 로 고정한 뒤엔 OVMS cgroup 13.0GB(부하 뒤 14.3GB)·available 18.0GB 라 **약 11GB 여유**. VM 을 켜면 약 6GB. 비전 xpu 학습은 약 13GB 를 쓴다(20:22 학습 중 available 6.1GB) — 시연 중 금지(§8-13). 스왑 4GB 는 어제 밀려난 오래된 페이지(Studio 1.85GB 등)라 여유 계산에서 뺐다.

모델별 단독 측정(`bench/mem_probe.py`, 한 모델만 띄워 첫 추론 뒤):

| 모델 | 장치 | OVMS 상주(CPU + GPU/NPU) | 시스템 가용 감소 | 적재(캐시 후) | 첫 추론 |
|---|---|---|---|---|---|
| qwen3-embedding | CPU | 0.59GB | 0.6GB | 3초 | 2.3초 |
| qwen3-reranker | CPU | 1.80GB | 1.6GB | 2초 | 3.0초 |
| whisper | NPU | 0.13 + 2.04GB | 2.0GB | 5초 | 2.2초 |
| whisper-gpu | GPU | 0.94 + 1.88GB | 3.1GB | 10초 | 7.7초 |
| qwen3-4b | GPU | 0.21 + 2.80GB | 2.9GB | 10초 | 0.9초 |
| qwen3-8b | GPU | 0.32 + 5.46GB | 4.9GB | 36초(첫 컴파일, 이때 최대 16.7GB·시스템 가용 9.2GB까지 내려감) | 3.3초 |

### 7.6 NPU 동시 사용 — Whisper(NPU) + YOLOv8n(NPU) + LLM(GPU) (10-21 20:45~20:56)

`bench/npu_concurrency.py`. 구간마다 20초(마지막 구간은 3분 한 번 더). Whisper = OVMS `/v3/audio/transcriptions`(§7.3 wav 6개를 번갈아), YOLO = 별도 프로세스에서 Ultralytics predict(`vision/models/synth_int8_openvino_model` = yolov8n 기반 5클래스 INT8, 사진 `vision/_test/scene.jpg`, 읽기만), LLM = `qwen3-4b` 스트리밍 160토큰. 다른 세션이 OVMS 와 주고받는 중이거나 비전 학습이 돌면 끝날 때까지 기다렸다 돌렸다 → 아래 전 구간 오염 없음. 20:13~20:20 1차 측정은 다른 세션 요청과 겹쳐 버렸고(`bench/out/npu_concurrency_run1_contaminated.json`), 20:22~20:33 GPU HANG 구간(§8-13)엔 측정이 없다.

| 구간 | STT 중앙값 (p90) | YOLO 추론 ms 중앙값 (p90) · fps | LLM 첫 토큰 · tok/s | 실패 |
|---|---|---|---|---|
| STT 혼자 | 1.56초 (1.71) | | | 0 |
| YOLO NPU 10fps(미리보기) 혼자 | | 13.7 (14.3) · 10 | | 0 |
| YOLO NPU 최대속도 혼자 | | 6.6 (6.8) · 113 | | 0 |
| LLM 혼자 | | | 0.111초 · 29.1 | 0 |
| STT + YOLO 10fps | 1.65초 (1.80) | 25.1 (38.8) · 10 | | 0 |
| STT + YOLO 최대 | 1.90초 (1.93) | 31.2 (32.0) · 32 | | 0 |
| STT + LLM | 2.03초 (2.22) | | 0.116초 · 27.7 | 0 |
| YOLO 10fps + LLM | | 16.7 (17.5) · 10 | 0.114초 · 28.6 | 0 |
| **STT + YOLO 10fps + LLM** | **2.22초 (2.47)** | **27.4 (40.6) · 10** | **0.115초 · 27.6** | 0 |
| STT + YOLO 최대 + LLM | 2.67초 (2.91) | 32.4 (34.2) · 29 | 0.118초 · 27.2 | 0 |
| YOLO GPU 30fps 혼자 | | 7.1 (7.5) · 30 | | 0 |
| YOLO GPU 30fps + LLM | | 9.1 (10.4) · 30 | 0.131초 · 24.6 | 0 |
| STT + YOLO GPU 30fps + LLM, **3분** | 2.03초 (2.25) · 88회 | 9.0 (10.0) · 30 | 0.133초 · 23.7 · 27회 | 0 |

- **둘 다 버틴다.** 모든 구간 실패 0, CER 0.056~0.086(혼자 0.068 과 같은 수준), 커널 로그(i915·ivpu) 깨끗.
- NPU 는 시간을 나눠 쓴다. YOLO 를 최대속도로 돌리면 STT 와 겹칠 때 YOLO 가 32fps 로 깎이고 STT 가 +22%. 미리보기를 10fps 로 묶으면 STT +6%, YOLO 는 25ms 로 늘어도 프레임 간격 100ms 안이다.
- STT 를 가장 많이 늦추는 건 NPU 가 아니라 LLM(GPU) 이다(+30%). 메모리 대역·CPU 를 같이 쓰는 탓으로 보인다<!-- TODO: 확인 필요 -->. 셋을 다 겹친 경우도 2.2초.
- LLM 은 NPU 부하에 거의 안 흔들린다(29.1 → 27.6 tok/s). GPU 를 다른 OpenVINO 프로세스와 나누면 −15~19%.
- **결론: `config.json` 장치 배치 유지**(Whisper NPU · LLM·리랭크 GPU · 임베딩 CPU). 비전 미리보기는 NPU 10fps 이하. Whisper 를 GPU 로 옮기면 혼자는 0.9초지만 GPU 를 LLM·Studio 정책과 나눠야 해서 옮기지 않는다.
- (위 표는 KV 캐시 동적 때 값. 1GB 고정 뒤 STT + YOLO GPU 30fps + LLM 1분: STT 2.03초·YOLO 9.1ms·LLM 23.2 tok/s, 실패 0 — 같다.)
- Studio ACT 정책은 OpenVINO GPU(`robot/policies.json` `defaults.device`)라 시연 땐 OVMS 와 GPU 를 같이 쓴다. 대용 시험(YOLO GPU 30fps 별도 프로세스 + LLM + STT, 3분)은 실패 0·행 없음 → OpenVINO 끼리는 버틴다. PyTorch xpu 학습과는 못 버틴다(§8-13). 실제 정책은 정책별 10회 시험 때 `journalctl -k | grep -i hang` 으로 한 번 더 본다.

### 7.7 KV 캐시 크기 — 의도 판별·보고 지연 (10-21 21:07~21:15)

에이전트 담당 관찰 "의도 판별이 0.8 → 1.6초로 느려진다"를 확인했다. 원본 graph(`--pull` 이 만든 `graph.pbtxt`)는 `cache_size: 0` = **동적 캐시**라 필요한 만큼만 잡는다(로그 `Cache type: dynamic, cache usage: 83.7% of 31.0 MB`). 크기 상한이 있는 건 아니지만, 작게 유지되는 탓에 다른 요청(보고·답) 뒤엔 의도 판별 시스템 프롬프트의 prefix 캐시(`enable_prefix_caching: true`)가 밀려나 매번 처음부터 계산한다.

`bench/kv_cache.py`: 시연 흐름을 흉내 낸 6회전 — 의도 판별(에이전트 `agent/router.py` `_llm_classify` 그대로) → 바로 한 번 더 → 결과 보고(`report_node` 꼴, 200토큰) → 기준서 답(`answer_node` 꼴, 근거 3조항) → 의도 판별. 다른 세션 요청·학습 없음 확인 뒤.

| 구간 (중앙값, 괄호 = 최대) | 동적 캐시 (원본) | **고정 1GB** (`graph_kv1.pbtxt`) |
|---|---|---|
| 보고·답 뒤 의도 판별 (12회) | 1601ms (1682) | **625ms (815)** |
| 의도 판별 바로 다시 (6회) | 554ms (620) | 573ms (627) |
| 결과 보고 200토큰 (6회) | 3034ms (3066) | 2402ms (2445) |
| 기준서 답 (6회) | 1597ms (1623) | 772ms (첫 회 1615) |

- 의도 판별 결과(의도·개수)는 두 경우 같다. 바꾼 뒤 `config.json` 으로 다시 띄워 3회전 재확인: 보고·답 뒤 판별 664ms.
- 보고·답이 빨라진 건 시험에서 같은 프롬프트를 되풀이해 근거·사실 부분까지 캐시에 맞았기 때문이라 **실제보다 좋게 나온 값**이다. 실제로는 시스템 프롬프트 몫만 준다. 의도 판별은 사용자 말이 짧아 실제와 거의 같다.
- 안정성: 고정 1GB 로 LLM 2줄 + 임베딩/리랭크 + STT 90초(실패 0), STT + YOLO GPU 30fps + LLM 60초(실패 0), 커널 로그 깨끗.
- 메모리: OVMS cgroup 11.7 → 13.0GB(부하 뒤 14.3GB), available 19.0 → 18.0GB. 1GB 는 약 7천 토큰 이상 담는다 — 에이전트 프롬프트(2천 토큰 이하)에 넉넉하다.
- 되돌리기: `config.json` 의 qwen3-4b 에서 `"graph_path": "graph_kv1.pbtxt"` 줄을 지우고 `systemctl --user stop ball-ovms && ./start_ovms.sh`.

### 7.8 시연 30분 연속 시험 — 에이전트(가짜 작업대) + STT + TTS + YOLO NPU (10-22 00:01~00:32)

`bench/soak.py` — 시연 날 대시보드 프로세스가 하는 일(에이전트·STT·TTS·비전)을 한 프로세스에 모으고 대시보드와 같은 인터넷 차단(`ui/netguard.py`)을 걸었다. 직전에 `start_demo.sh --mock` 으로 OVMS·대시보드를 처음부터 띄움(4모델·대시보드 응답까지 11.7초).

- 에이전트: `BALL_MOCK=1`(정책 1회 0.5초, 각 thread 3번째 시도에서 한 번 놓침), 한 바퀴 = [초기화 → 명령 1→5] + [초기화 → scene → plan → scene → 질문], 말투는 바퀴마다 바꿈(분홍·핑크·초록·녹색·연두, scene·plan 은 `docs/intents.csv`). 106바퀴.
- 동시에 STT 12초마다(`ui/samples` 34개 돌려 가며), TTS 15초마다(보고 문장 4개), YOLO `detect_objects` 0.5초마다(`e2e_int8` NPU, 사진 파일, 구역 `vision/_test/zones_test.json` — 블록 모델·`zones.json` 이 아직 없음).
- 다른 세션의 OVMS 사용 0회(겹친 구간 없음), xpu 학습 없음, 대시보드는 떠 있되 쉬는 중.

| 종류 | 횟수 | 실패 | 중앙 | p95 | 최대 | 5분 단위 중앙 (처음 → 끝) |
|---|---|---|---|---|---|---|
| 명령 defect | 106 | 0 | 0.63초 | 0.81 | 1.35 | 0.63 → 0.63 |
| 명령 red(분홍) | 106 | 0 | 2.21초 | 2.44 | 6.23 | 2.28 → 2.01 |
| 명령 blue(초록) | 106 | 0 | 0.63초 | 1.20 | 1.45 | 0.63 → 0.63 |
| 명령 handoff | 106 | 0 | 2.83초 | 2.86 | 3.25 | 2.83 → 2.84 (가짜 손 2초 대기 포함) |
| 명령 report | 106 | 0 | 2.45초 | 2.97 | 3.13 | 2.55 → 2.43 |
| 명령 scene | 212 | 0 | 1.62초 | 2.76 | 3.16 | 1.91 → 1.56 |
| 명령 plan | 106 | 0 | 2.26초 | 6.66 | 6.77 | 2.27 → 2.26 (끝까지 71 · 거절 35 = intents.csv 의 할 수 없는 요청 4종, 맞는 동작) |
| 명령 question | 106 | 0 | 2.37초 | 3.38 | 4.26 | 2.25 → 2.40 |
| STT (Whisper NPU) | 131 | 0 | 1.82초 | 2.08 | 2.22 | 1.84 → 1.72 · 정답 있는 96회 모두 CER 0 |
| TTS (Supertonic CPU) | 104 | 0 | 2.19초 | 2.80 | 2.87 | 2.31 → 2.22 · 아래 스레드 설정 참고 |
| YOLO (NPU) | 3,438 | 0 | 14ms | 42 | 76 | 14 → 14 · 3,438회 모두 NPU |

- 명령 시간 = 판별·말 만들기·가짜 정책(0.5초/회) 합계라 실물 시간표가 아니다. 볼 것은 **시간이 지나도 안 느는지**(안 늘었다)와 실패 0.
- 의도는 1,060개 모두 기대한 intent(scene·plan 포함), 오류 이벤트 0.
- 커널: `journalctl -k` 00:00 이후 hang·fence·error·fail·timeout 0줄. 인터넷 시도 0(netguard 차단 0회).

| 분 | 0 | 1 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|
| 가용 메모리 GB | 16.85 | 15.79 | 15.60 | 15.50 | 15.50 | 15.48 | 15.49 | 15.47 |
| OVMS GB | 8.37 | 9.40 | 9.56 | 9.64 | 9.65 | 9.65 | 9.65 | 9.66 |
| 시험 프로세스 GB | 1.02 | 1.06 | 1.08 | 1.09 | 1.11 | 1.12 | 1.13 | 1.15 |

- OVMS 는 처음 10분 동안 1.3GB 늘고(KV 캐시·버퍼가 차는 몫) 그 뒤 평평. 시험 프로세스는 30분 +0.13GB(명령 1,060개, 약 4MB/분) — 시연(명령 수십 개)에 영향 없음. 대시보드 0.68GB·스왑 3.99GB 그대로.
- **TTS 스레드**: 대시보드(`ui/tts.py`)는 `SUPERTONIC_INTRA_OP_THREADS` 를 안 준다(`ui/start_ui.sh` 도 `SUPERTONIC_*` 을 넘기지 않음) → 혼자 돌려도 한 문장 2.4~3.1초, 4 를 주면 0.74~1.2초(00:3x 같은 문장 6개씩). 연속 시험의 2.19초가 그 값이다. 고칠 곳은 UI 소관(§9).
- 부산물(에이전트 코드가 쓴 것): `runs/20261022/log.jsonl` 911줄(thread_id `soak-*`), `runs/agent_checkpoints.sqlite` 101MB. 보고는 같은 thread_id 만 세므로 시연 보고에는 안 섞인다.
- **판단: 시연 당일 안정.** 단 조건 — xpu 학습과 겹치지 않기(§8-13). 실물 ACT 정책(OpenVINO GPU)은 이 시험에 없다 → §7.6 의 YOLO GPU 30fps + LLM + STT 3분(실패 0)으로 갈음, 실물 통합 때 `journalctl -k | grep -i hang` 한 번.

### 7.9 자원표 — 대기·피크 (10-22 22:09~22:21, 22:01 재부팅 직후)

`bench/resmix.py` — 구간마다 부하를 켜고 1초마다 잰다. 다른 세션은 이 구간에 추론하지 않았다(총괄 규칙). 결과 `bench/out/resmix_night3.json`·`resmix_night3_real.json`.
- 메모리: 프로세스 RSS, GPU 메모리 = i915 `fdinfo` `drm-resident-system0`, NPU 메모리 = intel_vpu `fdinfo` `drm-resident-memory`(둘 다 시스템 RAM 에서 나온다).
- 바쁨: GPU 바쁨 = 1 − rc6 비율(`/sys/class/drm/card1/gt/gt0/rc6_residency_ms`), NPU 바쁨 = `npu_busy_time_us` 증가분.
- CPU 는 코어 수 단위(22 = 전부).
- 규칙대로 GPU 는 OVMS(LLM·리랭크)만 썼다. ACT 는 CPU(Studio 컨테이너 안)·NPU(호스트), YOLO·Whisper 는 NPU.
- 정책 간격은 실물과 같은 1.7초에 한 번 추론(아래 §7.10).

| 구간 | 지연 (중앙 / 최대) | GPU 바쁨 | NPU 바쁨 | 가용 메모리 |
|---|---|---|---|---|
| 대기 (OVMS 4모델 + Studio, 대시보드 없음) | — | 0.1% | 0% | 19.5GB (OVMS 전 28.1GB) |
| LLM 생성 (160토큰 반복) | 3.42초 / 3.91 | **98.5%** (1.84GHz) | 0% | 18.5GB |
| 리랭크 10문서 + 임베딩 1문장 | 리랭크 0.68초 · **임베딩 첫 요청 53.3초** | 순간 67% | 0% | 18.1GB |
| STT (Whisper NPU, 0.5초 간격) | 1.57초 / 1.75 | 0.1% | **74%** (최대 100) | 17.9GB |
| TTS (Supertonic CPU 4스레드) | 1.31초 / 1.90 | 0.2% | 0% | 17.4GB |
| YOLO NPU 10fps (`ball_int8`) | 18ms / 29 | 0.3% | 10% | 17.1GB |
| ACT CPU 단독 | 59.8ms / 72 | 0.1% | 0% | 16.9GB |
| ACT NPU 단독 | 27.8ms / 33 | 0.1% | 2~5% | 17.0GB |
| **시연 피크** (LLM+리랭크·임베딩+STT+TTS+YOLO+ACT CPU) | LLM 6.59초 · STT 2.51 · TTS 1.48 · 리랭크 0.77 · YOLO 29ms · ACT 102ms(최대 136) | 98% | 81% | **16.8GB** · 스왑 0 |
| 시연 피크 (ACT 만 NPU) | LLM 6.03 · STT 2.55 · ACT 34ms(최대 58) | 98% | 83% | 16.9GB |

| 누가 | 대기 RAM | 피크 RAM | GPU 메모리 | NPU 메모리 | CPU (평균 / 최대 코어) |
|---|---|---|---|---|---|
| OVMS (4B GPU·리랭크 GPU·임베딩 CPU·Whisper NPU) | 1.94GB | 3.20GB | 4.77 → 5.78GB | 2.02 → 2.08GB | 대기 0.01 · LLM 0.92~0.97 · 피크 0.7 / 1.1 |
| Studio 컨테이너 (serve + 작업자 2개, RSS 합) | 3.49GB | 3.52GB | 0 (i915 문맥은 열려 있으나 이번 부팅 사용 0ms) | 0 | 0.02 / 0.05 |
| ACT 정책 실행 프로세스 (CPU, 컨테이너 안) | — | 0.47GB | 0 | 0 | 0.5 / 4.3 (추론 60~100ms 동안 14스레드) |
| TTS + YOLO + ACT NPU (시험 프로세스 = 대시보드가 할 일) | — | 1.08GB | 0 | 0.11GB | 4.2 / 6.3 (거의 TTS) |
| 대시보드 (10-22 00:01 실측, 지금은 안 떠 있음) | 0.68GB | — | — | — | — |

- **시연 날 예상 가용 메모리 약 15GB.** 계산은 대기 19.5GB 에서 다음을 뺀 값이다.
  - 대시보드 프로세스(에이전트·TTS·YOLO 포함) 약 1.1GB(§7.8 시험 프로세스 1.02~1.15GB).
  - ACT 실행 0.5GB.
  - OVMS KV·버퍼 증가 1.3GB(§7.8).
  - 브라우저 약 1GB <!-- TODO: 확인 필요 — 대시보드 탭 실측 -->.
  - 메모리는 막는 요인이 아니다. 지금 자리에 모델 하나(3~4GB)를 더 올릴 여유도 있다.
- **막는 요인은 장치 시간이다.**
  - GPU: LLM 이 답을 만드는 동안 98~100% 를 혼자 쓴다. 리랭크도 같은 GPU 를 쓴다.
  - NPU: 받아쓰기 동안 74~100% 다. NPU 에 올린 것(YOLO·ACT)은 그동안 느려진다(YOLO 18 → 29ms, ACT NPU 28 → 34ms).
  - CPU: 남지만 TTS(4스레드)·LLM 이 메모리 대역폭을 같이 써서 CPU 추론이 1.3~1.7배 느려진다.
- **재부팅 뒤 첫 임베딩 53초**: OVMS 기동 뒤 첫 `/v3/embeddings` 요청 하나가 53.3초 걸렸다. 그다음부터는 37~53ms 다.
  - OVMS 로그엔 아무것도 안 남는다.
  - 10-22 새벽 에이전트 첫 색인(99조각) 때도 첫 요청이 18초였다(에이전트 기록).
  - → `start_demo.sh` 응답 확인에서 미리 치른다(§1).

### 7.10 ACT 정책 추론 — 장치별 (10-22 22:04~22:21, 로봇·카메라 없이)

- **모델**: `pick_red` ACT 내보내기(`/app/storage/models/708a44b7…/exports/openvino`, 네 정책 모두 같은 ACT 구조).
  - 입력: 상태 6 + 카메라 2대(640×480 → 384×384 레터박스).
  - 출력: 동작 100×6.
- **CPU**: Studio 컨테이너 안에서 `physicalai.inference.InferenceModel(device="CPU")` 를 `docker exec -i … python -I -`(stdin 스크립트, 컨테이너에 파일 안 남김)로 돌렸다. 전처리 포함.
- **NPU**: 같은 IR 을 `docker exec cat` 으로 꺼내(`/tmp/actx`, 읽기만) 호스트 `.venv` OpenVINO 로 돌렸다. 컨테이너 OpenVINO 의 장치는 `['CPU', 'GPU']` 뿐이다. `/dev/accel/accel0` 은 보이지만 NPU 사용자 드라이버가 이미지에 없다.
- **입력값**: 무작위 영상이다. ResNet18 + 트랜스포머 고정 계산량이라 지연이 입력값에 따라 달라지지 않는다.
- **결과 차이**: NPU 와 CPU 출력 차이는 최대 0.041(정규화 단위, 약 0.07°)이다.

| 장치 | 혼자, 연달아 | 혼자, 실물 간격 1.7초 | + TTS | + LLM 생성 | + STT | 시연 피크 |
|---|---|---|---|---|---|---|
| GPU (10-22 실물 52회 실행 중 추론 421회, `robot/logs/20261022/*.ticks.jsonl`) | — | 실행별 중앙의 중앙 **33.0ms**, 대개 26~40, 최대 60.5 | — | — | — | (오늘 밤 규칙상 안 잼) |
| CPU 기본 (14스레드, LATENCY) | 59.8ms (52~77) | 59.8 / 최대 72 | 78.1 / 102 | 77.3 / 89 | — | **102 / 136** |
| CPU 다른 설정 | P코어만 67.6 · 8스레드 76.2 · P코어·HT 끔 94.8 · THROUGHPUT 65.3 — 기본이 가장 빠름 | | | | | |
| NPU (호스트 `.venv`, LATENCY) | **12.9ms** (최대 18~25) | 27.8 / 33 (쉬는 동안 NPU 가 내려가 다시 올라오는 몫) | — | — | 30.2 / 49 | 34.1 / 58 |
| (참고) SmolVLA GPU | — | 중앙 487ms (17:43 실행) | | | | |

**런타임이 지연을 어떻게 쓰는가**(컨테이너 `physicalai/runtime` 소스 읽기)
- 지금 설정은 `SyncExecution(request_threshold 0.5)` 이다(`robot/run_policy.py:216`).
  - 동작 100개 묶음의 절반(50틱 = 1.67초)을 쓰면 **제어 루프 안에서** 다시 추론한다.
  - 그 틱은 추론 시간만큼 늦고, 다음 틱은 따라잡지 않고 그 자리에서 이어 간다(`core.py` `_tick`: 마감이 그 틱 시작 기준).
  - → 추론이 33ms 를 넘으면 그만큼 1.7초마다 한 번 팔이 마지막 명령에서 멈칫한다. CPU 면 1.7초마다 30~100ms(1~3틱)다.
  - GPU 도 지금 중앙 33ms 라 이미 경계에 있다(최대 60ms 면 1틱 멈칫).
- `AsyncExecution` 은 뒤 스레드에서 추론한다. 추론 중 소비한 틱 수만큼 새 묶음의 앞을 건너뛰어(offset) 이어 붙인다. 그래서 추론이 50틱(1.6초) 안에만 끝나면 멈칫이 없다.
  - CPU + Async 실행 설정(장치 CPU, `physicalai.runtime.AsyncExecution`, request_threshold 0.5)은 컨테이너 `physicalai run --print_config` 를 통과했다(설정 해석만, 로봇·카메라 연결 없음).

**판단**
- **33ms 안**: CPU 는 안 된다(60~136ms). NPU 는 혼자면 되지만(13~28ms) 받아쓰기와 겹치면 p95 46~56ms 다. 무엇보다 컨테이너가 NPU 를 못 써서 이번 주 안에는 못 쓴다(이미지 변경 = 계약 5절 금지).
- **정책 = CPU 로 GPU 를 LLM 전용으로 하는 안은 `AsyncExecution` 과 같이 바꿀 때만 권한다.** 지연 여유가 33ms → 1.6초가 되어 CPU 102ms(피크)도 문제없다.
  - 바꿀 곳은 로봇 소관이다: `robot/policies.json` `defaults.device` `"GPU"` → `"CPU"`, `robot/run_policy.py:216` `SyncExecution` → `AsyncExecution`.
  - 사용자 입회 실물 확인 1회가 필요하다: 동작 모양·조기 종료·멈춤이 그대로인지.
- Sync 그대로 CPU 로만 바꾸면 1.7초마다 멈칫이 생긴다(동작 시간 약 2~6% 늘어남, 모양 영향은 실물로만 확인 가능).
- 다만 §8-14 대로 오늘 GPU HANG 은 LLM·ACT 동시 사용과 시각이 안 맞는다. 정책을 GPU 에서 빼는 것은 보험이지 HANG 해결책으로 확인된 것은 아니다.

### 7.11 더 올릴 모델 후보 (설치·내려받기 안 함 — 명령만)

메모리 여유는 있다(§7.9 약 15GB). 그러나 오늘 밤 규칙상 GPU 는 LLM 전용이고 NPU 는 받아쓰기로 바쁘다. 그래서 후보는 "어디서 돌릴지"가 먼저다.

| 후보 | 기대 효과 | 위험·비용 | 판단 |
|---|---|---|---|
| Whisper large-v3 (turbo 의 원본, 디코더 4 → 32층) INT8 | 한국어 정확도 조금 나을 수 있음 <!-- TODO: 확인 필요 — 같은 실녹음으로 비교 --> | 디코더가 8배 → NPU 받아쓰기 1.6초 → 추정 3~6초(실측 아님). 크기 0.8 → 약 1.5GB, NPU 메모리 2.0 → 약 3GB(추정). 무음·잡음에 "감사합니다." 같은 환청은 large 가 더 잦다는 보고가 많음 | **시연엔 안 씀.** 10-22 실물 실패 원인(마이크 1m·교실 소음·녹음 시작 잘림·무음 환청)은 모델 크기로 안 풀린다 |
| 한국어로 미세조정한 turbo (크기·속도 그대로) | 한국어 받아쓰기 개선 가능 | HF 커뮤니티 모델이라 품질 편차, OpenVINO 변환 필요(optimum-intel 설치 = 수업 venv 금지, 프로젝트 .venv 에 추가 설치) | 시험해 볼 만함. 실녹음 비교 후 결정 |
| 장면 설명용 작은 VLM (Qwen2.5-VL-3B·InternVL2-2B·Phi-3.5-vision 급 INT4) | "책상에 뭐가 있어?" 같은 자유 질문 | CPU 에서 사진 1장 인코딩 + 40토큰 답 = 추정 6~10초. CPU 를 쓰면 ACT CPU·TTS 가 느려짐(§7.10). GPU 에 두 번째 LLM 은 10-21 HANG 2회(§8-3). 작은 3cm 블록 개수 세기는 VLM 이 약함 → 지금의 YOLO 개수 + 4B 답(scene)보다 틀릴 수 있음 | **시연엔 안 씀.** 쓰려면 질문 하나에만, CPU, 시간 제한 10초 |

- **받아쓰기 품질은 모델보다 입력 쪽**(UI 세션과 공유할 것).
  - 입 가까운 마이크, 녹음 시작 전 0.3초 미리 담기(잘림 방지), 말소리 없는 녹음은 보내지 않기(환청 차단).
  - 시연 낱말 프롬프트도 있다(`ui/stt_prompt_check.py` 가 이미 있음).
  - large-v3 를 쓴다면 NPU 대신 GPU 가 빠르다(turbo GPU 0.90 대 NPU 1.56초, §7.3). 다만 GPU 는 LLM 몫이다.
- 명령(인터넷 필요, 시연 당일 금지, `pull_models.sh` 형식).
  - 아래 저장소 이름은 OpenVINO 허브에 있는지 먼저 확인할 것 <!-- TODO: 확인 필요 -->.
  - 받은 뒤 `config.json` 에 항목을 넣으면 OVMS 가 1초 안에 다시 읽는다. 시험은 `whisper` 를 바꾸지 말고 이름을 따로(`whisper-large`) 둔다.

```bash
cd ~/dev/projects/ball-sorter && export LD_LIBRARY_PATH=$PWD/ovms/lib PATH=$PWD/ovms/bin:$PATH
ovms --pull --model_repository_path models --source_model OpenVINO/whisper-large-v3-int8-ov --task speech2text --target_device NPU
ovms --pull --model_repository_path models --source_model OpenVINO/Qwen2.5-VL-3B-Instruct-int4-ov --task text_generation --target_device CPU
# 미세조정 turbo 변환(프로젝트 .venv 에 optimum-intel 설치가 먼저 — README 기록 필요): optimum-cli export openvino --model <HF 저장소> --weight-format int8 models/whisper-ko-int8-ov
```

## 8. 겪은 문제

1. **8B 추가 중 시스템 메모리 부족 → 다른 프로세스가 스왑으로 밀림 (16:37)**. VM 이 켜져 있던 때(가용 약 11GB) OVMS 상한을 16GB 로 줘서, OVMS 가 자기 상한에 걸려 죽기 전에 시스템 전체가 모자라 Studio 프로세스 등이 스왑으로 밀렸다(스왑 8GB 가 거의 다 참, 죽은 프로세스는 없음). → 상한을 "기동 시 가용 − 3GB" 로 자동 계산하게 고침. 이후 사용자 지시로 VM 을 끔. 지금도 스왑 약 4GB 가 Studio 등의 오래된 페이지로 남아 있다(쓸 때 다시 올라옴).
2. **Docker 는 두 개다**: 현재 docker 기본 context 가 Docker Desktop(별도 VM, `/dev/dri`·`/dev/accel` 없음)이고, Studio 는 네이티브 엔진(`docker --context default`)에서 돈다. 호스트 드라이버(GPU 26.31·NPU 1.38)가 이미 최신이라 Docker 대신 **공식 Ubuntu 24 바이너리**를 택했다.
3. **GPU HANG 2회 — LLM 두 개를 한 서버에 올렸을 때**. ① 8B·4B 적재 뒤 Whisper-GPU 컴파일 중 `CL_INVALID_EVENT` → abort (16:48) ② 다른 세션의 8B 요청과 내 4B 요청이 겹친 추론 중 `GPU HANG ... context reset` → SEGV (17:20). 커널 로그 `i915 ... GPU HANG: ecode 12:10:...`. 단독 LLM 구성(4B·8B 각각)은 LLM 2줄 + 임베딩/리랭크 + STT 동시 부하 90~120초에서 실패 0, 행 없음. → **LLM 은 하나만**, 비교는 서버를 바꿔 띄워서.
4. **Qwen3 리랭커·임베딩은 지시 템플릿을 클라이언트가 붙여야 한다** — 안 붙이면 리랭크 순위가 틀린다(§3-5).
5. **CPU 리랭크가 느리다**(템플릿 포함 문서 8개 5.8초) → GPU 로 옮김(0.4초). LLM 과 같은 서버에서 부하 시험 통과.
6. **NPU 임베딩·리랭크 시도 실패**: 컴파일 중 메모리가 12.6GB 까지 늘어 상한에 걸려 죽음(긴 최대 문맥으로 정적 컴파일하는 것으로 보임 <!-- TODO: 확인 필요 — --max_length 로 줄이면 되는지 -->). 접음.
7. **Supertonic 기본 스레드가 오히려 느리다**(§6).
8. **측정이 다른 세션 작업과 겹침**: 비전 학습(GPU xpu, 16:47~17:00)과 에이전트 평가·CLI 가 같은 시간에 OVMS 를 썼다. 학습 중엔 내 OVMS 가 자기 상한에 걸려 스스로 죽었다(학습은 무사). 속도는 동시 요청 없는 회차만 골라 다시 쟀다.
9. **Whisper NPU 첫 컴파일 약 3분 25초**. `.ov_cache` 가 생긴 뒤로는 5초.
10. **thinking 이 기본으로 켜져 있다**(§3-1).
11. **8B 는 개수 규칙을 두 번 어김**(§7.1) — 4B 를 권장하는 이유 중 하나.
12. `sounddevice` 는 `libportaudio2` 가 있어야 import 된다 → 설치함(§5).
13. **GPU HANG 3회째 — 비전 xpu 학습과 OVMS LLM 이 GPU 를 같이 씀 (10-21 20:22~20:33)**. 비전 `train.py`(수업 venv, PyTorch xpu)가 20:22:16 시작 → 20:22:44 커널 `Fence expiration time out i915 ... ovms` → OVMS LLM 이 7분 동안 응답 없이 멈춤(다른 세션 요청은 `CANCELLED`) → 20:29:37 `GPU HANG: ecode 12:10:85def5fa, in ovms` · `context reset` → OVMS SEGV(코어 덤프). 학습은 무사히 끝났다. **LLM 하나 구성이어도 생긴다** → 계약서 5절 "젠북 GPU 는 한 번에 한 작업(xpu 학습은 OVMS 내린 뒤)". 20:32:23 인프라 세션이 학습이 끝나기를 기다렸다가 `start_ovms.sh` 로 다시 띄웠다(자동 재시작 아님). 로그 `logs/ovms-run8-gpuhang-xpu-train.log`.
14. **10-22 GPU 사건 6건 — 재현 조건 미확정, "LLM + ACT 동시"와는 시각이 안 맞음.** 이전 부팅(10-20 09:56 ~ 10-22 18:08)의 `journalctl -k -b -1` 을 실행 기록과 맞댔다(`bench/hang_timeline.py`).
    - 맞댄 기록: `robot/logs/20261022` 정책 52회·추론 421회, `runs/20261022/log.jsonl`, `logs/ball-ovms.log` 의 LLM 생성 줄, Studio `docker logs`.

    | 시각 | 커널 | 그때 LLM | 그때 ACT(GPU) | 그 밖 |
    |---|---|---|---|---|
    | 10:18:13 | GuC CT 오류 3줄(`Unsolicited response`·`-ENOKEY`) | 쉼(밤새 사용 없음) | 없음 | 10:07:02 절전 복귀 11분 뒤 |
    | 16:11:11 | `GPU HANG … in ovms` + context reset → OVMS SEGV | 쉼 (마지막 생성 16:04:52) | 없음 (직전 추론 222초 전, 다음 정책은 16:11:14 GPU 컴파일 — HANG 3초 뒤라 무사) | 좌표 집기 계획 2회(16:10:58·16:11:03, NPU·numpy) |
    | 16:28:42 | `Fence expiration … ovms` | 쉼 (마지막 16:27:05) | 없음 (220초 전) | Studio 카메라 미리보기 16:28:21 열고 닫음 |
    | 16:33:25 | 펜스 만료 2줄 → **OVMS 무응답 27분**(모델 상태는 AVAILABLE, CPU 190%) | 쉼 | 없음 | 17:00:27 내리기 시작 |
    | 17:00:52 | `DrogonIoLoop` segfault | — | — | OVMS 를 내리는 중 충돌(GPU 사건 아님), 17:00:56 새 OVMS |
    | 17:36:40 | 펜스 만료 1줄 (그 OVMS 는 뒤에도 답함) | 쉼 (마지막 17:26:54) | 없음 (699초 전) | Studio 는 SmolVLA 2.3GB 받는 중(네트워크) |

    - ACT GPU 추론 421회 중 LLM 생성과 1초 안으로 겹친 것은 5회(17:00:41~17:01:50)뿐이다. 그 뒤 HANG 은 없었다.
    - Studio 프로세스는 i915 문맥만 열고 GPU 를 안 쓴다(이번 부팅 0ms). 10-22 xpu 학습은 15:45:46~15:48:26 로 OVMS 를 내린 사이(15:45:38~15:50:36)였다 — 규칙대로.
    - **그래서 10-22 사건은 OVMS 가 혼자, 그것도 LLM 생성이 없을 때 났다.** 10-21 의 3건은 원인이 분명했다(LLM 2개 동시·xpu 학습 동시, §8-3·8-13).
    - 빈틈: OVMS 로그는 LLM 생성만 남기고 리랭크(GPU)·임베딩 요청은 안 남긴다. 그러니 "그때 OVMS GPU 가 놀았다"는 증명은 아니다. 다음 사건에 대비해 요청 시각을 남겨야 한다(아래).
    - 공통 배경: 그 부팅은 2.3일째였다. 그 사이 s2idle 절전·복귀 14회(10-22 아침 07:55→09:27·09:31→10:07), 10:18 GuC 통신 오류, 10-21 HANG 3건·OOM 4건 복구를 겪었다.
    - 가설: 오래 켜 둔 부팅에서 i915/GuC 상태가 나빠졌다(커널 7.0.0-38, GuC 70.36.0) <!-- TODO: 확인 필요 — 새 부팅에서 같은 부하로 재현 시험 -->.
    - **절전 복귀 실패(18:08:18)**
      - 마지막 기록은 `PM: suspend entry (s2idle)` → `Filesystems sync` 이고 복귀 기록이 없다. 그 뒤 기록이 디스크에 안 남아 원인 증거는 없다.
      - 앞 14번 중 4번(10-21 17:57·18:00·18:04·19:38)은 HANG 뒤였는데도 정상 복귀했다. 다만 그땐 HANG 난 OVMS 가 SEGV 로 죽고 새로 뜬 뒤였다.
      - 18:08 은 펜스 만료(17:36:40)를 겪은 OVMS 가 그대로 떠 있었다. 팔·카메라 2대가 꽂힌 Studio 도 실행 중이었다.
      - 관련 가능성은 있으나 확인 불가다. 절전은 18:04:59 덮개 닫음 → 18:05:14 열고 → 18:05:29 다시 닫음 → 18:08:18 이었다.
    - **막는 법**
      1. 시연·리허설 날 **아침에 재부팅**(새 부팅, 절전 0회) → `start_demo.sh`. 오늘 밤도 22:00 재부팅 뒤 측정 중 사건 0건.
      2. **시연 중 절전 금지**: 어댑터를 꽂고 덮개를 닫지 않는다(배터리일 때 덮개를 닫으면 잠든다 — 10-21 17:53 닫음 → 17:57 절전). 자리를 옮기기 전엔 `systemctl --user stop ball-ovms` 와 Studio 실행 세션 정리 뒤 덮개. 깨운 뒤 `./start_ovms.sh`·`./start_demo.sh`.
      3. **펜스 만료가 한 줄이라도 보이면 OVMS 를 바로 다시 띄운다.** 16:33 처럼 모델 상태는 AVAILABLE 인 채 답을 안 할 수 있다. `start_demo.sh` 가 실제 응답을 확인한다(§1). 지금 OVMS 기동 뒤 HANG·펜스 건수도 보여 준다. **10-23 13:19 부터는 감시가 이 줄을 보고 바로 KILL 한다**(§1 무응답 감시).
      4. 다음 사건 때 증거(사용자, root 필요 — 재부팅하면 사라짐): `sudo cat /sys/class/drm/card1/error > ~/gpu_error_$(date +%H%M).txt`. OVMS 요청 시각은 `--log_level DEBUG` 로 남길 수 있으나 로그가 커서 재현 시험 때만 쓴다.
    - **10-23 갱신(09:35 HANG·10:12 fence)** — 위 "오래 켠 부팅"·"아침 재부팅" 은 약해졌다: 10-22 22:00 새 부팅 68분째(23:08 fence), GuC 70.36→70.53 올린 09:57 부팅 15분째(10:12 fence)에도 났다.
      - 오류 기록(09:35): 연산 엔진 ccs0 에서 OVMS 커널 연쇄가 한 COMPUTE_WALKER 앞에서 7.5초 넘게 안 넘어가 GuC 리셋. 페이지 폴트·명령 오류 없음. ecode `12:10:85def5fa` = 10-21 17:20·20:29 와 같은 꼴, GPU HANG 5건 모두 연산 엔진(`12:10`).
      - "쉬었다가 첫 요청" 은 필요조건이 아니다: 10:12 는 감시 1토큰 요청(10:12:08)이 정상이었다. 멈춘 요청은 OVMS 로그에 한 줄도 안 남는다(첫 걸음에서 멈춤).
      - ~~가설: OVMS 연산 문맥이 다른 GPU 문맥(크롬·gnome-shell 렌더)과 동시에 돌 때~~ → **10-23 13:2x 철회**(아래). 크롬 하드웨어 가속 끄기 확인 시험도 하지 않는다.
    - **10-23 갱신 2 (13:2x) — 오늘 네 사건, 펜스 줄 정정, 렌더 가설 철회.** 근거 = 총괄 GPU 기록기 `~/gpu_trace/gpu_trace.py`(10:17 부터, root, 1초마다 주파수·RC6·프로세스별 엔진 ns, 커널 사건 때 `event_*` 폴더) + 에이전트 체크포인트(`runs/agent_checkpoints.sqlite`, 읽기만) + 오류 기록 판독.
      - **정정 — 펜스 줄의 뜻**: `Fence expiration time out i915-…:ovms[스레드]:…!` 은 "10초 넘게 기다린 펜스"가 아니라 **i915 요청 감시**의 메시지다(이 커널 `CONFIG_DRM_I915_REQUEST_TIMEOUT=20000`, 모듈 매개변수 `request_timeout_ms`; i915 모듈 문자열·`/boot/config-7.0.0-38-generic` 로 확인). GPU 요청 하나가 20초 안에 안 끝나면 **리셋 없이 취소**한다. 10초짜리는 `Asynchronous wait on fence … timed out` 이라는 다른 메시지. 그래서 **펜스 줄 시각 − 20초 = 멈춘 요청이 들어간 시각**이고, 이때는 오류 기록이 안 남는다(`card1/error` = "No error state collected", engine_info `Reset count 0`). 줄의 `ovms[…]` 은 프로세스가 아니라 GPU 문맥을 만든 스레드(모델 적재 스레드, 프로세스가 사는 동안 남음).
      - 두 꼴은 같은 뿌리(OVMS GPU 커널 연쇄가 안 넘어감)이고 멈춘 자리만 다르다:
        - **끊긴 꼴**(펜스 줄): 20초 동안 리셋이 없었다 = 2.5초마다 오는 heartbeat 선점에는 응했다(선점되는 자리에서 멈춤) → 요청 감시가 끊음 → OVMS 는 살아서 답을 못 함 → KILL 이 필요하다.
        - **리셋 꼴**(GPU HANG): 선점 안 되는 커널에서 멈춤 → 약 7.5초 뒤 GuC 리셋 → OVMS 가 `clWaitForEvents -14` → SEGV 로 1~2초 안에 스스로 죽고 Restart 가 받는다.
      - **오늘(10-23) 네 사건**:

        | 커널 시각 | 꼴 | 멈춘 요청(들어간 시각) | 그 뒤 | 다시 답함 |
        |---|---|---|---|---|
        | 09:35:15 | GPU HANG `12:10:85def5fa`, 리셋 | 09:35:07.4 사용자 '분홍블록을 좌표로 옮겨줘' 의도 판별(json_schema) — '지금 배치 어때?'(09:34:53, LLM 안 쓰는 템플릿 답)가 아니었다 | 09:35:15.6 `-14` → SEGV(Restart 넣기 전) → 09:36:28 다시 띄움 | 09:36:36(4모델 준비) |
        | 10:12:35 | 펜스(요청 감시 20초), 리셋 없음 | 10:12:15.2 사용자 '홈자세로 보내줘' 의도 판별(json_schema), 5초 시간 초과 | 감시 15초 시간 초과 2번 → 10:13:03 KILL → 10:13:07 새 프로세스(적재 7초) | 10:13:29 |
        | 12:49:00 | 펜스, 리셋 없음 | **12:48:40 감시 1토큰 물음**('안녕', max_tokens 1). 사용자 '그럼 초록블록 왼쪽에 있는걸 옮겨줘 좌표로'(12:48:42.4)는 그 뒤에 막혀 5.06초 시간 초과 → '잘 못 알아들었어요' | 12:49:15 감시 KILL → 12:49:18 새 프로세스, **LLM 적재 60초**(평소 2~4초) | 12:50:21 |
        | 12:54:31 | GPU HANG `12:10:85def5fa`, 리셋 | 총괄이 대시보드 밖에서 `agent.router.classify` 를 직접 부른 시험 요청(12:54:07 첫 번째는 정상, 12:54:21 두 번째가 생성 3초째에서 멈춤) — 12:49 에 다시 뜬 OVMS(뜬 지 5분) | 12:54:32 `-14` → 12:54:33 SEGV → 12:54:36 자동 재시작(NRestarts 3), 적재 15초 | 12:55:01 |

      - 12:54 오류 기록 판독(09:35 와 같은 방법): ACTHD = 배치 안 COMPUTE_WALKER(1,016개 중 867번째), IPEHR = 바로 앞 PIPE_CONTROL, 폴트·오류 레지스터 전부 0 → 09:35 와 같은 꼴. 멈춘 커널의 실행 모양(SIMD 폭·작업 그룹 수)은 09:35 와 달라 "특정 커널 하나"는 아니다. 12:54:25~30 엔 렌더를 포함해 어떤 문맥도 시간이 안 늘었다 → 리셋 꼴 땐 화면도 몇 초 굳을 수 있다 <!-- TODO: 확인 필요 — 화면이 실제로 굳었는지 -->. 사본·해독 스크립트는 인프라 세션 작업 폴더에만.
      - 12:49 정상 대비: 그날 10:17~13:01 기록(9,852초)의 OVMS 연산 구간 501개 중 468개 1초·28개 2초·가장 긴 정상 구간 7초 → 20초 내리 꽉 찬 것은 이것 하나. 그 사이 임베딩(판별 1단계)은 답했다.
      - **렌더 동시 가설 철회** — 근거:
        - 12:48:40 멈춘 순간 렌더는 1~2ms/s(화면 거의 정지), 요청은 1토큰. 12:48:42~53 렌더(10~60ms/s)는 사용자 명령에 대시보드가 반응한 것(결과).
        - 정상 연산 구간 501개 중 385개(77%)가 렌더 ≥10ms/s 와 같은 초에 시작했다. 10:39~10:53 은 크롬 렌더 80~95% + python 프로세스의 영상 엔진 사용 중에 OVMS 물음 약 40번, 사건 0.
        - 다른 후보도 가르지 못한다: GPU 잠듦(주파수 0)에서 시작한 것이 정상의 58%(291/501)이고 12:54 는 깨어 있을 때 시작. 1950MHz·스로틀 표시는 정상 연산 때도 같다. 프로세스 나이(2.5시간·5분)·요청 종류(json_schema 의도 판별·1토큰·긴 생성)도 제각각.
        - 앞 10-22 표의 "그때 LLM 쉼"은 OVMS 로그(생성 줄) 기준이다. 멈춘 요청은 로그에 안 남으므로 펜스 줄 20초 전에 요청이 있었을 수 있다.
      - **남는 그림**: 요청 종류·화면 사용과 상관없이 OVMS 의 GPU 요청에서 드물게 커널 연쇄가 멈춘다(기록 구간 GPU 요청 약 500번에 2번, 그중 487번은 감시 물음). 자리는 GPU 드라이버 층(OpenCL NEO `intel-opencl-icd 26.31.39395`·IGC 2.40·GuC 70.53·i915 커널 7.0.0-38)이라 사용자 공간 설정(크롬 가속·주파수·RC6·OVMS 큐 설정)으로 막을 근거가 없다. 막는 길은 발표 뒤 과제(드라이버·커널 판 바꿔 재현 시험 등) <!-- TODO: 확인 필요 -->. LLM 은 GPU 에 둔다(10-23 사용자 결정).
      - **대응(10-23 13:19, 빨리 되살리기)**: 감시가 펜스 줄을 보고 바로 KILL, 1토큰 물음 20초 → 60초(물음이 멈춤 기회를 늘리므로), 재시작·적재 중 2초마다 다시 봄(§1). `start_ovms.sh` StartLimitBurst 3 → 5. 12:49 였다면 KILL 12:49:15 → 12:49:00, 다시 답함 12:50:21 → 약 12:50:05. 적재가 평소대로면 멈춤부터 다시 답하기까지 약 30~40초(전: 감지 35~55초 + 재시작 + 적재 + 예열 차례 최대 20초).
      - 12:49 뒤 LLM 적재 60초의 까닭은 미확정: 그 1분 GPU 는 쉬었고(RC6 약 990ms/s), 그 프로세스가 평생 쓴 CPU 가 16.8초뿐이고 `.ov_cache` 도 안 바뀌었다 → 다시 컴파일이 아니라 2.66GB 컴파일 캐시 파일 읽기나 메모리 확보를 기다린 꼴(그때 메모리 30G 중 가용 약 11G). 또 나면 그 사이 `/proc/pressure/io`·`/proc/pressure/memory` 를 본다 <!-- TODO: 확인 필요 -->.

## 9. 다른 세션에 알릴 것 / 남은 일

- **UI**: TTS 를 부르는 프로세스에 `SUPERTONIC_INTRA_OP_THREADS=4`, `SUPERTONIC_INTER_OP_THREADS=1` 환경변수(§6). 10-22 00:3x 에도 아직 없음 — 대시보드 TTS 한 문장 2.4~3.1초 → 넣으면 0.74~1.2초(§7.8). `ui/tts.py:14`(`import os` 아래)에 `os.environ.setdefault(...)` 두 줄.
- **비전**: xpu 학습은 `systemctl --user stop ball-ovms` 뒤에, 끝나면 `start_ovms.sh`(§8-13). `vision/detect.py` 에 `YOLO_OFFLINE=1`(§11).
- **에이전트**: 서버엔 지금 `qwen3-4b` 만 있다. `LLM_PREFERRED` 가 8B → 4B 순이라 자동으로 4B 를 고른다. 8B 로 시연하려면 `systemctl --user stop ball-ovms` 후 `config-8b.json` 으로 띄운다(둘 다 올리기 금지).
- **VM**: 꺼진 상태, KIMS 경로(`vmroute kims`)도 끔. 다시 켜기 = `~/.local/bin/vpnvm start` → `~/.local/bin/vmroute kims on`(VPN 로그인은 사람이). VM 이 켜지면 가용 메모리가 약 4~5GB 줄어든다 — 권장 구성은 들어가지만 8B 구성은 빠듯하다.
- 재부팅 자동 시작이 필요하면 사용자 systemd 서비스로 바꿀 것(미설정).
- 스왑 정리(선택, RAM 여유 있을 때): `sudo swapoff -a && sudo swapon -a`

## 10. 다시 측정하기

```bash
cd ~/dev/projects/ball-sorter
.venv/bin/python bench/bench_llm.py qwen3-4b                 # 속도·의도 JSON·tools·thinking
.venv/bin/python bench/bench_speed_clean.py qwen3-4b         # 다른 요청과 안 겹친 회차만으로 속도
.venv/bin/python bench/bench_rag.py                          # 임베딩·리랭크(원문/템플릿)
.venv/bin/python bench/bench_tts.py                          # TTS → bench/out/tts_*.wav
.venv/bin/python bench/bench_stt.py whisper                  # 위 wav 받아쓰기
.venv/bin/python bench/stress.py 90 qwen3-4b                 # 동시 부하 안정성
.venv/bin/python bench/mem_probe.py qwen3-embedding          # 모델 하나만 8101 에 띄워 메모리 측정
.venv/bin/python bench/npu_concurrency.py --secs 20          # NPU·GPU 동시 사용 13구간(§7.6), 약 7분
.venv/bin/python bench/npu_concurrency.py --secs 180 --skip 12 --retries 0   # 마지막 구간(STT+YOLO GPU+LLM)만 3분
.venv/bin/python bench/kv_cache.py <태그>                    # 의도 판별·보고·답 지연(§7.7), bench/out/kv_<태그>.json
.venv/bin/python bench/soak.py --minutes 30 --tag <태그>     # 시연 연속 시험(§7.8), bench/out/soak_<태그>.json — SSH 가 끊겨도 돌게 systemd-run --user 로
.venv/bin/python bench/resmix.py --tag <태그>                # 자원표(§7.9) 약 9분, bench/out/resmix_<태그>.json. 구간 직접: --phases "정책CPU+TTS:45:act_cpu+tts,..."
.venv/bin/python bench/hang_timeline.py --boot -1 --day 20261022   # GPU 사건과 정책·LLM 실행 맞대기(§8-14), 읽기만
python3 -I bench/gpu_users.py 3                              # 프로세스별 GPU 엔진 사용률 3초 표본(크롬·gnome-shell 렌더, OVMS 연산), 읽기만(§8-14)
bench/ovms_watch_test.sh                                     # 무응답 감시 단독 시험(§1) 12항목 약 2분 — 가짜 서버 유닛 ball-watchtest-ovms + 가짜 커널 줄(tail -F)로만, 진짜 OVMS·GPU 안 건드림
```

## 11. 오프라인 점검 (10-21 밤 — 코드 읽기 + 확인)

와이파이는 끄지 않았다. 시작 스크립트·에이전트·UI·비전·로봇 코드와 그 코드가 쓰는 라이브러리 소스(`.venv`)를 읽어 인터넷을 부르는 곳을 찾았다. 대시보드는 임시 프로필의 headless Chrome 으로 열어 네트워크 로그를 봤다: 페이지 요청 318건이 전부 `127.0.0.1:7870` 이고 외부 CDN·Google Fonts 는 0건이다(외부로 나간 건 Chrome 자체의 업데이트·맞춤법 사전 요청뿐, 오프라인이면 조용히 실패). 마이크 녹음 편집(자르기) 기능은 열지 않아 확인하지 못했다.

| 위치 | 인터넷 호출 | 상태 |
|---|---|---|
| `start_ovms.sh` · OVMS | 없음. 모델·토크나이저는 `models/` 로컬(`config.json` 은 `base_path` 만), 실행 중 외부 연결 0(`ss`). 인터넷을 쓰는 건 서버 안 HF 내려받기 모듈(`--pull`)뿐 | ✅ **막음**: `HF_ENDPOINT=http://127.0.0.1:9`. 시험 pull 이 0.08초 만에 `Connection refused`. 20:32 기동부터 적용 |
| `pull_models.sh` | Hugging Face | 시연 당일 실행 금지(파일 머리 주석) |
| `ui/app.py:18` Gradio | 분석·버전 확인(api.gradio.app) | ✅ `GRADIO_ANALYTICS_ENABLED=False`, `share` 안 씀 |
| `ui/tts.py:116` Supertonic | 모델이 없으면 HF | ✅ `auto_download=False` |
| `agent/rag.py:129` Chroma | 사용 통계(posthog) | ✅ `anonymized_telemetry=False` |
| `agent/config.py:20` LangSmith | 추적 전송 | ✅ `LANGCHAIN_TRACING_V2=false` |
| `agent/llm.py:67` 임베딩 | tiktoken 사전 내려받기 | ✅ `check_embedding_ctx_length=False` |
| `vision/detect.py:13~14` Ultralytics | import 때 DNS 조회(one.one.one.one·dns.google), `predict` 마다 Google Analytics 이벤트(`~/.config/Ultralytics/settings.json` 의 `sync: true`, 온라인일 때) | ❌ `YOLO_OFFLINE=1` 필요 → 비전 담당. `vision/bench.py:17` 도 같음 |
| `robot/preflight.py:392` | 트레이너 `<트레이너-PC>:8001`(Tailscale) | ✅ 시연 모드는 WARN "(시연엔 필요 없음)" 으로 이미 처리 |
| `vision/train.py:33` (수업 venv) | COCO128 zip, AMP 확인용 가중치·글꼴 | 시연 범위 밖 — 학습은 와이파이 켠 채 |
| pip · uv | 실행 중 자동 설치 없음(`YOLO_AUTOINSTALL=false`) | ✅ |
| Studio 컨테이너 | 점검 밖(손대지 않음) | 오프라인 리허설 때 확인 |

## 재개 (2026-10-21 마감)

**끝난 것**
- OVMS 2026.4.1 바이너리 + 모델 5종(8B·4B·임베딩·리랭크·Whisper) 다운로드·graph 생성, 권장 구성(`config.json` = 4B·임베딩 CPU·리랭크 GPU·Whisper NPU)으로 `127.0.0.1:8100` 가동 중이었음.
- 에이전트 venv(`.venv`) 전 패키지 import 확인, `libportaudio2` 설치.
- Supertonic-3 한국어 TTS 동작·스레드 튜닝, Whisper NPU·GPU 받아쓰기 비교.
- 실측 전부(§7): 8B·4B 속도·의도 JSON·tools·thinking, 임베딩·리랭크, STT, TTS, 모델별·구성별 메모리. 4B 권장 근거 = §0·§7.1.
- 17:20 GPU 오류 관찰: 8B·4B 를 한 서버에 같이 둔 상태에서 추론이 겹치자 `i915 GPU HANG ... context reset` → OVMS SEGV(16:48 에도 같은 구성에서 1회). LLM 하나 구성은 부하 시험 통과(§8-3).

- (10-21 밤) NPU 동시 사용 시험(§7.6, 배치 유지), 오프라인 점검(§11), 시연 당일 메모리 여유(§7.5). 20:29 GPU HANG(§8-13) 뒤 20:32 OVMS 다시 띄움. 21:15 KV 캐시 1GB 고정을 `config.json` 에 반영하고 다시 띄움(§7.7).
- (10-22 밤샘) `start_demo.sh`(§1) — `--mock` 으로 네 경로 시험. 00:01 `start_demo.sh --mock` 으로 OVMS·대시보드를 처음부터 다시 띄움(대시보드 환경은 UI 세션 것 그대로 `BALL_MOCK=1 BALL_MOCK_POLICY_S=2 UI_TTS_OUT=silent`) — 지금 둘 다 가동 중. 30분 연속 시험 실패 0(§7.8).

**부팅 뒤 자동 시작: 안 함.** `ball-ovms` 는 `systemd-run --user` 일회성 유닛이라 젠북을 켜면 꺼져 있다. VMware VM 도 꺼진 상태 그대로다.

**남은 것 (선택)**
- 부팅 자동 시작이 필요하면 사용자 systemd 서비스로 바꾸기.
- 8B 를 쓰겠다면 few-shot 프롬프트로 의도 재평가(예시 없이 3/5).
- NPU 임베딩·리랭크를 `--max_length` 로 다시 시도(메모리 절약용, 급하지 않음).
- 스왑 정리 `sudo swapoff -a && sudo swapon -a`(선택).

**재개 첫 명령**
```bash
~/dev/projects/ball-sorter/start_demo.sh --mock     # 하드웨어 없이. 실물 시연은 옵션 없이(BALL_ROBOT=1)
```
