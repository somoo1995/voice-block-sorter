# robot/ — 정책 실행 (`run_policy` · `stop`)

> 담당: 로봇 세션. 계약서 3절의 `run_policy(name, timeout_s)` · `stop()`.
> 상태(2026-10-21): 코드·dry-run·오프라인 시험 완료. **실제 구동 시험은 안 했다**(사용자 입회 때). 정책 4개 모델은 아직 없고, `pick_red` 만 기존 ACT 모델(종이컵 집기)에 **임시 연결**해 dry-run 을 확인했다.
> 10-21 밤: 한 줄 점검 `robot/preflight.py` 추가, 녹화 당일 시간표는 맨 끝 "재개" 절.
> 10-21 밤 물체 변경: 부직포볼 → 스펀지 블록(약 3cm, **분홍·초록** — 코드의 red = 분홍, blue = 초록). 문서의 말만 바꿨고 정책 이름(`pick_red` 등)은 그대로. 눌리는 물체라 그리퍼 과열 주의(2.2 끝).
> 10-22 새벽: 배우지 않은 배치(숫자 모양 등)를 위한 **좌표 집기** 조사·시제품 `robot/coord_pick/` + 실행기 `executor.py`(가짜 로봇 시험만, 실제 송신 0회. 첫 구동은 10-23 입회) — [COORD_PICK.md](COORD_PICK.md). `stop()`·잠금·사전 점검은 이 README 의 정책 실행과 같은 것을 쓴다(`environment_robot`·`robot_preflight` 로 떼어 냄).

## 1. 쓰는 법

```python
from robot.run_policy import run_policy, stop

r = run_policy("pick_red")            # timeout_s=None → policies.json 값(기본 21 = 15초 에피소드 기준, --link 가 정책별 권장값을 넣음)
r = run_policy("pick_red", timeout_s=25)
stop()                                # 다른 스레드·다른 프로세스에서 불러도 된다
```

- 결과 `reason`: `home_reached`(홈 복귀 정지 감지로 일찍 끝남, ok) · `duration_elapsed`(timeout, ok) · `stopped` · `timeout`(벽시계 상한) · `error` · `refused`.
  - 그 밖에 붙는 것(2.1·2.3·2.4): `early_stop`({triggered, t_s, dist, moved, ticks, load_s = docker exec 시작 → 첫 틱}) · `end_dist` · `returned` · `tail_s`(실행이 끝나고 결과를 돌려주기까지).
  - `returncode` 는 docker exec 를 기다리지 않으므로, "Run complete" 를 봤으면 0 으로 적는다.

```bash
cd ~/dev/projects/ball-sorter
.venv/bin/python -m robot.run_policy pick_red --dry-run   # 로봇 안 움직임. 명령·설정·점검만
.venv/bin/python -m robot.run_policy pick_red --yes       # 실제 실행(사용자 입회). --yes 없으면 확인 질문
.venv/bin/python -m robot.run_policy --stop               # 즉시 중단(그 자리에 멈춤). --stop-home 은 시작 자세로 복귀 후 종료
.venv/bin/python -m robot.run_policy --status             # 실행 중 정책·로봇 잠금·Studio 세션
.venv/bin/python -m robot.run_policy --models             # Studio 모델 목록(연결된 정책 표시)
.venv/bin/python -m robot.run_policy --link pick_red <모델ID>
.venv/bin/python -m robot.selftest                        # 로봇 없이 대기 루프·stop·시간 상한·조기 종료·복귀·결과 먼저·연속 5회·print_config 캐시 시험(18건, 약 2분)
.venv/bin/python -m robot.home_stop_replay [데이터셋…]    # 녹화 관절 기록으로 조기 종료 판정 재생(잘못 걸림 0 이어야 함, 1초)
.venv/bin/python -m robot.preflight                       # 시연 직전 한 줄 점검(표 + 빨간 항목마다 고치는 법)
.venv/bin/python -m robot.preflight --record              # 녹화·학습 날 아침 점검
```

### 한 줄 점검 `preflight`

읽기만 한다(Studio 는 GET·`docker inspect`·읽기용 `docker exec`, 카메라는 열지 않음). 1~2초, 정책 설정 검사까지 하면 정책당 1초 안팎 더(`--quick` 이면 생략). 빨간(안됨) 항목이 있으면 종료 코드 1, `--json` 은 기계용.

| 항목 | 보는 것 | 시연 기준 | `--record` 기준 |
|---|---|---|---|
| 팔로워·리더 USB | `/dev/serial/by-id` 에 Studio 환경의 시리얼(팔로워 `5B90101138`, 리더 `5B79081525`) | 리더 없음 = 주의 | 둘 다 필수 |
| 카메라 2대 | `/dev/v4l/by-id` 에 환경 카메라 시리얼 + 누가 열고 있는지(`sudo -n` 으로 `/proc/*/fd`) | 누가 열고 있으면 주의 | 열려 있어도 정상(녹화 중) |
| Studio 컨테이너·API | `docker --context default inspect` · `GET /api/health` | 필수 | 필수 |
| Studio 세션·잠금 | `GET /api/runtime/sessions` · 컨테이너 잠금 파일(좀비 제외) · 우리 실행 | 세션·남의 잠금 = 안됨 | 세션은 정상, 우리 실행이 팔을 쥐면 주의 |
| 컨테이너 안 장치 | 호스트 장치 경로가 컨테이너 `/dev` 에 있는지 | 필수 | 필수 |
| Studio 학습 대기열 | `GET /api/jobs` 의 running·pending(데이터셋 이름·진행률) | 정보 | 정보 |
| 정책 4개 | `policies.json` 연결 → 모델·OpenVINO export·카메라 이름(`build_plan`) → `--print_config` | 미연결 = 안됨, 임시 = 주의 | 미연결 = 주의 |
| OVMS | `GET :8100/v3/models` 에 4모델 + `/v1/config` 상태 AVAILABLE, LLM 두 개면 안됨 | 필수 | 필수 |
| 대시보드 | `GET :7870/` + `ball-ui` 유닛 환경변수로 모드(`BALL_MOCK`·`BALL_ROBOT`) | 실제 모드 아니면 주의 | 정상 |
| 트레이너 | `GET :8001/health`·`/devices`(학습 중인지)·`/storage` | 꺼짐 = 주의 | 필수 |
| 디스크·전원 | 젠북 여유(10GB 미만 안됨, 25GB 미만 주의) · 어댑터 | | |

- 기대 장치는 정책이 쓰는 Studio 환경(연결된 모델의 데이터셋 환경, 없으면 모든 환경)에서 읽고 `robot/.state/devices.json` 에 남긴다. Studio 가 꺼져 있으면 그 기록으로 본다.
- 10-21 밤(팔·카메라 안 꽂힘) 실측: 시연 기준 안됨 6(팔로워·카메라 2·미연결 정책 3) · 주의 3(리더·임시 pick_red·대시보드 가짜 모드). Studio·OVMS·대시보드·트레이너 주소를 일부러 틀리게 준 시험에서도 넷 다 안됨 + 고치는 법이 나왔다. 카메라 사용 중 감지(다른 웹캠을 3초 열어 봄)·세션/잠금 판정 5가지도 확인.

`--timeout 초` · `--model 모델ID`(policies.json 무시) · `--device CPU|GPU|NPU` · `--no-home`(끝나고 복귀 안 함) · `--json`.

### 반환값

계약서의 키(`name ok started ended duration_s returncode log_path`)에 아래를 더한다.

| 키 | 뜻 |
|---|---|
| `reason` | `duration_elapsed`(정상, `ok=True` 는 이것뿐) · `stopped`(stop 요청) · `timeout`(벽시계 상한 초과) · `error`(실행 중 오류) · `refused`(시작 전 점검에서 거부, 로봇 안 움직임) · `dry_run` |
| `steps` | 제어 틱 수(`physicalai` 의 "Run complete: N steps") |
| `run_id` | 이번 실행 ID(로그·컨테이너 프로세스 표식) |
| `camera_released` | 우리 카메라 publisher 가 끝났는지. `True` 면 비전이 위 카메라를 바로 열 수 있다 |
| `error` | `refused`·`error` 일 때 이유(`studio_session: …`, `robot_busy: …`, `no_model: …` 등) |

로그: `robot/logs/YYYYMMDD/HHMMSS_<정책>.log` + 같은 이름의 `.runtime.json`(실제로 넘긴 설정)·`.runtime.yaml`(읽기용). dry-run 은 `robot/logs/dryrun/`.

## 2. 확인한 사실 (소스 근거)

`physicalai` 0.3.1.dev4+g9133822. 컨테이너(`/app/application/backend/.venv/…/site-packages/physicalai`)와 젠북 네이티브 venv(`~/dev/projects/physical-ai-studio/application/backend/.venv/…`)의 .py 166개가 md5 로 같아서 네이티브 쪽을 읽었다. 아래 줄 번호는 그 패키지 기준.

### 2.1 `physicalai run` 은 언제 끝나나

- 끝나는 조건은 넷뿐: **stop 요청 · `--run.duration_s` 경과 · Ctrl+C(SIGINT) · 예외**(`runtime/core.py` `run()` 356~451). `duration_s` 를 안 주면 멈출 때까지 돈다.
- **ACT 에는 "다 했다" 신호가 없다.** `PolicySource` 에 끝내는 조건이 없다(`runtime/action_sources/policy.py`). 그래서 그대로 두면 `run_policy` 는 항상 `timeout_s` 만큼 돈다 → **홈 복귀 정지 감지**(아래, 10-22 새벽)로 일찍 끝낸다. `timeout_s` 는 안전 상한으로 남는다.
- **홈 복귀 정지 감지**(`policies.json` `defaults.early_stop`, `robot/home_stop.py`):
  - 관절값 읽는 길(Studio 변경 없음): 런타임 설정에 내장 `physicalai.runtime.JsonlCallback(path="/dev/stdout")` 을 넣는다(`runtime/callbacks/jsonl.py` — 매 틱 `joint_positions` 를 JSON 한 줄로 쓰고 바로 flush).
    - 컨테이너에 파일을 만들지 않고 `docker exec` 출력으로 나온다.
    - `run_policy` 가 그 줄만 골라 감지기에 넣고 `.ticks.jsonl` 에 남긴다(나머지는 `.log`).
    - 컨테이너의 `physicalai run --print_config` 가 이 설정을 받는 것을 확인했다.
  - 판정(정규화, 몸통 5관절): 첫 틱 자세 = 홈.
    - 홈에서 `leave` 15 넘게 벗어난 뒤에만 본다 → 처음 홈 대기는 제외.
    - `near` 8(약 14°) 안으로 돌아와 `hold_s` 1.0초 동안 6관절 변화폭이 모두 `still` 3.0 이하이면 → SIGINT(정상 종료), 결과 `reason = "home_reached"`(ok).
  - 녹화 재생 시험(`python -m robot.home_stop_replay`, 10-20 녹화 40개, 끝 뒤 3초 버팀):
    - 잘못 걸림(작업 도중 걸림) 0, 안 걸림 1(끝 자세가 11 떨어진 덜 돌아온 에피소드 → timeout 이 끝냄).
    - 걸린 시점: 홈 도착 뒤 1.13초(중앙). 녹화 끝보다 0.5초(10개 묶음)·1.3초(30개 묶음) 앞(중앙, −4.4~−0.1).
    - 잡음 σ 0.6(정규화)을 섞어도 같다.
    - 기본값 근거: 작업 중 최대 거리 최소 44.6, 끝 자세 거리 중앙 2.3·90% 4.9. `near 6·still 1` 이면 잡음 σ 0.4 에서 하나도 안 걸려서 넓혔다.
  - 끄기: `"early_stop": {"enabled": false}` → 예전 동작(timeout 까지, 런타임 복귀 2.5초).
  - <!-- TODO: 확인 필요 — 정책(ACT)이 홈에서 얼마나 흔들리는지는 녹화가 아니라 실제 실행에서만 안다. 10-22 12:06 첫 시험의 `.ticks.jsonl` 로 본다 -->
  - 학습 에피소드보다 길게 주면 남는 시간에 **블록을 하나 더 집으러 갈 수 있다**. `timeout_s` ≈ 에피소드 90% 길이 + 3초로 맞춘다(`--link` 가 데이터셋에서 계산해 넣는다. 임시 모델은 90% 24.0초 → 권장 28초).
- `duration_s` 는 벽시계가 아니라 **제어 주기 수 × 1/30초**로 센다(`core.py:434`). 추론이 느려 30Hz 를 못 맞추면 실제로는 더 오래 돈다. 모델 로드·GPU 컴파일 시간도 안 들어간다.
- `run_policy` 의 벽시계 상한 = `startup_budget_s`(90) + 2×`timeout_s` + 복귀 2.5(런타임 복귀를 쓸 때만) + 15초. 넘기면 SIGINT(복귀 후 종료) → 12.5초 안에 안 끝나면 SIGKILL. 정상 경로가 아니라 멈춤 대비용이다.
- 표준 logging 에 핸들러가 없어서 "Shutdown complete — N steps (이유)"(`core.py:786`, INFO)는 화면에 안 나온다. 종료 이유는 `run_policy` 가 자기 기록(stop·상한)과 종료 코드로 정한다.
- `docker exec` 는 컨테이너 안 프로세스가 끝난 뒤 **약 2초 있다가** 돌아온다(자식이 출력을 물고 있어도 마찬가지, 실측 2.1초).
  - **`run_policy` 는 이것을 기다리지 않는다**(10-22 새벽).
  - "Run complete" 줄(CLI 가 런타임을 닫은 뒤 남김, `cli/run.py:169`)이나 종료 신호를 본 뒤, 컨테이너 조사(0.06초)를 0.1초마다 해서 실행 프로세스가 끝났는지 확인한다.
  - 우리 카메라 publisher 가 풀리면(구독자 없이 `camera_idle_timeout_s` 1.0초 뒤 스스로 끝남) 바로 결과를 돌려준다.
  - docker exec 의 나머지는 뒤 스레드가 거둔다(`.log` 끝에 `# docker exec 종료 rc=…`).
  - 결과의 `tail_s` = 실행 프로세스가 끝나서 결과를 돌려주기까지. 가짜 시험에서 1.05초였다(전에는 exec 2.1초 + 조사).
  - 신호 없이 "Run complete" 도 없이 끝났으면(오류일 수 있음) 예전처럼 종료 코드를 기다린다.

### 2.2 끝나거나 중단될 때 팔이 떨어지나 — 안 떨어진다 (전원을 끄지 않는 한)

- Studio 설정의 팔로워는 `SharedRobot` 이다. 실제 서보 연결은 **따로 떠 있는 소유 프로세스**(`robot/transport/_owner_worker.py`)가 쥐고, `physicalai run` 은 그 구독자다. 소유 프로세스는 새 세션으로 떠서(`_owner.py:78` `start_new_session=True`) `physicalai run` 이 죽어도 살아 있다.
- 소유 프로세스는 새 명령이 없으면 **마지막 목표 자세를 유지**한다(`_owner_worker.py:228`). 구독자가 없는 채 **10초**(`SharedRobot idle_timeout` 기본 10, `_shared_robot.py:229`) 지나면 끝나면서 `driver.disconnect()`(`_owner_worker.py:534`) → SO-101 팔로워는 현재 위치를 목표로 쓰고 **토크를 켠 채** 포트를 닫는다(`so101/so101.py:448~462`, `_hold_position` 704).
- 그래서 정상 종료 · `stop()`(SIGKILL) · 벽시계 상한 어느 경우에도 팔은 그 자세를 유지한다. **떨어지는 경우는 팔로워 전원을 끌 때**(그리고 `torque_on_disconnect=False` 로 바꿨을 때)뿐이다. 전원을 끄기 전엔 팔을 받친다.
- 주의: 연결할 때(`connect()` → `_configure_servos`) 레지스터를 쓰려고 **토크를 잠깐 껐다 켠다**(`so101.py:607` → `:412`). 높이 든 자세에서 시작하면 그 순간 살짝 처질 수 있다 <!-- TODO: 확인 필요 — 실제 구동 때 관찰 -->. 직전 실행이 10초 안에 끝났다면 소유 프로세스가 아직 있어 재연결이 없다(붙기만 함).
- **그리퍼 과열(블록 = 눌리는 스펀지)**: 보호값은 연결할 때마다 들어간다 — Studio 녹화·원격조작과 `run_policy` 모두 `SO101.connect()` → `_configure_servos()`(`so101.py:593~623`)가 그리퍼(ID 6)에 Max_Torque_Limit 500·Protection_Current 250·Overload_Torque 25(최대 토크 50%·최대 전류 50%·과부하 때 25%, lerobot `so_follower.py:171~173` 주석과 같은 값)를 쓰고, Studio 로그(10-20~21)의 팔로워 연결 23회에 쓰기 실패 경고는 0건이다. 단 이 드라이버는 EPROM 잠금(Lock, 55번)을 풀지 않고, 실제로 걸리는 토크 한도(Torque_Limit, 48번)는 서보 전원을 켤 때 16번 값을 복사하므로(Feetech STS 메모리 표) 50% 한도는 EPROM 에 500 이 저장돼 있어야 살아 있다 — Studio 로봇 설정 마법사로 보정했다면 저장돼 있다(보정 끝에 Lock=0 으로 씀, `workers/robots/so101_setup_worker.py:555`·`643`) <!-- TODO: 확인 필요 — 10-19 등록 때 보정 경로 기록 없음. 팔을 꽂고 아무도 포트를 안 쥘 때 컨테이너의 scservo_sdk 로 그리퍼 16·48번 읽기(쓰기 없음) -->. 위험: 리더 그리퍼를 끝까지 쥐면 팔로워 그리퍼는 블록에 막혀 목표에 못 간 채 계속 밀고(위치 제어라 남은 거리만큼 힘을 냄), 정책도 그 값을 배워 실행 때 똑같이 민다 — 쥔 채 기다리는 `handoff` 가 가장 길다. 대응: ① 녹화 때 리더 그리퍼는 블록이 살짝 눌릴 만큼만 쥔다 ② 쉬거나 정책을 바꿀 땐 그리퍼를 연 채로 둔다 ③ 정책 하나 녹화가 끝날 때마다 그리퍼 서보를 만져 보고 뜨거우면 식힌 뒤 계속한다 ④ 그리퍼가 갑자기 힘이 빠지거나 Studio 가 "stopped responding"·"Could not connect" 를 내면 과부하 보호가 걸린 것일 수 있다 → 팔을 받치고 팔로워 전원을 껐다 켠다(Studio 도 같은 안내, `robots/catalog/so101.py:215~221`).

### 2.3 홈 자세 복귀

- **조기 종료(2.1)를 켜면(기본) 런타임 복귀는 끄고, 끝 자세가 시작 자세에서 `return_far` 6(약 11°) 넘게 떨어졌을 때만 복귀한다**(10-22 새벽).
  - 복귀하는 경우: 시간 경과·벽시계 상한·`stop(home=True)`. 드물게 `home_reached` 인데 6~8 떨어진 경우도.
  - 복귀하지 않는 경우: `stop()`(SIGKILL, 그 자리 정지)·오류.
  - 방법: `physicalai run` 이 끝난 뒤 `robot/coord_pick/stream.py` 의 `container_return` 을 같은 `docker exec -i … python -I - coord_pick` 방식으로 돌린다(컨테이너에 파일 없음).
    - SharedRobot 같은 이름 → 소유 프로세스가 10초 안이면 그대로 붙는다.
    - 지금 자세 → 시작 자세, 관절 36°/s 상한, 3틱 전 명령과 8° 넘게 3틱 연속 어긋나면 그 자리 유지.
    - `stop()` 이 이 프로세스도 끊는다(`_PROBE` 가 `coord_pick` 을 run 으로 셈).
  - 결과에 `end_dist`(끝 자세 거리)·`returned`({ok, reason, duration_s} 또는 null)가 붙는다.
  - 녹화 기록으로 보면 끝 자세 거리 6 초과는 40개 중 2개다.
- 조기 종료를 끄면 예전처럼 `--run.return_to_initial_state=true` 를 준다: 끝날 때 **실행을 시작할 때 읽은 자세**로 2.5초 동안 선형 복귀한다(`core.py` `_return_to_initial_state` 724, `_RETURN_DURATION_S` 42). 관절 하나라도 이동량의 15% 넘게 못 따라오면(충돌) 그 자리에서 멈춘다(`_RETURN_MAX_TRACKING_ERROR` 43).
  - 복귀 기준이 "시작 자세"라서 **매 실행을 홈 자세에서 시작해야** 홈으로 돌아온다. 녹화 가이드(`docs/recording_guide.md` 3.1)의 "시작·끝 자세는 홈"과 맞물린다.
  - 시간 경과(정상)·SIGINT(`--stop-home`, 벽시계 상한) 때 복귀한다. **`stop()` 기본(SIGKILL)은 복귀하지 않는다** — "멈춰"는 그 자리 정지가 안전하다고 봤다.
- 멈춘 뒤 홈으로 돌리는 법(사람이 할 일):
  1. Studio 원격조작: **리더 팔을 먼저 홈 자세에 두고** 원격조작을 시작한다. 시작되면 팔로워가 리더 자세로 곧장 간다(`TeleopSource.update` 는 리더 관절값을 그대로 보내고 완만한 접근이 없다). 홈에 오면 화면에서 나온다.
  2. 전원을 끄고 손으로: 팔을 받친 채 전원을 끈다(토크가 풀려 떨어진다).
  - 그냥 다음 정책을 실행하면 그 멈춘 자세가 시작 자세(=복귀 기준)가 되므로 하지 않는다.

### 2.4 Studio 세션이 로봇을 쥐고 있을 때

- Studio 의 원격조작·녹화·모델 실행 화면은 런타임 세션을 띄우고 팔로워를 `SharedRobot`(이름 = 팔로워 ID)으로 쥔다. 같은 이름으로 붙으면 **같은 소유 프로세스에 둘이 명령을 보내게** 된다(`_shared_robot.py`: 같은 이름은 attach).
- 그래서 `run_policy` 는 이름을 `ball-sorter-follower` 로 바꿔 붙는다. Studio 가 팔을 쥐고 있으면 장치 잠금(`serial:ttyACM0`)에 걸려 **연결 단계에서 실패**한다(`RobotDeviceAlreadyOwned`, 움직이기 전).
- 그 전에 시작 점검에서 거부한다(`reason=refused`): `GET /api/runtime/sessions` 가 비어 있지 않으면 `studio_session`, 컨테이너의 잠금 파일(`/tmp/physicalai-1000/robot-locks`)에 다른 소유자가 살아 있으면 `robot_busy`.
- **Studio 세션은 끄지 않는다**(Studio 변경 금지). 사람이 Studio 화면에서 ← 로 나온다. 탭만 닫아도 세션은 45초 뒤 끝난다(`RUNTIME_IDLE_TIMEOUT_S`, `settings.py:368`).
- 반대 방향: 우리 실행이 끝난 뒤 10초 동안은 우리 소유 프로세스가 팔을 쥐고 있어 Studio 세션이 "장치를 이미 다른 곳이 씀"으로 실패할 수 있다 → 10초 기다린다.
- **다음 실행은 이전 실행이 풀릴 때까지 기다린다**(`wait_previous`, 10-22 새벽 — 결과를 먼저 돌려주기 때문). `run_policy`·좌표 집기 실행기 모두 사전 점검 전에 기다린다.
  - 기다리는 것: 컨테이너 안 실행 프로세스(정책·좌표 집기·복귀)와 우리 카메라 publisher 가 없어질 때까지. 최대 15초, 넘으면 `refused`(`previous_not_released`).
  - **기다리지 않는 것: 소유 프로세스(장치 잠금).** 같은 이름(`ball-sorter-follower`)으로 다시 붙는 게 정상 경로이고(2.2), 10초 뒤 스스로 끝난다. 기다리면 회당 10초가 늘고 다시 연결할 때 토크가 잠깐 풀린다.
  - 다른 소유자(Studio 세션)의 잠금은 이전처럼 사전 점검이 거부한다.
  - 겹칠 수 있는 것: 에이전트 쪽 detect·말. `run_policy` 는 카메라를 놓은 뒤 돌아오므로, 그 직후 detect 가 위 카메라를 열 수 있다.
  - `stop()` 은 끝나는 중인 실행(SIGINT 를 받고 정리 중)도 SIGKILL 로 끊는다. 조사 스크립트가 실행 ID 와 상관없이 우리 표시가 붙은 실행 프로세스를 모두 끊기 때문이다.
- 좀비: Studio 컨테이너의 PID 1(`uv run … serve`)이 끝난 자식을 거두지 않아 소유 프로세스가 좀비로 남는다(10-21 실측: 세션이 끝났는데 잠금 파일의 pid 2개가 `State: Z`). 좀비는 잠금이 아니다(flock 은 풀림) → 조사 스크립트가 좀비를 거른다.

### 2.5 카메라 (비전과 같은 위 카메라)

- 카메라는 `SharedCamera` 다. 서비스 이름이 카메라 설정에서 나와 Studio 와 같으므로, Studio 가 카메라를 열고 있으면 그 publisher 에 붙어 프레임을 같이 받는다(충돌 아님).
- 우리가 띄운 publisher 는 구독자가 없어지면 `idle_timeout` 뒤 끝난다. Studio 기본은 5초(`_shared_camera.py:147`)인데 우리는 **1초**로 준다. `run_policy` 는 publisher 가 끝날 때까지 기다렸다 돌아온다(`camera_released`) → 비전의 `detect_objects()` 가 바로 장치를 열 수 있다.
- 반대로 비전이 카메라를 열고 있으면 우리 실행은 카메라 연결에서 실패한다(로봇 연결 다음 단계, 팔은 안 움직임) → `reason=error`.

## 3. `run_policy` 가 하는 일

1. `policies.json` 에서 모델 ID 를 찾는다. 모델·데이터셋·환경을 Studio API 로 읽는다(GET 만).
2. 환경의 `GET /api/projects/{p}/environments/{e}/runtime-config`(팔로워·카메라·리더 원격조작 설정)를 받아 고친다.
   - `action_source` 를 Studio 의 "Run model" 과 같은 `PolicySource`(OpenVINO, `SyncExecution(request_threshold=0.5)`, `ChunkedActionQueue`+`LerpSmoother`)로 바꾼다(Studio `runtime/config_builder.py` `policy_source_fragment` 와 같은 내용). 리더는 빠진다.
   - 모델 경로 = 컨테이너 안 `/app/storage/models/<모델ID>/exports/openvino`(볼륨의 원본을 읽기만).
   - 팔로워 포트를 **시리얼**(`/dev/serial/by-id/*5B90101138*`)로 다시 찾는다. 이 API 는 팔이 안 보이면 등록 때 포트를 그대로 넣는다(`allow_stored_port=True`, `api/environments.py:71`) → ACM 번호가 바뀌면 리더를 팔로워로 몰 수 있어서.
   - 카메라는 모델 입력 이름(`images.<이름>`)에 맞춰 고른다. 이름이 다르면 `camera_map` 이 필요하다(4절).
3. 시작 점검(`preflight`): 컨테이너 실행 중 · Studio 세션 없음 · 팔로워 잠금 없음 · 이미 도는 정책 없음 · 장치 경로가 컨테이너 안에 있음 · 카메라 USB 있음 · **`physicalai run … --print_config` 로 설정 해석 통과**(해석만 하고 끝남, 로봇·카메라 연결 없음, 0.57초).
   - 실제 실행에서는 설정 해시(정책·모델 ID·runtime 설정 JSON(포트·카메라 장치·콜백 포함)·실행 인자)마다 **하루 한 번만** 검사한다(10-22 새벽).
   - 통과만 `robot/.state/validated.json` 에 적고, 같은 날 같은 해시면 건너뛴다. 설정·USB 번호·모델이 바뀌거나 날이 바뀌면 다시 검사하고, 실패는 매번 다시 검사한다.
   - dry-run·`--link`·`preflight.py` 는 캐시를 안 쓰고 늘 검사한다.
   - 결과의 `validate_cached` 와 `.log` 첫 줄의 `print_config=cached|checked` 로 어느 쪽이었는지 보인다.
4. 실행: `docker --context default exec -u appuser -e BALL_SORTER_RUN=<run_id> -w /tmp physical-ai-studio-xpu …/physicalai run --config '<JSON>' --run.duration_s=… --run.return_to_initial_state=true`. 설정은 `--config` 에 JSON 문자열로 넘겨 **컨테이너 안 파일·볼륨에 아무것도 쓰지 않는다**(`run_model.sh` 는 볼륨에 `docker cp` 한다).
5. 기다리는 동안 stop 요청·벽시계 상한을 본다. 끝나면 카메라가 풀릴 때까지 기다리고 결과를 돌려준다.

`stop()`: 컨테이너 안에서 `BALL_SORTER_RUN` 환경변수가 붙은 `physicalai run` 프로세스를 찾아 SIGKILL(`home=True` 면 SIGINT). 호스트 사용자(uid 1001)는 컨테이너 appuser(uid 1000) 프로세스에 신호를 못 보내서 `docker exec` 로 한다(약 0.2초). `.state/stop` 파일을 남겨 다른 프로세스에서 부른 stop 도 실행 쪽이 `stopped` 로 기록하고, 컨테이너 프로세스가 뜨기 직전에 온 stop 도 대기 루프가 다시 보낸다.

## 4. `policies.json`

| 키 | 뜻 |
|---|---|
| `studio_url` `docker_context` `container` | Studio 주소, docker context(`default` 고정 — 6절), 컨테이너 이름 |
| `robot_name` | 우리가 쓰는 `SharedRobot` 이름. Studio 세션과 일부러 다르게(2.4) |
| `defaults.device` | OpenVINO 장치. `GPU`(지난 추론 실측 장치) |
| `defaults.timeout_s` | 제어 루프 시간(초). 정책별 값이 없을 때. `--link` 가 정책별 권장값을 넣는다 |
| `defaults.startup_budget_s` | 모델 로드·GPU 컴파일·연결에 줄 시간(벽시계 상한 계산용). 90 은 추정 <!-- TODO: 확인 필요 — 첫 실제 실행 때 측정 --> |
| `defaults.return_to_initial_state` | 끝날 때 시작 자세로 복귀 |
| `defaults.camera_idle_timeout_s` | 우리 카메라 publisher 가 구독자 없이 버티는 시간 |
| `policies.<이름>.model_id` | Studio 모델 ID. `null` 이면 실행 거부(`no_model`) |
| `policies.<이름>.environment_id` | 비우면 모델이 학습한 데이터셋의 환경 |
| `policies.<이름>.camera_map` | 모델 카메라 이름 → 환경 카메라 이름. 같으면 비운다. 예: `{"tower cam": "front", "follow cam": "gripper"}` |
| `policies.<이름>.timeout_s` `device` `task` | 정책별 덮어쓰기(ACT 는 `task` 를 안 쓴다) |
| `temporary` `note` | 사람용 표시. `pick_red` 는 지금 임시 연결(`temporary: true`) |

## 5. 내일(10-22) Studio 절차

전부 젠북 Studio 화면(Docker 판은 UI·API 모두 `http://localhost:7860`, 프로젝트 `class_pick`)에서 한다. 녹화 방법 자체는 `docs/recording_guide.md` 를 따른다.

### 5.1 데이터셋 만들기 — 이름 규칙

| 정책 | Dataset name | Task |
|---|---|---|
| `pick_defect` | `bs_pick_defect` | `pick_defect` |
| `pick_red` | `bs_pick_red` | `pick_red` |
| `pick_blue` | `bs_pick_blue` | `pick_blue` |
| `handoff` | `bs_handoff` | `handoff` |

- Datasets 탭 → **Create new dataset** → Create → `Environment` · `Dataset name` · `Task` 를 위 표대로 → Create new dataset.
- **네 개 모두 같은 Environment** 로 만든다. 카메라 이름이 모델 입력 이름(`images.<카메라 이름>`)이 되므로 넷의 카메라 이름이 같아야 실행 설정 하나로 돈다.
- 녹화 가이드는 카메라 이름을 `tower cam`(위)·`follow cam`(손목)으로 정했다. 지금 `env_24` 는 `front`·`gripper` 다. 바꾸려면 **녹화 전에** Studio 카메라 설정에서 바꾼다(녹화 뒤에 바꾸면 그 데이터로 학습한 모델과 환경 이름이 달라져 `camera_map` 이 필요하다). 어느 쪽이든 `run_policy` 는 돈다.
- ACT 는 Task 문구를 안 쓴다(입력에 없음). 정책 하나 = 데이터셋 하나 = 모델 하나다.

### 5.2 녹화

- `docs/recording_guide.md` 대로 정책당 50 에피소드. 에피소드는 **15초 안팎**(같은 정책 안 ±3초, 10-22 사용자 확정) → `timeout_s` 약 21초(90% + 3초, `--link` 가 데이터로 계산).
- 발표 때 무엇을 몇 번 돌리는지는 5.7(발표 실연, 정책 실행 5회).
- 녹화가 끝나면 **녹화 화면에서 ← 로 나온다**(세션이 팔을 쥐고 있으면 `run_policy` 가 거부한다).

### 5.3 원격 트레이너(desktop-3060ti)로 학습 보내기

1. Models 탭 → **Train model**.
2. Setup: `Dataset` = `bs_<정책>`, `Run on` = **desktop-3060ti** (`http://<트레이너-PC>:8001`). 상태 표시가 **Remote trainer selected** 인지 본다. **Remote trainer unavailable** 이면 보내지 말고 데스크톱 트레이너가 켜져 있는지 확인을 요청한다(데스크톱이 재부팅되면 트레이너가 꺼진다 — 10-21 08:30 사례). `GPU` 선택, Policy = **ACT**.
3. Training parameters: `Max Epochs` 기본 5 → **10** 권장(10-21 판: 30 에피소드·10 epoch 에서 val/loss 최저가 7 epoch, 0.330), `Batch Size` 8, `Precision` BF16 Mixed, `Augment images` 켬(10-21 판과 같게).
4. Export: **OpenVINO** 가 체크돼 있는지 확인(기본은 지원 형식 전부). 없으면 `run_policy` 가 `no_openvino` 로 거부한다.
5. 시작. 정책당 약 **1시간 17분**(증강 켬, 15초 에피소드 50개·10 epoch). 근거·겹치는 순서는 맨 끝 "재개" 절. 같은 트레이너로 보낸 학습은 **하나씩 차례로** 돈다 — 녹화가 끝날 때마다 바로 보내 두면 Studio 가 대기열에 넣고 앞의 것이 끝나면 이어서 시작한다.

### 5.4 학습 중 Stop 금지 — 모델이 안 남는다

- 학습 표의 Stop 확인창은 "Your model checkpoint will be saved at the current step"(`ui/…/job-table.tsx:231`)이라고 하지만 **사실이 아니다**. 원격 학습에서 Stop → 트레이너에 cancel(`services/training_backends/remote.py:724` `_handle_stop_request`) → 트레이너가 `Training canceled` 로 끝내고(`trainer/runner.py:110`) 압축·전송을 안 한다 → **모델 목록에 아무것도 안 생긴다**(로컬 학습도 같음, 10-20 job bf4cbaaf 사례).
- 에폭을 줄이고 싶으면 Stop 하지 말고 처음부터 작게 보낸다.
- 노트북·Studio 가 꺼져도 학습은 데스크톱에서 계속되고, Studio 가 다시 켜지면 결과를 받아 모델로 등록한다(10-21 사례). 단 데스크톱 트레이너 프로세스는 살아 있어야 한다.

### 5.5 완료된 모델을 `policies.json` 에 연결

```bash
cd ~/dev/projects/ball-sorter
.venv/bin/python -m robot.run_policy --models                  # 데이터셋 bs_pick_red 로 만든 가장 최근 ACT 의 ID 를 찾는다
.venv/bin/python -m robot.run_policy --link pick_red <모델ID>  # dry-run 점검을 통과해야만 고친다
.venv/bin/python -m robot.run_policy pick_red --dry-run        # 무엇이 실행될지 확인
```

- `--link` 는 OpenVINO export·카메라 이름·설정 해석을 점검한 뒤 `model_id` 를 쓰고, 데이터셋 에피소드 길이로 `timeout_s`(90% + 3초)를 넣고, `temporary` 를 끈다. 이전 파일은 `robot/policies.json.bak`.
- 모델은 이름이 전부 "ACT" 라서 **데이터셋 이름으로 구분**한다(학습 화면에 모델 이름 칸이 없다).
- 카메라 이름이 안 맞으면 `camera_mismatch` 로 거부하고 무엇을 `camera_map` 에 적을지 알려 준다.

### 5.6 첫 실제 실행 (사용자 입회)

1. Studio 화면에서 나오기 → `.venv/bin/python -m robot.preflight` 에서 팔로워·카메라·Studio 세션·잠금·그 정책이 정상인지.
2. 팔을 홈 자세에 두기(복귀 기준 = 시작 자세).
3. 한 손은 팔로워 전원 스위치에.
4. 짧게 먼저: `.venv/bin/python -m robot.run_policy pick_red --timeout 10 --yes`. 로그의 첫 줄부터 팔이 움직이기 시작할 때까지 시간(**적재 시간 L**)을 재서 `startup_budget_s` 를 고친다. 회당 전체 시간(결과의 `duration_s`)도 적는다 — 5.7 발표 실연 시간 계산에 쓴다.
5. 다른 터미널에서 `--stop` 시험 → 그 자리 정지, 10초 뒤에도 안 떨어지는지 확인. `--stop-home` 도 한 번.
6. 끝나면 `camera_released: true` 와 `python vision/camera.py --busy` → `[]` 확인.
7. **`camera_idle_timeout_s` 0.5초 확인**(지금은 1.0 그대로 — 5.7 손잡이).
   - 순서: 1.0 으로 위 1~6 이 끝난 뒤 `policies.json` `defaults.camera_idle_timeout_s` 를 0.5 로 바꿔 같은 정책을 3회 돌린다.
   - 셋 다 아래면 0.5 를 둔다(회당 −0.5초): `home_reached`(또는 `duration_elapsed`)로 끝남, `.log` 에 카메라 오류 없음, `.ticks.jsonl` 에 틱이 끝까지 있음, 결과 `tail_s` 가 1.0초쯤에서 0.5초쯤으로 줄었음.
   - 하나라도 아니면 1.0 으로 되돌린다. publisher 가 시작 때 구독자보다 먼저 꺼진 것이다(`capture/transport/_publisher_worker.py:492`).

### 5.7 발표 실연 (3분 — 계약서 2절 끝, 사용자 확정 10-21 밤)

| # | 말 | 정책 실행 |
|---|---|---|
| 시작 | 투입에 분홍 2·초록 1·불량 1(블록 4개) | |
| ① | 불량품 골라내 줘 (`defect`) | `pick_defect` 1회 |
| ② | 지금 배치 어때? (`scene`) | 없음(detect 만) |
| ③ | 불량 빼고 나머지 다 정리해 줘 (`plan`) | `pick_red` 2회 + `pick_blue` 1회 |
| ④ | 분홍 블록 하나 내 손에 줘 (`handoff`) | `handoff` 1회(분홍 그릇 → 손) |
| ⑤ | 결과 알려 줘 (`report`) | 없음 |
| | 합계 | **정책 실행 5회** |

- 계약서 2절 표의 전체 흐름(블록 11개)은 미리 찍은 성공 영상으로 보여 준다.
- **시간 — 10-22 새벽 계산(조기 종료·먼 끝 자세만 복귀·결과 먼저 돌려주기 반영).** L = docker exec 시작 → 첫 틱(모델 적재 등, 결과 `early_stop.load_s` 로 잰다).

  | 1회 구성 | 시간 | 근거 |
  |---|---|---|
  | 준비(계획·`--print_config` 검사·세션·조사) | 0.7초(그날 그 설정 처음) / **0.15초**(같은 날 같은 설정 — 검사 건너뜀) | 실측(0.05 + 0.57 + 0.08 + 0.02) |
  | 적재 | L | 미측정 <!-- TODO: 확인 필요 — 10-22 12:06 첫 시험의 load_s --> |
  | 정책 | 약 14초 | 15초 에피소드, 홈 복귀 정지 감지가 녹화 끝보다 약 1초 앞(2.1) |
  | SIGINT·런타임 닫기 | 약 0.5초 | 감시 0.1 + 조사 0.06(실측) + 닫기 <!-- TODO: 확인 필요 — 닫기 시간 --> |
  | 끝 → 결과 | 약 1.1초 | 카메라 놓기(idle 1.0초 + 확인 0.1초). docker exec 2.1초는 안 기다린다(2.1) |
  | **합계** | **약 15.75 + L**(검사 건너뜀) / 16.3 + L(처음) | 전에는 17.4 + L |

  - **5회**(실연은 정책 4종, `pick_red` 2회):
    - 같은 날 리허설로 4종을 한 번씩 돌려 두면 다 건너뛴다 → **약 78.8 + 5L 초, L ≤ 1.2초면 85초에 든다.**
    - 리허설 없이 처음이면 검사 4번이 붙는다 → 약 81.0 + 5L.
    - 끝 자세가 멀어 복귀하는 회는 +3~4초(녹화로는 20회에 1회꼴).
  - 바로잡음: 지난 계산(17.4 + L)은 카메라 놓기 1초를 exec 2.1초와 따로 더했다(실제로는 겹친다). 대신 준비 0.7초·닫기 0.2초를 빠뜨렸다. 합계는 우연히 같았다.
  - 그래서 이번 변경으로 줄어든 것은 회당 약 1.1초다(2.1초가 아님). exec 를 기다리는 동안 카메라 놓기도 같이 끝나고 있었기 때문이다.
- 남은 손잡이(회당):
  - ~~매번 하는 `--print_config` 검사 빼기~~ → 설정 해시별 하루 한 번으로 했다(3절 3번, 10-22 새벽).
  - `camera_idle_timeout_s` 1.0 → 0.5: −0.5초. **지금은 안 바꾼다** → 12:06 첫 시험에서 확인(5.6 7번). 되면 5회 약 76.3 + 5L(L ≤ 1.7초면 든다).
  - 적재 L 자체를 없애려면 런타임을 띄워 둔 채 정책만 바꿔야 한다. 큰 변경이라 10-23 전에는 권하지 않는다.
- 그래서 L 을 잰 뒤의 결정(총괄): 카메라 0.5초를 쓸지, 정책 몫을 78.8 + 5L 초(리허설 뒤)로 잡을지, 말(TTS)·화면 갱신을 정책 실행과 겹칠지(에이전트·UI 담당).

## 6. 함정

| 함정 | 내용 |
|---|---|
| docker context | 젠북 docker CLI 의 현재 context 가 `desktop-linux`(Docker Desktop 실행 중)라 그냥 `docker ps` 에는 Studio 가 안 보인다. Studio 는 Docker Engine(`default`). 그래서 `docker --context default` 를 쓴다. `~/Downloads/run_model.sh` 는 그냥 `docker` 라서 지금 상태면 "컨테이너가 안 떠 있어요"로 멈출 수 있다 |
| uid | 컨테이너 appuser = uid 1000(받아 쓴 이미지라 `.env` 의 `APP_UID=1001` 이 안 먹음), 호스트 사용자 = 1001 → 호스트에서 컨테이너 프로세스에 kill 불가 → stop 은 `docker exec` |
| 좀비 잠금 | 2.4 끝 문단. `pid` 가 살아 있는지만 보면 끝난 세션의 잠금을 "사용 중"으로 오판한다 |
| `duration_s` | 벽시계가 아니라 제어 주기 수(2.1) |
| 지나치게 긴 `timeout_s` | 블록을 하나 더 집으러 간다(2.1). dry-run 이 메모로 경고한다 |
| 장치 번호 | 팔로워는 시리얼로 다시 찾는다. 카메라는 Studio 지문(`bus·index·serial`) 그대로 두고 `UVCCamera` 가 가장 잘 맞는 장치를 고른다 — 10-21 손목 카메라 USB 자리가 `14.0-1.1`→`2.1` 로 바뀌었는데 Studio 세션은 정상이었다. dry-run 은 by-id 로 지금 `/dev/videoN` 을 보여 준다 |
| 컨테이너 안 장치 | 장치 경로가 컨테이너 안에 없으면 `device_not_in_container` 로 거부(그때는 Studio 컨테이너 재시작, 사용자). 컨테이너는 privileged 이고 `/dev` 는 따로 마운트하지 않는다(tmpfs). 10-21 밤 확인: 컨테이너 기동(10-20 10:44 KST) 뒤 꽂은 팔의 `ttyACM0`·`1` 이 컨테이너 안에 10-21 16:49 에 새로 생겨 있었다 → 나중에 꽂은 장치도 들어온다. 반대로 **뺀 장치의 노드는 컨테이너 안에 남는다**(호스트에 없는 `video0~3`·`ttyACM0·1` 이 남아 있음) → 컨테이너 쪽 경로 검사는 "있다"만 믿을 수 있고, 꽂혔는지는 호스트 by-id 로 본다 |
| 실행 사이의 "멈춰" | `stop()` 은 **돌고 있는** 실행만 끊는다. 에이전트가 "0개 될 때까지 반복" 중 실행과 실행 사이에 온 멈춰는 에이전트 루프가 직접 멈춰야 한다(다음 `run_policy` 는 지난 stop 기록을 지우고 시작한다) |

## 7. 설치한 패키지

없음. 프로젝트 `.venv` 에 이미 있는 PyYAML 6.0.3 과 표준 라이브러리만 쓴다(`coord_pick` 은 이미 있는 numpy 2.5.2 도 쓴다. IK 를 직접 풀어 placo 는 필요 없다). `preflight` 의 카메라 사용 중 확인은 `sudo -n /usr/bin/python3` 로 `/proc/*/fd` 를 읽는다(비밀번호 없는 sudo 가 안 되면 "못 봄"으로 표시하고 넘어감).

## 8. 남은 일

- [ ] 실제 구동 시험(사용자 입회, 5.6): 시작 시간 → `startup_budget_s`, stop·복귀 확인, 연결 순간 토크 깜빡임 관찰, 그리퍼 토크 한도(16·48번) 읽기(2.2 끝).
- [ ] 조기 종료 실제 확인(10-22 12:06): 결과 `early_stop.t_s`(15초 에피소드면 약 14초), `.ticks.jsonl` 로 정책이 홈에서 흔들리는 폭(still 3.0 안인지), 안 걸리면 `still`·`near` 를 올리거나 `enabled: false`. 정책마다 녹화가 끝나면 `python -m robot.home_stop_replay bs_<정책>` 으로 새 15초 데이터를 재생해 잘못 걸림 0 확인.
- [ ] 정책 4개 학습 후 `--link`(`pick_red` 의 임시 모델 교체 포함).
- [ ] 에이전트와 맞추기: `timeout_s` 는 넘기지 말고(None) `policies.json` 정책별 값을 쓰게. 반복 루프의 멈춰 처리(6절 마지막 행).
- [x] 대시보드 실제 모드: 인프라의 루트 `start_demo.sh` 가 해결했다(옵션 없으면 BALL_ROBOT=1, `--restart` 면 떠 있어도 다시 띄움). `preflight` 의 고치는 법도 `./start_demo.sh --restart` 로 바꿨다(10-22 새벽).
- [ ] 좌표 집기(선택, 10-23 오후 판단): 실행기는 써 둠 → 입회 순서 COORD_PICK.md 8절(보정 → 빈 손 궤적 → 블록 1개 → 작은 1 → 7, 약 1시간 40분). 에이전트 도구 형태는 9절.

## 재개 (2026-10-21 밤, 10-22 새벽 갱신) — 녹화 당일 시간표

> 근거는 전부 젠북 Studio 기록(컨테이너 로그·`/api/jobs`·`/api/dataset/{id}/episodes`, 읽기만)에서 뽑은 실측이다. 추정은 따로 표시했다.
> 10-22 새벽: 사용자가 에피소드 목표를 **15초**(같은 정책 안 ±3초)로 확정 → 녹화·학습·도착 시각을 다시 계산했다(20초 기준의 마지막 모델 17:38 은 폐기 → **15:57**). 비전 방법 B(녹화 영상에서 사진 뽑기)와 첫 모델 실물 시험을 넣은 하루 시간표는 "하루 시간표".

### 아침 순서

1. 젠북·Studio·OVMS·대시보드 켜기(`RESUME.md` 1절) → 팔·카메라 꽂기.
2. `.venv/bin/python -m robot.preflight --record` → 안됨이 0 이 될 때까지 "고치는 법"대로. 트레이너가 꺼져 있으면 데스크톱에서 다시 켠다(줄에 명령이 나온다).
3. 녹화가 다 끝나고 모델을 연결한 뒤, 시연·리허설 직전에는 `--record` 없이 `preflight`.

### 에피소드 1개에 드는 시간 — 실측

10-20 오후 `class_pick_24_2` 에피소드 21~30(종이컵 집기, 두 사람)을 이어 찍은 기록:

| 항목 | 값 | 출처 |
|---|---|---|
| 녹화 구간 | 16:54:25(녹화 시작) → 17:01:41(10번째 저장 뒤 목록 갱신) = **436초 / 10개** | 컨테이너 로그 `start_recording_mutation`, `GET …/episodes` |
| 에피소드 길이 합 | 201.5초(평균 20.2초, 14.1~31.0초) | `/api/dataset/…/episodes` |
| **한 개당 실제 소요** | **43.6초** = 길이 20.2 + 저장·되돌리기·다시 시작 23.4 | |
| 30개 전체 길이 | 평균 19.8초(중앙 19.1, 90% 24.0) | 같은 API |

### 녹화 예상 — 에피소드 15초(10-22 새벽 사용자 확정, 같은 정책 안 ±3초), 정책당 50개 남기기

| | 계산 | 시간 |
|---|---|---|
| 한 개당 | 길이 15 + 저장·되돌리기·다시 시작 23.4(위 실측, 종이컵) <!-- TODO: 확인 필요 — 블록 되돌리기 시간은 첫 정책에서 실측 --> | 38.4초 |
| 찍는 개수 | 50 + 버리는 것 약 15%(녹화 가이드의 "17세트 → 18~20세트") ≈ **58개** | |
| 찍기 | 58 × 38.4초 | 약 37분 |
| 정책 바꿀 때 | 데이터셋 만들기·작업대를 그 정책 장면으로·처음 3세트로 기준 길이 잡기 <!-- TODO: 확인 필요 — 추정 --> | 약 10분 |
| **정책당** | `pick_defect`·`pick_red`·`pick_blue` 각 약 47분, `handoff` 약 52분(손바닥 위 블록 치우기) <!-- TODO: 확인 필요 — 추정 --> | **4개 약 3시간 13분** |

- 20초였다면 정책당 약 50분(넷 3시간 30분)이었다. 되돌리기가 한 개당 시간의 60% 라 길이를 줄여도 녹화는 6% 만 준다. 학습은 프레임에 비례해 25% 준다(아래).

### 3060 Ti ACT 학습 시간 — 실측

| job(10-20) | 데이터 | 설정 | 스텝 | 학습 속도 | 내보내기·압축 | 모델 받기 | 보낸 뒤 전체 |
|---|---|---|---|---|---|---|---|
| 6522e505 | 10개 4,097프레임 | 30 epoch, 증강 끔 | 13,980 | **7.05 step/s** | 4초 | 919.9MiB 101초 | 35분 54초 |
| 15a76755 | 20개 11,815프레임 | 12 epoch, 증강 끔 | 15,756 | **7.49 step/s** | 5초 | 722.7MiB 68초 | 37분 35초 |
| 56ceb544 | 30개 17,861프레임 | 10 epoch, **증강 켬** | 20,400 | **5.76 step/s** | — (젠북이 꺼져 있어 로그 없음) | 722.7MiB 135초 | (중간에 끊김) |

- 출처: 젠북 Studio 컨테이너 로그(`Training progress: step=…`, `Exporting to openvino`, `Downloaded model artifact`)와 `/api/jobs`. 학습 속도는 검증(epoch 마다 약 6초) 시간을 포함한 값이다.
- 기존 README 의 "내보내기 약 12분"은 틀렸다(위키 기록의 시각을 잘못 읽음). `Exporting to openvino format` → `Training finished` 가 4~5초다.
- 데이터 올리기는 1MB/s 안팎(19.6MiB 29초, 36.5MiB 40~45초, 52.3MiB 50초), 첫 스텝까지 25~40초.

### 50 에피소드 학습 추정 — 15초 에피소드 (정책당)

| | 증강 켬(5.3 권장) | 증강 끔 |
|---|---|---|
| 프레임 | 50 × 15초 × 30fps = 22,500. 검증에 에피소드 10%(5개)를 떼면 학습 20,250 | 같음 |
| 스텝 | 20,250 ÷ batch 8 = 2,531/epoch × 10 epoch = **25,310** | 같음 |
| 학습 | 25,310 ÷ 5.76 = 73분 | 25,310 ÷ 7.3 = 58분 |
| 올리기·시작·내보내기·받기 | 약 3.5분(약 66MB 올리기 1.1분, 시작 0.5분, 받기 1~2분) | 같음 |
| **정책당** | **약 1시간 17분** | **약 1시간 1분** |

<!-- TODO: 확인 필요 — 에피소드 평균 15초 가정. 실제 평균이 다르면 학습 시간이 비례해서 바뀐다(1초당 약 5분) -->

### 녹화와 학습 겹치기

- Studio 는 **트레이너 하나에 학습 하나만** 돌린다(`workers/training_worker.py:66` `_target_is_busy`, 키 = 트레이너+GPU `services/training_targets/remote.py:45`). 보내기는 막지 않고 대기(pending)로 쌓았다가 앞의 것이 끝나면 보낸 순서대로 시작한다(`services/job_service.py:46`). → **녹화 하나 끝날 때마다 바로 보낸다.**
- 병목은 여전히 트레이너다(녹화 47분 < 학습 77분). 그래서 녹화 사이에 비전 작업·첫 시험·점심을 넣어도, **녹화 k 를 학습 k−1 이 끝나기 전에만 끝내면** 모델 도착이 안 늦어진다. 아래 시간표의 녹화 마감은 12:04 · 13:21 · 14:38 이다.
- 대기 중인 학습은 시작할 때 젠북에서 데이터를 올리고 끝나면 젠북으로 받는다 → **마지막 모델을 받을 때까지 젠북·Studio 를 켜 두고 어댑터를 꽂아 둔다**(끊겨도 다시 켜면 이어서 받지만 그동안 줄이 멈춘다, 10-21 아침 사례).

### 하루 시간표 (10-22, 녹화 10:00 시작)

젠북 GPU 규칙(계약서 5절·비전 README 방법 B):
- YOLO 학습(xpu)은 **OVMS 를 내리고, Studio 녹화도 정책 시험도 없는 동안에만** 돌린다. Studio 는 녹화 영상을 GPU 로 압축하고, 정책 추론도 GPU 를 쓴다.
- 정책 시험(OpenVINO 추론)과 OVMS 는 같이 써도 된다.
- 녹화 영상에서 사진 뽑기는 **Studio 녹화 화면을 ← 로 닫은 뒤에만** 한다.

| 시각 | 사람·팔(녹화·시험) | 3060 Ti 학습(한 줄) | 젠북 GPU·비전 | 비고 |
|---|---|---|---|---|
| 09:00~09:20 | 작업대 세팅(흰 식탁보·그릇·블록 분류), 팔·카메라 꽂기, `preflight --record` | 트레이너 켜짐 확인 | | 안됨 0 이 될 때까지 |
| 09:20~09:50 | | | 비전 0~2단계(카메라 고정·60Hz·노출·구역) + 직접 촬영 20장(손 10·섞은 장면 10) | 위 카메라를 비전이 쓴다 → Studio 녹화 화면을 열지 않는다 |
| 09:50~10:00 | 여유(첫 데이터셋 `bs_pick_defect` 만들기) | | | |
| **10:00~10:47** | **녹화 ① `pick_defect`** | | | |
| 10:47 | 학습 보냄 | ① `pick_defect` **10:47~12:04** | | |
| 10:47~11:10 | | | 비전: `pick_defect` 뽑기 → 자동 라벨 → 10장 검수·`--calibrate` → 나머지 검수 | 녹화 화면 닫힌 동안 |
| **11:10~11:57** | **녹화 ② `pick_red`** | | | 마감 12:04 |
| 11:57 | 학습 보냄(대기) | | 비전: `pick_red` 뽑기·자동 라벨(검수는 점심에) | |
| 12:04 | `--models` → `--link pick_defect` | ② `pick_red` **12:04~13:21** | | 모델 ① 도착 |
| **12:06~12:26** | **★ `pick_defect` 첫 실물 시험**: 5.6(10초 짧게 → stop·복귀, **적재 시간 L·회당 전체 시간·조기 종료가 걸린 시점(`early_stop.t_s`)** 재기 — 5.7 실연 예산) → **10회 시험** | | | 실행 경로 전체를 녹화 ③ 전에 확인 |
| **12:26~13:13** | **녹화 ③ `pick_blue`** | | | 마감 13:21 |
| 13:13 | 학습 보냄(대기) | | 비전: `pick_blue` 뽑기 | |
| 13:13~13:48 | **점심** | ③ `pick_blue` **13:21~14:38** | 13:13~13:28 검수(`pick_red`·`pick_blue`) → **13:28~13:43 1차 YOLO 학습**(약 160장 12분, OVMS 내림) → NPU 확인 → OVMS 올림 | 13:21 모델 ② 도착 → `--link pick_red`(1분) |
| **13:48~14:40** | **녹화 ④ `handoff`** | | | 마감 14:38(2분 넘음 — 녹화를 2분 당기면 0) |
| 14:40 | 학습 보냄 → `--link pick_blue` | ④ `handoff` **14:40~15:57** | 비전: `handoff` 뽑기(시험 시작 전에) | 14:38 모델 ③ 도착 |
| 14:45~15:15 | `pick_red` 10회 → `pick_blue` 10회 | | (다른 한 사람) `handoff` 자동 라벨 → 검수(손 박스 약 100장) | 검수는 CPU·브라우저라 시험과 같이 해도 된다 |
| 15:15~15:42 | 쉼 | | **2차 YOLO 학습**(약 330장 25분, OVMS 내림) → NPU·bench → OVMS 올림 | 정책 시험·녹화 없음 |
| 15:42~15:57 | 대기(OVMS 올린 뒤 에이전트 명령 1~3 을 가짜 모드로 점검 가능) | | | |
| 15:57 | `--link handoff` | | | 모델 ④ 도착 = **마지막** |
| 16:00~16:15 | `handoff` 10회(손 자리에 B 의 손) | | 2차 모델로 `hand.present` 확인 | |
| 16:15~ | 명령 1→5 통합 리허설 | | | 정책 4개 × 10회 시험 끝 |

- 10회 시험 한 번 = 실행 21초(`timeout_s`) + 시작(모델 적재) + 복귀 2.5초 + 장면 되돌리기 약 20초 → 10회 약 12~15분으로 잡았다 <!-- TODO: 확인 필요 — 시작 시간은 12:06 첫 시험에서 잰다 -->. 결과(성공 수)는 정책마다 적어 둔다.
- 녹화가 밀릴 때 지킬 순서: ① 녹화 마감(12:04·13:21·14:38) → ② 첫 실물 시험 → ③ 비전·점심. 비전 검수와 점심은 뒤로 밀어도 모델 도착이 안 늦어진다. 녹화 마감을 넘기면 넘긴 만큼 뒤 모델이 다 늦어진다.
- 1차 YOLO 모델에는 손이 없다(`handoff` 전). 그 전까지 `hand.present` 는 늘 false 다.

### 시간을 줄이는 손잡이 (15초 에피소드, 위 녹화 일정 그대로일 때)

| 바꿀 것 | 학습 시간(정책당) | 마지막 모델 도착 | 대가 |
|---|---|---|---|
| 그대로(증강 켬, 10 epoch) | 1시간 17분 | **15:57** | |
| 증강 끔 | 1시간 1분 | 약 15:41 — 이때부터는 녹화가 병목 | 10-21 판(증강 켬)과 다른 조건. 조명·위치 변화에 약해질 수 있다 <!-- TODO: 확인 필요 --> |
| 증강 켬, 7 epoch | 약 55분 | 약 15:35 | 10-21 판 val/loss 최저가 7 epoch 였다. 단 val/loss 가 성공률과 같지는 않다 |
| 증강 끔, 7 epoch | 약 44분 | 약 15:24 | 위 둘 다 |

- 학습을 줄이면 녹화(점심·비전·첫 시험 포함)가 병목이 되어 도착이 크게 당겨지지 않는다. 그래서 **그대로(증강 켬, 10 epoch)를 권한다.**
- 어느 쪽이든 **학습 중 Stop 은 누르지 않는다**(5.4). 줄이고 싶으면 보낼 때 정한다.
- 젠북 Intel GPU 로 동시에 하나 더 돌리는 것은 권하지 않는다. 10-19 실측 1.1 step/s 라 25,310 스텝에 6시간 반이고, 비전 학습·OVMS·정책 추론과 GPU 를 나눠 쓴다.

### timeout_s — 15초 에피소드에 맞춤

- 규칙은 그대로 "에피소드 길이 90% + 3초"다(`episode_stats`: 올림, 2.1).
- 15±3초 에피소드의 90% 는 약 17~18초다 → **`policies.json` 기본값을 40 → 21초로 바꿨다**(10-22 새벽).
- `--link` 가 정책마다 실제 녹화 데이터로 다시 계산해 덮어쓴다(예: 90% 17.4초 → 21).
- 벽시계 상한 = 90 + 2×21 + 2.5 + 15 = 약 150초.
- 에이전트는 `timeout_s` 를 넘기지 않는다(None → policies.json 값). 계약서 예시의 40 을 넘기면 남는 19초 동안 블록을 하나 더 집으러 간다(2.1).
