#!/usr/bin/env bash
# 시연 날 한 줄 기동 — 카메라 고정값 → OVMS(+무응답 감시) → 대시보드 → 점검표(robot.preflight) → 마이크 → 브라우저 주소.
#   ./start_demo.sh            실제 로봇 모드(BALL_ROBOT=1). 기동만으로는 팔이 안 움직이고, 명령을 내려야 움직인다(사용자 입회 때만)
#   ./start_demo.sh --mock     가짜 작업대(BALL_MOCK=1) — 하드웨어 없이 리허설. 밤 시험은 UI_TTS_OUT=silent 를 앞에 붙인다
#   ./start_demo.sh --restart  대시보드가 떠 있어도 고른 모드로 다시 띄운다(OVMS 는 그대로)
# 이미 떠 있는 OVMS·대시보드는 건드리지 않고 상태만 보여 준다. 인터넷을 부르지 않는다(와이파이 꺼도 됨).
# 종료 코드: OVMS·대시보드가 안 뜨면 1. 실제 모드에선 점검에 "안됨"이 있어도 1(--mock 은 팔·카메라·Studio 항목을 안 본다).
set -u -o pipefail
cd "$(dirname "$(readlink -f "$0")")"
D=$PWD
PY=$D/.venv/bin/python
OVMS=http://127.0.0.1:8100
UI_PORT=${UI_PORT:-7870}
DASH=http://127.0.0.1:$UI_PORT/
WANT=(qwen3-4b qwen3-embedding qwen3-reranker whisper)
MODE=robot RESTART=0
for a in "$@"; do
  case $a in
    --mock) MODE=mock ;;
    --restart) RESTART=1 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "모르는 옵션: $a (쓸 수 있는 것: --mock --restart)"; exit 2 ;;
  esac
done
label() { case $1 in mock) echo "가짜 작업대(BALL_MOCK=1)" ;; robot) echo "실제 로봇(BALL_ROBOT=1)" ;; *) echo "$1" ;; esac; }

# 4모델이 모두 AVAILABLE 이면 0. "준비 n/4 · 떠 있는 것" 한 줄을 찍는다
ovms_ready() {
  "$PY" -I - "$OVMS" "${WANT[@]}" <<'PY'
import json, sys, urllib.request
base, want = sys.argv[1], sys.argv[2:]
try:
    ids = [m["id"] for m in json.load(urllib.request.urlopen(base + "/v3/models", timeout=3))["data"]]
    cfg = json.load(urllib.request.urlopen(base + "/v1/config", timeout=3))
except Exception as e:
    print(f"응답 없음({type(e).__name__})"); sys.exit(2)
st = {k: (v.get("model_version_status") or [{}])[0].get("state") for k, v in cfg.items()}
ok = [w for w in want if w in ids and st.get(w) == "AVAILABLE"]
print(f"준비 {len(ok)}/{len(want)} · " + ", ".join(i + ("" if st.get(i) == "AVAILABLE" else f"({st.get(i)})") for i in ids))
sys.exit(0 if len(ok) == len(want) else 1)
PY
}
# ball-ovms 유닛 상태. Restart=on-failure 로 다시 뜨는 동안(3초 대기 + 적재)은 activating 이라 is-active 가 실패한다 → 떠 있는 것으로 친다
ovms_state() { systemctl --user show ball-ovms -p ActiveState --value 2>/dev/null || echo inactive; }
ovms_up() { case $(ovms_state) in active|activating|reloading) return 0 ;; *) return 1 ;; esac; }
ui_up() { curl -s -o /dev/null -m 3 -w '%{http_code}' "$DASH" | grep -q '^200$'; }
ui_mode() {
  local env; env=$(systemctl --user show ball-ui -p Environment --value 2>/dev/null)
  case " $env " in
    *" BALL_MOCK=1 "*) echo mock ;;
    *" BALL_ROBOT=1 "*) echo robot ;;
    *" BALL_ROBOT=dry "*) echo "로봇 시험(BALL_ROBOT=dry, 안 움직임)" ;;
    *) echo "로봇 가짜(BALL_ROBOT 없음)" ;;
  esac
}

# 카메라 고정값(camera_ctrls.conf): 재부팅·USB 다시 꽂기 때마다 풀린다(10-23 09:32·09:39·09:58 손으로 넣음).
# 단계마다 따로 넣고(auto 를 먼저 꺼야 값이 먹음) 제어마다 다시 읽어 맞대 본다. Studio 가 카메라를 쥐고 있어도 -c 는 된다(10-23 10:25).
# 안 맞아도 기동은 계속(종료 코드에 안 넣음)
cam_set() {  # $1 장치, $2… 단계(쉼표로 이은 이름=값). 실패한 단계의 오류를 한 줄로 낸다
  local dev=$1 stage out; shift
  for stage in "$@"; do
    out=$(v4l2-ctl -d "$dev" -c "$stage" 2>&1 </dev/null) || printf ' %s' "${out//$'\n'/ }"
  done
}
cam_check() {  # $1 장치, $2… 단계. 다른 것을 "이름(바람 값·읽음 값)" 으로 낸다(메뉴형 "1 (Manual Mode)" 는 앞 숫자만)
  local dev=$1 kv k v got kvs; shift
  IFS=, read -ra kvs <<< "$(IFS=,; echo "$*")"   # 단계들을 이름=값 하나씩으로
  for kv in "${kvs[@]}"; do
    k=${kv%%=*}; v=${kv#*=}
    got=$(v4l2-ctl -d "$dev" --get-ctrl="$k" 2>/dev/null </dev/null); got=${got#*: }; got=${got%% *}
    [ "$got" = "$v" ] || printf ' %s(바람 %s·읽음 %s)' "$k" "$v" "${got:-실패}"
  done
}
echo "== 카메라 고정값 (camera_ctrls.conf)"
if ! command -v v4l2-ctl >/dev/null; then
  echo "  v4l2-ctl 없음 → 건너뜀 (v4l-utils)"
elif [ ! -f "$D/camera_ctrls.conf" ]; then
  echo "  camera_ctrls.conf 없음 → 건너뜀"
else
  while read -r name dev stages; do
    case $name in ''|'#'*) continue ;; esac
    read -ra st <<< "$stages"
    if [ ! -e "$dev" ]; then echo "  $name: 없음 → 건너뜀 ($(basename "$dev"))"; continue; fi
    err=$(cam_set "$dev" "${st[@]}"); bad=$(cam_check "$dev" "${st[@]}")
    if [ -z "$bad$err" ]; then echo "  $name: $(IFS=,; echo "${st[*]}" | tr , ' ') — 맞음"
    else echo "  ⚠ 주의 $name:${bad}${err:+ · 넣기 오류:$err}"; fi
  done < "$D/camera_ctrls.conf"
fi

echo "== 1. OVMS ($OVMS)"
if s=$(ovms_ready); then
  echo "  이미 떠 있음 — 건드리지 않음 · $s"
else
  case $(ovms_state) in
    active|reloading) echo "  적재 중 · $s" ;;
    activating) echo "  재시작 중(Restart=on-failure) → 기다림 · $s" ;;
    *)
      for i in $(seq 15); do [ "$(ovms_state)" = deactivating ] || break; sleep 2; done   # 멈추는 중이면 끝날 때까지
      echo "  꺼져 있음 → start_ovms.sh (첫 적재 7~14초)"
      ./start_ovms.sh | sed 's/^/  /' || { echo "  start_ovms.sh 실패"; exit 1; } ;;
  esac
  for i in $(seq 60); do
    s=$(ovms_ready) && break
    ovms_up || { echo "  OVMS 가 멈춤($(ovms_state)) — 로그: tail -30 logs/ball-ovms.log · 5분에 5번 넘게 죽으면 더 안 띄운다(journalctl --user -u ball-ovms)"; exit 1; }
    sleep 2
  done
  ovms_ready >/dev/null || { echo "  120초 안에 4모델이 안 뜸 · $s — 로그: logs/ball-ovms.log"; exit 1; }
  echo "  떴음 · $s"
fi
if systemctl --user is-active -q ball-ovms; then
  pid=$(systemctl --user show ball-ovms -p MainPID --value)
  mem=$(systemctl --user show ball-ovms -p MemoryCurrent --value)
  cfg=$(tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null | grep -A1 -- --config_path | tail -1)
  echo "  유닛 ball-ovms · 구성 $(basename "${cfg:-?}") · 메모리 $(( ${mem:-0} / 1024 / 1024 ))MB · 가용 $(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)GB"
fi
# AVAILABLE 이어도 답을 못 할 수 있다(10-22 16:33~17:00 펜스 만료 뒤 27분 무응답). 네 모델에 한 번씩 물어 확인하고 데운다.
# 재부팅 뒤 첫 임베딩은 53초 걸렸다(10-22 22:10, SETUP.md §7.9) — 첫 기준서 질문 대신 여기서 치른다
# OVMS 를 띄운 뒤 첫 구조화 출력(json_schema)은 문법 준비로 약 9초 걸린다(10-23 10:06 첫 의도 판별이 5초 제한에 걸려 "명령 아님").
# 스키마마다가 아니라 프로세스마다 한 번이라(새 스키마도 0.7초, 10-23 10:09) 의도 판별(IntentOut)과 비슷한 작은 스키마로 치른다
echo "  응답 확인(네 모델 한 번씩 + 구조화 출력 · 재부팅 뒤 첫 임베딩은 1분 가까이, 첫 구조화 출력은 10초 가까이 걸릴 수 있음)"
"$PY" -I - "$OVMS/v3" "$D/ui/samples/tts_F1_01.wav" <<'PY' || { echo "  ⚠ OVMS 가 답을 안 함 → systemctl --user stop ball-ovms; ./start_ovms.sh 뒤 다시"; exit 1; }
import json, sys, time, requests
base, wav = sys.argv[1], sys.argv[2]
def chat(): requests.post(f"{base}/chat/completions", timeout=30, json={"model": "qwen3-4b", "max_tokens": 1, "messages": [{"role": "user", "content": "안녕"}]}).raise_for_status()
def so():  # 에이전트 의도 판별과 같은 꼴: response_format json_schema(strict) · 생각 끔 · max_completion_tokens
    r = requests.post(f"{base}/chat/completions", timeout=60, json={
        "model": "qwen3-4b", "temperature": 0, "max_completion_tokens": 32, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "분홍 블록 두 개 옮겨"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "WarmIntent", "strict": True, "schema": {
            "type": "object", "additionalProperties": False, "required": ["intent", "count"],
            "properties": {"intent": {"type": "string", "enum": ["red", "blue", "none"]},
                           "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}}}})
    r.raise_for_status()
    json.loads(r.json()["choices"][0]["message"]["content"])  # 스키마대로 JSON 이 나와야 통과
def emb(): requests.post(f"{base}/embeddings", timeout=120, json={"model": "qwen3-embedding", "input": ["불량 블록"]}).raise_for_status()
def rr(): requests.post(f"{base}/rerank", timeout=30, json={"model": "qwen3-reranker", "query": "불량", "documents": ["불량 블록", "분홍 블록"]}).raise_for_status()
def stt():
    with open(wav, "rb") as f:
        requests.post(f"{base}/audio/transcriptions", timeout=30, files={"file": f}, data={"model": "whisper"}).raise_for_status()
out, ok = [], True
for name, fn in (("qwen3-4b", chat), ("구조화 출력", so), ("임베딩", emb), ("리랭크", rr), ("whisper", stt)):
    t0 = time.perf_counter()
    try:
        fn(); out.append(f"{name} {time.perf_counter() - t0:.1f}초")
    except Exception as e:
        ok = False; out.append(f"{name} 실패({type(e).__name__})"); break  # 하나가 막히면 나머지도 막혀 있다 — 시간 제한을 다 기다리지 않는다
print("  답 확인 · " + " · ".join(out))
sys.exit(0 if ok else 1)
PY
# GPU HANG(SETUP.md §8-3·8-13): 이번 부팅 전체와, 지금 OVMS 가 뜬 뒤를 따로 센다.
# 자동 재시작(Restart=on-failure)이 있으면 "뜬 뒤"는 마지막 재시작부터라 그 원인 사건이 빠진다 → 재시작 횟수(NRestarts, start_ovms.sh 기동 뒤)를 같이 찍는다
hang_log() { journalctl -k -b --no-pager -o short-iso "$@" 2>/dev/null | grep -iE 'gpu hang|fence expiration'; }
n=$(hang_log | wc -l); last=$(hang_log | tail -1 | cut -c12-19)
since=$(systemctl --user show ball-ovms -p ActiveEnterTimestamp --value 2>/dev/null | awk '{print $2" "$3}')
nr=$(systemctl --user show ball-ovms -p NRestarts --value 2>/dev/null)
tail_msg=""
[ -n "${since// }" ] && tail_msg=" · 지금 OVMS 기동(${since:11:5}) 뒤 $(hang_log --since "$since" | wc -l)건 · 자동 재시작 ${nr:-0}번"
echo "  커널 GPU HANG·fence: 이번 부팅 ${n}건${last:+(마지막 $last)}${tail_msg}"
# 무응답 감시(ovms_watch.py): 커널 펜스 만료 줄이면 바로, 아니면 60초마다 묻는 LLM 1토큰이 2번 연속 답이 없으면 OVMS 를 KILL → Restart=on-failure 가 다시 띄운다
W=ball-ovms-watch
if systemctl --user is-active -q $W; then
  echo "  무응답 감시: 이미 떠 있음 · 기록 logs/$W.log"
else
  systemctl --user reset-failed $W 2>/dev/null
  if systemd-run --user --unit=$W --collect -p Restart=on-failure -p RestartSec=10 --working-directory="$D" \
       -p StandardOutput=append:"$D/logs/$W.log" -p StandardError=append:"$D/logs/$W.log" \
       -E WATCH_INTERVAL_S="${WATCH_INTERVAL_S:-60}" "$PY" -I "$D/ovms_watch.py" >/dev/null 2>&1; then
    echo "  무응답 감시: 띄움(유닛 $W, 커널 펜스 줄 + ${WATCH_INTERVAL_S:-60}초마다 1토큰) · 기록 logs/$W.log · 끄기: systemctl --user stop $W"
  else
    echo "  ⚠ 무응답 감시를 못 띄움 — 시연은 계속 가능, OVMS 가 답을 안 하면 손으로: systemctl --user kill -s KILL ball-ovms"
  fi
fi

echo "== 2. 대시보드 ($DASH)"
start_ui() {
  # 10-23: 기본 = 좌표 집기(YOLO+IK). "학습한 동작으로"라고 하면 ACT
  if [ "$MODE" = mock ]; then (unset BALL_ROBOT; export BALL_MOCK=1; ui/start_ui.sh)
  else (unset BALL_MOCK; export BALL_ROBOT=1 BALL_COORD=${BALL_COORD:-1}; ui/start_ui.sh); fi 2>&1 | sed 's/^/  /'
  for i in $(seq 45); do
    ui_up && { echo "  응답 200 · 모드 $(label "$(ui_mode)")"; return 0; }
    systemctl --user is-active -q ball-ui || break
    sleep 2
  done
  echo "  90초 안에 응답 없음 — 로그: tail -30 ui/ui.log"; return 1
}
if systemctl --user is-active -q ball-ui; then
  for i in $(seq 15); do ui_up && break; sleep 2; done   # 막 띄운 참이면 잠깐 기다린다
fi
if systemctl --user is-active -q ball-ui && ui_up; then
  cur=$(ui_mode)
  if [ "$RESTART" = 1 ]; then
    echo "  떠 있음(모드 $(label "$cur")) → --restart: $(label $MODE) 로 다시 띄움"
    start_ui || exit 1
  else
    echo "  이미 떠 있음 — 건드리지 않음 · 모드 $(label "$cur")"
    [ "$cur" != "$MODE" ] && echo "  ⚠ 고른 모드($(label $MODE))와 다름 → ./start_demo.sh$([ $MODE = mock ] && echo ' --mock') --restart"
  fi
else
  systemctl --user is-active -q ball-ui && echo "  유닛은 떠 있는데 응답 없음 → 다시 띄움" || echo "  꺼져 있음 → $(label $MODE) 로 띄움"
  start_ui || exit 1
fi

echo "== 3. 점검 (robot.preflight, 시연 기준)"
[ "$MODE" = mock ] && echo "  (--mock: 팔·카메라·Studio·정책 항목은 가짜 작업대에선 안 쓰므로 안됨이어도 넘어간다)"
"$PY" -m robot.preflight; PF=$?

# 재부팅 뒤 기본 입력이 이어폰 단자 마이크로 잡히면 크롬 녹음이 완전 무음이 된다(10-22 밤) — 안됨이어도 기동은 계속
echo "== 마이크 (ui.mic_check)"
"$PY" -m ui.mic_check --device | sed "s/^/  /" || true

echo "== 4. 브라우저"
echo "  젠북에서:       $DASH"
echo "  다른 PC 에서:   ssh -L $UI_PORT:127.0.0.1:$UI_PORT robot-pc  →  $DASH  (대시보드는 127.0.0.1 에만 열림)"
if [ "$MODE" = mock ]; then exit 0; fi
exit $PF
