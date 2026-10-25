#!/usr/bin/env bash
# ovms_watch.py 단독 시험 — 진짜 OVMS·GPU 는 안 건드린다. 가짜 서버를 user 유닛 ball-watchtest-ovms 로 띄우고
# (Restart=on-failure·RestartSec·StartLimit 은 start_ovms.sh 와 같은 값) 감시를 그쪽에 붙인다. 시간만 줄였다(60초 → 1초 간격).
# 커널 줄은 가짜 파일을 tail -F 로 읽게 해서(WATCH_KMSG_CMD) 펜스 만료·GPU HANG 줄을 흉내 낸다.
# 확인: ① 적재 중(연결 거절) 안 셈 ② 2번 연속 답 없음 → KILL ③ 재시작 대기(activating) 안 셈·새 프로세스 다시 예열
#       ④ stop 으로 내린 것은 안 건드림 ⑤ 예열 유예가 지나도 못 답하면 셈 ⑥ 5분에 5번 넘게 뜨면 더 안 뜸(StartLimitBurst=5)
#       ⑦ 새 프로세스마다 첫 요청은 구조화 출력(json_schema) 1번, 그 뒤는 1토큰 ⑧ 구조화 출력이 거절돼도(400) 1토큰이 답하면 KILL 안 함
#       ⑨ 지금 프로세스 스레드의 펜스 만료 줄 → 물음 실패 없이 바로 KILL(2줄이어도 1번) ⑩ 남의 프로세스 펜스 줄·GPU HANG 줄은 안 건드림
#       ⑪ KILL 뒤 간격을 안 기다리고 재시작을 보고 적재 중 짧게 다시 물어 예열 ⑫ 예열 뒤 1토큰은 간격대로
# 사용: bench/ovms_watch_test.sh   (약 3분, 끝나면 시험 유닛·임시 폴더를 지운다)
#       WATCH_PY=<다른 사본> bench/ovms_watch_test.sh   (넣기 전 사본 시험)
set -u
cd "$(dirname "$(readlink -f "$0")")/.."
D=$PWD; PY=$D/.venv/bin/python; WPY=${WATCH_PY:-$D/ovms_watch.py}
U=ball-watchtest-ovms; PORT=18199
T=$(mktemp -d /tmp/ovms_watch_test.XXXXXX); LOG=$T/watch.log; LOG2=$T/watch2.log; FLAG=$T/hang; REQS=$T/reqs; NOSO=$T/noso; KM=$T/kmsg
ss -ltn | grep -q ":$PORT " && { echo "포트 $PORT 사용 중 — 시험 안 함"; exit 2; }
cat > "$T/fake_ovms.py" <<'PY'
import http.server, json, os, sys, threading, time
port, flag, delay, reqs, noso = int(sys.argv[1]), sys.argv[2], float(sys.argv[3]), sys.argv[4], sys.argv[5]
if os.path.exists(flag):
    os.remove(flag)          # 새 프로세스는 정상으로 시작한다(KILL 뒤 Restart 확인용)
threading.Thread(target=time.sleep, args=(10**6,), daemon=True).start()  # GPU 문맥을 만든 스레드 흉내(펜스 줄은 스레드 번호를 적는다, ⑨)
time.sleep(delay)            # 적재 흉내: 이동안 연결 거절
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        so = "response_format" in json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        with open(reqs, "a") as f:
            f.write(f"{os.getpid()} {'so' if so else 'chat'}\n")  # 요청 종류를 프로세스별로 남긴다(⑦·⑫)
        if os.path.exists(flag):
            time.sleep(3600)  # 답 없음 흉내(펜스 만료 뒤 OVMS)
        if so and os.path.exists(noso):  # 구조화 출력만 거절 흉내(⑧)
            self.send_response(400); self.send_header("content-length", "0"); self.end_headers(); return
        b = json.dumps({"choices": [{"message": {"content": json.dumps({"intent": "none", "count": None}) if so else "네"}}]}).encode()
        self.send_response(200); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self, *a):
        pass
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
PY
fake() {  # $1 = 적재 흉내 초
  systemctl --user reset-failed $U 2>/dev/null
  systemd-run --user --unit=$U --collect -p Restart=on-failure -p RestartSec=2 -p StartLimitIntervalSec=300 -p StartLimitBurst=5 \
    "$PY" -I "$T/fake_ovms.py" $PORT "$FLAG" "$1" "$REQS" "$NOSO" >/dev/null 2>&1
}
wait_for() {  # $1 = 기록에서 기다릴 정규식, $2 = 몇 번째, $3 = 최대 초, $4 = 기록 파일(기본 $LOG)
  local c f=${4:-$LOG}; for i in $(seq $(( $3 * 5 ))); do c=$(grep -cE "$1" "$f" 2>/dev/null); [ "${c:-0}" -ge "$2" ] && return 0; sleep 0.2; done; return 1
}
kl() { echo "$(date +%FT%T%:z) robot-pc kernel: $1" >> "$KM"; }  # 가짜 커널 줄 한 줄
FAIL=0
K='[0-9] KILL · '   # KILL 사건 줄(시각 바로 뒤). 시작 줄의 "실패면 KILL" 과 구별
check() { if eval "$2"; then echo "  통과 · $1"; else echo "  실패 · $1"; FAIL=1; fi; }
cleanup() { kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null; systemctl --user stop $U 2>/dev/null; systemctl --user reset-failed $U 2>/dev/null; rm -rf "$T"; }
trap cleanup EXIT
: > "$KM"

echo "== 가짜 OVMS(적재 3초) + 감시(1초 간격·실패 뒤 1초·1초 제한·2번·예열 유예 8초·적재 중 0.5초)"
fake 3
WATCH_UNIT=$U WATCH_URL=http://127.0.0.1:$PORT/v3/chat/completions WATCH_INTERVAL_S=1 WATCH_FAIL_INTERVAL_S=1 WATCH_TIMEOUT_S=1 WATCH_FAILS=2 \
  WATCH_WARM_S=8 WATCH_WARM_RETRY_S=0.5 WATCH_SUMMARY_S=100000 WATCH_LOG=$LOG WATCH_KMSG_CMD="tail -n 0 -F $T/kmsg_none" "$PY" -I "$WPY" & WPID=$!
wait_for '예열 끝' 1 15
check "① 적재 중 연결 거절은 안 셈(첫 예열 끝 전에 '답 없음' 0줄)" '[ "$(sed -n "1,/예열 끝/p" $LOG | grep -c "답 없음 [0-9]/")" = 0 ] && grep -q "예열 끝" $LOG'
touch "$FLAG"
wait_for "$K" 1 15
check "② 2번 연속 답 없음 → KILL" 'grep -q "답 없음 1/2" $LOG && grep -q "답 없음 2/2" $LOG && grep -q "$K" $LOG'
wait_for '예열 끝' 2 25
check "③ 재시작 대기는 안 셈 → 새 프로세스·자동 재시작 1번·다시 예열" 'grep -q "OVMS activating" $LOG && grep -q "자동 재시작 1번" $LOG && [ "$(sed -n "/$K/,\$p" $LOG | grep -c "답 없음 [0-9]/")" = 0 ]'
sleep 3   # 두 번째 프로세스도 1토큰 감시를 몇 번 받게
cp "$REQS" "$T/reqs1"
check "⑦ 새 프로세스마다 첫 요청은 구조화 출력 1번, 그 뒤는 1토큰(프로세스 2개)" 'awk "{n[\$1]++; if (!(\$1 in f)) f[\$1]=\$2; if (\$2==\"so\") s[\$1]++} END {c=0; ok=1; for (p in f) {c++; if (f[p]!=\"so\" || s[p]!=1 || n[p]<2) ok=0}; exit !(ok && c==2)}" $T/reqs1'
systemctl --user stop $U
wait_for 'OVMS inactive' 1 10; sleep 3
check "④ stop 으로 내린 것은 안 건드림(KILL 1번 그대로)" '[ "$(grep -c "$K" $LOG)" = 1 ] && grep -q "OVMS inactive" $LOG'

echo "== 적재가 안 끝나는 가짜 OVMS(적재 60초) — 예열 유예 8초 뒤부터 센다, 계속 죽으면 5번째 기동에서 멈춘다"
fake 60
wait_for '예열 유예 8초 지남' 1 20
check "⑤ 예열 유예가 지나도 못 답하면 셈 → KILL" 'wait_for "$K" 2 10'
wait_for '다시 안 뜸' 1 90
check "⑥ 5분에 5번 넘게 뜨면 더 안 뜸(둘째 묶음은 기동 5번·KILL 5번 뒤 6번째 기동 거절, '다시 안 뜸' 기록)" '[ "$(grep -c "$K" $LOG)" = 6 ] && grep -q "다시 안 뜸" $LOG'

echo "== 구조화 출력만 거절하는(400) 가짜 OVMS — 데우기만 못 할 뿐, 1토큰이 답하면 KILL 안 함"
touch "$NOSO"; fake 0
wait_for '예열 못 함\(HTTPError 400' 1 20; sleep 3
check "⑧ 구조화 출력 400 이어도 유예 뒤 1토큰이 답하면 KILL 안 함(KILL 6번 그대로)" '[ "$(grep -c "$K" $LOG)" = 6 ] && grep -q "예열 시도 실패(HTTPError 400" $LOG && grep -qE "예열 못 함\(HTTPError 400" $LOG && [ "$(tail -1 $REQS | cut -d" " -f2)" = chat ]'
systemctl --user stop $U
echo "  참고 · 최근 3분 시험 유닛 기동: $(journalctl --user -u $U --since "-3min" --no-pager -o cat 2>/dev/null | grep -cE '^Started ') 번 · $(journalctl --user -u $U --since "-3min" --no-pager -o cat 2>/dev/null | grep -m1 -E 'Start request repeated|start-limit' || echo '한도 기록 없음')"
kill "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null

echo "== 커널 줄(가짜 journalctl = tail -F) — 간격 30초 감시(적재 중 0.5초)에 펜스 만료·GPU HANG 줄을 흘린다"
rm -f "$NOSO"; fake 1
WATCH_UNIT=$U WATCH_URL=http://127.0.0.1:$PORT/v3/chat/completions WATCH_INTERVAL_S=30 WATCH_FAIL_INTERVAL_S=30 WATCH_TIMEOUT_S=1 WATCH_FAILS=2 \
  WATCH_WARM_S=8 WATCH_WARM_RETRY_S=0.5 WATCH_SUMMARY_S=100000 WATCH_LOG=$LOG2 WATCH_KMSG_CMD="tail -n 0 -F $KM" "$PY" -I "$WPY" & WPID=$!
wait_for '예열 끝' 1 15 "$LOG2"; sleep 1
MP=$(systemctl --user show $U -p MainPID --value); TID=$(ls /proc/$MP/task | grep -vx "$MP" | head -1)
kl "Fence expiration time out i915-0000:00:02.0:ovms[$$]:ea5!"
kl "i915 0000:00:02.0: [drm] GPU HANG: ecode 12:10:85def5fa, in ovms [$TID]"
wait_for '것이 아님' 1 5 "$LOG2"; wait_for 'GPU HANG 줄' 1 5 "$LOG2"; sleep 1
check "⑩ 남의 프로세스(시험 셸 $$) 펜스 줄·GPU HANG 줄은 KILL 안 함" '! grep -q "$K" $LOG2 && grep -q "주인 $$" $LOG2 && grep -q "GPU HANG 줄" $LOG2'
kl "Fence expiration time out i915-0000:00:02.0:ovms[$TID]:ea6!"; kl "Fence expiration time out i915-0000:00:02.0:ovms[$TID]:ea7!"   # 10-22 16:33 처럼 2줄
wait_for "$K" 1 3 "$LOG2"; sleep 0.5
check "⑨ 지금 프로세스($MP) 스레드($TID)의 펜스 줄 → 물음 실패 없이 3초 안에 KILL(2줄이어도 1번)" '[ "$(grep -c "$K" $LOG2)" = 1 ] && grep "$K" $LOG2 | grep -q "펜스" && ! grep -q "답 없음 [0-9]/" $LOG2'
wait_for '예열 끝' 2 8 "$LOG2"
check "⑪ KILL 뒤 간격(30초)을 안 기다리고 재시작을 보고 0.5초마다 다시 물어 8초 안에 예열" '[ "$(grep -c "예열 끝" $LOG2)" = 2 ] && grep -q "OVMS activating" $LOG2'
sleep 4; MP2=$(systemctl --user show $U -p MainPID --value)
check "⑫ 예열 뒤 1토큰은 간격대로(새 프로세스 $MP2 에 4초 동안 구조화 1번·1토큰 0번)" '[ "$MP2" != "$MP" ] && [ "$(grep -c "^$MP2 so" $REQS)" = 1 ] && [ "$(grep -c "^$MP2 chat" $REQS)" = 0 ]'
echo "== 감시 기록(앞 묶음)"
sed 's/^/  /' "$LOG"
echo "== 감시 기록(커널 줄 묶음)"
sed 's/^/  /' "$LOG2"
[ $FAIL = 0 ] && echo "결과: 전부 통과" || echo "결과: 실패 있음"
exit $FAIL
