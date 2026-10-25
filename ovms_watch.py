"""OVMS 무응답 감시 — 커널에 OVMS 펜스 만료 줄이 뜨면 바로, 아니면 60초마다 묻는 LLM 1토큰이 2번 연속 답이 없으면 ball-ovms 를 KILL 한다.

왜: GPU 사건 뒤 OVMS 가 죽지 않고 답만 안 하는 꼴이 있다(10-22 16:33 펜스 만료 뒤 27분, 23:08 30초, 10-23 10:12·12:49 — SETUP.md §8-14).
죽는 꼴(GPU HANG → SEGV)은 유닛의 Restart=on-failure 가 잡고, 이 감시는 안 죽는 꼴을 죽여서 Restart 에 넘긴다(KILL = 비정상 종료 → 3초 뒤 다시 뜸).
커널 줄: "Fence expiration time out i915-…:ovms[스레드]:…!" = i915 요청 감시(CONFIG_DRM_I915_REQUEST_TIMEOUT 20초)가 안 끝나는 OVMS GPU 요청을
리셋 없이 끊었다는 뜻이다. 그 뒤 OVMS 는 대개 답을 못 한다(10-23 12:49 는 15초 뒤 감시 물음이 KILL) → 줄에 적힌 스레드가 지금 OVMS 프로세스의 것이면
바로 KILL 한다. 예외: 10-22 17:36 은 펜스 뒤에도 답했다(그런 꼴이면 필요 없는 재시작 1번). GPU HANG 줄은 기록만 한다(OVMS 가 스스로 죽는다).
1토큰 물음은 커널 줄이 못 잡는 꼴의 안전망이다. 물음도 GPU 요청이라 멈춤을 부를 수 있어(10-23 12:48:40 에 멈춘 요청이 감시 물음) 60초마다 묻고,
한 번 답이 없으면 다음은 WATCH_FAIL_INTERVAL_S(20초) 뒤에 묻는다.
감시는 OVMS 를 띄우지 않는다 — stop 으로 내린 OVMS(xpu 학습 등, CONTRACT 5절)는 그대로 둔다.
세지 않는 때: 유닛이 active 가 아닐 때(재시작 대기·적재 중 activating, 멈춤), 새 OVMS 프로세스가 처음 답할 때까지(최대 WATCH_WARM_S초 — 적재·예열).
예열: 새 프로세스에 처음 묻는 것은 start_demo.sh 와 같은 구조화 출력(json_schema WarmIntent)이다. OVMS 를 띄운 뒤 첫 json_schema 는
문법 준비로 약 9초 걸려(10-23 10:06 첫 의도 판별이 5초 제한에 걸림) 자동 재시작 뒤엔 여기서 치른다. 답하면 그 뒤는 1토큰.
적재 중(연결 거절·404 — GPU 를 안 씀)엔 WATCH_WARM_RETRY_S(2초)마다 다시 묻는다. 다른 실패(시간 초과·400 등)는 WATCH_FAIL_INTERVAL_S 뒤.
구조화 출력은 데우기일 뿐 "살았나" 판단에는 안 쓴다 — 유예가 지나도 못 데웠으면 1토큰으로 살았는지 보고 센다.
기록: logs/ball-ovms-watch.log 에 사건마다 한 줄(시작·새 프로세스·예열 끝·실패·커널 줄·KILL·꺼짐), 30분마다 요약 한 줄.
띄우기: start_demo.sh 가 user 유닛 ball-ovms-watch 로 띄운다. 정지: systemctl --user stop ball-ovms-watch
시험: WATCH_UNIT·WATCH_URL·WATCH_KMSG_CMD 로 가짜 대상과 가짜 커널 줄을 가리킨다(bench/ovms_watch_test.sh). 표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import json, os, re, shlex, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNIT = os.getenv("WATCH_UNIT", "ball-ovms")
URL = os.getenv("WATCH_URL", "http://127.0.0.1:8100/v3/chat/completions")
MODEL = os.getenv("WATCH_MODEL", "qwen3-4b")
INTERVAL = float(os.getenv("WATCH_INTERVAL_S", "60"))
FAIL_INTERVAL = float(os.getenv("WATCH_FAIL_INTERVAL_S", "20"))  # 한 번 답이 없으면 다음 물음까지(안전망 감지 = 첫 실패 뒤 약 35초)
TIMEOUT = float(os.getenv("WATCH_TIMEOUT_S", "15"))
FAILS = int(os.getenv("WATCH_FAILS", "2"))
WARM = float(os.getenv("WATCH_WARM_S", "180"))      # 새 프로세스가 이 안에 한 번도 못 답하면 그때부터 센다(적재가 걸린 채 멈춘 꼴)
WARM_TIMEOUT = float(os.getenv("WATCH_WARM_TIMEOUT_S", "30"))  # 구조화 출력 예열 한 번의 제한(차가울 때 약 9초)
WARM_RETRY = float(os.getenv("WATCH_WARM_RETRY_S", "2"))  # 재시작 중·적재 중(연결 거절·404) 다시 볼 간격 — 이때는 GPU 를 안 쓴다
SUMMARY = float(os.getenv("WATCH_SUMMARY_S", "1800"))
LOG = Path(os.getenv("WATCH_LOG", ROOT / "logs" / f"{UNIT}-watch.log"))
KMSG_CMD = shlex.split(os.getenv("WATCH_KMSG_CMD", "journalctl -k -f -n 0 -o short-iso"))  # 빈 값이면 커널 줄을 안 본다
FENCE = re.compile(r"Fence expiration time out \S*:ovms\[(\d+)\]")
HANG = re.compile(r"GPU HANG: .*\bin ovms \[(\d+)\]")
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 로컬 주소만 — 프록시 환경변수를 타지 않는다
LOCK = threading.Lock()      # KILL 은 한 번에 한 곳에서(커널 줄 스레드·본 고리)
WAKE = threading.Event()     # KILL 뒤 본 고리를 깨워 재시작·예열을 바로 지켜보게
STOPPING = threading.Event()
killed = False               # 이 감시가 KILL 한 뒤 아직 새 프로세스를 못 봤다(다시 안 뜨면 한도 안내)
killed_pid = 0               # 마지막으로 KILL 한 프로세스 — 그 뒤 이 프로세스의 실패·펜스 줄은 세지 않는다
kmsg: subprocess.Popen | None = None


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError:  # 기록 파일을 못 쓰면 유닛 출력(같은 파일 또는 journal)으로
        sys.stderr.write(line)


def unit_state() -> tuple[str, str, int, str]:
    """(ActiveState, SubState, MainPID, NRestarts). 유닛이 없으면 inactive 로 나온다."""
    try:
        out = subprocess.run(["systemctl", "--user", "show", UNIT, "-p", "ActiveState", "-p", "SubState", "-p", "MainPID", "-p", "NRestarts"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "?", "?", 0, "?"
    d = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return d.get("ActiveState", "?"), d.get("SubState", "?"), int(d.get("MainPID") or 0), d.get("NRestarts") or "?"


def kill_ovms(pid: int, why: str) -> bool:
    """OVMS 가 아직 active 이고 같은 프로세스일 때만 KILL. 그새 누가 내리거나 새로 띄웠으면 건드리지 않는다."""
    global killed, killed_pid
    with LOCK:
        st, _, pid2, _ = unit_state()
        if st != "active" or pid2 != pid:
            log(f"KILL 안 함 — 그새 상태가 바뀜({st}, 프로세스 {pid2}) · {why}")
            return False
        r = subprocess.run(["systemctl", "--user", "kill", "-s", "KILL", UNIT], capture_output=True, text=True, timeout=10)
        log(f"KILL · 프로세스 {pid} · {why} → Restart=on-failure 가 다시 띄운다 (rc {r.returncode}{' ' + r.stderr.strip() if r.returncode else ''})")
        killed, killed_pid = True, pid
    WAKE.set()
    return True


def owner(tid: int) -> int:
    """스레드가 속한 프로세스(Tgid), 없으면 0. 펜스 줄은 프로세스가 아니라 GPU 문맥을 만든 스레드 번호를 적는다
    (10-23 12:49 ovms[49493] = 프로세스 49491 의 모델 적재 스레드, 이 스레드는 프로세스가 사는 동안 남아 있다)."""
    try:
        with open(f"/proc/{tid}/status") as f:
            for line in f:
                if line.startswith("Tgid:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


def on_kernel_line(line: str) -> None:
    m = FENCE.search(line)
    if m:
        tid = int(m.group(1))
        st, _, pid, _ = unit_state()
        who = owner(tid)
        if st == "active" and pid and who == pid and pid != killed_pid:
            kill_ovms(pid, f"커널 펜스 만료 줄(ovms[{tid}] — i915 요청 감시가 20초 넘은 GPU 요청을 끊음)")
        elif pid and pid == killed_pid:
            log(f"커널 펜스 만료 줄(ovms[{tid}]) · 프로세스 {pid} 는 이미 KILL 함 → 건드리지 않음")
        else:
            log(f"커널 펜스 만료 줄(ovms[{tid}]) · 지금 OVMS {st} 프로세스 {pid or '-'} 의 것이 아님(주인 {who or '없음'}) → 건드리지 않음")
        return
    m = HANG.search(line)
    if m:
        log(f"커널 GPU HANG 줄(ovms [{m.group(1)}]) · OVMS 가 스스로 죽는 꼴이라 기록만(Restart 가 다시 띄움)")


def watch_kernel() -> None:
    """커널 기록을 따라 읽는다(실행 사용자가 adm 그룹이면 sudo 없이 된다). 끊기면 5초 뒤 다시."""
    global kmsg
    while not STOPPING.is_set():
        try:
            kmsg = subprocess.Popen(KMSG_CMD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors="replace")
        except OSError as e:
            log(f"커널 줄 읽기를 못 띄움({e}) → 1토큰 물음만으로 감시")
            return
        for line in kmsg.stdout:
            try:
                on_kernel_line(line)
            except Exception as e:  # 줄 하나 처리 실패로 커널 감시가 멈추지 않게
                log(f"커널 줄 처리 오류({type(e).__name__}: {e})")
        rc = kmsg.wait()
        if not STOPPING.is_set():
            log(f"커널 줄 읽기 끊김(rc {rc}) → 5초 뒤 다시")
            time.sleep(5)


# start_demo.sh 응답 확인의 so() 와 같은 요청 — 에이전트 의도 판별(IntentOut, with_structured_output json_schema)과 같은 꼴
WARM_BODY = {
    "model": MODEL, "temperature": 0, "max_completion_tokens": 32, "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{"role": "user", "content": "분홍 블록 두 개 옮겨"}],
    "response_format": {"type": "json_schema", "json_schema": {"name": "WarmIntent", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["intent", "count"],
        "properties": {"intent": {"type": "string", "enum": ["red", "blue", "none"]},
                       "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}}}}}
LIVE_BODY = {"model": MODEL, "max_tokens": 1, "messages": [{"role": "user", "content": "안녕"}]}


def probe(structured: bool = False) -> tuple[bool, str, bool]:
    """structured=True 면 구조화 출력 예열(답이 JSON 으로 풀려야 성공), 아니면 1토큰.
    셋째 값 = 적재 중 신호(연결 거절·404 — 서버나 모델이 아직 안 올라옴)."""
    body = json.dumps(WARM_BODY if structured else LIVE_BODY).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    t0 = time.monotonic()
    try:
        with OPENER.open(req, timeout=WARM_TIMEOUT if structured else TIMEOUT) as r:
            raw = r.read()
        if structured:
            json.loads(json.loads(raw)["choices"][0]["message"]["content"])
        return True, f"{time.monotonic() - t0:.1f}초", False
    except Exception as e:  # 시간 초과·연결 거절·HTTP 오류 모두 "답 없음"
        code = getattr(e, "code", None)
        loading = code == 404 or isinstance(getattr(e, "reason", None), ConnectionRefusedError)
        return False, f"{type(e).__name__}{f' {code}' if code else ''} · {time.monotonic() - t0:.1f}초", loading


def main() -> None:
    global killed

    def stop(*_):
        STOPPING.set()
        log("감시 끝(stop)")
        if kmsg and kmsg.poll() is None:
            kmsg.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    log(f"감시 시작 · 대상 {UNIT} {URL} · 커널 펜스 만료 줄이면 바로 KILL({KMSG_CMD[0] if KMSG_CMD else '안 봄'}) · 1토큰 {INTERVAL:g}초마다"
        f"(실패 뒤 {FAIL_INTERVAL:g}초) {TIMEOUT:g}초 제한 · {FAILS}번 연속 실패면 KILL · 새 프로세스엔 먼저 구조화 출력 예열"
        f"({WARM_TIMEOUT:g}초 제한, 적재 중 {WARM_RETRY:g}초마다, 유예 최대 {WARM:g}초)")
    if KMSG_CMD:
        threading.Thread(target=watch_kernel, daemon=True).start()
    seen_pid, warm, born, fails, last_st = 0, False, 0.0, 0, ""
    tried: set[str] = set()  # 이 프로세스에서 이미 적은 예열 실패 종류(같은 실패를 재시도마다 적지 않게)
    n_ok = n_bad = 0
    t_sum = time.monotonic()
    while True:
        t_loop = time.monotonic()
        nap = INTERVAL
        st, sub, pid, nr = unit_state()
        if st != "active" or not pid:
            if st != last_st:
                hint = " — 다시 안 뜸(자동 재시작 한도 5분에 5번?) → 확인 뒤 ./start_ovms.sh" if killed and st in ("inactive", "failed") else ""
                log(f"OVMS {st}({sub}) · 세지 않음{hint}")
            last_st, seen_pid, fails = st, 0, 0
            if st in ("activating", "deactivating", "reloading"):
                nap = WARM_RETRY  # 재시작 중 — 새 프로세스를 바로 보게
        elif pid == killed_pid:
            nap = WARM_RETRY  # KILL 한 프로세스가 아직 active 로 보임 — 묻지 않고 사라질 때까지 짧게 다시 본다
        else:
            if pid != seen_pid:
                log(f"OVMS 프로세스 {pid} · 자동 재시작 {nr}번 · 구조화 출력 예열이 답할 때까지(최대 {WARM:g}초) 세지 않음")
                seen_pid, warm, born, fails, killed = pid, False, time.monotonic(), 0, False
                tried.clear()
            last_st = st
            ok, info, loading = probe(structured=not warm)
            if not warm:
                if ok:
                    warm, ok = True, None  # 이번 회차는 예열로 끝 — 세지 않는다
                    n_ok += 1
                    log(f"예열 끝 · 프로세스 {pid} · 뜬 지 {time.monotonic() - born:.0f}초 · 구조화 출력(json_schema) 첫 답 {info} → 이제 1토큰으로 감시")
                elif time.monotonic() - born < WARM:
                    kind = info.split(" ·")[0]
                    if kind not in tried:
                        tried.add(kind)
                        log(f"예열 시도 실패({info}) · 적재·예열 중이라 세지 않음")
                    ok = None
                    nap = WARM_RETRY if loading else FAIL_INTERVAL
                else:
                    warm = True
                    log(f"예열 유예 {WARM:g}초 지남 · 구조화 출력 예열 못 함({info}) → 지금부터 1토큰으로 센다")
                    ok, info, _ = probe()
            if ok:
                n_ok += 1
                if fails:
                    log(f"다시 답함 · {info} (실패 {fails}번 뒤)")
                fails = 0
            elif ok is False and pid != killed_pid:  # 커널 줄로 이미 KILL 한 프로세스의 실패는 세지 않는다
                fails += 1
                n_bad += 1
                log(f"답 없음 {fails}/{FAILS} · 프로세스 {pid} · {info}")
                if fails >= FAILS:
                    kill_ovms(pid, f"{FAILS}번 연속 답 없음")
                    fails = 0
                else:
                    nap = FAIL_INTERVAL
        if time.monotonic() - t_sum >= SUMMARY:
            log(f"요약 · 지난 {SUMMARY / 60:.0f}분 물음 {n_ok + n_bad}번 · 답 없음 {n_bad}번 · 지금 {st} 프로세스 {pid or '-'} · 자동 재시작 {nr}번")
            n_ok = n_bad = 0
            t_sum = time.monotonic()
        WAKE.wait(max(0.0, nap - (time.monotonic() - t_loop)))
        WAKE.clear()


if __name__ == "__main__":
    main()
