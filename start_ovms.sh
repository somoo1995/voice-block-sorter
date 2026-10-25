#!/usr/bin/env bash
# OVMS 기동 — 127.0.0.1:8100(REST, OpenAI 호환 /v3·/v1), gRPC 끔(--port 미지정).
# 메모리 상한(MemoryMax)·OOMScoreAdjust=1000: 모자라면 OVMS 만 죽고 VM·Studio·다른 작업은 안전.
# Restart=on-failure: GPU HANG 으로 죽으면(10-23 09:35 SEGV) 3초 뒤 혼자 다시 뜬다. stop 으로 끈 것은 다시 안 뜬다.
# StartLimitIntervalSec=300·StartLimitBurst=5: 5분 안에 기동이 5번을 넘으면 더 안 띄운다(GPU 가 망가진 채 끝없이 다시 뜨지 않게) → 원인 확인 뒤 이 스크립트로
#   (10-23 3 → 5: 12:49·12:54 처럼 사건이 5분 남짓 사이에 몰려도 시연 중 "다시 안 뜸"이 되지 않게)
# 답만 안 하는 꼴(펜스 만료)은 ovms_watch.py(유닛 ball-ovms-watch, start_demo.sh 가 띄움)가 커널 줄을 보고 KILL 해서 Restart 에 넘긴다
# 사용: ./start_ovms.sh [config.json]   정지: systemctl --user stop ball-ovms   로그: logs/ball-ovms.log
# 환경변수: OVMS_MEM_MAX(예 14G) OVMS_UNIT(기본 ball-ovms) OVMS_PORT(기본 8100)
cd "$(dirname "$(readlink -f "$0")")"
D=$PWD
CFG=$(readlink -f "${1:-config.json}")
UNIT=${OVMS_UNIT:-ball-ovms}
PORT=${OVMS_PORT:-8100}
case $(systemctl --user show "$UNIT" -p ActiveState --value 2>/dev/null) in
  active|reloading) echo "$UNIT 이미 실행 중 — 바꾸려면: systemctl --user stop $UNIT 후 다시 실행"; exit 0 ;;
  activating) echo "$UNIT 재시작 중(Restart=on-failure, 3초 대기 뒤 적재) — 곧 다시 뜬다. 상태: systemctl --user status $UNIT"; exit 0 ;;
  deactivating) echo "$UNIT 멈추는 중 — 끝난 뒤 다시 실행"; exit 1 ;;
esac
systemctl --user reset-failed "$UNIT" 2>/dev/null
# 상한 기본값 = 지금 가용 메모리 - 3GB. 시스템 전체가 모자라 다른 프로세스가 스왑으로 밀리기 전에 OVMS 가 먼저 막힌다.
MEM_MAX=${OVMS_MEM_MAX:-$(awk "/MemAvailable/{printf \"%dM\", \$2/1024 - 3072}" /proc/meminfo)}
echo "OVMS $UNIT: config=$CFG port=$PORT MemoryMax=$MEM_MAX"
# 오프라인: 모델은 models/ 에서만 읽는다. 서버 안의 HF 내려받기 모듈이 혹시 불려도 닫힌 로컬 포트로 가서 바로 실패한다.
exec systemd-run --user --unit="$UNIT" --collect \
  -p MemoryMax=$MEM_MAX -p MemorySwapMax=0 -p OOMScoreAdjust=1000 \
  -p Restart=on-failure -p RestartSec=3 -p StartLimitIntervalSec=300 -p StartLimitBurst=5 \
  -E LD_LIBRARY_PATH=$D/ovms/lib -E PATH=$D/ovms/bin:/usr/bin:/bin -E HF_ENDPOINT=http://127.0.0.1:9 \
  --working-directory=$D \
  $D/ovms/bin/ovms --config_path "$CFG" --rest_port "$PORT" --rest_bind_address 127.0.0.1 \
  --cache_dir $D/.ov_cache --log_path $D/logs/${UNIT}.log --log_level INFO
