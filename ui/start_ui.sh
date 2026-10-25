#!/usr/bin/env bash
# 대시보드 기동 — http://127.0.0.1:7870 (SSH 에서 띄워도 살아 있게 systemd-run --user)
# 정지: systemctl --user stop ball-ui   로그: ui/ui.log
# 넘기는 환경변수: 지금 셸에 설정된 UI_* FAKE_* BALL_* TTS_* STT_* LLM_* EMB_* RERANK_* 전부
#   예) UI_TTS_OUT=silent BALL_MOCK=1 BALL_MOCK_POLICY_S=2 ui/start_ui.sh   가짜 작업대 리허설(밤, 소리 없음)
#       BALL_ROBOT=1 ui/start_ui.sh   실제 로봇(사용자 입회 때만). BALL_ROBOT 이 있으면 BALL_MOCK 은 넘기지 않는다
cd "$(dirname "$(readlink -f "$0")")/.."
D=$PWD
ENVS=(-E PYTHONUNBUFFERED=1)
for v in $(compgen -e | grep -E '^(UI|FAKE|BALL|TTS|STT|LLM|EMB|RERANK)_'); do
  [ "$v" = BALL_MOCK ] && [ -n "${BALL_ROBOT:-}" ] && continue
  ENVS+=(-E "$v=${!v}")
done
echo "환경: ${ENVS[*]}"
systemctl --user stop ball-ui 2>/dev/null
systemctl --user reset-failed ball-ui 2>/dev/null
exec systemd-run --user --unit=ball-ui --collect --working-directory="$D" \
  -p StandardOutput=append:"$D/ui/ui.log" -p StandardError=append:"$D/ui/ui.log" \
  "${ENVS[@]}" "$D/.venv/bin/python" -m ui.app
