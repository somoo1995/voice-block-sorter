"""외부 접속 차단 — 대시보드 프로세스가 인터넷으로 나가는 연결을 막고 센다(와이파이를 끈 것과 같은 동작).

app.py 가 gradio 를 불러오기 전에 install() 한다. 파이썬 감사 훅(sys.addaudithook)으로
socket.getaddrinfo·connect·sendto 를 보고, 루프백(127.0.0.0/8·::1)·localhost·유닉스 소켓이 아니면
"이름 풀이 실패/네트워크 없음" 오류를 내서 막는다. 같은 프로세스에서 도는 에이전트·gradio·라이브러리 모두에 걸린다.
브라우저 쪽(글꼴·CDN)은 막지 못하므로 화면 자원이 전부 로컬인지는 따로 확인한다(ui/README.md).

환경변수
  UI_NET_GUARD   block(기본) | log(세기만) | off
  UI_NET_ALLOW   더 허용할 호스트·IP(쉼표 구분), 예) 10.0.0.5
"""
from __future__ import annotations

import errno
import ipaddress
import os
import socket
import sys
import threading
import time
from collections import deque

MODE = os.getenv("UI_NET_GUARD", "block")
ALLOW = {h.strip().lower() for h in os.getenv("UI_NET_ALLOW", "").split(",") if h.strip()}
LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback", socket.gethostname().lower()}

blocked = 0                      # 막은(또는 log 모드에서 본) 시도 수
recent: deque = deque(maxlen=20)  # (시각, 대상, 이벤트)
_lock = threading.Lock()
_installed = False


def _local(host) -> bool:
    if host is None:
        return True
    if isinstance(host, bytes):
        host = host.decode(errors="replace")
    h = str(host).strip("[]").lower()
    if not h or h in LOCAL_NAMES or h in ALLOW:
        return True
    try:
        ip = ipaddress.ip_address(h.split("%")[0])
    except ValueError:
        return False  # 이름인데 로컬이 아님 → 바깥
    return ip.is_loopback or ip.is_unspecified or str(ip) in ALLOW


_EVENTS = ("socket.getaddrinfo", "socket.connect", "socket.sendto")


def _hook(event: str, args) -> None:
    if event not in _EVENTS:
        return
    if event == "socket.getaddrinfo":
        host, port = args[0], args[1]
    else:
        sock, addr = args[0], args[1]
        if getattr(sock, "family", None) not in (socket.AF_INET, socket.AF_INET6) or not isinstance(addr, tuple):
            return  # 유닉스 소켓 등
        host, port = addr[0], addr[1]
    if _local(host):
        return
    global blocked
    with _lock:
        blocked += 1
        recent.append((time.strftime("%H:%M:%S"), f"{host}:{port}", event))
    print(f"[netguard] {'차단' if MODE == 'block' else '감지'} {event} {host}:{port}", file=sys.stderr, flush=True)
    if MODE == "block":
        if event == "socket.getaddrinfo":
            raise socket.gaierror(socket.EAI_NONAME, f"오프라인 차단(ui.netguard): {host}")
        raise OSError(errno.ENETUNREACH, f"오프라인 차단(ui.netguard): {host}:{port}")


def install() -> None:
    global _installed
    if _installed or MODE == "off":
        return
    sys.addaudithook(_hook)  # 한 번 걸면 뗄 수 없다(프로세스 끝까지)
    _installed = True


def status() -> dict:
    with _lock:
        return {"mode": MODE if _installed else "off", "blocked": blocked, "recent": list(recent)}
