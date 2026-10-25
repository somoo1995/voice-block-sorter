"""보기 전용 화면 — 교육장 사람들이 각자 노트북·폰으로 대시보드를 본다(발표 10-23 17:00, 작업대마다 시연이라 프로젝터 대신).

UI_VIEW=1 일 때만 ui.app 이 켠다(기본 꺼짐). 표준 라이브러리 HTTP 서버(스레드), 0.0.0.0:UI_VIEW_PORT(7880).
대시보드(Gradio 7870)는 로그인이 없어 그대로 열면 와이파이의 누구나 팔을 움직이므로 여는 것은 이 읽기 전용 페이지뿐이다.
입력칸·버튼·마이크·Gradio API 가 없고, 경로는 /v/<토큰> 아래 셋뿐이다. GET 만 받고 그 밖(다른 경로·POST 등 모든 메서드)은 404.
  /v/<토큰>        페이지(글자·그림만. 1초마다 아래 /s 를 부른다)
  /v/<토큰>/s      구역별 HTML 조각(JSON). ?h= 로 브라우저가 가진 구역 지문을 받아 같은 구역은 빼고 보낸다
  /v/<토큰>/snap   지금 위 카메라 사진 한 장(대시보드가 보이는 그 파일만 — 경로를 받지 않는다)
토큰은 시작 때마다 무작위(8자). 조각은 ui.app 의 html_* 를 그대로 쓴다(view_sections — BOARD 를 복사해 그 복사본으로 그린다).
시연을 깨지 않게: 이 서버의 예외는 스레드 안에서 전부 잡아 대시보드 로그에 한 줄(같은 종류는 60초에 한 번)만 남긴다.
렌더는 손님 수와 상관없이 1초에 한 번 — 그 결과를 캐시해 모두에게 나눠 준다. 사진 파일은 읽기만 한다.
"""
from __future__ import annotations

import gzip
import hmac
import json
import os
import re
import secrets
import socketserver
import subprocess
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.getenv("UI_VIEW_PORT", "7880"))
KEYS = ["hdr", "snapcap", "layout", "cards", "nodes", "cmds", "chat", "log", "dev", "report"]
ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"  # 폰으로 치기 쉽게 헷갈리는 글자(l·o·0·1) 뺌
IMG_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
# 페이지가 바깥으로 나가는 요청을 브라우저가 막는다(조각에 무엇이 섞여도 외부 요청 0). 보고서 iframe(srcdoc)도 이 규칙을 물려받는다
CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")

URL: str | None = None
_tok = ""
_collect = None  # () -> (조각 dict, 사진 경로|None) — ui.app.view_sections
_log = None  # (글) -> None — 대시보드 로그 한 줄
_warned: dict = {}  # 같은 종류 오류 → 마지막으로 로그에 남긴 시각
_page = b""
_cache = {"t": 0.0, "html": {}, "dig": {}, "snap": None, "snapv": None}
_cache_lock = threading.Lock()
_seen: dict = {}  # 보는 브라우저 IP → 마지막 갱신 시각


def _warn(where: str, e: BaseException):
    """예외를 로그에 한 줄(같은 종류는 60초에 한 번). 로그 쓰기마저 실패해도 삼킨다."""
    key = f"{where}:{e.__class__.__name__}"
    if time.time() - _warned.get(key, 0) < 60:
        return
    _warned[key] = time.time()
    try:
        _log(f"보기 전용 화면 {where} 오류 {e.__class__.__name__}: {str(e)[:120]} (대시보드는 그대로)")
    except Exception:
        pass


def viewers(window: float = 5.0) -> int:
    now = time.time()
    return sum(1 for t in list(_seen.values()) if now - t < window)


def _lan_ip() -> str:
    """머리에 찍을 와이파이 주소. UI_VIEW_HOST 가 있으면 그것. 외부로 접속해 알아내지 않는다(netguard)."""
    if os.getenv("UI_VIEW_HOST"):
        return os.environ["UI_VIEW_HOST"]
    try:
        out = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return "127.0.0.1"
    addrs = re.findall(r"^\d+:\s+(\S+)\s+inet\s+([\d.]+)/", out, re.M)
    for pref in ("wl", "en", "eth"):  # 와이파이 먼저, 다음 유선. docker·vmnet·tailscale 은 교육장에서 안 닿는다
        for dev, ip in addrs:
            if dev.startswith(pref):
                return ip
    return "127.0.0.1"


def _state(now: float) -> dict:
    """1초 캐시 — 보는 사람이 몇이든 조각은 1초에 한 번만 그린다(동시에 온 요청은 앞 요청이 그린 것을 받는다)."""
    with _cache_lock:
        if now - _cache["t"] >= 1.0:
            sections, snap = _collect()
            _cache["html"] = sections
            _cache["dig"] = {k: f"{zlib.crc32(v.encode()):08x}" for k, v in sections.items()}
            p = Path(snap) if snap else None
            _cache["snap"] = p if p and p.suffix.lower() in IMG_TYPES and p.is_file() else None
            _cache["snapv"] = f"{zlib.crc32(str(p).encode()):08x}{int(p.stat().st_mtime * 1000):x}" if _cache["snap"] else None
            _cache["t"] = now
        return dict(_cache)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64  # 기본 5 — 교육장 폰 수십 대가 한꺼번에 붙어도 접속 대기열이 넘치지 않게

    def server_bind(self):  # HTTPServer.server_bind 의 getfqdn(역방향 DNS 조회)을 건너뛴다
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "view", self.server_address[1]

    def handle_error(self, request, client_address):  # 접속 끊김(폰 잠금·창 닫기)은 조용히, 그 밖은 로그 한 줄
        e = sys.exc_info()[1]
        if e is not None and not isinstance(e, (ConnectionError, TimeoutError)):
            _warn("요청", e)


class _Handler(BaseHTTPRequestHandler):
    server_version, sys_version = "view", ""
    timeout = 15

    def log_message(self, *args):  # 접속마다 stderr 에 찍지 않는다
        pass

    def __getattr__(self, name):  # GET 말고 모든 메서드(POST·PUT·HEAD·OPTIONS …) = 404
        if name.startswith("do_"):
            return self._not_found
        raise AttributeError(name)

    def _not_found(self):
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def _send(self, code: int, body: bytes, ctype: str, gz: bool = False):
        if gz and len(body) > 1024 and "gzip" in (self.headers.get("Accept-Encoding") or ""):
            body = gzip.compress(body, 5)
        else:
            gz = False
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if gz:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            self._get()
        except (ConnectionError, TimeoutError):
            pass
        except Exception as e:
            _warn("요청", e)
            try:
                self._send(500, b"error", "text/plain; charset=utf-8")
            except Exception:
                pass

    def _get(self):
        path, _, query = self.path.partition("?")
        m = re.fullmatch(r"/v/([a-z0-9]{1,32})(/s|/snap|/)?", path)
        if not m or not hmac.compare_digest(m.group(1), _tok):
            return self._not_found()
        sub = m.group(2) or "/"
        if sub == "/":
            return self._send(200, _page, "text/html; charset=utf-8", gz=True)
        try:
            st = _state(time.time())
        except Exception as e:
            _warn("그리기", e)
            return self._send(503, b"busy", "text/plain; charset=utf-8")
        if sub == "/snap":
            p = st["snap"]
            if not p:
                return self._not_found()
            try:
                return self._send(200, p.read_bytes(), IMG_TYPES[p.suffix.lower()])
            except OSError:
                return self._not_found()
        _seen[self.client_address[0]] = time.time()
        have = re.search(r"(?:^|&)h=([0-9a-f.-]*)", query)
        have = dict(zip(KEYS, have.group(1).split("."))) if have else {}
        body = {"html": {k: v for k, v in st["html"].items() if have.get(k) != st["dig"][k]}, "h": st["dig"], "snap": st["snapv"]}
        self._send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json; charset=utf-8", gz=True)


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="referrer" content="no-referrer">
<title>스펀지 블록 선별 로봇 · 보기 전용</title><style>__CSS__
:root { --body-text-color: #1e293b; --body-text-color-subdued: #64748b; --border-color-primary: #e2e8f0;
  --color-accent: #6366f1; --block-background-fill: #fff; --background-fill-primary: #fff;
  --font-mono: "DejaVu Sans Mono", ui-monospace, monospace; color-scheme: light; }
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: #f1f5f9; color: var(--body-text-color); overflow-x: hidden;
  font: 15px/1.45 "Noto Sans CJK KR", "NanumBarunGothic", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif; }
.wrap { max-width: 1500px; margin: 0 auto; padding: 8px 16px 28px; }
.bar { display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--body-text-color-subdued); margin: 2px 0 6px; }
.bar .dot { width: 8px; height: 8px; } .bar.off .dot { background: var(--c-bad); animation: none; } .bar.off { color: var(--c-bad); }
.cols { display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; margin-top: 10px; }
@media (min-width: 960px) { .cols { grid-template-columns: minmax(0, 6fr) minmax(0, 5fr); } }
.col { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
.box { background: #fff; border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px; min-width: 0; }
#snapimg { width: 100%; height: auto; display: block; border-radius: 10px; }
.bench { max-width: 560px; margin: 0 auto; }
.pill { overflow-wrap: anywhere; }
.cards { margin-top: 0; }
.log { height: 300px; overflow-wrap: anywhere; }
.chatv { display: flex; flex-direction: column; gap: 6px; max-height: 340px; overflow-y: auto; }
.msg { padding: 6px 10px; border-radius: 10px; max-width: 92%; overflow-wrap: anywhere; font-size: 14px; }
.msg.user { align-self: flex-end; background: color-mix(in srgb, var(--color-accent) 12%, #fff); }
.msg.assistant { align-self: flex-start; background: #f1f5f9; }
.cmds { list-style: none; margin: 0; padding: 0; }
.cmds li { display: flex; flex-wrap: wrap; gap: 2px 8px; font-size: 14px; padding: 4px 0; border-bottom: 1px dashed var(--border-color-primary); }
.cmds .ts { color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums; }
.cmds em { margin-left: auto; font-style: normal; font-weight: 700; color: var(--c-ok); }
.cmds li.run em { color: var(--color-accent); }
.report-frame { height: 720px; }
@media (max-width: 640px) {
  .wrap { padding: 6px 12px 24px; }
  .hdr .ttl b { font-size: 18px; } .pill { font-size: 12px; }
  .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); } .card b { font-size: 34px; }
  .devtbl { font-size: 12px; } .devtbl td { white-space: normal; padding: 4px 3px; } .devtbl th { padding: 4px 3px; }
  .flow { gap: 4px; } .node { padding: 5px 7px 4px; font-size: 13px; }
  .report-frame { height: 560px; }
}
</style></head><body><div class="wrap">
<div class="bar" id="bar"><i class="dot"></i><span id="st">연결 중…</span><span>· 보기 전용(조작 안 됨) · 1초마다 새로 그림</span></div>
<div id="s-hdr"></div>
<div class="cols"><div class="col">
<div class="box"><img id="snapimg" alt="위 카메라 사진" hidden><div id="s-snapcap"></div></div>
<div class="box" id="s-layout"></div>
<div id="s-cards"></div>
</div><div class="col">
<div class="box" id="s-nodes"></div>
<div class="box" id="s-cmds"></div>
<div class="box" id="s-chat"></div>
<div class="box" id="s-log"></div>
<div class="box" id="s-dev"></div>
</div></div>
<div class="box" style="margin-top:12px"><div class="ptitle">결과 보고서</div><div id="s-report"></div></div>
</div><script>
(() => {
  const B = location.pathname.replace(/\\/+$/, ""), K = __KEYS__, h = {};
  const img = document.getElementById("snapimg"), st = document.getElementById("st"), bar = document.getElementById("bar");
  let sv = null;
  async function tick() {
    try {
      const r = await fetch(B + "/s?h=" + K.map(k => h[k] || "-").join("."), {cache: "no-store"});
      if (!r.ok) throw new Error(r.status);
      const j = await r.json();
      for (const k in j.html) { const el = document.getElementById("s-" + k); if (el) el.innerHTML = j.html[k]; }
      Object.assign(h, j.h);
      if (j.snap !== sv) { sv = j.snap; if (sv) { img.src = B + "/snap?" + sv; img.hidden = false; } else { img.hidden = true; img.removeAttribute("src"); } }
      bar.className = "bar"; st.textContent = "실시간 " + new Date().toLocaleTimeString("ko-KR", {hour12: false});
    } catch (e) { bar.className = "bar off"; st.textContent = "연결 끊김 — 다시 시도 중"; }
    setTimeout(tick, 1000);
  }
  tick();
})();
</script></body></html>"""


def _serve(srv):
    try:
        srv.serve_forever()
    except Exception as e:  # 여기까지 오면 보기 화면만 멈춘다
        _warn("서버", e)


def start(collect, css: str, log) -> str:
    """서버를 띄우고 주소를 돌려준다. 포트가 막혀 있으면 OSError — 부르는 쪽(ui.app._start_view)이 잡아 로그만 남긴다."""
    global URL, _tok, _collect, _log, _page
    _collect, _log, _tok = collect, log, "".join(secrets.choice(ALPHABET) for _ in range(8))
    _page = PAGE.replace("__CSS__", css).replace("__KEYS__", json.dumps(KEYS)).encode()
    srv = _Server(("0.0.0.0", PORT), _Handler)
    threading.Thread(target=_serve, args=(srv,), name="ui-view", daemon=True).start()
    URL = f"http://{_lan_ip()}:{PORT}/v/{_tok}"
    return URL
