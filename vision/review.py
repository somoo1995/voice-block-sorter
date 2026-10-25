"""자동 라벨 초안(labels_draft/<name>/draft.json) 검수 화면 — 사진을 넘기며 박스 승인·삭제·클래스 변경, 손 박스 직접 그리기.

  python vision/review.py day1                  # http://127.0.0.1:8791 (젠북 화면의 브라우저)
  ssh -N -L 8791:127.0.0.1:8791 robot-pc  # 다른 PC 에서 볼 때 — 그 PC 브라우저로 같은 주소
  python vision/review.py day1 --approve-all    # 화면 없이 일괄 승인(제외 아닌 사진 전부 통과) → 내보내기
  python vision/review.py day1 --status         # 상태 요약만

상태: auto(자동, 자신 있음) · review(검수 필요) · pass(사람이 통과) · exclude(제외). 내보내기는 auto·pass 만 넣는다.
labels_draft/<이름>/first.txt(사진 이름 한 줄에 하나)가 있으면 보기 '먼저 볼 사진'으로 그 사진부터 띄운다(보정 후보 10장 등).
바뀔 때마다 draft.json 을 원자적으로 저장한다(임시 파일 → 교체). 표준 라이브러리 http.server 만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

try:
    from . import autolabel
except ImportError:
    import autolabel

STATUSES = ("auto", "review", "pass", "exclude")
CTYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".bmp": "image/bmp"}


class Conflict(Exception):
    pass


class Draft:
    """draft.json 을 메모리에 들고 바뀔 때마다 저장. 밖에서 파일이 바뀌면(다른 창·autolabel 재실행) 저장을 거부한다."""

    def __init__(self, name: str):
        self.name = name
        self.path = autolabel.draft_path(name)
        if not self.path.exists():
            raise SystemExit(f"{self.path} 없음 — 먼저 python vision/autolabel.py <사진 폴더> --name {name}")
        self.lock = threading.Lock()
        self.thumbs: dict[int, bytes] = {}
        self.reload()

    def reload(self) -> None:
        self.data = json.loads(self.path.read_text())
        self.mtime = self.path.stat().st_mtime_ns

    def save(self) -> None:
        if self.path.stat().st_mtime_ns != self.mtime:
            self.reload()
            raise Conflict
        autolabel.save_draft(self.data, self.path)
        self.mtime = self.path.stat().st_mtime_ns

    def first(self) -> list[str]:
        """first.txt 의 사진 이름(stem) — 먼저 검수할 사진. 없으면 빈 목록."""
        f = self.path.with_name("first.txt")
        if not f.exists():
            return []
        stems = {im["stem"] for im in self.data["images"]}
        return [ln.strip() for ln in f.read_text().splitlines() if ln.strip() in stems]

    def image(self, i: int) -> dict:
        if not 0 <= i < len(self.data["images"]):
            raise IndexError(i)
        return self.data["images"][i]

    def thumb(self, i: int) -> bytes:
        if i not in self.thumbs:
            img = cv2.imread(self.image(i)["path"])
            h, w = img.shape[:2]
            ok, buf = cv2.imencode(".jpg", cv2.resize(img, (160, round(160 * h / w)), interpolation=cv2.INTER_AREA),
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            self.thumbs[i] = buf.tobytes()
        return self.thumbs[i]


def clean_boxes(boxes: list, w: int, h: int) -> list[dict]:
    """브라우저가 보낸 박스 검사: 클래스 이름, 사진 안으로 자르기, 2px 미만 버리기."""
    out = []
    for b in boxes:
        if b.get("cls") not in autolabel.NAMES:
            raise ValueError(f"클래스 {b.get('cls')!r}")
        x1, y1, x2, y2 = (float(v) for v in b["xyxy"])
        x1, x2 = sorted((min(max(x1, 0), w), min(max(x2, 0), w)))
        y1, y2 = sorted((min(max(y1, 0), h), min(max(y2, 0), h)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        ratio = b.get("ratio")
        nb = {"cls": b["cls"], "xyxy": [round(x1), round(y1), round(x2), round(y2)],
              "flags": [str(f) for f in b.get("flags") or []],
              "ratio": ratio if isinstance(ratio, (int, float)) else None}
        if b.get("by") == "human":
            nb["by"] = "human"
        out.append(nb)
    return out


def approve_all(name: str, make_zip: bool) -> None:
    d = Draft(name)
    n = 0
    for im in d.data["images"]:
        if im["status"] in ("auto", "review"):
            im["status"] = "pass"
            n += 1
    d.save()
    print(f"일괄 승인 {n}장 →", json.dumps(autolabel.summarize(d.data), ensure_ascii=False))
    print("데이터셋:", json.dumps(autolabel.export(name, make_zip), ensure_ascii=False))


def make_handler(draft: Draft):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 요청마다 찍지 않는다
            pass

        def _send(self, code: int, body: bytes, ctype: str, cache: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path == "/":
                    self._send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/api/draft":
                    with draft.lock:
                        d = draft.data
                        self._json({"name": d["name"], "marker": d.get("marker"), "names": d["names"],
                                    "summary": autolabel.summarize(d), "first": draft.first(),
                                    "images": [{k: im[k] for k in ("stem", "w", "h", "status", "reasons", "boxes")}
                                               for im in d["images"]]})
                elif path.startswith("/img/"):
                    p = draft.image(int(path[5:]))["path"]
                    with open(p, "rb") as f:
                        self._send(200, f.read(), CTYPES.get(p[p.rfind("."):].lower(), "image/jpeg"), cache=True)
                elif path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon")
                elif path.startswith("/thumb/"):
                    self._send(200, draft.thumb(int(path[7:])), "image/jpeg", cache=True)
                else:
                    self._json({"error": "없는 주소"}, 404)
            except (ValueError, IndexError, OSError) as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self):
            path = urlparse(self.path).path
            n = int(self.headers.get("Content-Length") or 0)
            if n > 2_000_000:
                return self._json({"error": "요청이 너무 큼"}, 413)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                with draft.lock:
                    if path.startswith("/api/image/"):
                        im = draft.image(int(path[11:]))
                        if "status" in body and body["status"] not in STATUSES:
                            raise ValueError(f"상태 {body['status']!r}")
                        boxes = clean_boxes(body["boxes"], im["w"], im["h"]) if "boxes" in body else im["boxes"]
                        im["boxes"] = boxes  # 둘 다 검사한 뒤에 바꾼다
                        im["status"] = body.get("status", im["status"])
                        im["reviewed"] = datetime.now().astimezone().isoformat(timespec="seconds")
                        draft.save()
                        return self._json({"ok": True, "summary": autolabel.summarize(draft.data)})
                    if path == "/api/bulk":
                        if body.get("status") not in STATUSES:
                            raise ValueError(f"상태 {body.get('status')!r}")
                        for i in body["indices"]:
                            draft.image(int(i))["status"] = body["status"]
                        draft.save()
                        return self._json({"ok": True, "summary": autolabel.summarize(draft.data)})
                    if path == "/api/export":
                        return self._json({"ok": True, **autolabel.export(draft.name, bool(body.get("zip")))})
                self._json({"error": "없는 주소"}, 404)
            except Conflict:
                self._json({"error": "draft.json 이 밖에서 바뀌었다(다른 창이나 autolabel 재실행). 새로 읽었으니 다시 해 줘."}, 409)
            except (ValueError, KeyError, IndexError, TypeError) as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="autolabel.py --name 으로 만든 초안 이름")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--approve-all", action="store_true", help="제외 아닌 사진 전부 통과 → 내보내기 (화면 없이)")
    ap.add_argument("--zip", action="store_true", help="--approve-all 때 Roboflow 업로드용 zip 도")
    ap.add_argument("--status", action="store_true", help="상태 요약만 출력")
    a = ap.parse_args()
    if a.approve_all:
        return approve_all(a.name, a.zip)
    draft = Draft(a.name)
    print(json.dumps(autolabel.summarize(draft.data), ensure_ascii=False))
    if draft.first():
        print(f"먼저 볼 사진 {len(draft.first())}장({draft.path.with_name('first.txt')}) — 화면의 보기 '먼저 볼 사진'")
    if a.status:
        return
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(draft))
    print(f"검수 화면: http://{a.host}:{a.port}  (끝내기 Ctrl+C, 저장은 바뀔 때마다 자동)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>블록 라벨 검수</title>
<style>
:root{--bg:#15171c;--panel:#1e222a;--line:#343b48;--text:#e7eaf0;--mut:#99a2b2;
 --auto:#3b82f6;--review:#f5b301;--pass:#22c55e;--exclude:#6b7280}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,"Noto Sans KR",sans-serif}
header{position:sticky;top:0;z-index:3;display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;
 padding:8px 14px;background:var(--panel);border-bottom:1px solid var(--line)}
header h1{font-size:15px;margin:0 6px 0 0}
.chip{padding:2px 9px;border-radius:999px;border:1px solid var(--line);font-size:13px;white-space:nowrap}
.chip b{font-variant-numeric:tabular-nums}
.c-auto{border-color:var(--auto)}.c-review{border-color:var(--review)}.c-pass{border-color:var(--pass)}.c-exclude{border-color:var(--exclude)}
select,button{font:inherit;color:var(--text);background:#2a303b;border:1px solid var(--line);border-radius:6px;padding:4px 10px}
button{cursor:pointer}button:hover{background:#353c49}
button.pri{background:#1f6f43;border-color:#2c9a5d}button.pri:hover{background:#25814f}
#save{color:var(--mut);font-size:13px;margin-left:auto}
main{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:14px;padding:10px 14px}
#wrap{min-width:0}
canvas{display:block;max-width:100%;background:#000;cursor:crosshair;border-radius:4px}
aside{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;align-self:start;
 max-height:calc(72vh - 70px);overflow-y:auto}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0}
.st{display:inline-block;padding:1px 8px;border-radius:4px;font-size:12px;color:#111}
.st-auto{background:var(--auto);color:#fff}.st-review{background:var(--review)}.st-pass{background:var(--pass)}.st-exclude{background:var(--exclude);color:#fff}
.mut{color:var(--mut);font-size:12.5px}
.cls{padding:3px 8px;border-radius:6px;border:2px solid transparent;background:#2a303b;font-size:12.5px;cursor:pointer}
.cls.on{border-color:#fff}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
td{padding:3px 4px;border-top:1px solid var(--line);vertical-align:middle}
tr.sel td{background:#2b3442}
td select{padding:1px 4px;font-size:12.5px}
td button{padding:1px 7px;font-size:12px}
.flag{color:#ffd400;font-size:12px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
#help{margin-top:10px;font-size:12px;color:var(--mut);line-height:1.6}
kbd{background:#2a303b;border:1px solid var(--line);border-radius:4px;padding:0 5px;font-size:11.5px;color:var(--text)}
#grid{position:relative;display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:6px;padding:6px 14px;
 height:28vh;overflow-y:auto;border-top:1px solid var(--line);align-content:start}
.th{border:3px solid var(--line);border-radius:6px;overflow:hidden;cursor:pointer;background:#000;position:relative}
.th img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover}
.th span{position:absolute;left:0;right:0;bottom:0;font-size:11px;background:rgba(0,0,0,.6);padding:1px 4px;white-space:nowrap;overflow:hidden}
.th.auto{border-color:var(--auto)}.th.review{border-color:var(--review)}.th.pass{border-color:var(--pass)}.th.exclude{border-color:var(--exclude);opacity:.5}
.th.cur{outline:3px solid #fff;outline-offset:1px}
#toast{position:fixed;right:16px;bottom:16px;max-width:560px;background:#263041;border:1px solid var(--line);
 border-radius:8px;padding:10px 14px;display:none;white-space:pre-wrap;z-index:5;font-size:13px}
@media (max-width:900px){main{grid-template-columns:1fr}aside{max-height:none}}
</style></head>
<body>
<header>
 <h1 id="title">블록 라벨 검수</h1>
 <span class="chip c-auto">자동 <b id="n-auto">0</b></span>
 <span class="chip c-review">검수 필요 <b id="n-review">0</b></span>
 <span class="chip c-pass">통과 <b id="n-pass">0</b></span>
 <span class="chip c-exclude">제외 <b id="n-exclude">0</b></span>
 <label>보기 <select id="filter">
  <option value="first" hidden>먼저 볼 사진</option>
  <option value="all">전체</option><option value="review" selected>검수 필요</option>
  <option value="auto">자동</option><option value="pass">통과</option><option value="exclude">제외</option>
  <option value="hand_hint">손 제안 있음</option><option value="no_hand">손 박스 없음</option>
 </select></label>
 <button id="bulk" title="지금 보기의 사진 중 제외가 아닌 것을 모두 통과">보이는 사진 모두 통과</button>
 <button id="export" class="pri">데이터셋 내보내기</button>
 <label class="mut"><input type="checkbox" id="zip"> zip 도</label>
 <span id="save">불러오는 중…</span>
</header>
<main>
 <div id="wrap"><canvas id="cv" width="640" height="480"></canvas></div>
 <aside>
  <div><b id="stem">-</b> <span id="pos" class="mut"></span> <span id="stbadge" class="st"></span></div>
  <div id="reasons" class="mut"></div>
  <div class="row">
   <button data-st="pass">통과 <kbd>P</kbd></button>
   <button data-st="exclude">제외 <kbd>X</kbd></button>
   <button data-st="review">검수로 <kbd>U</kbd></button>
  </div>
  <div class="mut">그릴 클래스 (빈 곳 드래그 = 새 박스)</div>
  <div class="row" id="classes"></div>
  <table><tbody id="boxes"></tbody></table>
  <div id="help">
   <kbd>←</kbd><kbd>→</kbd> 사진 넘기기 · <kbd>P</kbd> 통과하고 다음 · <kbd>X</kbd> 제외하고 다음 · <kbd>U</kbd> 검수로 되돌리기<br>
   박스 클릭 = 선택(끌면 이동, 모서리 끌면 크기) · <kbd>Tab</kbd> 다음 박스<br>
   <kbd>1</kbd>~<kbd>5</kbd> 선택한 박스 클래스 바꾸기(선택 없으면 그릴 클래스) · <kbd>Enter</kbd> 박스 승인 · <kbd>Del</kbd> 박스 삭제 · <kbd>Esc</kbd> 선택 해제<br>
   노란 점선 = 자동 라벨이 자신 없는 박스(split·defect?·hand? …). 고치거나 승인하면 점선이 사라진다.<br>
   손은 사람이 그린다: 손이 보이면 <kbd>5</kbd>(손) 상태에서 손목까지 드래그.
  </div>
 </aside>
</main>
<div id="grid"></div>
<div id="toast"></div>
<script>
"use strict";
const KEYS = ["red_ok", "blue_ok", "red_defect", "blue_defect", "hand"];   // 1~5
// 클래스 식별자는 그대로, 보이는 이름만: red_* = 분홍, blue_* = 초록 (10-21 밤 파스텔 스펀지)
const LABEL = {red_ok: "분홍 정상", blue_ok: "초록 정상", red_defect: "분홍 불량", blue_defect: "초록 불량", hand: "손"};
const COLOR = {red_ok: "#ff8fc8", blue_ok: "#8ce99a", red_defect: "#e0307a", blue_defect: "#2f9e44", hand: "#ffa94d"};
const ST = {auto: "자동", review: "검수 필요", pass: "통과", exclude: "제외"};
const $ = id => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
let D = null, view = [], pos = 0, sel = -1, drawCls = "hand", scale = 1, drag = null;
const pic = new Image();
const dirty = new Set();
let saveTimer = null, saving = false;

function toast(msg, ms = 4000) {
  const t = $("toast"); t.textContent = msg; t.style.display = "block";
  clearTimeout(toast.h); toast.h = setTimeout(() => t.style.display = "none", ms);
}
function cur() { return D.images[view[pos]]; }
function idx() { return view[pos]; }

async function load() {
  const r = await fetch("api/draft");
  D = await r.json();
  $("title").textContent = `블록 라벨 검수 — ${D.name}` + (D.marker ? ` (마커 ${D.marker})` : "");
  counts(D.summary);
  const fo = document.querySelector('#filter option[value="first"]');
  fo.hidden = !D.first.length;
  fo.textContent = `먼저 볼 사진 (${D.first.length})`;
  if (D.first.length && !load.done && D.images.some(im => D.first.includes(im.stem) && im.status === "review"))
    $("filter").value = "first";
  load.done = true;
  applyFilter();
  $("save").textContent = "저장됨";
}
function counts(s) { for (const k of Object.keys(ST)) $("n-" + k).textContent = s.status[k] || 0; }

function matches(im, f) {
  if (f === "all") return true;
  if (f === "first") return D.first.includes(im.stem);
  if (f in ST) return im.status === f;
  if (f === "hand_hint") return im.boxes.some(b => b.cls === "hand" && b.flags.length);
  if (f === "no_hand") return !im.boxes.some(b => b.cls === "hand");
  return true;
}
function applyFilter() {
  const keep = view.length ? idx() : -1;
  view = D.images.map((_, i) => i).filter(i => matches(D.images[i], $("filter").value));
  pos = Math.max(0, view.indexOf(keep));
  grid();
  show();
}
function grid() {
  $("grid").innerHTML = view.map((i, k) =>
    `<div class="th ${D.images[i].status}" data-k="${k}" id="th${i}"><img loading="lazy" src="thumb/${i}"><span>${D.images[i].stem}</span></div>`).join("");
}
$("grid").onclick = e => { const t = e.target.closest(".th"); if (t) { pos = +t.dataset.k; sel = -1; show(); } };

function show() {
  document.querySelectorAll(".th.cur").forEach(t => t.classList.remove("cur"));
  if (!view.length) { ctx.clearRect(0, 0, cv.width, cv.height); $("stem").textContent = "이 보기에 사진 없음"; $("pos").textContent = ""; $("boxes").innerHTML = ""; $("reasons").textContent = ""; $("stbadge").textContent = ""; return; }
  const th = $("th" + idx()), g = $("grid");
  if (th) {  // scrollIntoView 는 페이지까지 굴려 사진이 밀려 올라가므로 격자 상자만 굴린다
    th.classList.add("cur");
    if (th.offsetTop < g.scrollTop || th.offsetTop + th.offsetHeight > g.scrollTop + g.clientHeight)
      g.scrollTop = th.offsetTop - g.clientHeight / 2;
  }
  pic.onload = () => { fit(); draw(); };
  pic.src = "img/" + idx();
  panel();
}
function fit() {
  const im = cur();
  const w = $("wrap").clientWidth, h = window.innerHeight * 0.72 - 80;  // 아래 28vh 는 격자
  scale = Math.max(0.5, Math.min(2, w / im.w, h / im.h));
  cv.width = Math.round(im.w * scale); cv.height = Math.round(im.h * scale);
}
window.onresize = () => { if (view.length) { fit(); draw(); } };

function draw() {
  const im = cur(); if (!im) return;
  ctx.drawImage(pic, 0, 0, cv.width, cv.height);
  ctx.font = "12px system-ui"; ctx.textBaseline = "bottom";
  im.boxes.forEach((b, j) => {
    const [x1, y1, x2, y2] = b.xyxy.map(v => v * scale);
    ctx.setLineDash([]); ctx.lineWidth = j === sel ? 3 : 2; ctx.strokeStyle = COLOR[b.cls];
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (b.flags.length) {
      ctx.setLineDash([6, 4]); ctx.lineWidth = 2; ctx.strokeStyle = "#ffd400";
      ctx.strokeRect(x1 - 4, y1 - 4, x2 - x1 + 8, y2 - y1 + 8); ctx.setLineDash([]);
    }
    const txt = `${j + 1} ${LABEL[b.cls]}` + (b.flags.length ? " ⚠" + b.flags.join(",") : "");
    const tw = ctx.measureText(txt).width + 6, ty = Math.max(14, y1 - 4);
    ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(x1 - 1, ty - 14, tw, 15);
    ctx.fillStyle = b.flags.length ? "#ffd400" : COLOR[b.cls]; ctx.fillText(txt, x1 + 2, ty);
    if (j === sel) {
      ctx.fillStyle = "#fff";
      for (const [hx, hy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) ctx.fillRect(hx - 4, hy - 4, 8, 8);
    }
  });
  if (drag && drag.mode === "new") {
    const [a, b] = [drag.start, drag.end];
    ctx.setLineDash([4, 3]); ctx.lineWidth = 2; ctx.strokeStyle = COLOR[drawCls];
    ctx.strokeRect(Math.min(a[0], b[0]) * scale, Math.min(a[1], b[1]) * scale, Math.abs(b[0] - a[0]) * scale, Math.abs(b[1] - a[1]) * scale);
    ctx.setLineDash([]);
  }
}

function panel() {
  const im = cur(); if (!im) return;
  $("stem").textContent = im.stem;
  $("pos").textContent = `${pos + 1} / ${view.length}`;
  $("stbadge").textContent = ST[im.status]; $("stbadge").className = "st st-" + im.status;
  $("reasons").textContent = im.reasons.length ? "자동 라벨 사유: " + im.reasons.join(", ") : "";
  $("classes").innerHTML = KEYS.map((c, k) =>
    `<span class="cls ${c === drawCls ? "on" : ""}" data-c="${c}"><span class="sw" style="background:${COLOR[c]}"></span>${k + 1} ${LABEL[c]}</span>`).join("");
  $("boxes").innerHTML = im.boxes.map((b, j) => `<tr class="${j === sel ? "sel" : ""}" data-j="${j}">
    <td>${j + 1}</td>
    <td><select data-j="${j}">${KEYS.map(c => `<option value="${c}" ${c === b.cls ? "selected" : ""}>${LABEL[c]}</option>`).join("")}</select></td>
    <td class="flag">${b.flags.join(", ")}${b.ratio != null ? ` <span class="mut">마커 ${(b.ratio * 100).toFixed(1)}%</span>` : ""}${b.by === "human" ? ' <span class="mut">사람</span>' : ""}</td>
    <td style="white-space:nowrap">${b.flags.length ? `<button data-ok="${j}" title="Enter">승인</button>` : ""} <button data-del="${j}" title="Del">삭제</button></td>
  </tr>`).join("") || `<tr><td class="mut">박스 없음</td></tr>`;
}
$("classes").onclick = e => { const t = e.target.closest(".cls"); if (t) { setCls(t.dataset.c); } };
$("boxes").onclick = e => {
  const im = cur(), t = e.target;
  if (t.dataset.ok !== undefined) { im.boxes[+t.dataset.ok].flags = []; changed(); }
  else if (t.dataset.del !== undefined) { im.boxes.splice(+t.dataset.del, 1); sel = -1; changed(); }
  else { const tr = t.closest("tr"); if (tr && tr.dataset.j !== undefined && t.tagName !== "SELECT") sel = +tr.dataset.j; }
  panel(); draw();
};
$("boxes").onchange = e => {
  const t = e.target; if (t.tagName !== "SELECT") return;
  const b = cur().boxes[+t.dataset.j]; b.cls = t.value; b.flags = []; changed(); panel(); draw(); t.blur();
};
document.querySelectorAll("aside [data-st]").forEach(btn => btn.onclick = () => setStatus(btn.dataset.st, btn.dataset.st !== "review"));

function setCls(c) {
  if (sel >= 0) { const b = cur().boxes[sel]; b.cls = c; b.flags = []; changed(); }
  else drawCls = c;
  panel(); draw();
}
function setStatus(st, advance) {
  const im = cur(); if (!im) return;
  im.status = st; changed();
  const th = $("th" + idx()); if (th) th.className = "th " + st;
  if (advance) { if (pos < view.length - 1) go(1); else { toast("이 보기의 마지막 사진이다"); panel(); } }
  else panel();
}
function go(d) {
  if (!view.length) return;
  pos = Math.max(0, Math.min(view.length - 1, pos + d)); sel = -1; show();
}

// ---------------- 저장
function changed() {
  dirty.add(idx());
  $("save").textContent = "저장 대기…";
  clearTimeout(saveTimer); saveTimer = setTimeout(flush, 350);
}
async function flush() {
  if (saving) { saveTimer = setTimeout(flush, 200); return; }
  saving = true;
  try {
    for (const i of [...dirty]) {
      dirty.delete(i);
      const im = D.images[i];
      const r = await fetch("api/image/" + i, {method: "POST", headers: {"content-type": "application/json"},
        body: JSON.stringify({status: im.status, boxes: im.boxes})});
      const j = await r.json();
      if (!r.ok) { toast("저장 실패: " + j.error, 8000); if (r.status === 409) await load(); $("save").textContent = "저장 실패"; return; }
      counts(j.summary);
    }
    $("save").textContent = "저장됨 " + new Date().toLocaleTimeString();
  } catch (e) { $("save").textContent = "저장 실패"; toast("저장 실패: " + e, 8000); }
  finally { saving = false; }
}
window.addEventListener("beforeunload", e => { if (dirty.size || saving) { e.preventDefault(); e.returnValue = ""; } });

// ---------------- 마우스: 선택·이동·크기·새 박스
function pt(e) {
  const r = cv.getBoundingClientRect(), im = cur();
  const x = (e.clientX - r.left) * (cv.width / r.width) / scale, y = (e.clientY - r.top) * (cv.height / r.height) / scale;
  return [Math.max(0, Math.min(im.w, x)), Math.max(0, Math.min(im.h, y))];
}
function handleAt(b, p) {
  const t = 8 / scale, [x1, y1, x2, y2] = b.xyxy;
  const cs = [[x1, y1], [x2, y1], [x1, y2], [x2, y2]];
  return cs.findIndex(([x, y]) => Math.abs(p[0] - x) <= t && Math.abs(p[1] - y) <= t);
}
function boxAt(p) {
  let best = -1, area = Infinity;
  cur().boxes.forEach((b, j) => {
    const [x1, y1, x2, y2] = b.xyxy;
    if (p[0] >= x1 && p[0] <= x2 && p[1] >= y1 && p[1] <= y2 && (x2 - x1) * (y2 - y1) < area) { best = j; area = (x2 - x1) * (y2 - y1); }
  });
  return best;
}
cv.onmousedown = e => {
  if (e.button !== 0 || !cur()) return;
  e.preventDefault();
  const p = pt(e), im = cur();
  if (sel >= 0) { const h = handleAt(im.boxes[sel], p); if (h >= 0) { drag = {mode: "resize", h}; return; } }
  const j = boxAt(p);
  if (j >= 0) { sel = j; drag = {mode: "move", start: p, orig: [...im.boxes[j].xyxy]}; }
  else { sel = -1; drag = {mode: "new", start: p, end: p}; }
  panel(); draw();
};
window.onmousemove = e => {
  if (!drag) return;
  const p = pt(e), im = cur();
  if (drag.mode === "new") drag.end = p;
  else if (drag.mode === "move") {
    const o = drag.orig, w = o[2] - o[0], h = o[3] - o[1];
    const dx = Math.max(-o[0], Math.min(im.w - o[2], p[0] - drag.start[0]));
    const dy = Math.max(-o[1], Math.min(im.h - o[3], p[1] - drag.start[1]));
    im.boxes[sel].xyxy = [o[0] + dx, o[1] + dy, o[0] + dx + w, o[1] + dy + h];
    drag.moved = drag.moved || Math.abs(dx) + Math.abs(dy) > 1;
  } else {
    const b = im.boxes[sel].xyxy;
    if (drag.h & 1) b[2] = p[0]; else b[0] = p[0];
    if (drag.h & 2) b[3] = p[1]; else b[1] = p[1];
  }
  draw();
};
window.onmouseup = () => {
  if (!drag) return;
  const im = cur();
  if (drag.mode === "new") {
    const [a, b] = [drag.start, drag.end];
    const xyxy = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])].map(Math.round);
    if (xyxy[2] - xyxy[0] >= 4 && xyxy[3] - xyxy[1] >= 4) {
      im.boxes.push({cls: drawCls, xyxy, flags: [], ratio: null, by: "human"});
      sel = im.boxes.length - 1; changed();
    }
  } else if (drag.mode === "resize" || drag.moved) {
    const b = im.boxes[sel], v = b.xyxy;
    b.xyxy = [Math.min(v[0], v[2]), Math.min(v[1], v[3]), Math.max(v[0], v[2]), Math.max(v[1], v[3])].map(Math.round);
    b.flags = []; changed();  // 사람이 고친 박스 = 승인
  }
  drag = null; panel(); draw();
};

// ---------------- 키보드
document.onkeydown = e => {
  if (!D || e.target.tagName === "SELECT" || e.target.tagName === "INPUT" || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key, im = cur();
  if (k === "ArrowRight") go(1);
  else if (k === "ArrowLeft") go(-1);
  else if (!im) return;
  else if (k === "p" || k === "P") setStatus("pass", true);
  else if (k === "x" || k === "X") setStatus("exclude", true);
  else if (k === "u" || k === "U") setStatus("review", false);
  else if ("12345".includes(k) && k.length === 1) setCls(KEYS[+k - 1]);
  else if ((k === "Delete" || k === "Backspace") && sel >= 0) { im.boxes.splice(sel, 1); sel = -1; changed(); panel(); draw(); }
  else if (k === "Enter" && sel >= 0) { im.boxes[sel].flags = []; changed(); panel(); draw(); }
  else if (k === "Escape") { sel = -1; panel(); draw(); }
  else if (k === "Tab" && im.boxes.length) { sel = (sel + (e.shiftKey ? im.boxes.length - 1 : 1)) % im.boxes.length; panel(); draw(); }
  else return;
  e.preventDefault();
};

// ---------------- 위쪽 버튼
$("filter").onchange = () => { sel = -1; applyFilter(); $("filter").blur(); };
$("bulk").onclick = async () => {
  const ids = view.filter(i => D.images[i].status !== "exclude");
  if (!ids.length || !confirm(`지금 보기의 사진 ${ids.length}장을 모두 '통과'로 바꾼다(제외한 사진은 그대로).`)) return;
  await flush();
  const r = await fetch("api/bulk", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({indices: ids, status: "pass"})});
  const j = await r.json();
  if (!r.ok) { toast("실패: " + j.error, 8000); return; }
  ids.forEach(i => { D.images[i].status = "pass"; const t = $("th" + i); if (t) t.className = "th pass"; });
  counts(j.summary); panel(); show();
  toast(`${ids.length}장 통과`);
};
$("export").onclick = async () => {
  await flush();
  $("export").disabled = true;
  try {
    const r = await fetch("api/export", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify({zip: $("zip").checked})});
    const j = await r.json();
    if (!r.ok) { toast("내보내기 실패: " + j.error, 8000); return; }
    const b = Object.entries(j.boxes).map(([k, v]) => `${LABEL[k]} ${v}`).join(" · ");
    toast(`내보냄: train ${j.train}장 / valid ${j.valid}장 (빠짐 ${j.skipped}장 = 검수 필요·제외)\n박스: ${b}\n${j.dir}${j.zip ? "\n" + j.zip : ""}\n\n다음: ~/openvino-project/.venv/bin/python vision/train.py --dir ${j.dir}`, 15000);
  } finally { $("export").disabled = false; }
};
load().catch(e => { $("save").textContent = "불러오기 실패"; toast(String(e), 8000); });
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
