"""발표용 화면 캡처 + 음성 입력 경로 시험 — 헤드리스 크롬(CDP)으로 대시보드에서 발표 실연(계약서 2절 끝)을 돌리고 PNG 로 남긴다.

    UI_TTS_OUT=timed BALL_MOCK=1 BALL_MOCK_SCENE=demo4 BALL_MOCK_POLICY_S=19 ui/start_ui.sh   # 실연과 같은 시간으로 띄운다
    .venv/bin/python -m ui.screens                     # → docs/screens/*.png, report_*.html
    .venv/bin/python -m ui.screens --buttons           # 음성 파일 대신 빠른 명령 버튼으로
    UI_TTS_OUT=timed BALL_MOCK=1 BALL_MOCK_SCENE=full ui/start_ui.sh && .venv/bin/python -m ui.screens --part digit
                                                       # 숫자 7 계획·실행(블록 11개 장면, 아래)

--part main(기본, 블록 4개 demo4): 실연 ①~⑤ → 문서 질문 3개(작업 기준서·로봇 운용 설명서·시스템 안내서) → 초기화 뒤 ① 도중
음성 "멈춰" → 초기화 뒤 "분홍으로 숫자 7 만들어 줘"(분홍 정상 블록이 2개라 블록 부족 거절 화면).
--part digit(블록 11개 full): "블록으로 숫자 7 만들어 줘" → 칸 배치 계획 + "할까요?" → "그렇게 해"로 대답해 실행까지.
명령은 기본으로 ui/samples/demo/F1_NN.wav(python -m ui.demo_samples 가 만든 실연 음성)를 마이크 칸의 "파일 올리기"로 넣는다
= 녹음이 끝난 뒤와 같은 경로(STT → 대화창 🎙 → run_command). 크롬은 127.0.0.1 말고는 이름 풀이를 막아(와이파이 끈 것처럼)
화면이 로컬 자원만으로 그려지는지도 본다. 끝에 외부 요청·콘솔 오류 수를 출력한다. 설치 없음(google-chrome, websockets).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screens"
SAMPLES = ROOT / "ui" / "samples" / "demo"  # python -m ui.demo_samples
PORT = 9339
RUNNING = '(document.querySelector("#hdr")||{innerText:""}).innerText.includes("실행 중")'
ACTIVE = '(document.querySelector(".node.active")||{innerText:""}).innerText.includes(%s)'
PLAN_MID = ('document.querySelector(".plan li.st-active") && document.querySelectorAll(".plan li.st-done").length >= 1')
POLICY = ACTIVE % json.dumps("정책 실행")
# 발표 실연: (음성 번호, 빠른 명령 버튼 글, 실행 중에 찍을 조건 또는 None, 파일 이름)
STEPS = [("01", "① 불량품", POLICY, "01_defect"), ("02", "② 지금 배치", None, "02_scene"),
         ("03", "③ 불량 빼고", PLAN_MID, "03_plan"), ("04", "④ 분홍 블록 하나", POLICY, "04_handoff"),
         ("05", "⑤ 결과", None, "05_report")]
# 문서 질문: 문서마다 하나(작업 기준서 3.2 · 로봇 운용 설명서 4.3 · 시스템 안내서 5.1)
QUESTIONS = [("12", "? 초록 불량은", "06_question_standard"), ("08", "? 그리퍼가", "07_question_robot"),
             ("09", "? 인터넷", "08_question_system")]


class Page:
    def __init__(self, ws):
        self.ws, self.seq, self.pending = ws, 0, {}
        self.external, self.errors = [], []

    async def reader(self):
        async for msg in self.ws:
            m = json.loads(msg)
            if m.get("id") in self.pending:
                self.pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Network.requestWillBeSent":
                url = m["params"]["request"]["url"]
                if urlparse(url).hostname not in (None, "127.0.0.1", "localhost"):
                    self.external.append(url)
            elif m.get("method") == "Runtime.consoleAPICalled" and m["params"]["type"] == "error":
                self.errors.append(" ".join(str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])[:200])
            elif m.get("method") == "Runtime.exceptionThrown":
                self.errors.append(str(m["params"]["exceptionDetails"].get("exception", {}).get("description", ""))[:200])

    async def cmd(self, method, **params):
        self.seq += 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[self.seq] = fut
        await self.ws.send(json.dumps({"id": self.seq, "method": method, "params": params}))
        r = await asyncio.wait_for(fut, 60)
        if "error" in r:
            raise RuntimeError(f"{method}: {r['error']}")
        return r.get("result", {})

    async def js(self, expr):
        r = await self.cmd("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def until(self, expr, timeout=180):
        t0 = time.time()
        while not await self.js(expr):
            if time.time() - t0 > timeout:
                raise TimeoutError(expr[:80])
            await asyncio.sleep(0.25)
        return time.time() - t0

    async def shot(self, name, full=False):
        params = {"format": "png"}
        if full:
            w, h = json.loads(await self.js("JSON.stringify([document.documentElement.scrollWidth, document.documentElement.scrollHeight])"))
            params.update(captureBeyondViewport=True, clip={"x": 0, "y": 0, "width": w, "height": h, "scale": 1})
        r = await self.cmd("Page.captureScreenshot", **params)
        (OUT / f"{name}.png").write_bytes(base64.b64decode(r["data"]))
        print(f"  찍음 {OUT / name}.png")

    async def click(self, text):
        ok = await self.js(f"""(() => {{ const b = [...document.querySelectorAll('button')]
            .find(e => e.innerText.trim().startsWith({json.dumps(text)})); if (!b) return false; b.click(); return true; }})()""")
        if not ok:
            raise RuntimeError(f"버튼 없음: {text}")

    async def upload(self, wav: Path):
        await self.js("""(() => { const b = document.querySelector('#mic button[aria-label="Upload file"]'); if (b) b.click(); })()""")
        await asyncio.sleep(0.4)
        doc = await self.cmd("DOM.getDocument", depth=-1, pierce=True)
        node = await self.cmd("DOM.querySelector", nodeId=doc["root"]["nodeId"], selector="#mic input[type=file]")
        if not node.get("nodeId"):
            raise RuntimeError("마이크 칸에 파일 올리기 입력이 없음")
        await self.cmd("DOM.setFileInputFiles", files=[str(wav)], nodeId=node["nodeId"])


async def _settle(p: Page):
    """끝난 뒤: 마이크 칸을 녹음 모드로(발표 때 모습), 머리줄 타이머·카드 애니메이션이 가라앉게."""
    mode = await p.js("""(() => { const b = document.querySelector('#mic button[aria-label="Record audio"]');
        return b && b.classList.contains('selected') ? 'mic' : 'upload'; })()""")
    if mode != "mic":
        await p.js("""document.querySelector('#mic button[aria-label="Record audio"]').click()""")
    await asyncio.sleep(2.5)


async def _heard(p: Page) -> str:
    return await p.js("""(document.querySelector('#chat') || {innerText: ''}).innerText.split('\\n')
        .map(t => t.trim()).filter(t => t.startsWith('🎙')).pop() || ''""")


async def _say(p: Page, no: str, label: str | None, buttons: bool, voice: str):
    if buttons and label:
        await p.click(label)
    else:
        await p.upload(SAMPLES / f"{voice}_{no}.wav")


async def _step(p: Page, no, label, mid, name, buttons, voice, timeout=400):
    t0 = time.time()
    await _say(p, no, label, buttons, voice)
    await p.until(RUNNING, 60)
    if mid:
        try:
            await p.until(mid, 240)
            await asyncio.sleep(1.0)
            await p.shot(f"{name}_running")
        except TimeoutError:
            print(f"  ({name}: 실행 중 화면 조건이 안 와서 건너뜀)")
    await p.until("!" + RUNNING, timeout)
    await _settle(p)
    heard = await _heard(p)
    print(f"{name}: {time.time() - t0:.1f}초" + (f" · 대화창 {heard!r}" if heard else ""))
    await p.shot(f"{name}_done" if mid else name, full=True)  # 끝 장면은 전체 페이지: 1080 아래로 내려간 대화창(답·근거 줄)까지


async def _open_extra(p: Page):
    """접어 둔 "다른 명령"을 펼친다(이미 펼쳐져 있으면 그대로)."""
    await p.js("""(() => { const d = document.querySelector('#extra'); if (!d) return;
        if (![...d.querySelectorAll('button')].some(b => b.offsetParent && /분홍으로|그리퍼/.test(b.innerText))) {
          const h = d.querySelector('.label-wrap, button'); if (h) h.click(); } })()""")
    await asyncio.sleep(0.6)


async def _reset(p: Page):
    await p.click("초기화")
    await asyncio.sleep(2.0)


async def _open(p: Page, url: str):
    await p.cmd("Emulation.setDeviceMetricsOverride", width=1920, height=1080, deviceScaleFactor=1, mobile=False)
    await p.cmd("Page.navigate", url=url)
    await p.until('document.querySelector("#hdr") && document.querySelector("#hdr").innerText.includes("OVMS")', 60)
    await asyncio.sleep(1.5)
    await p.click("초기화")
    await p.until('document.querySelector(".log").innerText.includes("초기화")')
    await asyncio.sleep(2.5)


async def scenario(p: Page, url: str, buttons: bool, voice: str):
    await _open(p, url)
    await p.shot("00_ready")
    for no, label, mid, name in STEPS:  # 발표 실연 ①~⑤
        await _step(p, no, label, mid, name, buttons, voice)
    for no, label, name in QUESTIONS:  # 문서 질문(접어 둔 "다른 명령"에도 있음)
        if buttons:
            await _open_extra(p)
        await _step(p, no, label, None, name, buttons, voice)
    # 멈춤: 처음 상태로 돌려 ① 을 시작하고 정책 실행 중에 "멈춰"(음성, 버튼 모드면 ■ 멈춰)
    await _reset(p)
    await _say(p, "01", "① 불량품", buttons, voice)
    await p.until(POLICY, 90)
    await asyncio.sleep(2.0)
    if buttons:
        await p.click("■ 멈춰")
    else:
        await p.upload(SAMPLES / f"{voice}_06.wav")
    try:
        await p.until("!" + RUNNING, 30)
    except TimeoutError:
        print("  (음성 멈춰 뒤 30초 안에 안 멈춰 ■ 멈춰 버튼을 누름)")
        await p.click("■ 멈춰")
        await p.until("!" + RUNNING, 30)
    await _settle(p)
    await p.shot("09_stop", full=True)
    # 숫자 모양(좌표 집기) — 블록 4개 장면에서는 분홍 정상 블록이 모자라 거절한다
    await _reset(p)
    await _step(p, "10", None, None, "10_digit7_pink", buttons, voice)


async def scenario_digit(p: Page, url: str, buttons: bool, voice: str):
    """블록 11개(BALL_MOCK_SCENE=full)로 띄운 대시보드에서: 숫자 7 계획 → "할까요?"(확인 대기)면 대답해 실행까지."""
    await _open(p, url)
    await _step(p, "13", None, None, "11_digit7_plan", False, voice)  # 이 문장은 버튼이 없어 늘 음성 파일
    if await p.js("(() => { const r = document.querySelector('#confirm'); return !!(r && r.offsetParent); })()"):
        if buttons:
            await p.click("응, 해 줘")
        else:
            await p.upload(SAMPLES / f"{voice}_11.wav")
        await p.until(RUNNING, 60)
        await p.until("!" + RUNNING, 400)
        await _settle(p)
        await p.shot("12_digit7_run", full=True)
    else:
        print("  (숫자 7: 확인 대기가 아니라 대답 장면은 건너뜀 — 거절·계획만이면 11_digit7_plan 이 그 화면)")


async def main_async(a):
    OUT.mkdir(parents=True, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="screens-")
    args = ["google-chrome", "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={PORT}",
            f"--user-data-dir={prof}", "--window-size=1920,1080",
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]  # 마이크 칸이 "마이크 없음"으로 안 뜨게
    if not a.online:
        args.append("--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1")
    chrome = subprocess.Popen(args + ["about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t_start = time.time()
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            p = Page(ws)
            task = asyncio.create_task(p.reader())
            for d in ("Network", "Runtime", "Page"):
                await p.cmd(f"{d}.enable")
            run = scenario_digit if a.part == "digit" else scenario
            await run(p, f"http://127.0.0.1:{a.port}/?__theme={a.theme}", a.buttons, a.voice)
            # 이번에 만든 보고서(ui/report.py) HTML 을 같이 남기고, 새 창으로 열어 한 장 더
            reports = sorted((ROOT / "ui" / "reports").glob("report_*.html"), key=lambda x: x.stat().st_mtime)
            new = [r for r in reports if r.stat().st_mtime >= t_start]
            if new:
                dst = OUT / new[-1].name
                shutil.copy(new[-1], dst)
                print(f"  보고서 {dst}")
                await p.cmd("Page.navigate", url=f"http://127.0.0.1:{a.port}/gradio_api/file={new[-1]}")
                await asyncio.sleep(2)
                await p.cmd("Emulation.setDeviceMetricsOverride", width=1100, height=900, deviceScaleFactor=1, mobile=False)
                await p.shot("13_report_page", full=True)
            print(f"외부 요청 {len(p.external)}건 {p.external[:5]} · 콘솔 오류 {len(p.errors)}건 {p.errors[:5]}")
            task.cancel()
    finally:
        chrome.terminate()
        shutil.rmtree(prof, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="대시보드 화면 캡처(명령 1→5)")
    ap.add_argument("--buttons", action="store_true", help="음성 파일 대신 빠른 명령 버튼")
    ap.add_argument("--voice", default="F1", help="ui/samples 의 TTS 목소리(기본 F1)")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--online", action="store_true", help="크롬의 외부 이름 풀이를 막지 않음")
    ap.add_argument("--out", help="저장 폴더(기본 docs/screens)")
    ap.add_argument("--part", default="main", choices=["main", "digit"],
                    help="main = 실연·질문·멈춤·분홍 7(demo4), digit = 숫자 7 계획·실행(BALL_MOCK_SCENE=full 로 띄운 대시보드)")
    a = ap.parse_args()
    if a.out:
        global OUT
        OUT = Path(a.out).expanduser().resolve()
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
