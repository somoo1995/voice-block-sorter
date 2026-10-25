"""브라우저 마이크 길 점검 — 크롬 가짜 마이크(파일)로 대시보드와 같은 녹음 길을 돌린다:
getUserMedia → MediaRecorder(opus) → OfflineAudioContext 디코드(16kHz) → ui/stt.

    python -m ui.mic_check                       # sim 조건 2개 × 파일 8개 × 브라우저 음성 처리 4설정 → docs/stt_real_check.md 에 덧붙임
    python -m ui.mic_check --files 4 --conds babble10
    python -m ui.mic_check --device [--listen 1]  # 시연 전 점검: 기본 입력 장치·음소거·최근 녹음 무음(peak 0) (+ 지금 1초 소리 크기)

보는 것: 마이크 여는 데 걸린 시간(getUserMedia)·녹음 시작까지·앞에서 잃은 소리(원본과 맞대어)·
echoCancellation·noiseSuppression·autoGainControl 켬/끔에 따른 받아쓰기. 가짜 마이크는 열자마자 파일을 처음부터 튼다
(%noloop) — 실제 마이크를 여는 시간(장치·PipeWire)은 이 시험에 없다. 설치 없음(google-chrome, websockets).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

from ui import stt
from ui.screens import Page
from ui.stt_check import cer, norm
from ui.stt_real_check import ALL, CONDS, OUT, SIM

ROOT = Path(__file__).resolve().parents[1]
CDP_PORT = 9333
CONFIGS = [("on", "기본(모두 켬)", "true"),
           ("off", "모두 끔", "{echoCancellation:false,noiseSuppression:false,autoGainControl:false}"),
           ("ec_ns_off", "에코·잡음 제거 끔(자동 음량 켬)", "{echoCancellation:false,noiseSuppression:false}"),
           ("ns_off", "잡음 제거만 끔", "{noiseSuppression:false}"),
           ("agc_off", "자동 음량만 끔", "{autoGainControl:false}")]
JS = """(async () => {
  const cons = %s, recMs = %d;
  const t0 = performance.now();
  const s = await navigator.mediaDevices.getUserMedia({audio: cons});
  const t1 = performance.now();
  const rec = new MediaRecorder(s); const chunks = [];
  rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  await new Promise(r => { rec.onstart = r; rec.start(); });
  const t2 = performance.now();
  await new Promise(r => setTimeout(r, recMs));
  await new Promise(r => { rec.onstop = r; rec.stop(); });
  const settings = s.getAudioTracks()[0].getSettings();
  s.getTracks().forEach(t => t.stop());
  const buf = await new Blob(chunks, {type: rec.mimeType}).arrayBuffer();
  const ab = await new OfflineAudioContext(1, 1, 16000).decodeAudioData(buf);
  const x = ab.getChannelData(0), i16 = new Int16Array(x.length);
  for (let i = 0; i < x.length; i++) i16[i] = Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767)));
  const u8 = new Uint8Array(i16.buffer); let bin = '';
  for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return JSON.stringify({gum_ms: t1 - t0, start_ms: t2 - t1, mime: rec.mimeType, settings, b64: btoa(bin)});
})()"""


def _serve() -> tuple[http.server.ThreadingHTTPServer, str]:
    d = tempfile.mkdtemp(prefix="mic-")
    Path(d, "index.html").write_text("<!doctype html><title>mic</title>", encoding="utf-8")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                          functools.partial(http.server.SimpleHTTPRequestHandler, directory=d))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/index.html"


async def _record(src: Path, cons: str, rec_ms: int, url: str) -> dict:
    prof = tempfile.mkdtemp(prefix="mic-chrome-")
    args = ["google-chrome", "--headless=new", "--disable-gpu", f"--remote-debugging-port={CDP_PORT}", f"--user-data-dir={prof}",
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={src}%noloop",
            "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1", "about:blank"]
    chrome = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json"))
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            p = Page(ws)
            task = asyncio.create_task(p.reader())
            await p.cmd("Page.enable")
            await p.cmd("Page.navigate", url=url)
            await asyncio.sleep(0.8)
            r = await p.cmd("Runtime.evaluate", expression=JS % (cons, rec_ms), awaitPromise=True, returnByValue=True)
            task.cancel()
        val = r.get("result", {}).get("value")
        if not val:
            raise RuntimeError(f"크롬 녹음 실패: {json.dumps(r)[:300]}")
        out = json.loads(val)
        out["audio"] = np.frombuffer(base64.b64decode(out.pop("b64")), dtype=np.int16).astype("float32") / 32768
        return out
    finally:
        chrome.terminate()
        chrome.wait(timeout=10)
        shutil.rmtree(prof, ignore_errors=True)


def _lost_ms(rec: np.ndarray, ref: np.ndarray) -> float | None:
    """녹음 첫 샘플이 원본의 몇 ms 지점인지(맞대기). 음수면 원본보다 먼저 시작(앞에 무음이 더 있음)."""
    if len(rec) < 1600 or len(ref) < 1600:
        return None
    n = 1 << int(np.ceil(np.log2(len(rec) + len(ref))))
    xc = np.fft.irfft(np.fft.rfft(ref, n) * np.conj(np.fft.rfft(rec, n)), n)
    k = int(np.argmax(np.abs(xc)))
    k = k - n if k > n // 2 else k
    return round(k / 16, 1)


def main():
    ap = argparse.ArgumentParser(description="브라우저 마이크 길 점검(크롬 가짜 마이크)")
    ap.add_argument("--conds", default="room,babble10")
    ap.add_argument("--files", type=int, default=8, help="조건마다 파일 수(F1·M1 번갈아)")
    ap.add_argument("--configs", default=",".join(c[0] for c in CONFIGS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--device", action="store_true", help="마이크 입력 점검만(ui/micdev.py), 안됨이 있으면 종료 코드 1")
    ap.add_argument("--listen", type=float, default=0.0, help="--device 와 함께: 기본 입력에서 이만큼(초) 받아 크기만 봄(저장 안 함)")
    a = ap.parse_args()
    if a.device:
        from ui import micdev
        rows = micdev.check(a.listen)
        for st, item, note in rows:
            print(f"{st:<6}{item:<10}{note}")
        sys.exit(1 if any(st == "안됨" for st, _, _ in rows) else 0)
    srv, url = _serve()
    cfgs = [c for c in CONFIGS if c[0] in a.configs.split(",")]
    refs = {no: ref for no, ref, _ in ALL}
    rows = []
    t_start = time.time()
    with tempfile.TemporaryDirectory() as d:
        for cond in a.conds.split(","):
            files = sorted((SIM / cond).glob("*.wav"))
            files = sorted(files, key=lambda f: (f.stem[3:], f.stem[:2]))
            step = max(1, len(files) // max(1, a.files))  # 번호·목소리를 고르게
            pick = files[::step][:a.files]
            for f in pick:
                src44, sr = sf.read(f, dtype="float32")
                # 가짜 마이크 파일: 앞 0.6초·뒤 0.8초 방 소리(파일 첫 50ms 의 소리 바닥) + 원본
                floor = np.random.default_rng(0).standard_normal(int(sr * 0.6)).astype("float32") * float(np.std(src44[:int(sr * 0.05)]))
                padded = np.concatenate([floor, src44, floor, floor[:int(sr * 0.2)]])
                fake = Path(d) / f"fake_{cond}_{f.stem}.wav"
                sf.write(fake, padded, sr, subtype="PCM_16")
                ref16 = stt._load_16k_mono(str(fake), aa=True)
                for key, label, cons in cfgs:
                    r = asyncio.run(_record(fake, cons, int(len(padded) / sr * 1000) + 300, url))
                    wav = Path(d) / f"rec_{key}_{fake.stem}.wav"
                    sf.write(wav, r["audio"], 16000, subtype="PCM_16")
                    stt.transcribe(str(wav), prompt="")
                    hyp = stt.last_info.get("raw", "")
                    ref = refs[f.stem[3:]]
                    m = stt.last_info
                    rows.append({"cond": cond, "file": f.stem, "cfg": key, "ref": ref, "hyp": hyp, "exact": norm(ref) == norm(hyp),
                                 "cer": cer(ref, hyp), "gum_ms": round(r["gum_ms"], 1), "start_ms": round(r["start_ms"], 1),
                                 "lost_ms": _lost_ms(r["audio"], ref16), "peak": m.get("peak"), "speech_rms": m.get("speech_rms"),
                                 "mime": r["mime"], "settings": r["settings"]})
                    print(f"{cond} {f.stem} {key}: {hyp!r} gum {r['gum_ms']:.0f}ms start {r['start_ms']:.0f}ms "
                          f"lost {rows[-1]['lost_ms']}ms peak {m.get('peak')}", file=sys.stderr)
    srv.shutdown()
    cl = dict(CONDS)
    lab = {c[0]: c[1] for c in cfgs}
    s0 = rows[0]["settings"] if rows else {}
    out = [f"## {datetime.now():%Y-%m-%d %H:%M} — 브라우저 마이크 길(크롬 가짜 마이크, `python -m ui.mic_check`)", "",
           f"- 크롬 {subprocess.run(['google-chrome', '--version'], capture_output=True, text=True).stdout.strip()}, "
           f"MediaRecorder `{rows[0]['mime'] if rows else '?'}`, 디코드 16kHz, 받아쓰기 다듬기 `{','.join(stt.PREP)}`. "
           f"{(time.time() - t_start) / 60:.0f}분 걸림.",
           f"- 기본 설정에서 크롬이 알려 준 값: echoCancellation {s0.get('echoCancellation')}, noiseSuppression "
           f"{s0.get('noiseSuppression')}, autoGainControl {s0.get('autoGainControl')}, sampleRate {s0.get('sampleRate')}.", "",
           "| 조건 | 설정 | 완전 일치 / CER | 마이크 열기 ms | 녹음 시작 ms | 앞에서 잃은 소리 ms(중앙) | 말소리 크기 |", "|---|---|---|---|---|---|---|"]
    for cond in a.conds.split(","):
        for key, _, _ in cfgs:
            rs = [r for r in rows if r["cond"] == cond and r["cfg"] == key]
            if not rs:
                continue
            lost = [r["lost_ms"] for r in rs if r["lost_ms"] is not None]
            out.append(f"| {cl.get(cond, cond)} | {lab[key]} | {sum(r['exact'] for r in rs)}/{len(rs)} · "
                       f"{np.mean([r['cer'] for r in rs]):.3f} | {np.median([r['gum_ms'] for r in rs]):.0f} | "
                       f"{np.median([r['start_ms'] for r in rs]):.0f} | {np.median(lost) if lost else '-'} | "
                       f"{np.median([r['speech_rms'] or 0 for r in rs]):.3f} |")
    out += ["", "| 조건 | 파일 | 정답 | " + " | ".join(lab[c[0]] for c in cfgs) + " |", "|---|---|---|" + "---|" * len(cfgs)]
    for cond in a.conds.split(","):
        for stem in dict.fromkeys(r["file"] for r in rows if r["cond"] == cond):
            rs = {r["cfg"]: r for r in rows if r["cond"] == cond and r["file"] == stem}
            out.append(f"| {cl.get(cond, cond)} | {stem} | {next(iter(rs.values()))['ref']} | " + " | ".join(
                (rs[c[0]]["hyp"] if rs[c[0]]["exact"] else f"**{rs[c[0]]['hyp']}**") if c[0] in rs else "" for c in cfgs) + " |")
    md = "\n".join(out + [""])
    print(md)
    (SIM / f"mic_rows_{datetime.now():%Y%m%d_%H%M}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str),
                                                                      encoding="utf-8")
    if a.out:
        p = Path(a.out)
        old = p.read_text(encoding="utf-8") if p.exists() else "# 받아쓰기 실물 점검\n\n"
        i = old.find("\n## ")
        p.write_text((old[:i + 1] + md + "\n" + old[i + 1:]) if i >= 0 else old + "\n" + md, encoding="utf-8")
        print(f"\n→ {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
