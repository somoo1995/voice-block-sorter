"""위 카메라로 학습용 사진 찍기 (640x480 JPEG).

  창 모드(젠북 화면에서):  python vision/capture.py                 # 스페이스 = 저장, q/ESC = 끝
  간격 모드(창 없이):      python vision/capture.py --interval 2 --count 30

저장: vision/data/raw/<YYYYMMDD>/<prefix>_<HHMMSS>_<번호>.jpg  → autolabel.py 로 라벨 초안.
카메라를 다른 프로세스가 쓰는 중이면 장치를 열지 않는다. Studio 가 쓰는 중이면 Studio 스트림에서 받는다
(--source device 로 막을 수 있음).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

try:
    from . import camera
except ImportError:
    import camera

VISION = Path(__file__).resolve().parent


def save(frame, out: Path, prefix: str, n: int) -> Path:
    path = out / f"{prefix}_{datetime.now():%H%M%S}_{n:03d}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def run_window(stream, out: Path, prefix: str) -> int:
    win = "capture  [space]=save  [q]=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    n = 0
    flash = 0
    while True:
        frame = stream.read()
        view = frame.copy()
        cv2.putText(view, f"{n} saved  ({stream.source})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if flash:
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (255, 255, 255), 8)
            flash -= 1
        cv2.imshow(win, view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            n += 1
            print(save(frame, out, prefix, n), flush=True)
            flash = 3
        elif k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    return n


def run_interval(stream, out: Path, prefix: str, interval: float, count: int) -> int:
    n = 0
    next_t = time.monotonic()
    try:
        while count <= 0 or n < count:
            frame = stream.read()  # 계속 읽어야 오래된 장이 쌓이지 않는다
            if time.monotonic() >= next_t:
                n += 1
                print(save(frame, out, prefix, n), flush=True)
                next_t += interval
    except KeyboardInterrupt:
        pass
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=float, help="초 간격 자동 촬영(창 없음)")
    ap.add_argument("--count", type=int, default=0, help="간격 모드에서 찍을 장 수 (0 = Ctrl+C 까지)")
    ap.add_argument("--out", type=Path, default=VISION / "data" / "raw" / f"{datetime.now():%Y%m%d}")
    ap.add_argument("--prefix", default="block")
    ap.add_argument("--source", choices=["auto", "device", "studio"], default="auto")
    a = ap.parse_args()

    if a.interval is None and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        sys.exit("화면이 없음 — 젠북 화면의 터미널에서 실행하거나 --interval 로")
    a.out.mkdir(parents=True, exist_ok=True)
    stream = camera.open_stream(a.source)
    print(f"source={stream.source} → {a.out}", flush=True)
    try:
        if a.interval is None:
            n = run_window(stream, a.out, a.prefix)
        else:
            n = run_interval(stream, a.out, a.prefix, a.interval, a.count)
    finally:
        stream.close()
    print(f"{n}장 저장, 카메라 닫음")


if __name__ == "__main__":
    main()
