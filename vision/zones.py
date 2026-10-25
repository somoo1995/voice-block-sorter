"""스냅샷 위에 구역 5개 다각형을 클릭으로 찍어 vision/zones.json 저장.

  python vision/zones.py                    # 위 카메라에서 한 장 받아서
  python vision/zones.py --image snap.jpg   # 저장된 사진 위에
  python vision/zones.py --render out.jpg   # 창 없이 지금 zones.json 을 사진 위에 그려 확인

조작: 왼쪽 클릭 = 점 추가 · 오른쪽 클릭/Backspace = 마지막 점 취소 · Enter/Space = 이 구역 끝(점 3개 이상)
      1~5 = 그 구역 다시 찍기 · r = 지금 구역 점 지우기 · s = 저장 · q/ESC = 끝
다섯 구역을 다 찍으면 자동 저장. 이후 고친 것은 s 로 저장.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

VISION = Path(__file__).resolve().parent
ZONES_JSON = VISION / "zones.json"
ZONES = ["input", "bowl_red", "bowl_blue", "bowl_defect", "hand_spot"]
BOWL_ZONES = ZONES[:4]
COLORS = {  # BGR. bowl_red = 분홍 그릇, bowl_blue = 초록 그릇 (식별자는 그대로)
    "input": (0, 200, 255),
    "bowl_red": (203, 102, 255),
    "bowl_blue": (60, 200, 60),
    "bowl_defect": (40, 40, 40),
    "hand_spot": (255, 200, 0),
}


def load_zones(path: Path | str = ZONES_JSON, size: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """{구역: (N,2) int32 다각형}. size=(w,h) 를 주면 찍을 때 해상도에서 비례 변환."""
    d = json.loads(Path(path).read_text())
    zones = {k: np.array(v, np.float32) for k, v in d["zones"].items()}
    if size and tuple(d.get("image_size", size)) != tuple(size):
        sx, sy = size[0] / d["image_size"][0], size[1] / d["image_size"][1]
        zones = {k: v * (sx, sy) for k, v in zones.items()}
    return {k: v.round().astype(np.int32) for k, v in zones.items()}


def zone_of(pt: tuple[float, float], zones: dict[str, np.ndarray], names=BOWL_ZONES) -> str | None:
    """점이 들어 있는 첫 구역(names 순서)."""
    for name in names:
        if name in zones and cv2.pointPolygonTest(zones[name], (float(pt[0]), float(pt[1])), False) >= 0:
            return name
    return None


def draw_zones(img, zones: dict[str, np.ndarray], alpha: float = 0.18):
    overlay = img.copy()
    for name, poly in zones.items():
        cv2.fillPoly(overlay, [poly], COLORS.get(name, (200, 200, 200)))
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    for name, poly in zones.items():
        c = COLORS.get(name, (200, 200, 200))
        cv2.polylines(img, [poly], True, c, 2)
        x, y = poly.min(0)
        cv2.putText(img, name, (int(x) + 4, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(img, name, (int(x) + 4, int(y) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return img


class Editor:
    """창 없이도 시험할 수 있게 마우스·키 처리와 그리기를 분리했다."""

    def __init__(self, image, zones: dict[str, list] | None = None):
        self.image = image
        self.zones: dict[str, list] = {k: [list(map(int, p)) for p in v] for k, v in (zones or {}).items()}
        self.points: list[list[int]] = []
        self.mouse_pos: tuple[int, int] | None = None
        self.idx = self._next_missing(0)

    def _next_missing(self, start: int) -> int:
        for i in list(range(start, len(ZONES))) + list(range(start)):
            if ZONES[i] not in self.zones:
                return i
        return len(ZONES)

    @property
    def current(self) -> str | None:
        return ZONES[self.idx] if self.idx < len(ZONES) else None

    @property
    def complete(self) -> bool:
        return all(z in self.zones for z in ZONES)

    def mouse(self, event, x, y, flags=0, param=None) -> None:
        self.mouse_pos = (x, y)
        if self.current is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([int(x), int(y)])
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def key(self, k: int) -> str | None:
        """'finished'(마지막 구역까지 끝) · 'save' · 'quit' · None."""
        if k in (13, 10, 32):
            if self.current and len(self.points) >= 3:
                was_complete = self.complete
                self.zones[self.current] = self.points
                self.points = []
                self.idx = self._next_missing(self.idx)
                if self.complete and not was_complete:
                    return "finished"
        elif k == 8 and self.points:
            self.points.pop()
        elif ord("1") <= k <= ord("5"):
            self.idx, self.points = k - ord("1"), []
        elif k == ord("r"):
            self.points = []
        elif k == ord("s"):
            return "save"
        elif k in (ord("q"), 27):
            return "quit"
        return None

    def render(self):
        img = draw_zones(self.image.copy(), {k: np.array(v, np.int32) for k, v in self.zones.items()})
        if self.current:
            c = COLORS[self.current]
            pts = self.points + ([list(self.mouse_pos)] if self.mouse_pos and self.points else [])
            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts, np.int32)], False, c, 2)
            for p in self.points:
                cv2.circle(img, tuple(p), 4, c, -1)
            msg = f"[{self.idx + 1}/5] {self.current}: {len(self.points)} pts  Enter=done  RClick=undo  s=save  q=quit"
        else:
            msg = "all 5 zones done  (1-5 = redo a zone, s = save, q = quit)"
        cv2.rectangle(img, (0, img.shape[0] - 26), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.putText(img, msg, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img


def save_zones(zones: dict[str, list], image, source: str, path: Path = ZONES_JSON) -> Path:
    snap = path.with_name(f"{path.stem}_snapshot.jpg")  # zones.json → zones_snapshot.jpg
    cv2.imwrite(str(snap), image)
    data = {
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "camera": "front (ZVCAI USB_Camera zvcai231100001)",
        "source_image": source,
        "snapshot": snap.name,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "zones": {z: zones[z] for z in ZONES if z in zones},
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, help="이 사진 위에 찍기 (없으면 카메라에서 한 장)")
    ap.add_argument("--zones", type=Path, default=ZONES_JSON)
    ap.add_argument("--render", type=Path, metavar="OUT_JPG", help="창 없이 zones 를 그려 저장")
    ap.add_argument("--source", choices=["auto", "device", "studio"], default="auto")
    a = ap.parse_args()

    if a.image:
        image, source = cv2.imread(str(a.image)), str(a.image)
        if image is None:
            sys.exit(f"사진을 못 읽음: {a.image}")
    elif a.render and a.zones.with_name(f"{a.zones.stem}_snapshot.jpg").exists():
        snap = a.zones.with_name(f"{a.zones.stem}_snapshot.jpg")
        image, source = cv2.imread(str(snap)), str(snap)
    else:
        try:
            from . import camera
        except ImportError:
            import camera
        image, src = camera.grab_frame(a.source)
        source = f"camera ({src}) {datetime.now():%Y-%m-%d %H:%M:%S}"

    if a.render:
        zones = load_zones(a.zones, (image.shape[1], image.shape[0]))
        cv2.imwrite(str(a.render), draw_zones(image, zones))
        print(a.render, {k: len(v) for k, v in zones.items()})
        return

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        sys.exit("화면이 없음 — 젠북 화면의 터미널에서 실행")
    old = json.loads(a.zones.read_text())["zones"] if a.zones.exists() else None
    ed = Editor(image, old)
    win = "zones"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, ed.mouse)
    while True:
        cv2.imshow(win, ed.render())
        act = ed.key(cv2.waitKey(30) & 0xFF)
        if act in ("finished", "save"):
            if ed.complete:
                print("저장:", save_zones(ed.zones, image, source, a.zones))
            else:
                print("아직 안 찍은 구역:", [z for z in ZONES if z not in ed.zones])
        elif act == "quit" or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
