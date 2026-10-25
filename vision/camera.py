"""위 카메라(Studio 'front' = ZVCAI USB_Camera) 공용 도구.

- busy_pids(): 다른 프로세스(Studio 컨테이너·정책 실행)가 카메라를 쓰는지 확인. 쓰는 중이면 열지 않는다.
- grab_frame(): 한 장 찍고 바로 닫기 (640x480 BGR). Studio 가 카메라를 잡고 있으면 장치를 열지 않고
  Studio 미리보기 웹소켓에서 한 장 받는다(source="auto").
- list_controls()/set_controls(): v4l2-ctl 없이 노출·화이트밸런스 컨트롤 조회/설정 (ioctl).

CLI:
  python vision/camera.py --busy
  python vision/camera.py --list
  python vision/camera.py --set auto_exposure=1 exposure_time_absolute=150 white_balance_automatic=0
  python vision/camera.py --snap /tmp/x.jpg [--source auto|device|studio]
"""
from __future__ import annotations

import argparse
import fcntl
import os
import struct
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request

# /dev/videoN 번호는 꽂는 순서로 바뀌므로 by-id(시리얼) 로 찾는다. Studio 카메라 설정의 serial 과 같다.
FRONT_CAM = "/dev/v4l/by-id/usb-ZVCAI_USB_Camera_zvcai231100001-video-index0"
FRONT_SERIAL = "zvcai231100001"
WIDTH, HEIGHT = 640, 480
STUDIO = os.environ.get("BALL_STUDIO_URL", "http://127.0.0.1:7860")


class CameraBusy(RuntimeError):
    pass


def resolve(dev: str = FRONT_CAM) -> str:
    real = os.path.realpath(dev)
    if not os.path.exists(real):
        raise FileNotFoundError(f"카메라 없음: {dev} (USB 연결 확인)")
    return real


# root 로 돌릴 스캔. Studio 는 privileged 컨테이너라 /dev 가 따로 있어 경로·inode 가 호스트와 다르다
# (fuser·lsof 가 못 찾음, 2026-10-21 실측) → 열린 fd 를 stat 해서 장치 번호(rdev)로 비교한다.
_SCAN = r"""
import os, stat, sys
rdev, me, unreadable = int(sys.argv[1]), int(sys.argv[2]), False
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        fds = os.listdir(f"/proc/{pid}/fd")
    except PermissionError:
        unreadable = True
        continue
    except OSError:
        continue
    for fd in fds:
        try:
            st = os.stat(f"/proc/{pid}/fd/{fd}")
        except OSError:
            continue
        if stat.S_ISCHR(st.st_mode) and st.st_rdev == rdev and int(pid) != me:
            print(pid)
            break
if unreadable:
    print("UNREADABLE")
"""


def busy_pids(dev: str = FRONT_CAM) -> list[int] | None:
    """dev 를 열고 있는 PID 목록. 확인할 수 없으면 None."""
    args = [_SCAN, str(os.stat(resolve(dev)).st_rdev), str(os.getpid())]
    for cmd in (["sudo", "-n", "/usr/bin/python3", "-I", "-S", "-c"], [sys.executable, "-I", "-S", "-c"]):
        try:
            r = subprocess.run(cmd + args, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode != 0:
            continue
        out = r.stdout.split()
        pids = [int(p) for p in out if p.isdigit()]
        if pids or "UNREADABLE" not in out:
            return pids
        return None  # sudo 없이 돌았고 남의 프로세스는 못 봤다
    return None


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def held_by_studio(pids: list[int]) -> bool:
    return bool(pids) and all("physicalai" in cmdline(p) for p in pids)


def held_by_chrome_probe(pids: list[int]) -> bool:
    """크롬의 video_capture 도우미 프로세스뿐인가. 대시보드가 마이크를 열고 닫을 때 크롬이 장치 목록을 만들며
    카메라를 잠깐 연다(10-23 14:17 리허설: 음성 명령 직후 detect 가 CameraBusy) → 잠깐 기다리면 풀린다."""
    return bool(pids) and all("video_capture" in cmdline(p) for p in pids)


def ensure_free(dev: str = FRONT_CAM, chrome_wait_s: float = 1.0) -> None:
    pids = busy_pids(dev)
    t_end = time.monotonic() + chrome_wait_s
    while pids and held_by_chrome_probe(pids) and time.monotonic() < t_end:
        time.sleep(0.25)
        pids = busy_pids(dev)
    if pids and held_by_chrome_probe(pids):
        # 크롬 도우미는 장치를 열어 목록만 만들고 영상은 안 받는다(14:31 리허설: 4초를 기다려도 남음) → 막지 않고 연다.
        # 정말 영상을 받고 있으면 아래 open_camera 가 "카메라 열기 실패"로 알린다
        return
    if pids:
        raise CameraBusy(f"카메라 사용 중(PID {pids}: {cmdline(pids[0])[:80]})")
    if pids is None:
        raise CameraBusy("카메라 사용 여부를 확인할 수 없음(sudo -n 실패) — 열지 않음")


def open_camera(dev: str = FRONT_CAM, check: bool = True):
    import cv2

    if check:
        ensure_free(dev)
    # OpenCV 5 의 V4L2 백엔드는 경로 문자열로 못 연다 → /dev/videoN 의 N 으로 연다
    cap = cv2.VideoCapture(int(resolve(dev).removeprefix("/dev/video")), cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"카메라 열기 실패: {dev} (그 사이 다른 프로세스가 잡았을 수 있음)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return cap


def fit(frame):
    import cv2

    if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return frame


class DeviceStream:
    """카메라 장치를 직접 연다. 처음 몇 장은 노출이 안 잡혀 버린다."""

    source = "device"

    def __init__(self, dev: str = FRONT_CAM, warmup: int = 8, check: bool = True):
        self.cap = open_camera(dev, check)
        for _ in range(warmup):
            self.cap.grab()

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("프레임 읽기 실패")
        return fit(frame)

    def close(self) -> None:
        self.cap.release()


def studio_camera_config(serial: str = FRONT_SERIAL) -> dict:
    """Studio 에 등록된 카메라 설정(GET 만). 같은 설정으로 붙어야 publisher 설정을 바꾸지 않는다."""
    def get(path):
        with urllib.request.urlopen(f"{STUDIO}{path}", timeout=5) as r:
            return json.load(r)

    for p in get("/api/projects"):
        for c in get(f"/api/projects/{p['id']}/cameras"):
            if (c.get("fingerprint") or {}).get("serial") == serial:
                return c
    raise CameraBusy(f"Studio 에 serial {serial} 카메라 설정이 없음")


class StudioStream:
    """Studio 가 카메라를 잡고 있을 때: 미리보기와 같은 웹소켓(/api/cameras/ws)에 구독자로 붙는다.

    장치는 열지 않는다. 이미 도는 공유 publisher 의 프레임(JPEG q80)을 받는다.
    """

    source = "studio"

    def __init__(self, serial: str = FRONT_SERIAL, timeout: float = 5.0):
        import websocket  # websocket-client

        cfg = studio_camera_config(serial)
        url = STUDIO.replace("http", "ws", 1) + "/api/cameras/ws?camera=" + urllib.parse.quote(json.dumps(cfg))
        self.ws = websocket.create_connection(url, timeout=timeout)
        # 구독 직후 몇 장은 아직 빈(검은) 버퍼다
        deadline = time.monotonic() + timeout
        while self._recv().max() == 0:
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError("Studio 스트림에서 빈 화면만 옴")

    def _recv(self):
        import cv2
        import numpy as np

        try:
            data = self.ws.recv()
        except Exception as e:
            raise RuntimeError(f"Studio 스트림 끊김: {e}") from e
        if isinstance(data, str):
            raise RuntimeError(f"Studio 응답: {data[:200]}")
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Studio 프레임 디코딩 실패")
        return frame

    def read(self):
        return fit(self._recv())

    def close(self) -> None:
        self.ws.close()


def open_stream(source: str = "auto", dev: str = FRONT_CAM, warmup: int = 8):
    """source: auto(비어 있으면 장치, Studio 가 쓰는 중이면 Studio 스트림) | device | studio."""
    if source == "studio":
        return StudioStream()
    pids = busy_pids(dev)
    t_end = time.monotonic() + 1.0
    while pids and held_by_chrome_probe(pids) and time.monotonic() < t_end:
        time.sleep(0.2)
        pids = busy_pids(dev)
    if pids == [] or (pids and held_by_chrome_probe(pids)):
        # 크롬 장치 목록 도우미(마이크를 켤 때 카메라를 잠깐 연다)는 영상을 안 받으므로 막지 않는다(10-23 16:00, ensure_free 와 같은 규칙)
        return DeviceStream(dev, warmup, check=False)
    if pids is None:
        raise CameraBusy("카메라 사용 여부를 확인할 수 없음(sudo -n 실패) — 열지 않음")
    if source == "auto" and held_by_studio(pids):
        return StudioStream()
    raise CameraBusy(f"카메라 사용 중(PID {pids}: {cmdline(pids[0])[:80]})")


def grab_frame(source: str = "auto", dev: str = FRONT_CAM, warmup: int = 8):
    """한 장 찍고 바로 닫는다. (frame, 실제 경로 'device'|'studio') 를 돌려준다."""
    stream = open_stream(source, dev, warmup)
    try:
        stream.read()  # 큐에 남은 이전 장 버리기
        return stream.read(), stream.source
    finally:
        stream.close()


# ---- V4L2 컨트롤 (linux/videodev2.h) ----

def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (ord("V") << 8) | nr


_QUERYCTRL = struct.Struct("<II32siiiiI2I")   # struct v4l2_queryctrl (68 bytes)
_CONTROL = struct.Struct("<Ii")              # struct v4l2_control
_QUERYMENU = struct.Struct("<II32sI")        # struct v4l2_querymenu (44 bytes, name 쪽)
VIDIOC_QUERYCTRL = _iowr(36, _QUERYCTRL.size)
VIDIOC_G_CTRL = _iowr(27, _CONTROL.size)
VIDIOC_S_CTRL = _iowr(28, _CONTROL.size)
VIDIOC_QUERYMENU = _iowr(37, _QUERYMENU.size)
NEXT_CTRL = 0x80000000
FLAG_DISABLED, FLAG_INACTIVE = 0x0001, 0x0010
TYPES = {1: "int", 2: "bool", 3: "menu", 4: "button", 5: "int64", 6: "class"}


def _var_name(name: str) -> str:
    """v4l2-ctl 과 같은 이름 규칙: 'White Balance, Automatic' → white_balance_automatic."""
    out = ""
    for ch in name:
        if ch.isalnum():
            out += ch.lower()
        elif out and not out.endswith("_"):
            out += "_"
    return out.rstrip("_")


def list_controls(dev: str = FRONT_CAM) -> list[dict]:
    ensure_free(dev)
    fd = os.open(resolve(dev), os.O_RDWR | os.O_NONBLOCK)
    ctrls = []
    try:
        cid = NEXT_CTRL
        while True:
            buf = bytearray(_QUERYCTRL.pack(cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
            try:
                fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buf)
            except OSError:
                break
            qid, typ, name, mn, mx, step, default, flags, _, _ = _QUERYCTRL.unpack(buf)
            cid = qid | NEXT_CTRL
            if flags & FLAG_DISABLED or typ == 6:
                continue
            c = {"name": _var_name(name.split(b"\0")[0].decode()), "id": qid, "type": TYPES.get(typ, typ),
                 "min": mn, "max": mx, "step": step, "default": default,
                 "inactive": bool(flags & FLAG_INACTIVE)}
            v = bytearray(_CONTROL.pack(qid, 0))
            try:
                fcntl.ioctl(fd, VIDIOC_G_CTRL, v)
                c["value"] = _CONTROL.unpack(v)[1]
            except OSError:
                c["value"] = None
            if typ == 3:
                c["menu"] = {}
                for i in range(mn, mx + 1):
                    m = bytearray(_QUERYMENU.pack(qid, i, b"", 0))
                    try:
                        fcntl.ioctl(fd, VIDIOC_QUERYMENU, m)
                        c["menu"][i] = _QUERYMENU.unpack(m)[2].split(b"\0")[0].decode()
                    except OSError:
                        pass
            ctrls.append(c)
    finally:
        os.close(fd)
    return ctrls


def set_controls(values: dict[str, int], dev: str = FRONT_CAM) -> None:
    """적힌 순서대로 설정한다(auto_exposure=1 을 먼저 해야 exposure_time_absolute 가 먹는다)."""
    ids = {c["name"]: c["id"] for c in list_controls(dev)}
    fd = os.open(resolve(dev), os.O_RDWR | os.O_NONBLOCK)
    try:
        for name, value in values.items():
            if name not in ids:
                raise KeyError(f"컨트롤 없음: {name} (python vision/camera.py --list)")
            fcntl.ioctl(fd, VIDIOC_S_CTRL, bytearray(_CONTROL.pack(ids[name], int(value))))
    finally:
        os.close(fd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default=FRONT_CAM)
    ap.add_argument("--busy", action="store_true", help="사용 중인 PID 출력")
    ap.add_argument("--list", action="store_true", help="컨트롤 목록·현재값")
    ap.add_argument("--set", nargs="+", metavar="NAME=VALUE")
    ap.add_argument("--snap", metavar="JPG", help="한 장 찍어 저장")
    a = ap.parse_args()
    if a.busy:
        print(busy_pids(a.dev))
    if a.set:
        set_controls({k: int(v) for k, v in (s.split("=", 1) for s in a.set)}, a.dev)
    if a.list or a.set:
        for c in list_controls(a.dev):
            extra = f" menu={c['menu']}" if "menu" in c else f" min={c['min']} max={c['max']} step={c['step']}"
            flag = " (inactive)" if c["inactive"] else ""
            print(f"{c['name']:32s} {c['type']:5s} value={c['value']} default={c['default']}{extra}{flag}")
    if a.snap:
        import cv2

        frame, src = grab_frame(dev=a.dev)
        cv2.imwrite(a.snap, frame)
        print(a.snap, src)
    if not (a.busy or a.list or a.set or a.snap):
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
