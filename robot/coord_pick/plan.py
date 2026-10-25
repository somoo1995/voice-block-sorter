"""블록 위치(위 카메라 픽셀) → 집기·놓기 순서 → 경유점 → IK → 관절 궤적(정규화, 30Hz). 로봇은 안 건드린다.

계획 두 가지:
- plan_digit: 흩은 블록으로 숫자 모양 만들기(10-21 밤).
- plan_sort:  투입 구역의 분홍·초록(·불량) 블록을 제 그릇에 넣기(10-22 r14).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robot.coord_pick import digits
from robot.coord_pick.geometry import Frames, apply_h, fit_homography
from robot.coord_pick.kinematics import ARM, JointMap, SO101Kinematics, finger_points_mm, jaw_gap_mm

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
EXAMPLE_CONFIG_PATH = HERE / "config_example.json"  # 10-21 밤 예시값(가짜 작업대) — selftest 의 숫자 배치 시험용
ZONES_JSON = HERE.parent.parent / "vision" / "zones.json"
CAMERA_SIZE = (640, 480)  # vision/camera.py 가 주는 사진 크기 = 호모그래피 픽셀 좌표
OK_CLASSES = ("red_ok", "blue_ok")
COLOR_KO = {"red_ok": "분홍", "blue_ok": "초록", "red_defect": "분홍 불량", "blue_defect": "초록 불량"}
BOWL_KO = {"bowl_red": "분홍 그릇", "bowl_blue": "초록 그릇", "bowl_defect": "불량 그릇"}
BLOCK_R_MM = 21.2  # 3cm 정육면체 대각선의 절반 — 놓인 방향을 모르니 원으로 본다
YAW_ASPECT_MIN = 1.2  # 위 카메라·손목 덩어리 가로세로비가 이 이상이어야 긴 축 방향을 믿는다
PLACED_R_MM = 17.0  # 우리가 놓은 블록은 칸에 맞춰 놓으므로 반 변 + 오차 2mm


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def studio_ranges() -> tuple[dict[str, tuple[int, int]] | None, str]:
    """Studio 의 팔로워 보정 범위(GET 만). 못 읽으면 None."""
    def items(d):
        return d.get("items", d) if isinstance(d, dict) else d

    try:
        from robot import run_policy as rp
        cfg = rp.load_config()
        for p in items(rp._get(cfg, "/api/projects", timeout=3)):
            for r in items(rp._get(cfg, f"/api/projects/{p['id']}/robots", timeout=3)):
                if r.get("type") == "SO101_Follower" and r["payload"].get("calibration"):
                    cal = r["payload"]["calibration"]
                    return {k: (cal[k]["range_min"], cal[k]["range_max"]) for k in ARM}, f"Studio {r['name']}"
    except Exception as e:  # noqa: BLE001 — 오프라인이면 기본값
        return None, f"Studio 못 읽음({type(e).__name__})"
    return None, "Studio 에 SO101 팔로워 보정 없음"


@dataclass
class Setup:
    cfg: dict
    kin: SO101Kinematics
    jm: JointMap
    frames: Frames
    H: np.ndarray            # 픽셀 → 작업대 mm
    range_src: str
    tcp: np.ndarray = field(default_factory=lambda: np.zeros(3))   # 집게 가운데(끝 좌표 m)
    tilt: dict = field(default_factory=dict)                         # 기울기 한도(rad)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, cfg: dict | None = None, *, use_studio: bool = True) -> Setup:
        cfg = cfg or load_config()
        j = cfg["joint"]
        lim = {k: tuple(v) for k, v in j.get("limits_deg", {}).items() if k in ARM}
        kin = SO101Kinematics(limits_deg=lim)
        ranges, src = studio_ranges() if use_studio else (None, "--offline")
        if ranges is None:
            ranges = {k: tuple(j["range_default"]) for k in ARM}
            src += f" → 기본 {j['range_default']}"
        jm = JointMap(ranges, {k: int(j["sign"][k]) for k in ARM}, {k: float(j["offset_deg"][k]) for k in ARM})
        t = cfg["table"]
        b = t["base"]
        H = fit_homography(t["markers_px"], t["markers_mm"])
        td = cfg.get("tilt_deg", {})
        tilt = {"grasp": math.radians(td.get("grasp_max", 0)), "release": math.radians(td.get("release_max", 0)),
                "step": math.radians(td.get("step", 5)), "jaw_incline": math.radians(td.get("jaw_incline_max", 90)),
                "margin_deg": td.get("margin_min", 0)}
        s = cls(cfg, kin, jm, Frames(b["x"], b["y"], b["yaw_deg"]), H, src,
                np.array(cfg.get("tcp_mm", [0, 0, 0]), float) / 1000, tilt)
        if not cfg.get("calibrated"):
            s.warnings.append("config calibrated=false — 보정 확정 전(실제 송신은 공중 리허설만, COORD_PICK.md 8절)")
        return s

    # 좌표 변환 -------------------------------------------------------------
    def px_to_w(self, px) -> np.ndarray:
        return apply_h(self.H, np.atleast_2d(px))

    def w_to_px(self, w) -> np.ndarray:
        return apply_h(np.linalg.inv(self.H), np.atleast_2d(w))

    def w_to_r3(self, w_xy, z_mm: float) -> np.ndarray:
        r = self.frames.w2r(w_xy)
        return np.array([r[0], r[1], z_mm])

    def tcp_pos(self, q: np.ndarray) -> np.ndarray:
        """집게 가운데(tcp)의 로봇 좌표(mm)."""
        T = self.kin.fk(q)
        return (T[:3, 3] + T[:3, :3] @ self.tcp) * 1000

    def finger_low(self, q: np.ndarray, grip_n: float) -> float:
        """집게 끝(고정·움직이는 집게 모서리) 가운데 가장 낮은 점의 높이(mm)."""
        T = self.kin.fk(q)
        pts = finger_points_mm(grip_n) / 1000 @ T[:3, :3].T + T[:3, 3]
        return float(pts[:, 2].min() * 1000)

    def tangent_yaw(self, p_r_mm) -> float:
        """그 점에서 팔 평면에 수직인 수평 방향(로봇 좌표 rad) — 기울인 집게의 두 끝이 같은 높이가 되는 집게 방향."""
        o = self.kin.pan_origin * 1000
        return math.atan2(p_r_mm[1] - o[1], p_r_mm[0] - o[0]) + math.pi / 2

    # IK ---------------------------------------------------------------------
    def solve(self, p_r_mm: np.ndarray, yaw_r: float | None, q_prev: np.ndarray, *, strict: bool = False,
              tilt: float | None = 0.0, tilt_max: float | None = None, grip_n: float | None = None,
              jaw_limit: bool = True, near: float | None = None, only_prev: bool = False,
              finger_check: bool = True) -> tuple[np.ndarray | None, dict]:
        """tcp 를 p_r_mm(로봇 좌표 mm)에, 집게 방향 yaw(로봇 좌표). 안 되고 strict 가 아니면 +90° 로 해 본다(정육면체라 같음).

        tilt: 집게 기울기(rad, 0 = 위에서 수직). None 이면 0 부터 tilt_max 까지 step 씩 늘려 관절 한계 여유가
        tilt_deg.margin_min 이상인 첫 기울기(없으면 여유가 가장 큰 것).
        집게는 180° 돌려도 같아서 IK 는 yaw 를 180° 단위로 본다. info["yaw_used"]·info["tilt_used"] = 실제로 맞춘 값.
        jaw_limit: 기울인 채 집게축이 수평에서 기운 각 한도를 볼지(집는 자세만 — 옮기는 중에는 상관없다).
        near: tilt=None 일 때 이 기울기에 가까운 것부터 처음 되는 것(이동 중 — 기울기가 오락가락하지 않게).
        only_prev: 시작값을 q_prev 만 — 손목 롤이 같은 갈래로 이어지게(전형 자세 시작값은 롤 0 쪽 갈래로 갈 수 있다).
        finger_check=False: 집게 끝 바닥 검사를 빼고 finger_low_mm 만 알려 준다(column 이 높이를 올릴 때).
        안 되면 (None, {"ok": False, "why": 이유}).
        """
        hz = self.cfg["heights_mm"]
        floor = hz.get("finger_min", 0)  # 계획 바닥(실행기 safety.floor_mm 는 따로 다시 본다)
        grip_n = self.cfg["gripper"]["open"] if grip_n is None else grip_n
        if tilt is None:
            tmax = self.tilt["grasp"] if tilt_max is None else tilt_max
            tilts = list(np.arange(0.0, tmax + 1e-9, self.tilt["step"])) or [0.0]
            if near is not None:
                tilts = sorted(tilts, key=lambda v: abs(v - near))
        else:
            tilts = [tilt]
        az = math.degrees(math.atan2(p_r_mm[1], p_r_mm[0]))
        seeds = (np.radians([az, 0, 30, 65, 0]), np.radians([az, 55, -60, 55, 0]))  # 가까운·먼 전형 자세
        why, best = "IK 안 풀림", None
        for t in tilts:
            for cand in ((yaw_r,) if strict or yaw_r is None else (yaw_r, yaw_r + math.pi / 2)):
                info = {"ok": False}
                for q0 in ((q_prev,) if only_prev else (q_prev, *seeds)):
                    q, info = self.kin.ik(p_r_mm / 1000, yaw=cand, q0=q0, tilt=t, tcp=self.tcp)
                    if info["ok"]:
                        break
                if not info["ok"]:
                    why = (f"팔이 안 닿음(위치 오차 {info['pos_err_mm']:.0f}mm, {info['limit_joint']} 한계 여유 "
                           f"{info['limit_margin_deg']:.0f}°)" if info["pos_err_mm"] >= 2 else
                           f"기울기 {math.degrees(t):.0f}° 를 못 맞춤(오차 {info['tilt_err_deg']:.0f}°)")
                    continue
                if jaw_limit and abs(math.radians(info["jaw_incline_deg"])) > self.tilt["jaw_incline"] + 1e-9:
                    why = f"집게 두 끝 높이 차가 큼(집게축 {info['jaw_incline_deg']:.0f}°)"
                    continue
                pts = self.kin.link_points(q)
                info["min_link_z_mm"] = round(float(pts[:-1, 2].min() * 1000), 1)
                if info["min_link_z_mm"] < hz["min_link"]:
                    why = f"관절이 탁자에 가까움({info['min_link_z_mm']:.0f}mm)"
                    continue
                info["finger_low_mm"] = round(self.finger_low(q, grip_n), 1)
                if finger_check and info["finger_low_mm"] < floor:
                    why = f"집게 끝이 탁자 아래({info['finger_low_mm']:.0f}mm)"
                    continue
                found = (q, {**info, "yaw_used": cand, "tilt_used": float(t)})
                if len(tilts) == 1 or near is not None or info["limit_margin_deg"] >= self.tilt["margin_deg"]:
                    return found
                if best is None or info["limit_margin_deg"] > best[1]["limit_margin_deg"]:
                    best = found
        return best if best is not None else (None, {"ok": False, "why": why})

    def column(self, w_xy, yaw_w: float, heights: tuple[float, ...], q_prev: np.ndarray, *, tilt_max: float,
               grip_n: float | None = None, yaws: tuple[float, ...] | None = None,
               finger_target: float | None = None, close_n: float | None = None) -> dict | None:
        """한 자리에서 같은 기울기·같은 집게 방향으로 heights 높이를 모두 지날 수 있나(수직으로 내려갔다 올라오기).

        기울기는 0 부터 step 씩 늘려 관절 한계 여유가 tilt_deg.margin_min 이상인 첫 값(먼 점일수록 기운다, 없으면
        여유가 가장 큰 것). 집게 방향 후보 = yaws(작업대 rad, 기본 yaw·+90°). finger_target: 기울어 집게 끝 모서리가
        이보다 낮아지면 그 높이를 그만큼 올린다(heights_mm.grasp_finger). close_n 을 주면 집게를 그 값까지 닫은 모양
        (기울어 있으면 반쯤 닫힌 움직이는 집게 끝이 가장 낮다 — 10-22 시연 배치 25° 에서 6 → 0.8mm)도 같이 본다. 반환 {tilt, yaw_w, qs, infos, zs, margin}
        또는 None(이유는 self.last_why).
        """
        tilts = list(np.arange(0.0, tilt_max + 1e-9, self.tilt["step"])) or [0.0]
        yaws = yaws or (yaw_w, yaw_w + math.pi / 2)
        self.last_why = "IK 안 풀림"
        best = None
        for t in tilts:
            for yw in yaws:
                q, qs, infos, zs = q_prev, [], [], []
                for z in heights:
                    for _ in range(3):
                        qn, info = self.solve(self.w_to_r3(w_xy, z), yw - self.frames.yaw, q, strict=True,
                                              tilt=float(t), grip_n=grip_n, finger_check=finger_target is None)
                        if qn is not None and close_n is not None:
                            # 닫는 도중(벌림 → 닫힘)의 모든 그리퍼 값 × 롤 두 갈래(집게는 180° 돌려도 같아 궤적이 다른
                            # 갈래로 올 수 있다 — 그러면 움직이는 집게가 반대쪽)
                            g0 = self.cfg["gripper"]["open"] if grip_n is None else grip_n
                            branches = [qn]
                            lim = self.kin.limits()[4]
                            for d in (math.pi, -math.pi):
                                if lim[0] <= qn[4] + d <= lim[1]:
                                    q2, i2 = self.kin.ik(self.w_to_r3(w_xy, z) / 1000, yaw=yw - self.frames.yaw,
                                                         q0=np.r_[qn[:4], qn[4] + d], tilt=float(t), tcp=self.tcp)
                                    if i2["ok"] and abs(q2[4] - qn[4]) > 1.0:
                                        branches.append(q2)
                            info["finger_low_mm"] = round(min([info["finger_low_mm"]] + [
                                self.finger_low(qb, v) for qb in branches
                                for v in np.linspace(min(g0, close_n), max(g0, close_n), 7)]), 1)
                        if qn is None or finger_target is None or info["finger_low_mm"] >= finger_target - 0.5:
                            break
                        z += finger_target - info["finger_low_mm"]
                    if qn is not None and info["finger_low_mm"] < self.cfg["heights_mm"].get("finger_min", 0):
                        qn, info = None, {"why": f"집게 끝이 탁자 아래({info['finger_low_mm']:.0f}mm)"}
                    if qn is None:
                        self.last_why = f"{info['why']}(높이 {z:g}mm)"
                        break
                    qs.append(qn)
                    infos.append(info)
                    zs.append(round(float(z), 1))
                    q = qn
                if len(qs) == len(heights):
                    margin = min(i["limit_margin_deg"] for i in infos)
                    col = {"tilt": float(t), "yaw_w": yw, "qs": qs, "infos": infos, "zs": zs, "margin": margin}
                    if margin >= self.tilt["margin_deg"]:
                        return col
                    if best is None or margin > best["margin"]:
                        best = col
        return best

    def q_to_n(self, q: np.ndarray, gripper_n: float) -> np.ndarray:
        return np.r_[self.jm.to_norm(q), gripper_n]

    def n_to_q(self, n: np.ndarray) -> np.ndarray:
        return self.jm.to_rad(np.asarray(n)[:5])


# 장면 --------------------------------------------------------------------------
def blocks_from_detection(det: dict) -> list[dict]:
    """detect_objects()·detect_hsv 결과 → [{cls, px, conf, zone}]. 계약 objects 가 있으면 그것, 없으면 detections 박스 가운데."""
    out = []
    if det.get("objects"):
        for o in det["objects"]:
            out.append({"cls": o["cls"], "px": [float(v) for v in o["px"]], "conf": o.get("conf"), "zone": o.get("zone"),
                        **{k: o[k] for k in ("yaw_w", "aspect") if k in o}})
    else:
        for o in det.get("detections", []):
            x1, y1, x2, y2 = o["box"]
            out.append({"cls": o.get("key") or o["cls"], "px": [(x1 + x2) / 2, (y1 + y2) / 2], "conf": o.get("conf"),
                        "zone": o.get("zone")})
    return [b for b in out if b["cls"] != "hand"]


BOWL_ZONES = ("bowl_red", "bowl_blue", "bowl_defect")
# 그릇 → 투입 구역 되돌리기(10-23, config unsort 로 바꿈): floor_mm = 종이 접시 바닥 높이(집는 높이에 더함), overlap_mm = 이보다
# 가까운 블록끼리는 겹치거나 붙은 것으로 보고 안 꺼냄, edge_mm = 그릇 다각형 가장자리에서 블록 가운데가 이보다 가까우면 테두리에
# 걸친 것으로 보고 안 꺼냄, cell_gap_mm = 칸 가운데에서 이 안에 블록이 있으면 찬 칸, place_yaw_deg = 놓을 때 집게축(작업대 °,
# 90 = 블록 긴 축 0° — 테이프 가로와 나란히)
# max_grasp_mm = 접시 바닥에서 집는 tcp 높이 상한(먼 그릇은 기울여야 해 집게 끝이 탁자에 닿지 않게 높이를 올리는데, 16mm 두께
# 블록은 tcp 가 바닥 + 16 을 넘으면 집게 끝이 블록 옆까지 못 내려가 헛집는다)
UNSORT_TRAVEL_EXTRA_MM = 45.0  # 되돌리기 운반 길: 집게 손가락·손목이 tcp 에서 옆으로 닿는 만큼 금지 구역 여유를 더 둔다(불량 그릇
                               # → 칸 길이 손 자리 가장자리를 지나 손가락이 60mm 높이로 걸렸다 — 10-23 dry-run)
UNSORT_DEFAULTS = {"floor_mm": 3.0, "overlap_mm": 45.0, "edge_mm": 15.0, "cell_gap_mm": 60.0, "place_yaw_deg": 90.0,
                   "max_grasp_mm": 16.0}


def input_cells(s: Setup) -> list[tuple[int, np.ndarray]]:
    """투입 구역(config table.input_area_mm) 3×3 칸 가운데(작업대 mm). 번호 = 위 카메라 화면 기준(1 왼쪽 위 · 3 오른쪽 위 ·
    7 왼쪽 아래 = 로봇 쪽, robot/grasp_layout.csv 와 같음). 화면 왼쪽 = 작업대 x 작은 쪽, 화면 위 = y 큰 쪽."""
    (x0, y0), (x1, y1) = s.cfg["table"]["input_area_mm"]
    return [(r * 3 + c + 1, np.array([x0 + (x1 - x0) * (2 * c + 1) / 6, y1 - (y1 - y0) * (2 * r + 1) / 6]))
            for r in range(3) for c in range(3)]


def free_cells(s: Setup, blocks: list[dict], gap_mm: float, zw: dict | None = None) -> list[tuple[int, np.ndarray]]:
    """빈 칸(가운데에서 gap_mm 안에 투입 구역 블록이 없음) — 로봇에 가까운 칸부터."""
    zw = zones_w(s) if zw is None else zw
    occ = [np.asarray(b["w"] if "w" in b else s.px_to_w(b["px"])[0], float) for b in blocks
           if in_input(s, {**b, "w": b["w"] if "w" in b else s.px_to_w(b["px"])[0]}, zw)]
    free = [(n, c) for n, c in input_cells(s) if all(float(np.linalg.norm(c - o)) >= gap_mm for o in occ)]
    return sorted(free, key=lambda nc: float(np.linalg.norm(s.frames.w2r(nc[1]))))


def select_unsort(s: Setup, blocks: list[dict], classes: tuple[str, ...], zw: dict | None = None) -> list[dict]:
    """그릇 → 투입 구역 되돌리기 후보: 블록마다 zone(검출 zone, 없으면 그릇 다각형 안인지)·w 를 달고, 그릇 안 classes 블록 중
    못 꺼낼 것에 skip(이유)을 단다 — 다른 블록과 overlap_mm 안(겹치거나 붙음), 검출 박스가 다른 블록 중앙의 1.8배 넘음(붙은
    덩어리), 그릇 가장자리 edge_mm 안(테두리에 걸침)."""
    zw = zones_w(s) if zw is None else zw
    u = {**UNSORT_DEFAULTS, **s.cfg.get("unsort", {})}
    out = []
    for b in blocks:
        b = dict(b, w=s.px_to_w(b["px"])[0])
        if b.get("zone") not in (*BOWL_ZONES, "input"):
            b["zone"] = next((k for k in BOWL_ZONES if k in zw and _inside(b["w"], zw[k])), b.get("zone"))
        out.append(b)
    areas = [b["box_area"] for b in out if b.get("box_area")]
    big = 1.8 * float(np.median(areas)) if len(areas) >= 2 else None
    for b in out:
        if b["cls"] not in classes or b.get("zone") not in BOWL_ZONES:
            continue
        near = [float(np.linalg.norm(b["w"] - o["w"])) for o in out if o is not b]
        if near and min(near) < u["overlap_mm"]:
            b["skip"] = f"겹쳐 있어 못 꺼냄(옆 블록과 {min(near):.0f}mm)"
        elif big and b.get("box_area", 0) > big:
            b["skip"] = "겹쳐 있어 못 꺼냄(붙은 덩어리 — 검출 박스가 다른 블록의 1.8배 넘음)"
        elif b["zone"] in zw and _edge_dist(b["w"], zw[b["zone"]]) < u["edge_mm"]:
            b["skip"] = f"그릇 가장자리에 걸쳐 있어 못 꺼냄(가장자리 {_edge_dist(b['w'], zw[b['zone']]):.0f}mm)"
    return out


def demo_unsort_scene(s: Setup) -> list[dict]:
    """되돌리기 시험 장면: 그릇마다 1~2개(분홍 그릇 2·초록 그릇 1·불량 그릇 2) + 투입 구역에 1개(그 칸은 찬 칸)."""
    zw = zones_w(s)
    bowls = bowl_targets(s, zw)
    out = []
    # 분홍 그릇 둘은 35mm 떨어져 겹친 것으로 남아야 하고, 불량 그릇 둘(53mm)은 꺼낸다
    for name, cls, off in (("bowl_red", "red_ok", (0, 0)), ("bowl_red", "red_ok", (-24, 26)), ("bowl_blue", "blue_ok", (0, 0)),
                           ("bowl_defect", "red_defect", (-22, 0)), ("bowl_defect", "blue_defect", (30, 10))):
        c = bowls[name] + np.array(off, float)
        out.append({"cls": cls, "px": s.w_to_px(c)[0].round(1).tolist(), "conf": 0.9, "zone": name})
    n, c = input_cells(s)[4]
    out.append({"cls": "blue_ok", "px": s.w_to_px(c + np.array([5.0, -4.0]))[0].round(1).tolist(), "conf": 0.9,
                "zone": "input"})
    return out


WHICH = ("left", "right", "middle", "top", "bottom", "near", "far")
WHICH_KO = {"left": "왼쪽(위 카메라 u 작은 것부터)", "right": "오른쪽(u 큰 것부터)", "middle": "가운데(u 순위 가운데부터)",
            "top": "위(위 카메라 v 작은 것부터 = 로봇에서 먼 쪽)", "bottom": "아래(v 큰 것부터 = 로봇 쪽)",
            "near": "로봇에 가까운 것부터(로봇 베이스까지 작업대 거리)", "far": "로봇에서 먼 것부터"}
NEXT_TO = {"red": ("red_ok",), "blue": ("blue_ok",), "red_defect": ("red_defect",), "blue_defect": ("blue_defect",)}


def which_error(which) -> str | None:
    """which 값 검사 — 모르는 값이면 이유(엉뚱한 블록을 고르지 않게 거부한다), 맞으면 None."""
    if isinstance(which, str):
        return None if which in WHICH else f"which 는 {', '.join(WHICH)} 또는 {{'next_to': 색}} (받은 값 {which!r})"
    if isinstance(which, dict) and set(which) == {"next_to"} and which["next_to"] in NEXT_TO:
        return None
    return f"which 관계는 {{'next_to': {'|'.join(NEXT_TO)}}} 만 (받은 값 {which!r})"


def select_which(blocks: list[dict], classes: tuple[str, ...], which, n: int | None = 1, *,
                 flip: bool = False, dist_fn=None, w_fn=None) -> tuple[list[dict], list[dict], dict]:
    """위치·관계로 고르기(10-23): 투입 구역(zone input)의 classes 후보를 줄 세워 앞에서 n 개(None = 모두).
    위치(위 카메라 사진 박스 가운데 px): left = u 작은 것부터 · right = u 큰 것부터 · middle = u 순위 가운데부터 ·
    top = v 작은 것부터(로봇에서 먼 쪽) · bottom = v 큰 것부터(로봇 쪽) · near·far = 로봇 베이스까지 작업대 거리(dist_fn).
    관계: {"next_to": 색} = 그 색(투입 구역) 블록에 가장 가까운 것부터(작업대 mm, w_fn — 자기 자신은 뺀다).
    flip(config which_flip) = 사용자 시점이 카메라와 반대일 때 left·right 만 맞바꾼다(여기 한 곳).
    고른 블록에 which_rank(0 부터 = 집는 순서)를 달고, 안 고른 같은 색 후보는 cls "obstacle"(원래 색 = which_cls)로 바꿔
    이웃으로만 본다. 모르는 which 는 ValueError. 반환 (바꾼 blocks, 고른 것, 기준 {rule, ref_px?})."""
    err = which_error(which)
    if err:
        raise ValueError(err)
    cand = [b for b in blocks if b["cls"] in classes and b.get("zone", "input") == "input"]
    info: dict = {}
    if isinstance(which, dict):
        refs = [b for b in blocks if b["cls"] in NEXT_TO[which["next_to"]] and b.get("zone", "input") == "input"]
        info["rule"] = f"{which['next_to']} 블록 옆(가장 가까운 것부터)"
        info["ref_px"] = [b["px"] for b in refs]

        def d_ref(b):
            others = [r for r in refs if r is not b]
            return min((float(np.linalg.norm(w_fn(b) - w_fn(r))) for r in others), default=math.inf)
        order = sorted((b for b in cand if d_ref(b) < math.inf), key=d_ref)
        info["ref_dist_mm"] = [round(d_ref(b), 1) for b in order]
    else:
        side = which if not flip or which not in ("left", "right") else {"left": "right", "right": "left"}[which]
        info["rule"] = WHICH_KO[side] + (" — which_flip 으로 뒤집음" if side != which else "")
        if side in ("left", "right", "middle"):
            order = sorted(cand, key=lambda b: float(b["px"][0]), reverse=side == "right")
            if side == "middle":
                mid = (len(order) - 1) / 2
                order = [b for _, b in sorted(enumerate(order), key=lambda ib: (abs(ib[0] - mid), ib[0]))]
        elif side in ("top", "bottom"):
            order = sorted(cand, key=lambda b: float(b["px"][1]), reverse=side == "bottom")
        else:
            if dist_fn is None:
                raise ValueError("near·far 는 로봇까지 거리(dist_fn)가 필요")
            order = sorted(cand, key=dist_fn, reverse=side == "far")
    chosen = order if n is None else order[:max(0, int(n))]
    rank = {id(b): i for i, b in enumerate(chosen)}
    out = []
    for b in blocks:
        if id(b) in rank:
            out.append(dict(b, which_rank=rank[id(b)]))
        elif b in cand:
            out.append(dict(b, cls="obstacle", which_cls=b["cls"]))
        else:
            out.append(b)
    return out, sorted((o for o in out if "which_rank" in o), key=lambda o: o["which_rank"]), info


def zones_path(cfg: dict) -> Path | None:
    """구역 파일: config zones_json(null = 안 씀 — 예시 작업대), 없으면 vision/zones.json."""
    if "zones_json" not in cfg:
        return ZONES_JSON
    v = cfg["zones_json"]
    return None if v is None else (HERE.parent.parent / v if not Path(v).is_absolute() else Path(v))


def zones_w(s: Setup, path: Path | None = None) -> dict[str, np.ndarray]:
    """vision/zones.json 의 구역 다각형(픽셀) → 작업대 mm. 없으면 {}."""
    path = path or zones_path(s.cfg)
    if path is None or not Path(path).exists():
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    w, h = d.get("image_size", CAMERA_SIZE)
    return {k: s.px_to_w(np.array(v, float) * (CAMERA_SIZE[0] / w, CAMERA_SIZE[1] / h)) for k, v in d["zones"].items()}


def _inside(p, poly) -> bool:
    x, y, inside = p[0], p[1], False
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def in_input(s: Setup, b: dict, zw: dict) -> bool:
    """투입 구역 안의 블록인가: 검출 결과의 zone, 없으면 zones.json input(작업대 mm), 없으면 config input_area_mm."""
    if b.get("zone"):
        return b["zone"] == "input"
    if "input" in zw:
        return _inside(b["w"], zw["input"])
    (x0, y0), (x1, y1) = s.cfg["table"]["input_area_mm"]
    return x0 <= b["w"][0] <= x1 and y0 <= b["w"][1] <= y1


def demo_scene(s: Setup, n: int, seed: int = 7) -> list[dict]:
    """입력 구역(작업대 mm)에 정상 블록 n 개 + 불량 2개를 흩는다: 55mm 격자 칸을 섞어 고르고 ±4mm 흔든 뒤,
    위 카메라 픽셀(±1.5px 잡음)로 만든다. 팔이 안 닿는 자리는 뺀다. 자리가 모자라면 불량부터 뺀다."""
    rnd = random.Random(seed)
    (x0, y0), (x1, y1) = s.cfg["table"]["input_area_mm"]
    hz = s.cfg["heights_mm"]
    seed_q = s.n_to_q(np.array(s.cfg["home_n"]))

    def reachable(p) -> bool:  # 사람이 입력 구역에 놓는 블록은 팔이 닿는 곳이어야 한다
        return s.column(p, 0.0, (hz["approach"], hz["grasp"]), seed_q, tilt_max=s.tilt["grasp"]) is not None

    grid = [(x, y) for x in np.arange(x0 + 15, x1 - 14, 55) for y in np.arange(y0 + 15, y1 - 14, 55)]
    rnd.shuffle(grid)
    pts = []
    for gx, gy in grid:
        p = (gx + rnd.uniform(-4, 4), gy + rnd.uniform(-4, 4))
        if reachable(p):
            pts.append(p)
        if len(pts) == n + 2:
            break
    pts, defects = pts[:n], pts[n:]
    classes = ["red_ok", "blue_ok"] * n
    blocks = []
    for c, p in list(zip(classes, pts)) + list(zip(("red_defect", "blue_defect"), defects)):
        q = s.w_to_px(np.array([p]))[0]
        blocks.append({"cls": c, "px": [float(q[0] + rnd.uniform(-1.5, 1.5)), float(q[1] + rnd.uniform(-1.5, 1.5))],
                       "conf": 0.9})
    return blocks


def demo_sort_scene(s: Setup) -> list[dict]:
    """시연 배치(블록 4개: 분홍 2·초록 1·불량 1)를 투입 구역 안쪽에 둔 가짜 검출. 서로 77mm 이상 떨어지게(집게 여유)."""
    (x0, y0), (x1, y1) = s.cfg["table"]["input_area_mm"]
    w, h = x1 - x0, y1 - y0
    layout = (("red_ok", 0.15, 0.55), ("blue_ok", 0.45, 0.25), ("red_ok", 0.80, 0.50), ("red_defect", 0.50, 0.75))
    out = []
    for cls, fx, fy in layout:
        px = s.w_to_px(np.array([[x0 + fx * w, y0 + fy * h]]))[0]
        out.append({"cls": cls, "px": [round(float(px[0]), 1), round(float(px[1]), 1)], "conf": 0.9, "zone": "input"})
    return out


# 집게 여유 -----------------------------------------------------------------------
def jaw_clearance(c: np.ndarray, yaw: float, others: list[tuple[np.ndarray, float]], half_gap: float,
                  finger: float, finger_w: float = 20.0) -> float:
    """집게 손가락(c ± u·[half_gap, half_gap+finger])과 이웃 블록(중심, 반지름) 사이 최소 여유(mm). 이웃이 없으면 99."""
    u = np.array([math.cos(yaw), math.sin(yaw)])
    v = np.array([-u[1], u[0]])
    best = 99.0
    for o, r in others:
        d = o - c
        along, perp = abs(float(d @ u)), abs(float(d @ v))
        if perp >= r + finger_w / 2:
            continue
        best = min(best, along - r - (half_gap + finger))
    return best


# 계획 ---------------------------------------------------------------------------
@dataclass
class Step:
    what: str
    n: np.ndarray          # 끝 정규화 6값
    dur_s: float
    p_r: np.ndarray | None = None
    info: dict | None = None
    fast: bool = False     # 공중 구간(블록·그릇에서 떨어진 이동) — 실행기가 safety.air_deg_s·가감속으로 따로 시간을 맞춘다


def pose_hits(s: Setup, q: np.ndarray, grips, zones: dict, below: float, margin: float) -> list[str]:
    """한 자세에서 집게 끝 최저점(grips 의 그리퍼 값마다)·손목이 zones(작업대 mm 다각형) 위 below(mm) 아래에 있는 구역들 —
    실행기 check_workspace 와 같은 기준(여유 margin)."""
    T = s.kin.fk(q)
    pts = [s.kin.link_points(q)[-2] * 1000]
    for gv in grips:
        f = finger_points_mm(gv) @ T[:3, :3].T + T[:3, 3] * 1000
        pts.append(f[int(np.argmin(f[:, 2]))])
    hit = []
    for p in pts:
        if p[2] >= below:
            continue
        pw = s.frames.r2w(p[:2])
        for name, poly in zones.items():
            if name not in hit and (_inside(pw, poly) or _edge_dist(pw, poly) < margin):
                hit.append(name)
    return hit


class _Path:
    """경유점 쌓기(홈 → … → 홈). 직선 구간은 10mm 간격 IK, 기울기·집게 방향은 구간 처음에서 끝까지 고르게 바꾼다."""

    def __init__(self, s: Setup, q0: np.ndarray | None = None, grip: float | None = None, *, yaw: float = 0.0,
                 tilt: float = 0.0):
        """q0·grip 을 주면 그 자세에서 시작(손목 확인처럼 실행을 여러 구간으로 나눌 때), 없으면 홈."""
        self.s, self.cfg = s, s.cfg
        self.home = np.array(s.cfg["home_n"], float)
        self.q = s.n_to_q(self.home) if q0 is None else np.array(q0, float)
        self.grip = self.home[5] if grip is None else float(grip)
        self.yaw = yaw     # 지금 집게 방향(로봇 좌표, 실제로 맞춘 값)
        self.tilt = tilt
        self.steps: list[Step] = [Step("시작(홈 자세)" if q0 is None else "시작", s.q_to_n(self.q, self.grip), 0.0)]  # 첫 목표 = 지금 자세
        self.failures: list[str] = []
        self.flips: list[str] = []

    def line(self, p_from: np.ndarray, p_to: np.ndarray, yaw_to: float, tilt_to: float, what: str, *,
             strict: bool, tilt_max: float | None = None, q_goal: np.ndarray | None = None,
             keep_roll: float | None = None, fast: bool = False) -> bool:
        """끝을 직선으로. 각 구간 시간 = max(거리/직선속도, 관절 변화/관절속도).

        집게 방향은 지금 방향에서 yaw_to(180° 주기)까지, 기울기는 지금 기울기에서 tilt_to 까지 고르게. 이동 구간(strict
        아님)은 180° 다른 방향 후보(손목 롤이 반대로 도는 길)도 끝까지 풀어 보고 관절 한계 여유가 tilt_deg.margin_min 이상인
        것 가운데 롤을 가장 덜 돌리는 길을 고른다. q_goal(다음 구간이 쓰는 자세)이 있으면 끝 롤이 그 갈래에 가까운 길.
        한 걸음(10mm)에 롤이 60° 넘게 튀는 길(IK 가 다른 갈래로 넘어감)은 다른 길이 없을 때만 쓴다.
        도중에 그 기울기로 안 되면 tilt_max 안에서 가까운 기울기를 찾는다. 구간 끝은 항상 yaw_to·tilt_to 를 지킨다.
        keep_roll(rad): 손목 롤을 지금 갈래 그대로(시작값 = 직전 자세만) 두고, 구간 안에서 이만큼 넘게 돌면 실패 —
        손목 카메라로 맞춘 뒤 내려갈 때(tcp 가 롤 축에서 떨어져 있어 롤이 돌면 맞춘 자리가 틀어진다, 10-22 17:29 실물).
        fast: 공중 구간 — 시간을 motion.air_mm_s·air_deg_s 로 잡고 걸음에 표시(실행기가 빠른 한도·가감속으로 맞춤).
        """
        s = self.s
        y0 = self.yaw
        # 롤 변화 ≈ 방향 변화 − 팬 변화(방위 변화) — 그게 가장 작은 쪽이 첫 후보
        o = s.kin.pan_origin * 1000
        daz = math.atan2(p_to[1] - o[1], p_to[0] - o[0]) - math.atan2(p_from[1] - o[1], p_from[0] - o[0])
        daz = (daz + math.pi) % (2 * math.pi) - math.pi
        k0 = round((y0 + daz - yaw_to) / math.pi)
        cands = (k0,) if strict or keep_roll is not None else (k0, k0 + 1, k0 - 1)
        best, why = None, ""
        for k in cands:
            r = self._trace(p_from, p_to, yaw_to + k * math.pi, tilt_to, what, strict, tilt_max, keep_roll, fast)
            if r["fail"]:
                why = why or r["fail"]
                continue
            goal = -abs(float(r["q"][4] - q_goal[4])) if q_goal is not None else 0.0
            key = (not r["jump"], r["margin"] >= s.tilt["margin_deg"], round(goal, 1), -r["roll"])
            if best is None or key > best[0]:
                best = (key, r)
            if key[0] and key[1] and k == k0 and (q_goal is None or goal > -0.5):
                break
        if best is None:
            self.failures.append(why)
            return False
        r = best[1]
        self.steps += r["steps"]
        self.flips += r["flips"]
        self.q, self.yaw, self.tilt = r["q"], r["yaw"], r["tilt"]
        return True

    def _trace(self, p_from, p_to, y1, tilt_to, what, strict, tilt_max, keep_roll=None, fast=False) -> dict:
        s, hm = self.s, self.cfg["motion"]
        mm_s, deg_s = air_speeds(hm) if fast else (hm["cart_mm_s"], hm["max_deg_s"])
        q, y, t = self.q, self.yaw, self.tilt
        y0, t0 = y, t
        dist = float(np.linalg.norm(p_to - p_from))
        k = max(1, int(math.ceil(dist / 10)))
        steps, flips, margin, jump = [], [], 99.0, False
        for i in range(1, k + 1):
            u = i / k
            p = p_from + (p_to - p_from) * u
            want, tw = y0 + (y1 - y0) * u, t0 + (tilt_to - t0) * u
            last = i == k
            qn, info = s.solve(p, want, q, strict=strict or last, tilt=tw, grip_n=self.grip, jaw_limit=False,
                               only_prev=keep_roll is not None)
            if qn is not None and keep_roll is not None and abs(float(qn[4] - self.q[4])) > keep_roll:
                return {"fail": f"{what}: 손목 롤이 {math.degrees(float(qn[4] - self.q[4])):+.0f}° 돌아야 함(롤 고정 구간)"}
            if qn is None and not (strict or last) and tilt_max is not None:
                qn, info = s.solve(p, want, q, tilt=None, tilt_max=tilt_max, grip_n=self.grip, jaw_limit=False,
                                   near=tw)
            if qn is None:
                return {"fail": f"{what}: IK 실패 (로봇 좌표 {np.round(p).tolist()} mm, {info.get('why', '')})"}
            jump = jump or abs(float(qn[4] - q[4])) > math.radians(60)
            if abs(info["yaw_used"] - want) > 1e-9:
                flips.append(what)
            dur = max(dist / k / mm_s, math.degrees(float(np.abs(qn - q).max())) / deg_s)
            q, y, t = qn, info["yaw_used"], info["tilt_used"]
            margin = min(margin, info["limit_margin_deg"])
            steps.append(Step(what if last else what + "…", s.q_to_n(q, self.grip), dur, p, info, fast))
        return {"fail": None, "steps": steps, "flips": flips, "q": q, "yaw": y, "tilt": t, "margin": margin,
                "roll": abs(float(q[4] - self.q[4])), "jump": jump}

    def joint_move(self, q_to: np.ndarray, what: str, *, yaw: float | None = None, tilt: float | None = None,
                   fast: bool = False, links_only: bool = False) -> None:
        """관절 공간 직선(홈 ↔ 작업대 위). 도중 링크 높이를 20점 샘플로 본다. fast = 공중 구간(line 과 같음).
        links_only = 끝(집게) 점은 빼고 관절만 본다(바닥을 누른 집게를 수직으로 뺄 때 — 집게 끝은 실행기가 따로 본다)."""
        s, hm, hz = self.s, self.cfg["motion"], self.cfg["heights_mm"]
        deg_s = air_speeds(hm)[1] if fast else hm["max_deg_s"]
        dur = max(0.5, math.degrees(float(np.abs(q_to - self.q).max())) / deg_s)
        sl = slice(None, -1) if links_only else slice(None)
        low = min(float(s.kin.link_points(self.q + (q_to - self.q) * t / 20)[sl, 2].min() * 1000) for t in range(21))
        if low < hz["min_link"]:
            self.failures.append(f"{what}: 관절 공간 이동 중 링크가 탁자 {low:.0f}mm 까지 내려감")
        self.q = q_to
        if yaw is not None:
            self.yaw = yaw
        if tilt is not None:
            self.tilt = tilt
        self.steps.append(Step(what, s.q_to_n(self.q, self.grip), dur, info={"path_min_z_mm": round(low, 1)}, fast=fast))

    def joint_hits(self, q_to: np.ndarray, zones: dict, below: float, margin: float) -> list[str]:
        """지금 자세 → q_to 관절 공간 직선(20점)에서 금지 구역 위를 낮게 지나는 구역 이름들(pose_hits 기준)."""
        hit: list[str] = []
        for t in range(21):
            for h in pose_hits(self.s, self.q + (q_to - self.q) * t / 20, (self.grip,), zones, below, margin):
                if h not in hit:
                    hit.append(h)
        return hit

    def hold(self, seconds: float, what: str) -> None:
        """그 자세로 seconds 초 멈춤(목표값을 그대로 되풀이)."""
        if seconds > 0:
            self.steps.append(Step(what, self.s.q_to_n(self.q, self.grip), seconds))

    def gripper(self, value: float, what: str, *, settle: bool = False) -> None:
        """그리퍼 값 바꾸기(grip_s 초). settle 이면 그 전에 motion.settle_s 초 그 자세로 멈춘다(팔이 따라와 멈춘 뒤 쥐기·놓기)."""
        hold = self.cfg["motion"].get("settle_s", 0) if settle else 0
        if hold > 0:
            self.steps.append(Step(f"{what} 전 멈춤", self.s.q_to_n(self.q, self.grip), hold))
        self.grip = value
        self.steps.append(Step(what, self.s.q_to_n(self.q, self.grip), self.cfg["motion"]["grip_s"]))

    def finish(self, zones: dict | None = None, below: float = 100.0, margin: float = 20.0,
               z_high: float | None = None, keep_grip: bool = False) -> None:
        """홈으로(관절 공간). zones 를 주면 그 위를 낮게 지나는지 보고, 지나면 먼저 그 자리에서 z_high 로 올라간다.
        keep_grip = 집게를 쉼 자세로 바꾸지 않는다(블록을 쥔 채 홈에 갈 때 — 떨어뜨리지 않게)."""
        home_q = self.s.n_to_q(self.home)
        if zones and z_high is not None and self.joint_hits(home_q, zones, below, margin):
            p = self.s.tcp_pos(self.q)
            if p[2] < z_high:
                self.line(p, np.array([p[0], p[1], z_high]), self.yaw, self.tilt, "홈 가기 전 올라감", strict=False,
                          tilt_max=max(self.s.tilt["grasp"], self.s.tilt["release"]), fast=True)
        self.joint_move(home_q, "홈으로", fast=True)
        if not keep_grip:
            self.gripper(self.home[5], "집게 쉼 자세")

    def checks(self) -> tuple[dict, float]:
        steps = self.steps
        total = sum(st.dur_s for st in steps)
        infos = [st.info for st in steps if st.info and "pos_err_mm" in st.info]
        jm_low = [st.info["path_min_z_mm"] for st in steps if st.info and "path_min_z_mm" in st.info]
        return {
            "실패": len(self.failures),
            "최대 위치 오차 mm": max((x["pos_err_mm"] for x in infos), default=0),
            "최대 기울기 오차 °": max((x["tilt_err_deg"] for x in infos), default=0),
            "최대 기울기 °": round(max((x["tilt_deg"] for x in infos), default=0), 1),
            "관절 한계 최소 여유 °": min((x["limit_margin_deg"] for x in infos), default=99),
            "링크 최저 높이 mm(직선 구간)": min((x["min_link_z_mm"] for x in infos), default=999),
            "링크 최저 높이 mm(홈 이동)": min(jm_low, default=999),
            "집게 끝 최저 높이 mm": min((x["finger_low_mm"] for x in infos), default=999),
            "이동 중 집게 방향 90° 바꿈": len(self.flips),
        }, total

    def tight_warning(self) -> list[str]:
        tight = [st for st in self.steps if st.info and st.info.get("limit_margin_deg", 99) < 3]
        if not tight:
            return []
        joints = sorted({st.info["limit_joint"] for st in tight})
        return [f"관절({', '.join(joints)})이 녹화로 확인된 범위 끝 3° 안에 들어가는 자세 {len(tight)}곳"
                f"(예: {tight[0].what.rstrip('…')}) — 그 블록을 로봇 쪽으로 옮기거나 approach 높이를 낮춘다"]


def plan_digit(s: Setup, digit: str, blocks: list[dict], *, color: str = "any", small: bool = False,
               max_blocks: int | None = None) -> dict:
    """small = 3행 시험 글꼴(digits.SMALL), max_blocks = 앞의 칸 몇 개만(입회 시험: 블록 1개)."""
    cfg = s.cfg
    hz, g = cfg["heights_mm"], cfg["gripper"]
    dg = cfg["digit"]
    warnings = list(s.warnings)
    cells = digits.cells(digit, small)
    if max_blocks is not None:
        cells = cells[:max_blocks]

    # 블록: 픽셀 → 작업대 → 로봇
    for b in blocks:
        b["w"] = s.px_to_w(b["px"])[0]
        b["r"] = s.frames.w2r(b["w"])
    usable = [b for b in blocks if b["cls"] in OK_CLASSES and (color == "any" or b["cls"].startswith(color))]
    needed = len(cells)
    if len(usable) < len(cells):
        warnings.append(f"블록이 모자람: 숫자 {digit} 은 {len(cells)}개, 쓸 수 있는 정상 블록 {len(usable)}개")
        cells = cells[:len(usable)]

    # 칸 중심(작업대 mm): origin = 위·왼쪽 칸, 열 → +x, 행 → −y (작업대 좌표를 yaw 만큼 돌림)
    a = math.radians(dg["yaw_deg"])
    ex, ey = np.array([math.cos(a), math.sin(a)]), np.array([math.sin(a), -math.cos(a)])
    org, (px_, py_) = np.array(dg["origin_mm"], float), dg["pitch_mm"]
    targets = {(r, c): org + c * px_ * ex + r * py_ * ey for r, c in cells}

    # 순서: 남은 칸 중 놓을 때 집게 여유가 가장 큰 칸 먼저(같으면 로봇에서 먼 칸) → 가장 가까운 블록
    placed: list[tuple[np.ndarray, float]] = []
    remaining, pool, order, used = dict(targets), list(usable), [], []
    path = _Path(s)
    grid_yaw_w = a
    seed_q = s.n_to_q(np.array(cfg["home_n"]))
    _col: dict = {}

    def column(w_xy: np.ndarray, yaw_w: float, z_lo: float) -> dict | None:
        """그 자리에서 그 집게 방향으로 approach ↔ z_lo 를 같은 기울기로 오르내릴 수 있나."""
        key = (round(float(w_xy[0]), 1), round(float(w_xy[1]), 1), round(yaw_w, 4), z_lo)
        if key not in _col:
            _col[key] = s.column(w_xy, yaw_w, (hz["approach"], z_lo), seed_q, tilt_max=s.tilt["grasp"], yaws=(yaw_w,))
        return _col[key]

    def best_yaw(w_xy, neigh, half_gap, z_lo):
        """두 집게 방향(격자 방향, +90°) 중 닿는 것 가운데 이웃 여유가 큰 것. 둘 다 안 닿으면 (−999, None)."""
        opts = [(jaw_clearance(w_xy, y, neigh, half_gap, g["finger_mm"]), y)
                for y in (grid_yaw_w, grid_yaw_w + math.pi / 2) if column(w_xy, y, z_lo) is not None]
        return max(opts) if opts else (-999.0, None)

    open_gap, release_gap = g.get("open_gap_mm") or jaw_gap_mm(g["open"]), g.get("release_gap_mm") or jaw_gap_mm(g["release"])
    while remaining:
        def place_score(item):
            cell, w = item
            return (round(best_yaw(w, placed, release_gap / 2, hz["place"])[0], 1),
                    float(np.linalg.norm(s.frames.w2r(w))))
        cell, w = max(remaining.items(), key=place_score)

        def pick_option(b):
            """집을 때 이웃 = 다른 안 집은 블록(불량 포함) + 이미 놓은 블록. (여유, 집게 방향)"""
            neigh = [(o["w"], BLOCK_R_MM) for o in blocks if o is not b and o not in used] + placed
            return best_yaw(b["w"], neigh, open_gap / 2, hz["grasp"])

        # 블록: 집게 여유 3mm 이상인 것 중 칸에서 가장 가까운 것, 없으면 여유가 가장 큰 것
        opts = [(b, *pick_option(b)) for b in pool]
        free = [o for o in opts if o[1] >= 3]
        b, pick_cl, pick_yaw_w = (min(free, key=lambda o: float(np.linalg.norm(o[0]["w"] - w))) if free
                                  else max(opts, key=lambda o: o[1]))
        place_cl, place_yaw_w = best_yaw(w, placed, release_gap / 2, hz["place"])
        pool.remove(b)
        used.append(b)
        del remaining[cell]
        if pick_yaw_w is None or place_yaw_w is None:
            where = "블록" if pick_yaw_w is None else f"칸 {cell}"
            path.failures.append(f"{where} 자리에 닿지 않는다(기울기 {math.degrees(s.tilt['grasp']):.0f}° 까지) — "
                                 f"그 블록을 옮기거나 digit 위치를 바꾼다")
            continue
        order.append({"cell": cell, "target_w": w, "block": b, "pick_yaw_w": pick_yaw_w, "place_yaw_w": place_yaw_w,
                      "pick_tilt": column(b["w"], pick_yaw_w, hz["grasp"])["tilt"],
                      "place_tilt": column(w, place_yaw_w, hz["place"])["tilt"],
                      "pick_clear_mm": round(pick_cl, 1), "place_clear_mm": round(place_cl, 1)})
        placed.append((w, PLACED_R_MM))

    # 궤적: 홈 → 첫 블록 위는 관절 공간으로 한 번(시작 = 녹화 홈 자세), 그다음부터는 끝을 직선으로
    cur_p = None
    tmax = s.tilt["grasp"]
    for i, o in enumerate(order):
        b = o["block"]
        pick_hi, pick_lo = s.w_to_r3(b["w"], hz["approach"]), s.w_to_r3(b["w"], hz["grasp"])
        place_hi, place_lo = s.w_to_r3(o["target_w"], hz["approach"]), s.w_to_r3(o["target_w"], hz["place"])
        py, ly = o["pick_yaw_w"] - s.frames.yaw, o["place_yaw_w"] - s.frames.yaw
        pt, lt = o["pick_tilt"], o["place_tilt"]
        tag = f"[{i + 1}/{len(order)}] {COLOR_KO[b['cls']]}"
        if cur_p is None:
            qa, info = s.solve(pick_hi, py, path.q, strict=True, tilt=pt)
            if qa is None:
                path.failures.append(f"{tag} 첫 블록 위 IK 실패 (로봇 좌표 {np.round(pick_hi).tolist()} mm)")
                break
            path.gripper(g["open"], f"{tag} 집게 벌림")
            path.joint_move(qa, f"{tag} 홈 → 블록 위", yaw=py, tilt=pt)
        elif not path.line(cur_p, pick_hi, py, pt, f"{tag} → 블록 위", strict=False, tilt_max=tmax):
            break
        if not path.line(pick_hi, pick_lo, py, pt, f"{tag} 내려감", strict=True):
            break
        path.gripper(g["close"], f"{tag} 집음", settle=True)
        if not (path.line(pick_lo, pick_hi, py, pt, f"{tag} 들어 올림", strict=True)
                and path.line(pick_hi, place_hi, ly, lt, f"{tag} → 칸 {o['cell']} 위", strict=False, tilt_max=tmax)
                and path.line(place_hi, place_lo, ly, lt, f"{tag} 내려놓음", strict=True)):
            break
        path.gripper(g["release"], f"{tag} 놓음", settle=True)
        if not path.line(place_lo, place_hi, ly, lt, f"{tag} 올라감", strict=True):
            break
        cur_p = place_hi
        if i + 1 < len(order):
            path.gripper(g["open"], f"{tag} 집게 벌림")
    path.finish()

    checks, total = path.checks()
    checks["집을 때 집게 여유 최소 mm"] = min((o["pick_clear_mm"] for o in order), default=99)
    checks["놓을 때 집게 여유 최소 mm"] = min((o["place_clear_mm"] for o in order), default=99)
    if checks["집을 때 집게 여유 최소 mm"] < 3:
        warnings.append("집을 때 집게 손가락이 이웃 블록에 닿을 수 있다(표의 집기 여유 < 3mm) — 그 블록 둘레를 사람이 벌려 놓거나 "
                        "open_gap 을 줄인다(대각선 42mm 보다는 커야 함)")
    if checks["놓을 때 집게 여유 최소 mm"] < 3:
        warnings.append("놓을 때 집게 손가락이 이미 놓은 블록에 닿을 수 있다(표의 놓기 여유 < 3mm) — pitch_mm 를 늘리거나 release_gap 을 줄인다")
    warnings += path.tight_warning()
    return {"digit": str(digit), "pattern": digits.pattern(digit, small), "cells": len(cells), "needed": needed,
            "order": order, "steps": path.steps, "failures": path.failures, "warnings": warnings, "checks": checks,
            "total_s": total, "blocks": blocks}


def bowl_targets(s: Setup, zw: dict | None = None) -> dict[str, np.ndarray]:
    """그릇 가운데(작업대 mm). config bowls.center_mm 에 있으면 그것, 없으면 zones.json 다각형(픽셀) 가운데의 호모그래피."""
    zw = zones_w(s) if zw is None else zw
    out = {}
    for name in ("bowl_red", "bowl_blue", "bowl_defect"):
        fixed = s.cfg.get("bowls", {}).get("center_mm", {}).get(name)
        if fixed:
            out[name] = np.array(fixed, float)
        elif name in zw:
            out[name] = zw[name].mean(axis=0)
    return out


def _edge_dist(p, poly) -> float:
    best = math.inf
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ab, ap = b - a, np.asarray(p, float) - a
        t = float(np.clip(ap @ ab / max(ab @ ab, 1e-9), 0, 1))
        best = min(best, float(np.linalg.norm(ap - t * ab)))
    return best


def drop_points(s: Setup, bowl: str, bowls: dict, zw: dict, side_mm: float = 0.0, edge_mm: float = 25.0):
    """놓는 점 후보(작업대 mm): 그릇 가운데(+ 접선으로 side_mm) → 닿지 않으면 15mm 씩 로봇 쪽으로, 그릇 다각형 가장자리에서
    edge_mm 안쪽까지만. 10-22 녹화에서 사람도 불량 그릇은 로봇 쪽 가장자리에 놓았다."""
    c = np.asarray(bowls[bowl], float)
    r_c = s.frames.w2r(c)
    tan = np.array([-r_c[1], r_c[0]]) / max(1e-9, float(np.linalg.norm(r_c)))
    start = s.frames.r2w(r_c + tan * side_mm)
    toward = s.frames.t - start
    toward = toward / max(1e-9, float(np.linalg.norm(toward)))
    poly = zw.get(bowl)
    for k in range(0, 8):
        p = start + toward * 15 * k
        if k and poly is not None and not (_inside(p, poly) and _edge_dist(p, poly) >= edge_mm):
            break
        yield p


def crosses(a_w, b_w, zones: dict, skip, margin: float) -> list[str]:
    """작업대 직선 a → b 가 지나는 금지 구역들(skip 은 뺀다, 가장자리 여유 margin + 10mm)."""
    a_w, b_w = np.asarray(a_w, float), np.asarray(b_w, float)
    hit = []
    for name, poly in zones.items():
        if name in skip:
            continue
        for u in np.linspace(0, 1, max(2, int(np.linalg.norm(b_w - a_w) / 10) + 1)):
            pt = a_w + (b_w - a_w) * u
            if _inside(pt, poly) or _edge_dist(pt, poly) < margin + 10:
                hit.append(name)
                break
    return hit


def travel(s: Setup, path: "_Path", p_from_w, z_from, p_to_w, z_to, yaw_to, tilt_to, what, *, zones: dict, skip,
           z_high: float, margin: float, tilt_max: float, q_goal=None, fast: bool = False) -> bool:
    """작업대 p_from(높이 z_from) → p_to(높이 z_to) 직선. 금지 구역을 지나면 위로 올라 z_high 로 건너 내려온다.
    fast = 공중 구간(블록을 들고 그릇으로·그릇에서 다음 블록 위로 — 나르는 높이 이상)."""
    a3, b3 = s.w_to_r3(p_from_w, z_from), s.w_to_r3(p_to_w, z_to)
    hit = crosses(p_from_w, p_to_w, zones, skip, margin)
    if not hit:
        return path.line(a3, b3, yaw_to, tilt_to, what, strict=False, tilt_max=tilt_max, q_goal=q_goal, fast=fast)
    _, i1 = s.solve(s.w_to_r3(p_from_w, z_high), path.yaw, path.q, tilt=None, tilt_max=tilt_max, grip_n=path.grip,
                    jaw_limit=False)
    _, i2 = s.solve(s.w_to_r3(p_to_w, z_high), yaw_to, path.q, tilt=None, tilt_max=tilt_max, grip_n=path.grip,
                    jaw_limit=False)
    if not i1.get("ok") or not i2.get("ok"):
        path.failures.append(f"{what}: {', '.join(hit)} 위를 지나야 하는데 {z_high:g}mm 높이에 안 닿음")
        return False
    up3, over3 = s.w_to_r3(p_from_w, z_high), s.w_to_r3(p_to_w, z_high)
    return (path.line(a3, up3, path.yaw, i1["tilt_used"], f"{what}({'·'.join(hit)} 위라 올라감)", strict=False,
                      tilt_max=tilt_max, fast=fast)
            and path.line(up3, over3, yaw_to, i2["tilt_used"], f"{what}(높이 {z_high:g})", strict=False, tilt_max=tilt_max,
                          fast=fast)
            and path.line(over3, b3, yaw_to, tilt_to, what, strict=False, tilt_max=tilt_max, q_goal=q_goal, fast=fast))


def plan_sort(s: Setup, blocks: list[dict], *, classes: tuple[str, ...] = OK_CLASSES, max_blocks: int | None = None,
              lift_mm: float = 0.0, zones: dict | None = None, unsort: bool = False,
              cells: list[tuple[int, np.ndarray]] | None = None) -> dict:
    """투입 구역의 classes 블록을 하나씩 집어 제 그릇(config bowls.targets) 위 release 높이에서 놓는다.

    순서: 집게 여유 3mm 이상인 블록 중 로봇에 가까운 것부터. 한 그릇에 여러 개면 놓는 점을 그릇 가운데에서 spread_mm 씩
    좌우(접선)로 벌린다. lift_mm > 0 = 공중 리허설(모든 높이를 그만큼 올려 블록·그릇에 닿지 않게, 첫 실제 구동용).
    unsort=True(되돌리기, 10-23): 그릇 안 블록(zone = 그릇, select_unsort 가 단 skip 은 뺌)을 집어 cells(빈 칸, free_cells)에
    차례로 놓는다 — 집는 높이 + unsort.floor_mm(접시 바닥), 집는 그 그릇은 그 블록 집기·떠나기에서만 금지 구역에서 뺀다,
    놓을 때 집게축 unsort.place_yaw_deg(블록 긴 축 0°)·높이 heights_mm.place. 칸이 모자라면 남은 블록은 두고 경고.
    """
    cfg = s.cfg
    hz, g, bw = cfg["heights_mm"], cfg["gripper"], cfg["bowls"]
    warnings = list(s.warnings)
    zw = zones_w(s) if zones is None else zones
    bowls = bowl_targets(s, zw)
    for b in blocks:
        b["w"] = s.px_to_w(b["px"])[0]
        b["r"] = s.frames.w2r(b["w"])
    inside = [b for b in blocks if in_input(s, b, zw)]
    u = {**UNSORT_DEFAULTS, **cfg.get("unsort", {})}
    unreachable: list[dict] = []
    if unsort:
        todo = [b for b in blocks if b["cls"] in classes and b.get("zone") in BOWL_ZONES and not b.get("skip")]
        for b in blocks:
            if b["cls"] in classes and b.get("skip"):
                warnings.append(f"{COLOR_KO.get(b['cls'], b['cls'])}(작업대 {np.round(b['w']).tolist()} mm, "
                                f"{BOWL_KO.get(b.get('zone'), b.get('zone'))}): {b['skip']}")
        cells_left = list(cells or [])
    else:
        todo = [b for b in inside if b["cls"] in classes]
        skipped = [b for b in blocks if b["cls"] in classes and b not in inside]
        if skipped:
            warnings.append(f"투입 구역 밖이라 안 옮김: {', '.join(COLOR_KO.get(b['cls'], b['cls']) for b in skipped)} "
                            f"(이미 그릇에 있거나 구역 밖)")
    if max_blocks is not None:
        todo = sorted(todo, key=lambda b: (b.get("which_rank", 0), float(np.linalg.norm(b["r"]))))[:max_blocks]
    z = {k: hz[k] + lift_mm for k in ("grasp", "approach", "carry")}
    z_rel = bw["rim_mm"] + bw["release_above_mm"] + lift_mm
    if z_rel > z["carry"]:
        z["carry"] = z_rel
    seed_q = s.n_to_q(np.array(cfg["home_n"]))
    open_gap = jaw_gap_mm(g["open"])
    tg, tr = s.tilt["grasp"], s.tilt["release"]

    # 집기 자리: 작업대 축 방향 두 개(블록은 보통 작업대 모서리에 나란히 놓인다) 중 닿고 이웃 여유가 큰 것
    picked: list[dict] = []
    order, path = [], _Path(s)
    per_bowl: dict[str, int] = {}
    # 금지 구역(목표가 아닌 그릇·손 자리, 실행기와 같은 다각형) 위를 지나는 이동은 heights_mm.over_zone 높이로 올려서 간다
    sf = {"forbid_margin_mm": 20, "forbid_below_mm": 100, **cfg.get("safety", {})}
    zones_forbid = {k: v for k, v in zw.items() if k in ("bowl_red", "bowl_blue", "bowl_defect", "hand_spot")}
    z_high = hz.get("over_zone", 130) + lift_mm

    cache: dict[int, dict] = {}
    extra = [math.radians(d) for d in (15, -15, 30, -30, 75, 105)]
    extra_off = [math.radians(d) for d in (15, -15, 30, -30)]

    def cell_hits(dcol) -> list[str]:
        """칸에 놓는 자세(나르는 높이 ↔ 놓는 높이)에서 집게 끝·손목이 금지 구역 여유 안 낮게 걸리는 구역(실행기 검사와 같은 기준)."""
        return sorted({h for q in dcol["qs"] for h in pose_hits(s, q, (g["close"], g["release"]), zones_forbid,
                                                                 sf["forbid_below_mm"], sf["forbid_margin_mm"])})

    def zf_for(b) -> dict:
        """그 블록을 집을 때 보는 금지 구역 — 되돌리기는 집는 그 그릇만 뺀다."""
        return {k: v for k, v in zones_forbid.items() if not (unsort and k == b.get("zone"))}

    def try_yaw(b, yw):
        add = u["floor_mm"] if unsort else 0.0   # 종이 접시 바닥
        col = s.column(b["w"], yw, (z["approach"] + add, z["grasp"] + add), seed_q, tilt_max=tg, yaws=(yw,),
                       finger_target=hz.get("grasp_finger", 0) + lift_mm + add, close_n=g["close"])
        if col is None:
            return None
        if unsort and col["zs"][-1] - add - lift_mm > u["max_grasp_mm"]:
            s.last_why = (f"그릇이 멀어 기울여 집어야 해 집는 높이가 바닥 위 {col['zs'][-1] - add - lift_mm:.0f}mm — 집게 끝이 블록 "
                          f"옆까지 못 내려감(상한 unsort.max_grasp_mm {u['max_grasp_mm']:g})")
            return None
        up, uinfo = s.solve(s.w_to_r3(b["w"], z["carry"]), yw - s.frames.yaw, col["qs"][0], strict=True,
                            tilt=None, tilt_max=max(tg, tr), grip_n=g["close"], near=col["tilt"], only_prev=True)
        if up is None:  # 집은 뒤 나르는 높이로 들어 올릴 수 있어야 한다(기울기는 바뀌어도 됨)
            s.last_why = f"들어 올리는 높이 {z['carry']:g}mm 에 안 닿음"
            return None
        col["carry_tilt"], col["q_carry"] = uinfo["tilt_used"], up
        grips = np.linspace(g["close"], g["open"], 5)  # 벌린 채 내려가고 닫는 동안
        hit = sorted({h for q in col["qs"] for h in pose_hits(s, q, grips, zf_for(b), sf["forbid_below_mm"],
                                                              sf["forbid_margin_mm"])})
        if hit:
            s.last_why = f"벌린 집게·손목이 {', '.join(hit)} 위로 나감(집게 방향 {math.degrees(yw):.0f}°)"
            return None
        return col

    def first_yaws(b) -> list[float]:
        """먼저 해 볼 집게 방향(작업대 rad). 위 카메라로 잰 블록 긴 축(yaw_w)이 있고 길쭉하면 긴 축 + 90°(짧은 변을 끼움 —
        10-22 녹화 사람 집기: 긴 축과 집게축 |각| 중앙 77°, 사분위 62~85), 아니면 작업대 축 0°·90°."""
        if b.get("yaw_w") is not None and b.get("aspect", 1.0) >= YAW_ASPECT_MIN:
            return [b["yaw_w"] + math.pi / 2]
        return [0.0, math.pi / 2]

    def options(b, need_extra: bool) -> list:
        """블록 b 의 집기 자세 후보 [col] — first_yaws 먼저, need_extra 면 ±15°·30°(긴 축을 모를 땐 작업대 축에서, 알 땐
        수직 방향에서) 비튼 방향도(스펀지라 30° 어긋나도 쥔다 — 기울인 먼 자리는 반경 방향 집게가 두 끝 높이 차 한도에
        걸려 접선 쪽만 남는다). 블록마다 한 번만 푼다."""
        c = cache.setdefault(id(b), {})
        fy = first_yaws(b)
        if "axis" not in c:
            c["axis"] = [col for col in (try_yaw(b, yw) for yw in fy) if col]
            c["why"] = s.last_why
        if need_extra and "extra" not in c:
            more = ([fy[0] + v for v in extra_off] + [0.0, math.pi / 2]) if len(fy) == 1 else extra
            c["extra"] = [col for col in (try_yaw(b, yw) for yw in more) if col]
            for col in c["extra"]:
                col["twisted"] = True
        return c["axis"] + (c.get("extra") or [])

    while True:
        rest = [b for b in todo if b not in picked]
        if not rest:
            break
        opts = []
        for b in rest:
            peers = [o for o in blocks if o.get("zone") == b.get("zone")] if unsort else inside
            neigh = [(o["w"], BLOCK_R_MM) for o in peers if o is not b and o not in picked]

            def scored(cols):
                out = []
                for col in cols:
                    cl = jaw_clearance(b["w"], col["yaw_w"], neigh, open_gap / 2, g["finger_mm"])
                    # 이웃 여유 3mm 이상 → 관절 여유 → 비틀지 않은 방향 → 덜 기울기 → 집게 두 끝 높이 차 작은 것 → 여유
                    out.append(((cl >= 3, col["margin"] >= s.tilt["margin_deg"], not col.get("twisted"),
                                 -col["tilt"], -abs(col["infos"][-1]["jaw_incline_deg"]), cl), col, cl))
                return out
            sc = scored(options(b, False))
            if not any(x[2] >= 3 for x in sc):
                sc = scored(options(b, True))
            best = max(sc, key=lambda x: x[0]) if sc else None
            opts.append((b, best, cache[id(b)].get("why", "") if best is None else ""))
        ok = [o for o in opts if o[1] is not None]
        if not ok:
            for b, _, why in opts:
                if unsort:  # 되돌리기는 못 꺼내는 블록을 남기고 나머지만(전체 실패가 아니다)
                    unreachable.append({"block": b, "why": why})
                else:
                    path.failures.append(f"{COLOR_KO[b['cls']]} 블록(작업대 {np.round(b['w']).tolist()} mm)에 닿지 않는다 — {why}")
            break
        clear = [o for o in ok if o[1][2] >= 3] or ok
        b, (key, col, cl), _ = min(clear, key=lambda o: (o[0].get("which_rank", 0), float(np.linalg.norm(o[0]["r"]))))
        picked.append(b)
        if unsort:  # 빈 칸 차례로 — 닿는 첫 칸(놓을 때 집게축 고정 = 블록 긴 축 0°)
            z_pl = hz.get("place", 10) + lift_mm
            yaw_pl = math.radians(u["place_yaw_deg"])
            dcol = cell = None
            for cell in list(cells_left):
                # 기울여 놓는 먼 칸은 집게 끝이 탁자에 닿지 않게 놓는 높이를 올린다(집기의 grasp_finger 와 같은 식)
                dcol = s.column(cell[1], yaw_pl, (z["carry"], z_pl), seed_q, tilt_max=tr, grip_n=g["close"],
                                yaws=(yaw_pl,), finger_target=hz.get("grasp_finger", 0) + lift_mm, close_n=g["close"])
                if dcol is not None and cell_hits(dcol):  # 로봇 쪽 칸은 손가락이 손 자리 여유에 걸릴 수 있다 → 다음 칸
                    dcol = None
                if dcol is not None:
                    break
            if not cells_left or dcol is None:
                picked.pop()
                left = [x for x in todo if x not in picked]
                warnings.append(f"놓을 빈 칸이 {'모자라' if not cells_left else '닿지 않아'} 남김: "
                                f"{', '.join(COLOR_KO[x['cls']] for x in left)}")
                break
            cells_left.remove(cell)
            order.append({"block": b, "bowl": None, "src": b["zone"], "cell": cell[0], "drop_w": cell[1], "drop_shift_mm": 0.0,
                          "pick_yaw_w": col["yaw_w"], "pick_tilt": col["tilt"], "place_z": dcol["zs"][-1],
                          "carry_tilt": col["carry_tilt"], "grasp_z": col["zs"][-1], "approach_z": col["zs"][0],
                          "q_carry": col["q_carry"], "q_approach": col["qs"][0], "q_drop": dcol["qs"][0],
                          "drop_yaw_w": dcol["yaw_w"], "drop_tilt": dcol["tilt"], "pick_clear_mm": round(cl, 1),
                          "pick_jaw_incline": col["infos"][-1]["jaw_incline_deg"]})
            continue
        bowl = bw["targets"].get(b["cls"])
        if bowl not in bowls:
            path.failures.append(f"{COLOR_KO[b['cls']]} 블록을 넣을 그릇({bowl}) 위치가 없다(zones.json·bowls.center_mm)")
            continue
        k = per_bowl.get(bowl, 0)
        per_bowl[bowl] = k + 1
        side = (0, 1, -1, 2, -2)[k % 5]
        dcol = drop_w = None
        for drop_w in drop_points(s, bowl, bowls, zw, side * bw.get("spread_mm", 20), bw.get("edge_mm", 25)):
            # 놓을 때 집게 방향: 집은 방향 그대로(손목 롤을 덜 돌림), 안 되면 팔 평면에 수직(두 끝 같은 높이)
            yaw_drop = s.tangent_yaw(s.w_to_r3(drop_w, z_rel)) + s.frames.yaw
            dcol = s.column(drop_w, yaw_drop, (z["carry"], z_rel), seed_q, tilt_max=tr, grip_n=g["close"],
                            yaws=(col["yaw_w"], yaw_drop))
            if dcol is not None:
                break
        if dcol is None:
            path.failures.append(f"{BOWL_KO.get(bowl, bowl)} 위(가운데 {np.round(bowls[bowl]).tolist()} mm 에서 로봇 쪽까지, "
                                 f"높이 {z_rel:g})에 닿지 않는다 — {s.last_why}")
            continue
        order.append({"block": b, "bowl": bowl, "drop_w": drop_w, "drop_shift_mm": round(float(np.linalg.norm(drop_w - bowls[bowl])), 1),
                      "pick_yaw_w": col["yaw_w"], "pick_tilt": col["tilt"],
                      "carry_tilt": col["carry_tilt"], "grasp_z": col["zs"][-1], "approach_z": col["zs"][0],
                      "q_carry": col["q_carry"], "q_approach": col["qs"][0], "q_drop": dcol["qs"][0],
                      "drop_yaw_w": dcol["yaw_w"], "drop_tilt": dcol["tilt"], "pick_clear_mm": round(cl, 1),
                      "pick_jaw_incline": col["infos"][-1]["jaw_incline_deg"]})


    def go(p_from_w, z_from, p_to_w, z_to, yaw_to, tilt_to, what, skip, q_goal=None) -> bool:
        return travel(s, path, p_from_w, z_from, p_to_w, z_to, yaw_to, tilt_to, what, zones=zones_forbid, skip=skip,
                      z_high=z_high, margin=sf["forbid_margin_mm"] + (UNSORT_TRAVEL_EXTRA_MM if unsort else 0.0),
                      tilt_max=max(tg, tr), q_goal=q_goal, fast=True)

    # 궤적
    cur = None
    prev_bowl = None
    for i, o in enumerate(order):
        b = o["block"]
        hi, lo, top = (s.w_to_r3(b["w"], o["approach_z"]), s.w_to_r3(b["w"], o["grasp_z"]), s.w_to_r3(b["w"], z["carry"]))
        d_top, d_lo = s.w_to_r3(o["drop_w"], z["carry"]), s.w_to_r3(o["drop_w"], o.get("place_z", z_rel))
        src = o.get("src")      # 되돌리기: 집는 그 그릇(이 블록 집기·떠나기에서만 금지 구역에서 뺌)
        zf_b = {k: v for k, v in zones_forbid.items() if k != src}
        where = f"칸 {o['cell']}" if unsort else BOWL_KO.get(o["bowl"], o["bowl"])
        py, dy = o["pick_yaw_w"] - s.frames.yaw, o["drop_yaw_w"] - s.frames.yaw
        pt, dt, ct = o["pick_tilt"], o["drop_tilt"], o["carry_tilt"]
        tag = f"[{i + 1}/{len(order)}] {COLOR_KO[b['cls']]}"
        path.gripper(g["open"], f"{tag} 집게 벌림")
        if cur is None and path.joint_hits(o["q_carry"], zf_b, sf["forbid_below_mm"], sf["forbid_margin_mm"]):
            # 홈 → 블록 위가 금지 구역(손 자리 등) 위를 낮게 지나면: 블록 위 높은 곳(over_zone)으로 간 뒤 내려온다
            hq, hinfo = s.solve(s.w_to_r3(b["w"], z_high), py, o["q_carry"], strict=True, tilt=None,
                                tilt_max=max(tg, tr), near=ct, only_prev=True)
            hit = path.joint_hits(hq, zf_b, sf["forbid_below_mm"], sf["forbid_margin_mm"]) if hq is not None else ["IK"]
            if hit:
                path.failures.append(f"{tag} 홈 → 블록 위 관절 이동이 {', '.join(hit)} 위를 낮게 지난다(높이 {z_high:g} 로도)")
                break
            path.joint_move(hq, f"{tag} 홈 → 블록 위(높이 {z_high:g})", yaw=py, tilt=hinfo["tilt_used"], fast=True)
            cur = s.w_to_r3(b["w"], z_high)
        if cur is None:
            qa, info = o["q_carry"], {"ok": True}
            if qa is None:
                path.failures.append(f"{tag} 첫 블록 위 IK 실패 (로봇 좌표 {np.round(top).tolist()} mm, {info['why']})")
                break
            path.joint_move(qa, f"{tag} 홈 → 블록 위", yaw=py, tilt=ct, fast=True)
            cur = top
        if not ((cur is top or go(s.frames.r2w(cur[:2]), cur[2], b["w"], z["carry"], py, ct, f"{tag} → 블록 위",
                                  {prev_bowl, src}, o["q_carry"]))
                and path.line(top, hi, py, pt, f"{tag} 집기 높이 위로", strict=False, tilt_max=max(tg, tr),
                              q_goal=o["q_approach"])
                and path.line(hi, lo, py, pt, f"{tag} 내려감", strict=True)):
            break
        path.gripper(g["close"], f"{tag} 집음", settle=True)
        if not (path.line(lo, hi, py, pt, f"{tag} 들어 올림", strict=True)
                and path.line(hi, top, py, ct, f"{tag} 나르는 높이로", strict=False, tilt_max=max(tg, tr),
                              q_goal=o["q_carry"])
                and go(b["w"], z["carry"], o["drop_w"], z["carry"], dy, dt, f"{tag} → {where} 위",
                       {o["bowl"], src}, o["q_drop"])
                and path.line(d_top, d_lo, dy, dt, f"{tag} 놓는 높이로", strict=True)):
            break
        path.gripper(g["release"], f"{tag} 놓음", settle=True)
        if not path.line(d_lo, d_top, dy, dt, f"{tag} 올라감", strict=True, fast=True):
            break
        cur, prev_bowl = d_top, o["bowl"]
    path.finish({k: v for k, v in zones_forbid.items() if k != prev_bowl}, sf["forbid_below_mm"], sf["forbid_margin_mm"],
                z_high)
    checks, total = path.checks()
    checks["집을 때 집게 여유 최소 mm"] = min((o["pick_clear_mm"] for o in order), default=99)
    checks["집을 때 집게 두 끝 높이 차 최대 °"] = max((abs(o["pick_jaw_incline"]) for o in order), default=0)
    checks["집는 tcp 높이 mm(기울기로 올림)"] = [o["grasp_z"] for o in order]
    if checks["집을 때 집게 여유 최소 mm"] < 3:
        warnings.append("집을 때 집게 손가락이 이웃 블록에 닿을 수 있다(집기 여유 < 3mm) — 블록 사이를 벌려 놓는다")
    warnings += path.tight_warning()
    if unsort:
        checks["놓은 칸"] = [o["cell"] for o in order]
        for x in unreachable:
            warnings.append(f"{COLOR_KO[x['block']['cls']]}(작업대 {np.round(x['block']['w']).tolist()} mm) 못 꺼냄 — {x['why']}")
    return {"kind": "unsort" if unsort else "sort", "classes": list(classes), "order": order, "unreachable": unreachable, "steps": path.steps, "failures": path.failures,
            "warnings": warnings, "checks": checks, "total_s": total, "blocks": blocks, "needed": len(todo),
            "heights": {**z, "release": z_rel, **({"place": hz.get("place", 10) + lift_mm, "floor": u["floor_mm"]}
                                                 if unsort else {})}, "lift_mm": lift_mm}


def trajectory(steps: list[Step], rate_hz: float, *, with_marks: bool = False, with_fast: bool = False):
    """경유점 → 일정 간격 목표값(정규화 6열). 실제 실행기가 send_action 으로 보낼 모양.

    with_marks=True 면 (목표값, [(목표값 번호, 단계 이름)]) — 실행기가 단계마다 기록을 남길 때 쓴다.
    with_fast=True 면 끝에 틱마다 공중 구간 여부(bool 배열, 그 틱에 도착하는 걸음의 fast)를 더 돌려준다.
    """
    out, cur, marks, fast = [], None, [], []
    for st in steps:
        if cur is None:
            cur = st.n
            out.append(cur)
            fast.append(False)
        else:
            k = max(1, int(math.ceil(st.dur_s * rate_hz - 1e-6)))
            for i in range(1, k + 1):
                out.append(cur + (st.n - cur) * i / k)
                fast.append(bool(st.fast))
            cur = st.n
        if not st.what.endswith("…"):
            marks.append((len(out) - 1, st.what))
    arr = np.array(out)
    res = (arr, marks) if with_marks else (arr,)
    if with_fast:
        res = res + (np.array(fast, bool),)
    return res if len(res) > 1 else res[0]


def air_speeds(motion: dict) -> tuple[float, float]:
    """공중 구간 계획 속도(직선 mm/s, 관절 °/s): motion.air_mm_s·air_deg_s, 없으면 보통 값(cart_mm_s·max_deg_s) —
    설정이 없으면 예전과 같다(10-23)."""
    return float(motion.get("air_mm_s") or motion["cart_mm_s"]), float(motion.get("air_deg_s") or motion["max_deg_s"])
