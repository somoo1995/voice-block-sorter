"""SO-101 기구학 — Studio 에 들어 있는 URDF(`so101_new_calib.urdf`)로 순·역기구학을 numpy 만으로 푼다.

단위 세 가지(COORD_PICK.md 2절):
- 정규화(n): Studio·physicalai 가 주고받는 값. 몸통 관절 -100~100, 그리퍼 0~100.
- 틱: 서보 값. 보정 범위가 0~4095 라서(follower_24) n = 틱/4095×200−100.
- URDF 각(rad): 보정 범위 가운데가 0. lerobot 의 degrees 모드와 같은 식 → 각(도) = n × (range_max−range_min)/200 × 360/4095.
  범위가 0~4095 면 정확히 각(도) = 1.8 × n. 관절마다 부호(sign)는 config 로 둔다(입회 보정 1단계에서 확인).

끝 = gripper_frame_link(고정 집게 끝에서 6.3mm 안쪽, 집게 방향 = 끝 z, 집게가 벌어지는 축 = 끝 x — 움직이는 집게는 −x 쪽).
tcp = 끝 좌표로 준 점(mm). 10-22 녹화 102번 집기에서 블록 가운데는 끝에서 (−28.5, 12.7, 0)mm 였다(COORD_PICK.md 2.2).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URDF_PATH = Path(__file__).resolve().parent / "so101_new_calib.urdf"
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")  # IK 가 푸는 5개
TIP = "gripper_frame_link"
G = 9.81

# 집게 끝 모양(끝 좌표 mm) — Studio 메시(wrist_roll_follower_so101_v1·moving_jaw_so101_v1.stl)에서 잼.
# 고정 집게 끝 3mm 단면: x 0~6.1, y ±3.7, z 6.3 (끝 5~20mm 안쪽은 y ±7.5). 바닥 검사는 이 네 모서리.
FIXED_TIP_MM = np.array([[0.0, -7.5, 6.3], [0.0, 7.5, 6.3], [13.6, -7.5, 6.3], [13.6, 7.5, 6.3]])
# 움직이는 집게: 그리퍼 정규화값 n → (끝 z, 끝 x 범위, 두 집게 안쪽 틈 mm). n 1 = 3.6°
JAW_TABLE = ((45, 0.6, 0.9, 8.7, 0.0), (50, 7.3, -22.2, -15.8, 15.8), (52, 8.1, -32.0, -25.8, 24.9),
             (55, 6.8, -46.7, -40.7, 36.0), (58, 2.7, -61.0, -55.1, 46.0), (60, -1.6, -70.0, -63.9, 55.6),
             (63, -10.0, -82.3, -76.3, 66.2), (66, -20.8, -92.7, -86.2, 84.2), (70, -37.9, -103.3, -92.8, 95.0))


def jaw_gap_mm(n: float) -> float:
    """그리퍼 정규화값 → 두 집게 안쪽 틈(mm, 메시 계산). 45 이하는 닫힘."""
    ns = [r[0] for r in JAW_TABLE]
    return float(np.interp(n, ns, [r[4] for r in JAW_TABLE]))


def finger_points_mm(n_grip: float) -> np.ndarray:
    """바닥·물체 검사용 집게 끝 점들(끝 좌표 mm): 고정 집게 끝 4모서리 + 움직이는 집게 끝 4모서리."""
    ns = [r[0] for r in JAW_TABLE]
    z = float(np.interp(n_grip, ns, [r[1] for r in JAW_TABLE]))
    x0 = float(np.interp(n_grip, ns, [r[2] for r in JAW_TABLE]))
    x1 = float(np.interp(n_grip, ns, [r[3] for r in JAW_TABLE]))
    moving = np.array([[x0, -7.5, z], [x0, 7.5, z], [x1, -7.5, z], [x1, 7.5, z]])
    return np.vstack([FIXED_TIP_MM, moving])


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis: np.ndarray, q: float) -> np.ndarray:
    x, y, z = axis
    c, s, t = math.cos(q), math.sin(q), 1 - math.cos(q)
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    T: np.ndarray        # 부모 링크 → 관절 원점(4×4)
    axis: np.ndarray
    lower: float
    upper: float
    child: str = ""


class SO101Kinematics:
    """base_link → gripper_frame_link 사슬. 길이 단위 m, 각 단위 rad."""

    def __init__(self, urdf_path: str | Path = URDF_PATH, tip: str = TIP,
                 limits_deg: dict[str, tuple[float, float]] | None = None):
        """limits_deg: IK 가 쓸 관절 범위(도). 없으면 URDF 한계. 녹화로 확인된 범위를 쓰는 게 안전하다(config.json)."""
        root = ET.parse(urdf_path).getroot()
        by_child = {}
        for j in root.findall("joint"):  # <transmission> 안의 <joint> 는 빼고 최상위만
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3] = _rpy(*rpy)
            T[:3, 3] = xyz
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
            lim = j.find("limit")
            lo = float(lim.get("lower", "-inf")) if lim is not None else -math.inf
            hi = float(lim.get("upper", "inf")) if lim is not None else math.inf
            child = j.find("child").get("link")
            by_child[child] = (j.find("parent").get("link"), Joint(j.get("name"), j.get("type"), T, axis, lo, hi, child))
        # 링크 질량·무게중심(링크 좌표) — 중력 처짐 모형(gravity_torque)에 쓴다. 움직이는 집게(가지)는 gripper_link 에 얹는다.
        self.mass: dict[str, tuple[float, np.ndarray]] = {}
        for ln in root.findall("link"):
            ine = ln.find("inertial")
            if ine is None:
                continue
            m = float(ine.find("mass").get("value"))
            o = ine.find("origin")
            com = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
            self.mass[ln.get("name")] = (m, com)
        for child, (parent, jt) in by_child.items():
            if parent == "gripper_link" and child in self.mass and jt.kind == "revolute":
                m, com = self.mass.pop(child)
                gm, gc = self.mass["gripper_link"]
                pc = jt.T[:3, 3] + jt.T[:3, :3] @ com
                self.mass["gripper_link"] = (gm + m, (gm * gc + m * pc) / (gm + m))
        chain, link = [], tip
        while link in by_child:
            parent, joint = by_child[link]
            chain.append(joint)
            link = parent
        self.chain = list(reversed(chain))
        self.joints = {j.name: j for j in self.chain if j.kind == "revolute"}
        missing = [n for n in ARM if n not in self.joints]
        if missing:
            raise ValueError(f"URDF 사슬에 관절이 없다: {missing}")
        self._limits = np.array([[self.joints[n].lower, self.joints[n].upper] for n in ARM])
        for i, n in enumerate(ARM):
            if limits_deg and n in limits_deg:
                self._limits[i] = np.radians(limits_deg[n])
        self.pan_origin = self.chain[0].T[:3, 3].copy()  # 받침 위 회전축(팬) 원점

    def _walk(self, q):
        """(관절, 관절 원점 자세(회전 전), 자식 링크 자세) 를 사슬 순서대로."""
        if not isinstance(q, dict):
            q = dict(zip(ARM, q))
        T = np.eye(4)
        for j in self.chain:
            Tj = T @ j.T
            if j.kind == "revolute":
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j.axis, q.get(j.name, 0.0))
                T = Tj @ R
            else:
                T = Tj
            yield j, Tj, T

    def fk(self, q: dict[str, float] | np.ndarray) -> np.ndarray:
        """끝(gripper_frame_link)의 4×4 자세. q 는 ARM 순서 배열 또는 {이름: rad}."""
        T = np.eye(4)
        for _, _, T in self._walk(q):
            pass
        return T

    def limits(self) -> np.ndarray:
        return self._limits.copy()

    def link_points(self, q: np.ndarray) -> np.ndarray:
        """관절 원점들(어깨 이후)과 끝의 위치(m). 탁자와 부딪히는지 볼 때 쓴다."""
        pts = [T[:3, 3].copy() for _, _, T in self._walk(q)]
        return np.array(pts[1:])  # shoulder_pan 원점(받침 위)은 빼고

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """ARM 5관절에 걸리는 중력 토크(N·m, 관절 축 방향 +). URDF 질량·무게중심으로."""
        frames, joints = {"base_link": np.eye(4)}, []
        for j, Tj, T in self._walk(q):
            frames[j.child] = T
            if j.kind == "revolute":
                joints.append((j.name, Tj[:3, 3], Tj[:3, :3] @ j.axis, len(frames) - 1))
        names = list(frames)
        tau = {}
        for name, o, a, idx in joints:
            t = 0.0
            for ln in names[idx:]:
                if ln in self.mass:
                    m, com = self.mass[ln]
                    p = frames[ln][:3, :3] @ com + frames[ln][:3, 3]
                    t += float(a @ np.cross(p - o, np.array([0.0, 0.0, -m * G])))
            tau[name] = t
        return np.array([tau[n] for n in ARM])

    def _state(self, qv) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(끝 자세, ARM 관절 축 세계 방향 5×3, 관절 원점 5×3, shoulder_lift 축) — 사슬을 한 번 돈다(해석 야코비안용)."""
        axes, origins, lift = [], [], None
        T = np.eye(4)
        for j, Tj, T in self._walk(qv):
            if j.kind == "revolute" and j.name in ARM:
                a = Tj[:3, :3] @ j.axis
                axes.append(a)
                origins.append(Tj[:3, 3])
                if j.name == "shoulder_lift":
                    lift = a
        return T, np.array(axes), np.array(origins), lift

    def _out_dir(self, a: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """팔 평면(shoulder_lift 축 a 에 수직)의 바깥쪽 수평 단위 벡터."""
        h = np.array([-a[1], a[0], 0.0])
        n = np.linalg.norm(h)
        h = h / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        return h if h[:2] @ (pos[:2] - self.pan_origin[:2]) >= 0 else -h

    def ik(self, pos: np.ndarray, yaw: float | None = None, q0: np.ndarray | None = None, *,
           tilt: float = 0.0, tcp: np.ndarray | None = None, iters: int = 60,
           tol_mm: float = 0.5) -> tuple[np.ndarray, dict]:
        """tcp(끝 좌표 m, 기본 끝 자체)를 pos(m, base_link 기준)에 둔다.

        집게 방향(끝 z)은 아래(−Z)에서 팔 평면 바깥쪽으로 tilt(rad) 기운다(0 = 위에서 수직). 5축 팔이라 끝 z 는 늘
        팔 평면 안에 있다 → 위치 3 + 기울기 1 + 집게 방향 1 = 5 로 딱 맞는다.
        yaw: 집게가 벌어지는 축(끝 x)을 수평면에 투영한 방향. None 이면 roll 자유. 기울어 있을 때 yaw 가 팔 평면의
        접선이 아니면 두 집게 끝 높이가 달라진다(info["jaw_incline_deg"]).
        반환: (q[5] rad, 정보{pos_err_mm, tilt_deg, tilt_err_deg, yaw_err_deg, jaw_incline_deg, ok, iters, ...})
        감쇠 최소제곱(Levenberg–Marquardt) + 해석 야코비안(관절 축 × 거리). 한계를 넘으면 잘라낸다.
        """
        lim = self.limits()
        q = np.zeros(5) if q0 is None else np.array(q0, dtype=float)
        tcp = np.zeros(3) if tcp is None else np.asarray(tcp, float)
        st, ct = math.sin(tilt), math.cos(tilt)

        def evaluate(qv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            T, A, O, lift = self._state(qv)
            R = T[:3, :3]
            p = T[:3, 3] + R @ tcp
            z, x = R[:, 2], R[:, 0]
            h = self._out_dir(lift, pos) if st else np.zeros(3)
            want = st * h + np.array([0.0, 0.0, -ct])
            r = [(p - pos) / 0.01, (z - want) * 2.0]          # 1cm 를 1 로 · 원하는 기울기에서 벗어난 정도
            Jp = np.cross(A, p - O).T / 0.01
            Jd = np.cross(A, z).T * 2.0
            if st:
                Jd[:, 0] -= 2.0 * st * np.cross(A[0], h)      # 팬이 돌면 팔 평면(h)도 같이 돈다
            J = [Jp, Jd]
            if yaw is not None:
                d = math.atan2(x[1], x[0]) - yaw
                r.append([(d + math.pi / 2) % math.pi - math.pi / 2])  # 집게는 180° 돌려도 같다
                dx = np.cross(A, x)
                J.append(((x[0] * dx[:, 1] - x[1] * dx[:, 0]) / max(1e-12, x[0] ** 2 + x[1] ** 2))[None, :])
            return np.concatenate([np.atleast_1d(np.asarray(v, dtype=float)) for v in r]), np.vstack(J)

        lam, it = 1e-2, 0
        r, J = evaluate(q)
        for it in range(1, iters + 1):
            step = np.linalg.solve(J.T @ J + lam * np.eye(5), -J.T @ r)
            qn = np.clip(q + step, lim[:, 0], lim[:, 1])
            rn, Jn = evaluate(qn)
            if rn @ rn < r @ r:
                q, r, J, lam = qn, rn, Jn, max(lam / 3, 1e-6)
                if np.linalg.norm(r[:3]) * 10 < tol_mm and np.linalg.norm(r[3:6]) < 0.01:
                    break
            else:
                lam *= 4
                if lam > 1e6:
                    break
        T, _, _, lift = self._state(q)
        z = T[:3, 2]
        tilt_now = math.degrees(math.acos(max(-1.0, min(1.0, -z[2]))))
        h = self._out_dir(lift, pos)
        want = math.sin(tilt) * h + np.array([0.0, 0.0, -math.cos(tilt)])
        tilt_err = math.degrees(math.acos(max(-1.0, min(1.0, float(z @ want)))))
        err = float(np.linalg.norm(T[:3, 3] + T[:3, :3] @ tcp - pos) * 1000)
        info = {"pos_err_mm": round(err, 2), "tilt_deg": round(tilt_now, 2), "tilt_err_deg": round(tilt_err, 2),
                "jaw_incline_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(T[2, 0]))))), 1), "iters": it}
        if yaw is not None:
            info["yaw_err_deg"] = round(math.degrees(abs(r[-1])), 2)
        margin = np.minimum(q - lim[:, 0], lim[:, 1] - q)
        info["limit_margin_deg"] = round(math.degrees(float(margin.min())), 1)
        info["limit_joint"] = ARM[int(margin.argmin())]
        info["ok"] = err < 2.0 and tilt_err < 3.0 and info.get("yaw_err_deg", 0) < 3.0
        return q, info


@dataclass(frozen=True)
class JointMap:
    """정규화 ↔ URDF 각. sign·offset 은 입회 보정 1단계에서 확인해 config 에 넣는다."""

    ranges: dict[str, tuple[int, int]]          # Studio 보정의 range_min·range_max
    sign: dict[str, int]
    offset_deg: dict[str, float]

    def deg_per_unit(self, name: str) -> float:
        lo, hi = self.ranges[name]
        return (hi - lo) / 200 * 360 / 4095

    def to_rad(self, n: np.ndarray) -> np.ndarray:
        return np.array([math.radians(self.sign[k] * n[i] * self.deg_per_unit(k) + self.offset_deg[k])
                         for i, k in enumerate(ARM)])

    def to_norm(self, q: np.ndarray) -> np.ndarray:
        return np.array([(math.degrees(q[i]) - self.offset_deg[k]) / self.deg_per_unit(k) * self.sign[k]
                         for i, k in enumerate(ARM)])
