"""집기 정책 끝 판정 — '블록 위에서 시작해 내려가 쥐고 들어 올린 뒤 멈춤'을 관절값으로 잡는다(home_stop 의 집기판).

집기 정책(policies.json grasp, 10-23 녹화)은 IK 가 블록 위(집게 가운데 50mm, 집게 벌림)에 세워 둔 자세에서 시작해
"내려가 쥐기 → 30~50mm 들기 → 1초 정지"로 끝나는 짧은 에피소드로 배운다. 홈으로 돌아오지 않으므로 홈 복귀 정지 대신:

1. 닫음: 그리퍼 값이 지금까지 최댓값에서 close_drop 넘게 떨어짐(집게 벌림 63 → 쥔 값 50~52).
2. 들림: 닫기 시작한 뒤 집게 가운데(FK) 가장 낮았던 높이보다 lift_mm 넘게 올라감.
3. 멈춤: 그 뒤 hold_s 초 동안 6관절 변화폭이 모두 ≤ still.
→ 끝(정상 종료 SIGINT). 판정에 쓰는 높이 함수 tcp_z(관절 6개 정규화) → mm 는 부르는 쪽이 준다(coord_pick 기구학).

바닥 누름(10-23 12:59 실물: 정책이 닫지 않고 tcp −6.6mm 까지 내려가 10초를 다 씀): 닫기 전에 tcp 가 floor_mm 아래에
floor_hold_s 넘게 있거나 floor_hard_mm 아래로 가면 바로 끝(kind "floor" — 실행기가 못 쥔 것으로 보고 IK 로 넘긴다).
floor_mm None 이면 안 본다. 녹화 사람 39개: 닫기 전 tcp 최저 −10~18mm(쥘 때 잠깐) — 기본값은 거기 걸리지 않게 잡는다.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

DEFAULTS = {"close_drop": 6.0, "lift_mm": 20.0, "still": 1.5, "hold_s": 0.5, "floor_mm": None, "floor_hold_s": 0.4,
            "floor_hard_mm": None}


@dataclass
class GraspTrigger:
    t: float            # 판정 시각(관측 시각 기준)
    t_closed: float     # 닫기 시작을 본 시각
    lift_mm: float      # 그때 들린 높이(닫은 뒤 최저점 기준)
    z_low: float        # 닫은 뒤 집게 가운데 최저 높이(mm)
    kind: str = "grasp"  # "grasp"(닫고·들고·멈춤) | "floor"(닫기 전에 바닥을 누름)


class GraspStopDetector:
    def __init__(self, tcp_z, *, close_drop: float = DEFAULTS["close_drop"], lift_mm: float = DEFAULTS["lift_mm"],
                 still: float = DEFAULTS["still"], hold_s: float = DEFAULTS["hold_s"], floor_mm: float | None = None,
                 floor_hold_s: float = DEFAULTS["floor_hold_s"], floor_hard_mm: float | None = None):
        self.tcp_z = tcp_z
        self.close_drop, self.lift, self.still, self.hold_s = close_drop, lift_mm, still, hold_s
        self.floor, self.floor_hold, self.floor_hard = floor_mm, floor_hold_s, floor_hard_mm
        self.floor_since: float | None = None
        self.z_min: float | None = None
        self.g_max: float | None = None
        self.closed_at: float | None = None
        self.z_low: float | None = None
        self.lifted_at: float | None = None
        self.last_z: float | None = None
        self.trigger: GraspTrigger | None = None
        self._win: deque[tuple[float, list[float]]] = deque()

    @property
    def closed(self) -> bool:
        return self.closed_at is not None

    @property
    def lifted(self) -> bool:
        return self.lifted_at is not None

    def feed(self, t: float, q) -> bool:
        """관측 하나(시각 초, 관절 6개). 끝 판정이 난 뒤로는 계속 True."""
        if self.trigger is not None:
            return True
        q = [float(v) for v in q[:6]]
        z = float(self.tcp_z(q))
        self.last_z = z
        self.z_min = z if self.z_min is None else min(self.z_min, z)
        g = q[5]
        self.g_max = g if self.g_max is None else max(self.g_max, g)
        if self.closed_at is None:
            if self.g_max - g > self.close_drop:
                self.closed_at, self.z_low = t, z
                return False
            if self.floor is not None:  # 닫기 전 바닥 누름
                if z < self.floor:
                    self.floor_since = t if self.floor_since is None else self.floor_since
                    if t - self.floor_since >= self.floor_hold or (self.floor_hard is not None and z < self.floor_hard):
                        self.trigger = GraspTrigger(t=t, t_closed=-1.0, lift_mm=0.0, z_low=round(z, 1), kind="floor")
                        return True
                else:
                    self.floor_since = None
            return False
        if self.lifted_at is None:
            self.z_low = min(self.z_low, z)
            if z - self.z_low > self.lift:
                self.lifted_at = t
                self._win.clear()
            return False
        self._win.append((t, q))
        while self._win and self._win[0][0] < t - self.hold_s:
            self._win.popleft()
        if self._win[-1][0] - self._win[0][0] < self.hold_s - 1.5 / 30:
            return False
        span = max(max(s[1][j] for s in self._win) - min(s[1][j] for s in self._win) for j in range(6))
        if span <= self.still:
            self.trigger = GraspTrigger(t=t, t_closed=self.closed_at, lift_mm=round(z - self.z_low, 1),
                                        z_low=round(self.z_low, 1))
            return True
        return False

    def summary(self) -> dict:
        tr = self.trigger
        return {"closed": self.closed, "lifted": self.lifted, "z_low_mm": None if self.z_low is None else round(self.z_low, 1),
                "last_z_mm": None if self.last_z is None else round(self.last_z, 1),
                "z_min_mm": None if self.z_min is None else round(self.z_min, 1),
                "lift_mm": tr.lift_mm if tr else None, "stop_kind": tr.kind if tr else None}


def tcp_z_fn():
    """coord_pick 기구학으로 관절 6개(정규화) → 집게 가운데 높이(mm). config.json 은 읽기만."""
    import numpy as np

    from robot.coord_pick.plan import Setup, load_config
    s = Setup.load(load_config(), use_studio=False)
    return lambda q: float(s.tcp_pos(s.n_to_q(np.asarray(q, float)))[2])
