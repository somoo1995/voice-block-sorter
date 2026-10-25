"""홈 복귀 정지 감지 — 정책이 일을 마치고 홈으로 돌아와 멈추면 timeout 을 기다리지 않고 끝내기 위한 판정.

ACT 에는 "다 했다" 신호가 없어 `--run.duration_s`(timeout) 까지 돈다(README 2.1). 녹화 에피소드는
"놓고 → 시작 자세 1~2초 정지"로 끝나므로, 정책도 같은 자리에서 멈춘다. 그 순간을 관절값으로 잡는다.

판정(관절은 정규화 단위, 몸통 5개):
1. 홈 = 첫 관측 자세(런타임 복귀 기준과 같다).
2. 벗어남: 홈과의 거리(관절별 차의 최댓값)가 leave 를 넘은 적이 있어야 한다 → 처음 홈 대기 구간은 판정하지 않는다.
3. 그 뒤 거리 ≤ near 이고, 마지막 hold_s 초 동안 6관절(그리퍼 포함) 변화폭이 모두 ≤ still 이면 → 끝.

기본값은 10-20 녹화 40개로 정했다(README 5.7): 작업 중 최대 거리 최소 44.6, 끝 자세 거리 중앙 2.3·90% 4.9.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

DEFAULTS = {"leave": 15.0, "near": 6.0, "still": 1.0, "hold_s": 1.0}


@dataclass
class Trigger:
    t: float            # 판정 시각(관측 시각 기준)
    t_home: float       # 홈 근처에 들어온 시각(이번 정지 구간의 시작)
    dist: float         # 그때 홈 거리


class HomeStopDetector:
    def __init__(self, *, leave: float = DEFAULTS["leave"], near: float = DEFAULTS["near"],
                 still: float = DEFAULTS["still"], hold_s: float = DEFAULTS["hold_s"]):
        if near >= leave:
            raise ValueError("near 는 leave 보다 작아야 한다")
        self.leave, self.near, self.still, self.hold_s = leave, near, still, hold_s
        self.home: list[float] | None = None
        self.t0: float | None = None
        self.left_at: float | None = None
        self.max_dist = 0.0
        self.last_dist = 0.0
        self.trigger: Trigger | None = None
        self._win: deque[tuple[float, list[float]]] = deque()
        self._near_since: float | None = None

    def dist(self, q) -> float:
        return max(abs(float(q[j]) - self.home[j]) for j in range(5))

    def feed(self, t: float, q) -> bool:
        """관측 하나(시각 초, 관절 6개). 끝 판정이 난 뒤로는 계속 True."""
        if self.trigger is not None:
            return True
        q = [float(v) for v in q[:6]]
        if self.home is None:
            self.home, self.t0 = q, t
        d = self.dist(q)
        self.last_dist = d
        self.max_dist = max(self.max_dist, d)
        if self.left_at is None:
            if d > self.leave:
                self.left_at = t
            return False
        if d > self.near:
            self._win.clear()
            self._near_since = None
            return False
        if self._near_since is None:
            self._near_since = t
        self._win.append((t, q))
        while self._win and self._win[0][0] < t - self.hold_s:
            self._win.popleft()
        if t - self._near_since < self.hold_s - 1e-9:
            return False
        span = max(max(s[1][j] for s in self._win) - min(s[1][j] for s in self._win) for j in range(6))
        if span <= self.still:
            self.trigger = Trigger(t=t, t_home=self._near_since, dist=round(d, 2))
            return True
        return False
