"""시험용 가짜 SharedRobot·가짜 시계 — stream.py 를 로봇 없이 돌린다(표준 라이브러리 + numpy).

FakeSharedRobot: 1차 지연(tau)으로 명령을 따라가는 팔. 오차 주입:
- stuck=(관절, 틱): 그 틱부터 그 관절이 멈춘다(충돌·걸림 흉내)
- bias=(관절, 정규화): 그 관절이 늘 그만큼 덜 간다(중력 처짐 흉내)
- stale_from=틱: 그 틱부터 관측 시각이 안 바뀐다(소유 프로세스 끊김 흉내)
- interrupt_at=틱: 그 틱의 send_action 에서 KeyboardInterrupt(SIGINT 흉내)
- grip_block=정규화: 그리퍼가 그 값 아래로 안 닫힌다(블록을 쥔 것 흉내)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class FakeClock:
    """sleep 하면 시간만 흐른다(실제로 기다리지 않음)."""

    def __init__(self, t0: float = 1000.0):
        self.t = t0

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(0.0, dt)


class RealClock:
    """실제 시계(실시간 시험용)."""

    @staticmethod
    def now() -> float:
        import time
        return time.monotonic()


@dataclass
class FakeObs:
    joint_positions: np.ndarray
    timestamp: float


class FakeSharedRobot:
    def __init__(self, start, clock, *, tau: float = 0.05, stuck=None, bias=None, stale_from=None,
                 interrupt_at=None, grip_block=None, refuse: str | None = None):
        self.q = np.array(start, float)
        self.cmd = self.q.copy()
        self.clock = clock
        self.tau = tau
        self.stuck, self.bias, self.stale_from = stuck, bias, stale_from
        self.interrupt_at, self.grip_block, self.refuse = interrupt_at, grip_block, refuse
        self.sends: list[np.ndarray] = []
        self._t = clock.now()
        self._ts = self._t
        self.connected = False

    def connect(self) -> None:
        if self.refuse:
            raise RuntimeError(self.refuse)
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def send_action(self, action, goal_time: float = 0.1) -> None:
        if self.interrupt_at is not None and len(self.sends) == self.interrupt_at:
            raise KeyboardInterrupt
        self._advance()
        self.cmd = np.array(action, float)
        self.sends.append(self.cmd.copy())

    def _advance(self) -> None:
        now = self.clock.now()
        dt, self._t = now - self._t, now
        goal = self.cmd.copy()
        if self.bias is not None:
            j, b = self.bias
            goal[j] -= b
        if self.grip_block is not None:
            goal[5] = max(goal[5], self.grip_block)
        a = 1 - math.exp(-dt / self.tau) if dt > 0 else 0.0
        new = self.q + a * (goal - self.q)
        if self.stuck is not None and len(self.sends) >= self.stuck[1]:
            new[self.stuck[0]] = self.q[self.stuck[0]]
        self.q = new
        if self.stale_from is None or len(self.sends) < self.stale_from:
            self._ts = now

    def get_observation(self) -> FakeObs:
        self._advance()
        return FakeObs(self.q.copy(), self._ts)
