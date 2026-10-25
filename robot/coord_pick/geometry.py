"""작업대 좌표 — 위 카메라 픽셀 → 작업대 mm(호모그래피), 작업대 mm → 로봇 base_link(평면 강체 변환).

작업대 좌표(W): 테이프 표시 4점으로 정한 평면 좌표(mm). 사용자가 줄자로 잰 값을 config 에 적는다.
로봇 좌표(R): URDF base_link(mm). x = 팔이 뻗는 앞, y = 왼쪽, z = 위. 녹화 데이터로 보면 z=0 이 탁자면이다.
"""
from __future__ import annotations

import math

import numpy as np


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """정규화 DLT. src·dst 는 (N,2), N≥4. src → dst 3×3."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("호모그래피는 대응점 4개 이상이 필요하다")

    def norm(p):
        c = p.mean(0)
        s = math.sqrt(2) / max(np.sqrt(((p - c) ** 2).sum(1)).mean(), 1e-9)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Ts, Td = norm(src), norm(dst)
    a = (Ts @ np.c_[src, np.ones(len(src))].T).T
    b = (Td @ np.c_[dst, np.ones(len(dst))].T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    H = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    return H / H[2, 2]


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def fit_rigid2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, float]:
    """src → dst 평면 회전+이동(Kabsch). 반환 (tx, ty, yaw_rad). 입회 보정 3단계(팔 끝으로 테이프 점 짚기)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    t = cd - R @ cs
    return float(t[0]), float(t[1]), math.atan2(R[1, 0], R[0, 0])


class Frames:
    """W(작업대 mm) ↔ R(로봇 mm). base = 작업대 좌표에서 본 로봇 base_link 원점·방향(yaw: 작업대 x 축에서 로봇 x 축까지)."""

    def __init__(self, base_x: float, base_y: float, base_yaw_deg: float):
        self.t = np.array([base_x, base_y])
        y = math.radians(base_yaw_deg)
        self.yaw = y
        self.R = np.array([[math.cos(y), -math.sin(y)], [math.sin(y), math.cos(y)]])  # R → W 회전

    def w2r(self, p) -> np.ndarray:
        return (np.asarray(p, float) - self.t) @ self.R  # R^T (p - t)

    def r2w(self, p) -> np.ndarray:
        return np.asarray(p, float) @ self.R.T + self.t
