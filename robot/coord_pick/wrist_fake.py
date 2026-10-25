"""손목 확인 실행(executor._sort_wrist)의 가짜 세계 — 시험용. 실제 송신·카메라 없이.

FakeWorld.install(ex) 가 executor._run_real(송신)·wrist.capture(손목 사진)를 바꾼다:
- 송신: 받은 궤적의 마지막 목표를 지금 자세로 둔다. "집고 들어 올림" 구간이면 tcp 가 진짜 블록에서 grip_ok_mm 안일 때만
  그리퍼 측정값 = 닫힘 + 2.5(쥠), 아니면 닫힘 + 0.1(빈손). fail_first 면 첫 집기는 미끄러진 것으로.
- 손목 사진: 지금 자세에서 진짜 블록이 보일 자리(wrist_ref 화면 모형)에 분홍/초록 사각형(가까울수록 크게)을 그린 640×480.
"""
from __future__ import annotations

import numpy as np

from robot.coord_pick import wrist


class FakeWorld:
    def __init__(self, s, true_w, color="red", *, grip_ok_mm=12.0, fail_first=False, px_gain=1.0, true_yaw_w=None,
                 aspect=1.5, hover_bias_px=(0.0, 0.0)):
        """px_gain: 실제 화면 감도가 화면 모형의 몇 배인지(10-22 17:29 실물 1.4~1.9배).
        true_yaw_w: 블록 긴 축 작업대 방위(rad, None = 정육면체). 있으면 집게축이 긴 축과 60° 이상 벌어져야(짧은 변을
        끼워야) 쥔다(10-22 실물: 세로 방향 블록을 못 잡음).
        hover_bias_px: 높은 곳(hover)에서만 화면 모형과 실제가 이만큼 어긋남(닫기 직전 높이에선 0) — 17:32 실물처럼
        hover 에서 맞춰도 내려가면 벗어나는 경우."""
        self.s, self.true_w, self.color, self.px_gain = s, np.asarray(true_w, float), color, px_gain
        self.true_yaw_w, self.aspect = true_yaw_w, aspect
        self.hover_bias = np.asarray(hover_bias_px, float)
        self.rolls: list = []
        self.grip_ok_mm, self.fail_first = grip_ok_mm, fail_first
        self.q = None
        self.sends, self.shots, self.grasps = [], 0, []
        ref = wrist.load_ref()
        self.sm = wrist.screen_model(ref)
        ds = "pick_red" if color == "red" else "pick_green"
        self.after = ref["after"]["by_dataset"][ds]
        c = ref["close"]["by_dataset"][ds]
        self.center_off = np.array([c["u_med"], c["v_med"]]) - self.sm.predict([[0.0, 0.0, 0.0]])[0]
        self.dropped = False

    def install(self, ex):
        self._orig = (ex._run_real, wrist.capture)
        ex._run_real = self.run_real
        wrist.capture = self.capture
        self.ex = ex

    def uninstall(self):
        self.ex._run_real, wrist.capture = self._orig

    def run_real(self, s, targets, marks, lim, log, *, name, **kw):
        last = np.array(targets[-1], float)
        self.q = s.n_to_q(last)
        r = np.asarray(targets, float)[:, 4] * 1.8
        self.rolls.append((name, float(r.min()), float(r.max())))
        meas = last.copy()
        if "집고" in name:
            tcp = s.tcp_pos(self.q)
            dist = float(np.linalg.norm(s.frames.r2w(tcp[:2]) - self.true_w))
            rho = None
            if self.true_yaw_w is not None:
                T = s.kin.fk(self.q)
                gam = np.arctan2(T[1, 0], T[0, 0])
                beta = self.true_yaw_w - s.frames.yaw
                rho = float(np.degrees((beta - gam + np.pi / 2) % np.pi - np.pi / 2))
            held = (dist < self.grip_ok_mm and not (self.fail_first and not self.grasps)
                    and (rho is None or abs(rho) >= 60))
            self.grasps.append({"dist_mm": round(dist, 1), "held": held, "rho_deg": None if rho is None else round(rho, 1)})
            meas[5] = last[5] + (2.5 if held else 0.1)
        if "놓기" in name or "홈으로(다시)" in name:
            self.dropped = True
        elif "hover" in name:
            self.dropped = False
        self.sends.append(name)
        return {"reason": "done", "run_id": f"fake-{len(self.sends)}",
                "run": {"end": {"reason": "done"}, "last_step": {"measured": meas.round(2).tolist()}}}

    def capture(self, dev=None):
        import cv2
        self.shots += 1
        img = np.full((480, 640, 3), 225, np.uint8)
        bgr = (200, 140, 230) if self.color == "red" else (150, 220, 160)
        if self.grasps and self.grasps[-1]["held"] and not self.dropped:  # 쥔 블록은 두 날 사이(녹화 "집은 뒤" 자리)
            a = self.after
            side = a["area_med"] ** 0.5
            cv2.rectangle(img, (int(a["u_med"] - side / 2), int(a["v_med"] - side / 2)),
                          (int(a["u_med"] + side / 2), int(a["v_med"] + side / 2)), bgr, -1)
            return img
        T = self.s.kin.fk(self.q)
        tcp = (T[:3, 3] + T[:3, :3] @ self.s.tcp) * 1000
        P = np.r_[self.s.frames.w2r(self.true_w), wrist.BLOCK_Z_MM]
        d = (P - tcp) @ T[:3, :3]
        p0 = (self.sm.predict([[0.0, 0.0, d[2]]])[0] + self.center_off  # 가운데면 보일 자리(녹화 닫기 직전 기준과 같은 점)
              + self.hover_bias * float(np.clip((d[2] - 10) / 23, 0, 1)))
        u, v = p0 + self.px_gain * (self.sm.predict([d])[0] - self.sm.predict([[0.0, 0.0, d[2]]])[0])
        side = 150 * 60 / (60 + max(0.0, d[2]))
        if self.true_yaw_w is None:
            cv2.rectangle(img, (int(u - side / 2), int(v - side / 2)), (int(u + side / 2), int(v + side / 2)), bgr, -1)
            return img
        # 긴 축: 로봇 좌표 방향 → 끝 좌표 → 화면(화면 모형 J) — 실제 카메라와 같은 대응
        beta = self.true_yaw_w - self.s.frames.yaw
        t = T[:3, :3].T @ np.array([np.cos(beta), np.sin(beta), 0.0])
        dz = d[2]
        J = np.array([[self.sm.cu[1] + self.sm.cu[4] * dz, self.sm.cu[2] + self.sm.cu[5] * dz],
                      [self.sm.cv[1] + self.sm.cv[4] * dz, self.sm.cv[2] + self.sm.cv[5] * dz]])
        a = J @ t[:2]
        ang = float(np.degrees(np.arctan2(a[1], a[0])))
        short = side / np.sqrt(self.aspect)
        box = cv2.boxPoints(((float(u), float(v)), (short * self.aspect, short), ang)).astype(np.int32)
        cv2.fillConvexPoly(img, box, bgr)
        return img
