"""6단계: 발표용 그림(PNG, 밝은 배경·한글 축). results.json 을 읽는다.

    .venv/bin/python -m robot.compare.report      # → docs/compare_20261022/fig1~4_*.png
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from robot.compare.common import OUT, ROOT, setup  # noqa: E402
from robot.coord_pick.plan import zones_w  # noqa: E402

DOCS = ROOT / "docs" / "compare_20261022"
FONT = next((f for f in ("NanumSquare", "Noto Sans CJK JP", "DejaVu Sans")
             if any(x.name == f for x in font_manager.fontManager.ttflist)), "DejaVu Sans")
plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False, "figure.facecolor": "white",
                     "axes.facecolor": "white", "savefig.facecolor": "white", "font.size": 11})
ZONE_KO = {"input": "투입 구역", "bowl_red": "분홍 그릇", "bowl_blue": "초록 그릇", "bowl_defect": "불량 그릇",
           "hand_spot": "손 자리"}
DS_KO = {"pick_red": "분홍 옮기기", "pick_green": "초록 옮기기", "pick_defect": "불량 골라내기"}
POL_KO = {"act_red": "ACT 분홍", "smolvla_red": "SmolVLA 분홍", "act_green": "ACT 초록", "act_defect": "ACT 불량"}
POL_DS = {"act_red": "pick_red", "smolvla_red": "pick_red", "act_green": "pick_green", "act_defect": "pick_defect"}
TASK_COL = {"pick_red": "#e8559a", "pick_green": "#2e9e4f", "pick_defect": "#7b2d8b"}


def _table(ax, zw, s, *, xlim=(-20, 340), ylim=(-250, 200)):
    for k, poly in zw.items():
        p = np.vstack([poly, poly[:1]])
        ax.plot(p[:, 0], p[:, 1], color="#f08c1a" if k == "input" else "#a0a0a0", lw=2 if k == "input" else 1,
                ls="-" if k == "input" else "--", zorder=1)
        if k != "input":
            c = poly.mean(0)
            if xlim[0] < c[0] < xlim[1] and ylim[0] < c[1] < ylim[1]:
                ax.text(c[0], c[1], ZONE_KO[k], ha="center", va="center", color="#888", fontsize=9, zorder=1)
    b = s.frames.t
    if ylim[0] < b[1] < ylim[1]:
        ax.plot(*b, marker="s", ms=12, color="#444", zorder=3)
        ax.text(b[0], b[1] + 14, "로봇", ha="center", va="bottom", fontsize=10, color="#444")
    else:  # 로봇 받침이 그림 아래 밖 — 방향만 표시
        ax.text(b[0], ylim[0] + 6, f"▼ 로봇(y {b[1]:.0f})", ha="center", va="bottom", fontsize=9, color="#444")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("작업대 x (mm)")
    ax.set_ylabel("작업대 y (mm)")
    ax.grid(alpha=0.25)


def fig_recordings(res, zw, s):
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.6))
    for ax, ds in zip(axes, ("pick_red", "pick_green", "pick_defect")):
        h = res["human"][ds]
        _table(ax, zw, s, ylim=(-250, 200))
        if h["others"]:
            o = np.array(h["others"])
            ax.scatter(o[:, 0], o[:, 1], s=12, color="#cfcfcf", zorder=2, label="그 장면의 다른 블록")
        P = np.array(h["points"])
        F = h["nearest_color_flags"]
        near = np.array([p for p, f in zip(P, F) if f])
        far = np.array([p for p, f in zip(P, F) if f is False])
        if len(near):
            ax.scatter(near[:, 0], near[:, 1], s=60, color=TASK_COL[ds], edgecolors="black", linewidths=0.6, zorder=4,
                       label=f"같은 색 중 로봇에 가장 가까운 것 {len(near)}")
        if len(far):
            ax.scatter(far[:, 0], far[:, 1], s=70, marker="X", color="white", edgecolors=TASK_COL[ds], linewidths=1.6,
                       zorder=4, label=f"더 가까운 같은 색이 있었음 {len(far)}")
        v = max(h["valid"], 1)
        ax.set_title(f"{DS_KO[ds]} 녹화 {h['n']}개 — 사람이 집은 블록이\n같은 색 중 가장 가까움 "
                     f"{h['nearest_same_color']}/{h['valid']} ({100 * h['nearest_same_color'] / v:.0f}%) · "
                     f"모든 블록 중 가장 가까움 {h['nearest']}/{h['valid']} ({100 * h['nearest'] / v:.0f}%)", fontsize=11)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=9, frameon=False)
    fig.suptitle("녹화 데이터: 사람이 집은 블록 자리(작업대 좌표, 로봇은 아래)", fontsize=14)
    fig.tight_layout()
    fig.savefig(DOCS / "fig1_recordings.png", dpi=130)
    plt.close(fig)


def fig_policy_points(res, zw, s):
    pols = [p for p in ("act_red", "smolvla_red", "act_green", "act_defect") if p in res["policies"]]
    fig, axes = plt.subplots(1, len(pols), figsize=(5.2 * len(pols), 7.2))
    for ax, pol in zip(np.atleast_1d(axes), pols):
        r = res["policies"][pol]
        ds = POL_DS[pol]
        _table(ax, zw, s, ylim=(-120, 200))
        T = np.array(r["own"]["targets"])
        ax.scatter(T[:, 0], T[:, 1], s=110, color="#ffd27a", alpha=0.7, zorder=2, label="정답: 사람이 집은 자리(녹화)")
        P = np.array(r["own"]["points"])
        ax.scatter(P[:, 0], P[:, 1], s=30, color="#1c7ed6", zorder=4, label="정책 예측(같은 녹화 장면)")
        ph = r["photo"]
        if ph["points"]:
            Q = np.array(ph["points"])
            ok = np.array(ph["ok_flags"], bool)
            ax.scatter(Q[ok, 0], Q[ok, 1], s=45, marker="o", facecolors="none", edgecolors="#2b8a3e", linewidths=1.6,
                       zorder=5, label=f"새 사진 예측 — 정답 {ok.sum()}")
            ax.scatter(Q[~ok, 0], Q[~ok, 1], s=45, marker="x", color="#c92a2a", linewidths=1.6, zorder=5,
                       label=f"새 사진 예측 — 오답·허공 {(~ok).sum()}")
        rs = res.get("real_summary", {}).get(pol)
        if rs:
            Rp = np.array(rs["points"])
            ax.scatter(Rp[:, 0], Rp[:, 1], s=70, marker="^", color="#212529", edgecolors="white", linewidths=0.8,
                       zorder=6, label=f"실물 첫 닫힘 {len(Rp)}회")
        o = r["own"]
        q = ph["pred"] or {"s": 0}
        t = ph["target"] or {"s": 0}
        ax.set_title(f"{POL_KO[pol]}\n녹화 장면 상관 x {o['rx']} · y {o['ry']} | 새 사진 예측 퍼짐 {q['s']:.0f}mm "
                     f"(정답 {t['s']:.0f}mm)", fontsize=11)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5, frameon=False)
    fig.suptitle("정책이 집으러 간 자리(예측) vs 정답 — 예측이 한 곳에 몰리면 영상을 덜 본다", fontsize=14)
    fig.tight_layout()
    fig.savefig(DOCS / "fig2_policy_points.png", dpi=130)
    plt.close(fig)


def fig_mirror_example(res, s):
    """같은 녹화 장면의 원본과 투입 구역 좌우 반전 사진에 정책 예측 집기 자리를 찍는다."""
    pols = [p for p in ("act_red", "smolvla_red") if p in res["policies"]]
    if not pols:
        return
    r0 = res["policies"][pols[0]]
    scenes = {x["id"]: x for x in json.loads((OUT / "scenes.json").read_text())["scenes"]}
    det = json.loads((OUT / "det.json").read_text())
    from robot.compare.evaluate import mirror_w
    best, gap = None, 0.0
    for sid, v in r0["by_id"].items():
        sc = scenes.get(sid)
        if not sc or sc["set"] != "rec" or sc["ds"] != "pick_red" or "mirror" not in v or "own" not in v:
            continue
        if not all(sid in res["policies"][p]["by_id"] and "mirror" in res["policies"][p]["by_id"][sid] for p in pols):
            continue
        g = float(np.linalg.norm(mirror_w(sc["human_w"], v["mirror_xc"]) - np.asarray(sc["human_w"])))
        tx = float(s.w_to_px(sc["human_w"])[0][0])
        if abs(tx - v["mirror_xc"]) > 60:  # 목표가 테이프 가장자리면 반전 사진에서 잘린다 — 안쪽 예시만
            continue
        # 반전 사진에서 예측 자리에 다른 블록이 있으면 '우연히 맞음'처럼 보인다 — 예측 자리가 빈 장면만
        mb = [s.px_to_w([2 * v["mirror_xc"] - o["px"][0], o["px"][1]])[0] for o in det[sid]["yolo"]
              if o.get("zone") == "input"]
        pm = np.asarray(v["mirror"][:2])
        if mb and min(float(np.linalg.norm(b - pm)) for b in mb) < 50:
            continue
        if g > gap:
            best, gap = sid, g
    if best is None:
        return
    sc = scenes[best]
    imgs = {"own": cv2.imread(str(ROOT / sc["front"])), "mirror": cv2.imread(str(OUT / "scenes" / f"{best}_mirror.jpg"))}
    tgt = {"own": np.asarray(sc["human_w"]), "mirror": mirror_w(sc["human_w"], r0["by_id"][best]["mirror_xc"])}
    marks = {"act_red": ("o", "#1c7ed6"), "smolvla_red": ("D", "#f08c00")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, key, title in zip(axes, ("own", "mirror"), ("원본 녹화 장면", "투입 구역만 좌우 반전(블록 자리만 바뀜)")):
        img = imgs[key]
        ax.imshow(cv2.cvtColor(img[120:300, 100:380], cv2.COLOR_BGR2RGB), extent=(100, 380, 300, 120))
        t = s.w_to_px(tgt[key])[0]
        ax.scatter([t[0]], [t[1]], s=380, facecolors="none", edgecolors="#ffd000", linewidths=3, label="정답 블록(사람이 집은 것)")
        for pol in pols:
            p = res["policies"][pol]["by_id"][best][key]
            q = s.w_to_px(p[:2])[0]
            mk, c = marks[pol]
            ax.scatter([q[0]], [q[1]], s=120, marker=mk, color=c, edgecolors="white", linewidths=1.2,
                       label=f"{POL_KO[pol]} 예측")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"좌우 반전 시험 예시({best}) — 영상을 보면 예측도 따라 움직여야 한다", fontsize=13)
    fig.tight_layout()
    fig.savefig(DOCS / "fig3_mirror_example.png", dpi=130)
    plt.close(fig)


def fig_accuracy(res):
    """사진 장면(known4·front) + 실물 실행에서 목표 블록을 집으러 간 비율 — 쉬움/방해."""
    rows = []
    for pol in ("act_red", "smolvla_red", "act_green", "act_defect"):
        if pol in res["policies"]:
            ph = res["policies"][pol]["photo"]
            rows.append((f"{POL_KO[pol]}\n(사진·오프라인)", ph["easy"], ph["hard"]))
        R = [r for r in res["real"] if r["policy"] == pol]
        if R:
            e = [sum(r["ok"] for r in R if r["kind"] == "쉬움"), sum(r["kind"] == "쉬움" for r in R)]
            h = [sum(r["ok"] for r in R if r["kind"] == "방해"), sum(r["kind"] == "방해" for r in R)]
            rows.append((f"{POL_KO[pol]}\n(실물)", e, h))
    mod = res.get("modular", {}).get("photo", {})
    for task, ko in (("red", "분홍"), ("green", "초록"), ("defect", "불량")):
        if task in mod:
            rows.append((f"YOLO+좌표 {ko}\n(사진·계획)", mod[task]["easy"], mod[task]["hard"]))
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(12, 1.5 * len(rows)), 5.8))
    pe = [100 * e[0] / e[1] if e[1] else 0 for _, e, _ in rows]
    phd = [100 * h[0] / h[1] if h[1] else 0 for _, _, h in rows]
    b1 = ax.bar(x - 0.2, pe, 0.4, color="#74b9ff", label="쉬움: 로봇에 가장 가까운 블록이 목표")
    b2 = ax.bar(x + 0.2, phd, 0.4, color="#e17055", label="방해: 목표보다 가까운 다른 블록이 있음")
    for bars, idx in ((b1, 1), (b2, 2)):
        for bar, row in zip(bars, rows):
            v = row[idx]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{v[0]}/{v[1]}" if v[1] else "-",
                    ha="center", fontsize=9)
    ax.set_xticks(x, [r[0] for r in rows], fontsize=9.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("목표 블록을 집으러 간 비율 (%)")
    ax.set_title("새 장면에서 목표 블록을 집으러 간 비율 — 쉬움(가장 가까운 블록이 목표) / 방해(더 가까운 다른 블록)",
                 fontsize=12.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=10, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(DOCS / "fig4_accuracy.png", dpi=130)
    plt.close(fig)


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    res = json.loads((OUT / "results.json").read_text())
    s = setup()
    zw = zones_w(s)
    fig_recordings(res, zw, s)
    fig_policy_points(res, zw, s)
    fig_mirror_example(res, s)
    fig_accuracy(res)
    print(sorted(p.name for p in DOCS.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
