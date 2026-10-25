"""5단계: 판정. 앞 단계 결과(scenes·det·truth_fix·jobs_*·infer_*·modular)를 읽어 숫자를 낸다 → robot/logs/n2/results.json

    .venv/bin/python -m robot.compare.evaluate

예측 집기 자리
- ACT: 첫 묶음(100스텝 = 3.3초) 안에서 그리퍼가 열렸다 닫히기 시작한 순간(집게 가운데 높이 60mm 아래)의 관절 → FK →
  집게 가운데 작업대 좌표. 없으면 첫 묶음 끝 자세. 두 번째 묶음부터는 사진이 첫 장면 그대로라 정책이 '아직 홈'으로 보고
  그리퍼를 되돌리는 가짜 닫힘이 생겨(10-22 밤 실측) 쓰지 않는다. 학습 녹화 장면에서 첫 묶음 끝과 사람이 집은 자리의
  거리 중앙 40mm.
- SmolVLA: 묶음이 50스텝(1.7초)이고 실물 첫 닫힘이 8.1초라, 묶음을 이어 붙인 8.3초 안의 닫힘, 없으면 끝 자세.

정답(목표 블록)
- 녹화 장면: 사람이 그 에피소드에서 실제로 집은 자리(그리퍼 닫힘 시작의 FK, r14 grasps.json) — 블록 라벨이 필요 없다.
- 사진·실물 장면: 투입 구역 블록(known4·front = YOLO, 실물 직전 사진 = 그때 그려진 박스 되읽기 + 손 고침) 중 목표 색에서
  로봇에 가장 가까운 것(녹화 때 사람의 순서). 예측 자리에서 가장 가까운 블록이 목표 색이고 70mm 안이면 정답(HIT_MM).
"""
from __future__ import annotations

import json

import numpy as np

from robot.compare.common import OUT, POLICY_TASK, TARGETS, TASK_OF_DS, blocks, closing, scene_kind, setup, tcp_w
from robot.compare.infer import SUB_GRIPS, mirror_px

HIT_MM = 70.0        # 어느 블록으로 갔나 판정 한도. 오프라인 예측(열린 고리 첫 묶음) 자리 오차가 학습 장면 중앙 40mm·75% 51mm,
                     # 실물 첫 닫힘도 블록 중심에서 25~55mm(첫 시도가 블록 위에서 닫고 다시 집음) — 블록 간격은 40~60mm
SAME_MM = 40.0       # 사람이 집은 자리와 이만큼 안이면 '같은 블록을 향함'
APART_MM = 60.0      # 두 목표가 이만큼 떨어진 짝만 '어느 쪽을 따라가나'를 센다
SKIP_PHOTO = {"front_174916"}  # 팔이 화면 안(홈 자세 아님)


def pred(policy: str, job: dict, z, n: int) -> tuple[np.ndarray, bool]:
    seq = z[f"{n}_first"] if policy.startswith("act") else z[f"{n}_stream"]
    i = closing(seq[:, 5], float(job["state"][5]))
    if i is not None:
        p = tcp_w(seq[i])
        if p[2] < 60:
            return p, True
    return tcp_w(seq[-1]), False


def mirror_w(w, xc: float) -> np.ndarray:
    """작업대 점 → 위 카메라 px → 테이프 안쪽 가운데(xc) 기준 좌우 반전 → 작업대."""
    s = setup()
    return s.px_to_w(mirror_px(s.w_to_px(np.asarray(w, float))[0], xc))[0]


def corr(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 2)


def spread(P) -> dict:
    P = np.asarray(P, float)[:, :2]
    return {"sx": round(float(P[:, 0].std()), 1), "sy": round(float(P[:, 1].std()), 1),
            "s": round(float(np.sqrt(P.var(0).sum())), 1)}


def target_of(bl: list[dict], task: str) -> dict | None:
    tg = [b for b in bl if b["cls"] in TARGETS[task]]
    return tg[0] if tg else None


def judge_photo(p, bl, task) -> dict:
    if not bl:
        return {"ok": False, "air": True, "nearest_robot": False, "hit": None, "d": None}
    d = [float(np.linalg.norm(np.asarray(b["w"]) - p[:2])) for b in bl]
    k = int(np.argmin(d))
    air = d[k] > HIT_MM
    return {"ok": (not air) and bl[k]["cls"] in TARGETS[task], "air": air, "nearest_robot": (not air) and k == 0,
            "hit": None if air else bl[k]["cls"], "hit_k": k, "d": round(d[k], 1)}


def policy_eval(policy: str, tb: dict) -> dict:
    sets = [(json.loads((OUT / f"jobs_{policy}{t}.json").read_text()), np.load(OUT / f"infer_{policy}{t}.npz"))
            for t in [""] + [f"_{x.stem.split(policy + '_', 1)[1]}" for x in sorted(OUT.glob(f"jobs_{policy}_*.json"))]]
    task = POLICY_TASK[policy]
    rows = []
    for J, Z in sets:
        for n, j in enumerate(J):
            p, closed = pred(policy, j, Z, n)
            rows.append({**{k: j.get(k) for k in ("group", "id", "ds", "human_w", "swap", "swap_human_w", "sub", "mirror_xc")},
                         "p": p.round(1).tolist(), "closed": closed, "lat_ms": round(float(Z[f"{n}_lat"].mean()) * 1000)})
    own = {r["id"]: r for r in rows if r["group"] == "own"}
    mine = [r for r in own.values() if TASK_OF_DS[r["ds"]] == task and r["human_w"]]
    out: dict = {"policy": policy, "task": task, "lat_ms": int(np.median([r["lat_ms"] for r in rows]))}
    # 1) 학습 녹화 장면: 사람이 집은 자리 vs 예측
    P = np.array([r["p"] for r in mine])
    T = np.array([r["human_w"] for r in mine])
    D = np.linalg.norm(P[:, :2] - T, axis=1)
    out["own"] = {"n": len(mine), "d_med": round(float(np.median(D))), "d_p75": round(float(np.percentile(D, 75))),
                  "same": int((D <= SAME_MM).sum()), "rx": corr(P[:, 0], T[:, 0]), "ry": corr(P[:, 1], T[:, 1]),
                  "pred": spread(P), "target": spread(T), "closed": sum(r["closed"] for r in mine),
                  "points": P[:, :2].round(1).tolist(), "targets": T.round(1).tolist()}
    allrec = [r for r in own.values()]
    out["all_rec_pred"] = spread([r["p"] for r in allrec])
    out["by_id"] = {}
    for r in rows:
        if r["group"] in ("own", "mirror", "empty", "photo_mirror") or (r["group"] == "photo" and r["sub"] == SUB_GRIPS[0]):
            out["by_id"].setdefault(r["id"], {})[r["group"]] = r["p"]
            if r["group"] == "mirror":
                out["by_id"][r["id"]]["mirror_xc"] = r["mirror_xc"]
    # 2) 좌우 반전: 예측이 반전된 목표를 따라가나
    mir = {r["id"]: r for r in rows if r["group"] == "mirror"}
    dp, dt, follow, cnt, mpts = [], [], 0, 0, []
    for r in mine:
        m = mir.get(r["id"])
        if not m:
            continue
        t0, t1 = np.asarray(r["human_w"]), mirror_w(r["human_w"], m["mirror_xc"])
        dp.append(m["p"][0] - r["p"][0])
        dt.append(t1[0] - t0[0])
        mpts.append(m["p"][:2])
        if np.linalg.norm(t1 - t0) > APART_MM:
            cnt += 1
            follow += np.linalg.norm(np.asarray(m["p"][:2]) - t1) < np.linalg.norm(np.asarray(m["p"][:2]) - t0)
    dp, dt = np.array(dp), np.array(dt)
    out["mirror"] = {"n": len(dp), "slope": round(float((dp * dt).sum() / max((dt * dt).sum(), 1e-9)), 2),
                     "moved_med": round(float(np.median(np.abs(dp)))), "target_moved_med": round(float(np.median(np.abs(dt)))),
                     "follow": int(follow), "pairs": cnt, "points": np.round(mpts, 1).tolist()}
    # 3) 맞바꿈: swap = 손목 사진만, bgswap = 위 사진의 교실 배경(작업대 너머)만, inswap = 위 사진의 테이프 안쪽(블록)만
    #    짝 장면 것. 예측이 이 장면의 목표와 짝 장면의 목표 중 어느 쪽으로 가나(inswap 은 짝 = 블록 쪽)
    for g in ("swap", "bgswap", "inswap"):
        sw = [r for r in rows if r["group"] == g and TASK_OF_DS[r["ds"]] == task and r["human_w"] and r["swap_human_w"]]
        f_this = f_other = pairs = 0
        for r in sw:
            a, b = np.asarray(r["human_w"]), np.asarray(r["swap_human_w"])
            if np.linalg.norm(a - b) <= APART_MM:
                continue
            pairs += 1
            p = np.asarray(r["p"][:2])
            if np.linalg.norm(p - a) < np.linalg.norm(p - b):
                f_this += 1
            else:
                f_other += 1
        out[g] = {"pairs": pairs, "follow_this": f_this, "follow_other": f_other}
    # 4) 블록 지움: 그래도 같은 자리로 가서 닫나
    em = [(r, own[r["id"]]) for r in rows if r["group"] == "empty" and r["id"] in own
          and TASK_OF_DS[r["ds"]] == task and r["human_w"]]
    if em:
        de = [float(np.linalg.norm(np.asarray(e["p"][:2]) - np.asarray(o["p"][:2]))) for e, o in em]
        dh = [float(np.linalg.norm(np.asarray(e["p"][:2]) - np.asarray(o["human_w"]))) for e, o in em]
        out["empty"] = {"n": len(em), "closed": sum(e["closed"] for e, _ in em),
                        "low": sum(e["p"][2] < 40 for e, _ in em), "d_from_own_med": round(float(np.median(de))),
                        "d_from_target_med": round(float(np.median(dh))), "points": [e["p"][:2] for e, _ in em]}
    # 4b) 블록 없는 실제 작업대(calibrate) + 각 장면의 손목 사진·관절값: 예측이 손목 사진 에피소드의 목표로 가나
    cb = [(r, own.get(r["id"])) for r in rows if r["group"] == "calib" and TASK_OF_DS[r["ds"]] == task and r["human_w"]]
    if cb:
        P = np.array([r["p"] for r, _ in cb])
        T = np.array([r["human_w"] for r, _ in cb])
        D = np.linalg.norm(P[:, :2] - T, axis=1)
        do = [float(np.linalg.norm(np.asarray(r["p"][:2]) - np.asarray(o["p"][:2]))) for r, o in cb if o]
        out["calib"] = {"n": len(cb), "rx": corr(P[:, 0], T[:, 0]), "ry": corr(P[:, 1], T[:, 1]),
                        "d_med": round(float(np.median(D))), "same": int((D <= SAME_MM).sum()),
                        "d_from_own_med": round(float(np.median(do))) if do else None, "pred": spread(P),
                        "closed": sum(r["closed"] for r, _ in cb), "low": int((P[:, 2] < 40).sum())}
    # 5) 사진 장면(known4·front): 색 정답
    ph = [r for r in rows if r["group"] == "photo" and r["id"] not in SKIP_PHOTO]
    res = []
    for r in ph:
        bl = tb[r["id"]]
        tg = target_of(bl, task)
        if tg is None:
            continue
        j = judge_photo(np.asarray(r["p"]), bl, task)
        res.append({**r, **j, "kind": scene_kind(bl, task), "target_w": tg["w"]})
    by_sub: dict = {}
    for r in res:
        by_sub.setdefault(r["sub"], []).append(r)
    subs = list(by_sub)
    same = 0
    ids = sorted({r["id"] for r in res})
    for sid in ids:
        hits = {(r["hit"], r.get("hit_k")) for r in res if r["id"] == sid}
        same += len(hits) == 1
    first = by_sub[subs[0]] if subs else []
    out["photo"] = {"n_scenes": len(ids), "subs": subs, "same_choice_all_subs": same,
                    "ok": sum(r["ok"] for r in res), "n": len(res),
                    "easy": [sum(r["ok"] for r in res if r["kind"] == "쉬움"), sum(r["kind"] == "쉬움" for r in res)],
                    "hard": [sum(r["ok"] for r in res if r["kind"] == "방해"), sum(r["kind"] == "방해" for r in res)],
                    "nearest": sum(r["nearest_robot"] for r in res), "air": sum(r["air"] for r in res),
                    "rx": corr([r["p"][0] for r in first], [r["target_w"][0] for r in first]),
                    "ry": corr([r["p"][1] for r in first], [r["target_w"][1] for r in first]),
                    "pred": spread([r["p"] for r in first]) if first else None,
                    "target": spread([r["target_w"] for r in first]) if first else None,
                    "points": [r["p"][:2] for r in first], "targets": [r["target_w"] for r in first],
                    "ok_flags": [r["ok"] for r in first]}
    # 사진 좌우 반전: 목표가 옮겨 간 만큼 예측도 옮겨 가나(기울기 1 = 다 따라감, 0 = 안 움직임). '반전 목표에 더
    # 가까워졌나' 개수는 한 점에 머무는 예측도 우연히 맞히므로 기울기로 본다
    pm = {r["id"]: r for r in rows if r["group"] == "photo_mirror"}
    base = {r["id"]: r for r in first}
    dp, dt = [], []
    for sid, m in pm.items():
        tg = target_of(tb[sid], task) if sid in base else None
        if tg is None:
            continue
        dp.append(m["p"][0] - base[sid]["p"][0])
        dt.append(mirror_w(tg["w"], m["mirror_xc"])[0] - tg["w"][0])
    dp, dt = np.array(dp), np.array(dt)
    out["photo_mirror"] = {"n": len(dp), "slope": round(float((dp * dt).sum() / max((dt * dt).sum(), 1e-9)), 2) if len(dp) else None,
                           "moved_med": round(float(np.median(np.abs(dp)))) if len(dp) else None,
                           "target_moved_med": round(float(np.median(np.abs(dt)))) if len(dt) else None}
    return out


def real_eval(real: list[dict], tb: dict) -> list[dict]:
    rows = []
    for r in real:
        if r["id"] not in tb:
            continue
        task = POLICY_TASK[r["policy"]]
        bl = tb[r["id"]]
        q = np.array([x[1] for x in r["ticks"]], float)
        i = closing(q[:, 5])
        p = tcp_w(q[i]) if i is not None else None
        tg = target_of(bl, task)
        j = judge_photo(p, bl, task) if p is not None else {"ok": False, "air": False, "nearest_robot": False,
                                                            "hit": None, "d": None}
        rows.append({"id": r["id"], "start": r["start"], "policy": r["policy"], "task": task,
                     "kind": scene_kind(bl, task), "closed": i is not None, "t_close": None if i is None else
                     r["ticks"][i][0], "p": None if p is None else p.round(1).tolist(),
                     "target_w": tg["w"] if tg else None, **j})
    return rows


def modular_eval(scenes: dict, tb: dict, det: dict) -> dict:
    plans = {(r["id"], r["task"]): r for r in json.loads((OUT / "modular.json").read_text())}
    out = {"rec": {}, "photo": {}, "real": {}}
    for task in TARGETS:
        # 녹화 장면(YOLO 가 학습에 안 본 사진): YOLO 가 고른 블록 = 사람이 집은 블록인가. 블록 단위로 짝짓는다:
        # 사람 블록 = YOLO·HSV 합집합 블록 중 사람이 닫은 자리(FK)에 가장 가까운 것(50mm 안), 고른 블록 = 계획의 첫 블록
        R = [s for s in scenes.values() if s["set"] == "rec" and TASK_OF_DS[s["ds"]] == task and s.get("human_w")]
        same = plan_ok = none = nohuman = 0
        diff_ids = []
        for s in R:
            yb = [b for b in blocks(det[s["id"]]["yolo"]) if b["cls"] in TARGETS[task]]
            p = plans.get((s["id"], task))
            plan_ok += bool(p and p["ok"])
            if not yb:
                none += 1
                diff_ids.append(s["id"])
                continue
            uw = _union_w(det[s["id"]])
            dh = np.linalg.norm(uw - np.asarray(s["human_w"]), axis=1)
            if dh.min() > 50:
                nohuman += 1
                continue
            w = np.asarray(p["from_mm"] if p and p.get("from_mm") else yb[0]["w"])
            if int(np.argmin(np.linalg.norm(uw - w, axis=1))) == int(np.argmin(dh)):
                same += 1
            else:
                diff_ids.append(s["id"])
        out["rec"][task] = {"n": len(R), "same_as_human": same, "no_target_found": none, "human_block_unclear": nohuman,
                            "plan_ok": plan_ok, "diff": diff_ids}
        # 사진 장면: 목표 블록을 골랐나 + 계획
        P = [s for s in scenes.values() if s["set"] in ("known4", "front") and target_of(tb[s["id"]], task)]
        ok = pok = 0
        kinds = {"쉬움": [0, 0], "방해": [0, 0]}
        for s in P:
            p = plans.get((s["id"], task))
            tg = target_of(tb[s["id"]], task)
            good = bool(p and p.get("from_mm") and np.linalg.norm(np.asarray(p["from_mm"]) - tg["w"]) < 15)
            ok += good
            pok += bool(p and p["ok"])
            k = scene_kind(tb[s["id"]], task)
            kinds[k][0] += good and bool(p["ok"])
            kinds[k][1] += 1
        out["photo"][task] = {"n": len(P), "pick_ok": ok, "plan_ok": pok, "easy": kinds["쉬움"], "hard": kinds["방해"]}
    fails = [r for r in plans.values() if not r["ok"]]
    out["plan_total"] = {"n": len(plans), "ok": len(plans) - len(fails),
                         "why": sorted({(r["failures"] or r["violations"] or [r["reason"]])[0] for r in fails})[:8]}
    return out


def _union_w(d: dict) -> np.ndarray:
    """투입 구역 블록 자리(작업대 mm) — YOLO·HSV 합집합, 15px 안은 같은 블록."""
    s = setup()
    uniq: list[dict] = []
    for o in [o for o in d["yolo"] + d["hsv"] if o.get("zone") == "input"]:
        if all(np.hypot(o["px"][0] - u["px"][0], o["px"][1] - u["px"][1]) > 15 for u in uniq):
            uniq.append(o)
    return np.array([s.px_to_w(o["px"])[0] for o in uniq]) if uniq else np.zeros((0, 2))


def human_eval(scenes: dict, det: dict) -> dict:
    """녹화에서 사람이 집은 블록이 로봇에 가장 가까운 블록이었나(블록 자리 = YOLO·HSV 합집합, 15px 안은 같은 블록)."""
    s = setup()
    out = {}
    for ds, task in TASK_OF_DS.items():
        R = [x for x in scenes.values() if x["set"] == "rec" and x["ds"] == ds and x.get("human_w")]
        near = valid = near_color = 0
        pts, others, flags, cflags = [], [], [], []
        for x in R:
            objs = [o for o in det[x["id"]]["yolo"] + det[x["id"]]["hsv"] if o.get("zone") == "input"]
            uniq: list[dict] = []
            for o in objs:
                if all(np.hypot(o["px"][0] - u["px"][0], o["px"][1] - u["px"][1]) > 15 for u in uniq):
                    uniq.append(o)
            bl = [{"w": s.px_to_w(o["px"])[0], "color": o["cls"].split("_")[0]} for o in uniq]
            for b in bl:
                b["r"] = float(np.linalg.norm(s.frames.w2r(b["w"])))
            bl.sort(key=lambda b: b["r"])
            h = np.asarray(x["human_w"])
            d = [float(np.linalg.norm(b["w"] - h)) for b in bl]
            pts.append(h.round(1).tolist())
            if not bl or min(d) > 50:
                flags.append(None)
                cflags.append(None)
                continue
            valid += 1
            k = int(np.argmin(d))
            near += k == 0
            col = bl[k]["color"]
            same_col = [i for i, b in enumerate(bl) if b["color"] == col]
            near_color += k == same_col[0]
            flags.append(k == 0)
            cflags.append(k == same_col[0])
            others.extend(b["w"].round(1).tolist() for i, b in enumerate(bl) if i != k)
        out[ds] = {"n": len(R), "valid": valid, "nearest": near, "nearest_same_color": near_color, "points": pts,
                   "nearest_flags": flags, "nearest_color_flags": cflags, "others": others}
    return out


def main() -> int:
    doc = json.loads((OUT / "scenes.json").read_text())
    scenes = {s["id"]: s for s in doc["scenes"]}
    det = json.loads((OUT / "det.json").read_text())
    fx = OUT / "truth_fix.json"
    fix = {k: v for k, v in (json.loads(fx.read_text()) if fx.exists() else {}).items() if not k.startswith("_")}
    tb = {sid: blocks(fix.get(sid, d["yolo"])) for sid, d in det.items()}
    for sid in fix:
        tb.setdefault(sid, blocks(fix[sid]))
    res = {"policies": {}, "human": human_eval(scenes, det), "real": real_eval(doc["real"], tb)}
    # 실물: 첫 닫힘 자리의 퍼짐·목표와의 상관(정책별)
    res["real_summary"] = {}
    for pol in POLICY_TASK:
        R = [r for r in res["real"] if r["policy"] == pol and r["p"] is not None and r["target_w"] is not None]
        if R:
            P = np.array([r["p"][:2] for r in R])
            T = np.array([r["target_w"] for r in R])
            res["real_summary"][pol] = {"n": len(R), "pred": spread(P), "target": spread(T), "rx": corr(P[:, 0], T[:, 0]),
                                        "ry": corr(P[:, 1], T[:, 1]), "points": P.round(1).tolist(),
                                        "targets": T.round(1).tolist()}
    for pol in POLICY_TASK:
        if (OUT / f"infer_{pol}.npz").exists():
            res["policies"][pol] = policy_eval(pol, tb)
    if (OUT / "modular.json").exists():
        res["modular"] = modular_eval(scenes, tb, det)
    res["truth"] = {k: v for k, v in tb.items()}
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, default=float))
    for pol, r in res["policies"].items():
        print(pol, {k: v for k, v in r["own"].items() if k not in ("points", "targets")})
        print("   mirror", {k: v for k, v in r["mirror"].items() if k != "points"}, "swap", r["swap"], "bgswap", r["bgswap"], "inswap", r["inswap"])
        print("   empty", {k: v for k, v in r.get("empty", {}).items() if k != "points"}, "calib", r.get("calib"))
        print("   photo", {k: v for k, v in r["photo"].items() if k not in ("points", "targets", "ok_flags")},
              "photo_mirror", r["photo_mirror"])
    for r in res["real"]:
        print("real", r["start"], r["policy"], r["kind"], "ok" if r["ok"] else "x", r["hit"], r["d"],
              "nearest" if r["nearest_robot"] else "")
    for pol, x in res["real_summary"].items():
        print("real_summary", pol, {k: v for k, v in x.items() if k not in ("points", "targets")})
    for ds, h in res["human"].items():
        print("human", ds, {k: v for k, v in h.items() if k in ("n", "valid", "nearest", "nearest_same_color")})
    if "modular" in res:
        m = res["modular"]
        print("modular rec", {t: {k: v for k, v in x.items() if k != "diff"} for t, x in m["rec"].items()})
        print("modular photo", m["photo"], "plan", m["plan_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
