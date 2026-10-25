"""4단계: 모듈형(YOLO + 좌표 집기) — 장면마다 YOLO 검출로 sort_blocks(색, dry_run=True, max_blocks=1) 계획만 세운다.
로봇·카메라 없음(config.json 읽기만, 계획 기록은 robot/logs/n2/plan/).

    .venv/bin/python -m robot.compare.modular [--workers 4]     # → robot/logs/n2/modular.json
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

from robot.compare.common import OUT, ROOT, TARGETS

COLOR = {"red": "red", "green": "blue", "defect": "defect"}


def _one(job: tuple[str, str, str, list]) -> dict:
    os.nice(10)
    from robot.coord_pick.executor import sort_blocks
    sid, task, image, objects = job
    r = sort_blocks(COLOR[task], dry_run=True, detection={"objects": objects}, max_blocks=1, front_image=str(ROOT / image),
                    log_dir=OUT / "plan")
    first = (r.get("plan") or [{}])[0]
    return {"id": sid, "task": task, "ok": bool(r.get("ok")), "reason": r.get("reason"), "error": r.get("error"),
            "failures": r.get("failures", []), "violations": [v.get("kind") for v in r.get("violations", [])],
            "from_mm": first.get("from_mm"), "time_s": r.get("time_s")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m robot.compare.modular")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    det = json.loads((OUT / "det.json").read_text())
    scenes = json.loads((OUT / "scenes.json").read_text())["scenes"]
    jobs = []
    for s in scenes:
        objs = det[s["id"]]["yolo"]
        for task, cls in TARGETS.items():
            if any(o["zone"] == "input" and o["cls"] in cls for o in objs):
                jobs.append((s["id"], task, s["front"], objs))
    (OUT / "plan").mkdir(exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_one, jobs))
    (OUT / "modular.json").write_text(json.dumps(res, ensure_ascii=False))
    print(f"계획 {len(res)}건: 성공 {sum(r['ok'] for r in res)}")
    for r in res:
        if not r["ok"]:
            print(" ", r["id"], r["task"], r["reason"], r["error"], r["failures"][:2], r["violations"][:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
