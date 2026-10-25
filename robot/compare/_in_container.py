"""Studio 컨테이너 안에서 도는 오프라인 정책 추론(읽기만 — 컨테이너에 파일을 쓰지 않는다). infer.py 가 이 파일을
`python -I -c` 로 넘기고, 사진·작업 묶음(npz)을 stdin 으로 주고 결과(npz)를 stdout 으로 받는다.

    argv: <export_dir> <task 또는 -> <스레드 수> <최대 스텝>

작업 = (위 사진 번호, 손목 사진 번호, 관절값). 묶음 하나를 다 쓴 뒤 그 끝 관절값을 다음 관측 상태로 넣어 다시 추론한다
(사진은 처음 한 장 그대로 — 팔이 움직인 뒤의 사진은 없다. 묶음 절반마다 다시 물으면 움직이기 전 구간만 되풀이된다).
그리퍼가 열렸다가 닫히기 시작하면 멈춘다.
"""
import io
import sys
import time

import cv2
import numpy as np

real_stdout = sys.stdout.buffer
sys.stdout = sys.stderr  # 로그는 stderr 로, stdout 은 결과 npz 만

from physicalai.inference import InferenceModel  # noqa: E402

export_dir, task, threads, max_steps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
task = None if task == "-" else task


def closing(g, rest):
    i = 0
    while i < len(g) and g[i] < rest + 6:
        i += 1
    if i == len(g):
        return None
    m = g[i]
    while i < len(g) and g[i] >= m - 2:
        m = max(m, g[i])
        i += 1
    return i if i < len(g) else None


data = np.load(io.BytesIO(sys.stdin.buffer.read()))
fronts = [np.ascontiguousarray(cv2.imdecode(data[f"f{i}"], cv2.IMREAD_COLOR)[..., ::-1]) for i in range(int(data["nf"]))]
grips = [np.ascontiguousarray(cv2.imdecode(data[f"g{i}"], cv2.IMREAD_COLOR)[..., ::-1]) for i in range(int(data["ng"]))]
jobs = data["jobs"]          # (N, 2) 위·손목 사진 번호
states = data["states"]      # (N, 6)
model = InferenceModel(export_dir, backend="openvino", device="CPU", INFERENCE_NUM_THREADS=threads)
out = {}
for n in range(len(jobs)):
    front, grip = fronts[int(jobs[n, 0])], grips[int(jobs[n, 1])]
    state = states[n].astype(np.float32)
    rest = float(state[5])
    model.reset()
    stream, lat, first = [], [], None
    while len(stream) < max_steps:
        obs = {"state": state[None], "images.front": front[None], "images.gripper": grip[None]}
        if task:
            obs["task"] = [task]
        t0 = time.perf_counter()
        chunk = model.predict_action_chunk(obs)
        lat.append(time.perf_counter() - t0)
        if first is None:
            first = chunk
        stream.extend(chunk)
        state = np.asarray(chunk[-1], np.float32)
        if closing([a[5] for a in stream], rest) is not None:
            break
    out[f"{n}_first"] = np.asarray(first, np.float32)
    out[f"{n}_stream"] = np.asarray(stream, np.float32)
    out[f"{n}_lat"] = np.asarray(lat, np.float32)
    if (n + 1) % 20 == 0 or n + 1 == len(jobs):
        print(f"{n + 1}/{len(jobs)} 추론 {len(lat)}회 {np.mean(lat) * 1000:.0f}ms", file=sys.stderr, flush=True)
buf = io.BytesIO()
np.savez_compressed(buf, **out)
real_stdout.write(buf.getvalue())
real_stdout.flush()
