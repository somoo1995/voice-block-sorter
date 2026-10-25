# 비전 모델 장치별 속도 (2026-10-21T17:00)

- 입력 1x3x640x640, OpenVINO 2026.4.1, 전원 AC, OVMS inactive
- CPU = Intel(R) Core(TM) Ultra 9 185H · GPU = Intel(R) Arc(TM) Graphics (iGPU) · NPU = Intel(R) AI Boost
- 지연 = 요청 1개 동기 추론(LATENCY 힌트), 처리량 = 비동기 요청 여러 개(THROUGHPUT 힌트), e2e = Ultralytics 전처리+추론+후처리(detect_objects 경로)

| 모델 | 장치 | 첫 컴파일 ms | 지연 중앙값 ms | 지연 p90 ms | FPS (동기) | FPS (처리량) | e2e ms |
|---|---|---:|---:|---:|---:|---:|---:|
| coco_int8 | CPU | 290 | 12.74 | 16.21 | 73.7 | 110.0 | 19.13 |
| coco_int8 | GPU | 2,678 | 3.31 | 3.66 | 289.7 | 454.0 | 9.0 |
| coco_int8 | NPU | 1,047 | 7.05 | 7.23 | 141.2 | 163.8 | 10.32 |
| coco_fp16 | CPU | 119 | 27.49 | 35.37 | 34.5 | 33.2 | 34.12 |
| coco_fp16 | GPU | 209 | 4.47 | 4.57 | 225.4 | 217.6 | 11.31 |
| coco_fp16 | NPU | 660 | 7.81 | 8.17 | 126.8 | 145.3 | 10.84 |
