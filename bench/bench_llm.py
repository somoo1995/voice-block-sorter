"""LLM 실측: 첫 토큰 시간·초당 토큰, 한국어 의도 JSON 정답률, tools 호출.

사용: .venv/bin/python bench/bench_llm.py qwen3-8b [qwen3-4b ...]
결과는 화면 + bench/out/llm_<모델>.json
"""
import json
import statistics
import sys
import time
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8100/v3"
OUT = Path(__file__).resolve().parent / "out"
NO_THINK = {"enable_thinking": False}

SPEED_PROMPT = "부직포 공을 색깔별로 골라 담는 로봇 팔이 하는 일을 다섯 문장으로 차근차근 설명해 줘."

INTENT_SYSTEM = (
    "너는 부직포 공 선별 로봇의 명령 해석기다. 사용자 말을 아래 의도 중 하나로 분류해 JSON 으로만 답한다.\n"
    "- defect: 불량품(찢어지거나 모양이 나쁜 공)을 골라낸다\n"
    "- red: 빨간 공을 상자로 옮긴다\n"
    "- blue: 파란 공을 상자로 옮긴다\n"
    "- handoff: 공을 사람 손에 건네준다(색이 언급돼도 손에 주는 것이면 handoff)\n"
    "- report: 지금까지 작업 결과를 말로 보고한다\n"
    "count 는 사용자가 개수를 말했을 때만 정수(예: 하나=1, 두 개=2), 말하지 않았거나 '다/전부'면 null."
)
INTENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "robot_intent",
        "schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["defect", "red", "blue", "handoff", "report"]},
                "count": {"type": ["integer", "null"]},
            },
            "required": ["intent", "count"],
            "additionalProperties": False,
        },
    },
}
INTENT_CASES = [
    ("불량품 골라내 줘", "defect", None),
    ("빨간 볼 옮겨 줘", "red", None),
    ("파란 공 다 담아", "blue", None),
    ("빨간 거 하나 내 손에 줘", "handoff", 1),
    ("결과 알려 줘", "report", None),
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "pick_balls",
        "description": "지정한 색의 부직포 공을 집어 상자로 옮긴다",
        "parameters": {
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": ["red", "blue"], "description": "공 색깔"},
                "count": {"type": "integer", "description": "옮길 개수"},
            },
            "required": ["color", "count"],
        },
    },
}]
TOOL_PROMPT = "파란 공 두 개 상자에 넣어 줘."


def stream_once(model, prompt, max_tokens=256):
    """스트리밍으로 첫 토큰 시간(TTFT)과 생성 속도를 잰다."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": NO_THINK,
    }
    t0 = time.perf_counter()
    t_first = None
    usage = None
    text = []
    with requests.post(f"{BASE}/chat/completions", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                piece = ch.get("delta", {}).get("content")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    text.append(piece)
    t_end = time.perf_counter()
    n = usage["completion_tokens"]
    return {
        "ttft_s": t_first - t0,
        "total_s": t_end - t0,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": n,
        "decode_tok_s": (n - 1) / (t_end - t_first),
        "text_head": "".join(text)[:80],
    }


def chat(model, messages, **extra):
    body = {"model": model, "messages": messages, "temperature": 0, "chat_template_kwargs": NO_THINK}
    body.update(extra)
    t0 = time.perf_counter()
    r = requests.post(f"{BASE}/chat/completions", json=body, timeout=600)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), dt


def run(model):
    res = {"model": model}
    print(f"\n######## {model}")

    # ① 속도: 예열 1회 + 본측정 3회 중앙값
    stream_once(model, "안녕", max_tokens=8)
    runs = [stream_once(model, SPEED_PROMPT) for _ in range(3)]
    res["speed_runs"] = runs
    res["ttft_s"] = statistics.median(x["ttft_s"] for x in runs)
    res["decode_tok_s"] = statistics.median(x["decode_tok_s"] for x in runs)
    print(f"[속도] 프롬프트 {runs[0]['prompt_tokens']}토큰 → 생성 {runs[0]['completion_tokens']}토큰 | "
          f"TTFT 중앙값 {res['ttft_s']:.3f}s | 생성 {res['decode_tok_s']:.1f} tok/s")
    print(f"       예: {runs[0]['text_head']!r}")

    # ② 의도 JSON (json_schema 강제)
    ok = 0
    cases = []
    for text, want_intent, want_count in INTENT_CASES:
        j, dt = chat(model, [{"role": "system", "content": INTENT_SYSTEM}, {"role": "user", "content": text}],
                     response_format=INTENT_SCHEMA, max_tokens=64)
        raw = j["choices"][0]["message"]["content"]
        try:
            got = json.loads(raw)
            hit = got.get("intent") == want_intent and got.get("count") == want_count
        except json.JSONDecodeError:
            got, hit = None, False
        ok += hit
        cases.append({"text": text, "want": [want_intent, want_count], "raw": raw, "ok": hit, "latency_s": dt,
                      "completion_tokens": j["usage"]["completion_tokens"]})
        print(f"[의도] {'O' if hit else 'X'} {dt*1000:6.0f}ms  {text!r:24} → {raw}")
    res["intent_cases"] = cases
    res["intent_acc"] = ok / len(INTENT_CASES)
    res["intent_latency_median_s"] = statistics.median(c["latency_s"] for c in cases)
    print(f"[의도] 정답 {ok}/{len(INTENT_CASES)} | 지연 중앙값 {res['intent_latency_median_s']*1000:.0f}ms")

    # ③ tools 호출
    j, dt = chat(model, [{"role": "user", "content": TOOL_PROMPT}], tools=TOOLS, tool_choice="auto", max_tokens=128)
    msg = j["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    parsed = None
    if calls:
        fn = calls[0]["function"]
        parsed = {"name": fn["name"], "arguments": json.loads(fn["arguments"])}
    tool_ok = parsed == {"name": "pick_balls", "arguments": {"color": "blue", "count": 2}}
    res["tool"] = {"ok": tool_ok, "latency_s": dt, "finish_reason": j["choices"][0]["finish_reason"],
                   "tool_calls": calls, "content": msg.get("content")}
    print(f"[도구] {'O' if tool_ok else 'X'} {dt*1000:.0f}ms finish={j['choices'][0]['finish_reason']} "
          f"tool_calls={json.dumps(parsed, ensure_ascii=False)} content={msg.get('content')!r}")

    # ④ thinking 기본값 확인(끄지 않으면 reasoning_content 가 생기는지)
    body = {"model": model, "messages": [{"role": "user", "content": "3 더하기 4는?"}], "max_tokens": 400, "temperature": 0}
    t0 = time.perf_counter()
    j = requests.post(f"{BASE}/chat/completions", json=body, timeout=600).json()
    dt = time.perf_counter() - t0
    m = j["choices"][0]["message"]
    res["thinking_default"] = {"latency_s": dt, "has_reasoning": bool(m.get("reasoning_content")),
                               "completion_tokens": j["usage"]["completion_tokens"], "content": m.get("content")}
    print(f"[생각] 기본값: reasoning_content {'있음' if m.get('reasoning_content') else '없음'}, "
          f"{j['usage']['completion_tokens']}토큰 {dt:.1f}s, 답={m.get('content')!r}")

    OUT.mkdir(exist_ok=True)
    (OUT / f"llm_{model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    for m in sys.argv[1:] or ["qwen3-8b", "qwen3-4b"]:
        run(m)
