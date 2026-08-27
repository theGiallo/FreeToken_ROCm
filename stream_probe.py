#!/usr/bin/env python3
"""Stream one completion and print reasoning vs content deltas."""
import json
import sys


def main():
    import urllib.request

    prompt = sys.argv[1]
    temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    top_p = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    body = {
        "model": "qwen3.6-35b-a3b.gguf",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "top_k": top_k,
        "top_p": top_p,
        "stream": True,
    }
    req = urllib.request.Request(
        "http://127.0.0.1:1919/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    reasoning = []
    content = []
    fr = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = d["choices"][0]
            delta = ch.get("delta", {})
            if "reasoning_content" in delta and delta["reasoning_content"]:
                reasoning.append(delta["reasoning_content"])
            if "content" in delta and delta["content"]:
                content.append(delta["content"])
            if ch.get("finish_reason"):
                fr = ch["finish_reason"]
    rtext = "".join(reasoning)
    ctext = "".join(content)
    print(f"finish_reason={fr}")
    print(f"reasoning ({len(rtext)} chars):")
    print(f"  {rtext!r}")
    print(f"content ({len(ctext)} chars):")
    print(f"  {ctext!r}")


if __name__ == "__main__":
    main()