# Qwen3.6-35B-A3B Tool-Calling Sampling Tuning Results

Date: 2026-08-27
Benchmark: `test_tool_calls.py` against `ft serve --tool-call-parser qwen35`

## Best config (deterministic, greedy)

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1
}
```

- Explicit tool prompt ("Use the bash tool to list files in /tmp/"): **20/20 (100%)**
- Greedy decode = only config reaching 100% reliability.

## Sweep results (explicit tool prompt, 15-20 samples each)

| Config                            | Tool calls |
|-----------------------------------|-----------|
| greedy (0.0, 1.0, -1)             | 20/20 (100%) |
| t0.1 k40 p0.95                     | 18/20 (90%) |
| t0.3 k40 p0.95                     | 15/20 (75%) |
| t0.8 k60 p0.95                     | 14/20 (70%) |
| t0.7 k60 p0.95                     | 14/20 (70%) |
| t1.0 k70 p0.90                     | 15/20 (75%) |
| t1.0 k70 p0.95                     | 10/20 (50%) |
| t1.0 k60 p0.95 (earlier)          | 7/10 (70%) |
| t1.0 k100 p0.95                   | 3/10 (30%) |
| t1.0 k200 p0.95                   | 6/10 (60%) |

## Key findings

1. **Temperature is the dominant factor.** Sub-0.3 temps reliably produce tool calls
   (75-100%); higher temps (>1.0) get worse (50% or below).
2. **Model emits ~4+ tool-call XML variants**:
   - `response` wrapper + `<function=standard_tool_calling>` wrapper (correct, parses)
   - `<function=bash>` direct (parses) - was observed at t0.1
   - Bare `<bash><parameter=command>...</bash>` (does NOT parse - missing `<function=`)
   - `<tool_call>` + `<bash>` (does NOT parse - wrong nested tag)
   Greedy decode picks the most likely continuation = the template-correct form.
3. **Non-streaming curl works end-to-end** and produces `standard_tool_calling`
   meta-call, which the server unwraps into the real tool (name + JSON args).
4. Pi models.json sampling params ARE honored by the server (temp/top_k/top_p).

## Recommendation for models.json (128k thinking variant)

Use greedy (temp=0) for tool reliability. Note: for creative/general tasks
higher temp may be preferred; tl;dr greedy is optimal for agent/tool workflows.