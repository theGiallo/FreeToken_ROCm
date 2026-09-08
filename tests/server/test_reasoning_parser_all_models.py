from __future__ import annotations

from freetoken.server.reasoning_parser import ReasoningParser


def _stream(parser: ReasoningParser, chunks):
    reasoning, content = "", ""
    for chunk in chunks:
        r, c = parser.parse_stream_chunk(chunk)
        reasoning += r
        content += c
    fr, fc = parser.flush()
    reasoning += fr
    content += fc
    return reasoning, content


# --------------------------------------------------------------- gpt-oss harmony
ANALYSIS_FINAL = (
    "<|channel|>analysis<|message|>The user says hi. Greet back.<|end|>"
    "<|start|>assistant<|channel|>final<|message|>Hello! How can I help you today?"
)


def test_harmony_non_stream_splits_analysis_and_final():
    parser = ReasoningParser("gpt_oss")
    reasoning, content = parser.parse_non_stream(ANALYSIS_FINAL)
    assert reasoning == "The user says hi. Greet back."
    assert content == "Hello! How can I help you today?"
    assert "<|channel|>" not in content


def test_harmony_non_stream_analysis_only():
    parser = ReasoningParser("gpt_oss")
    reasoning, content = parser.parse_non_stream(
        "<|channel|>analysis<|message|>still thinking"
    )
    assert reasoning == "still thinking"
    assert content == ""


def test_harmony_non_stream_final_only():
    parser = ReasoningParser("gpt_oss")
    reasoning, content = parser.parse_non_stream(
        "<|channel|>final<|message|>Just the answer.<|return|>"
    )
    assert reasoning == ""
    assert content == "Just the answer."


def test_harmony_non_stream_preserves_commentary_tool_block_verbatim():
    text = (
        "<|channel|>analysis<|message|>need weather<|end|>"
        '<|start|>assistant<|channel|>commentary to=functions.get_weather '
        '<|message|>{"city":"Paris"}<|call|>'
    )
    parser = ReasoningParser("gpt_oss")
    reasoning, content = parser.parse_non_stream(text)
    assert reasoning == "need weather"
    # The whole commentary block (markers included) survives for the tool parser.
    assert content.startswith("<|channel|>commentary to=functions.get_weather")
    assert '{"city":"Paris"}' in content
    # The closing marker must be present verbatim so the tool parser sees it.
    assert content.endswith("<|call|>")


def test_harmony_non_stream_passthrough_when_no_channels():
    parser = ReasoningParser("gpt_oss")
    reasoning, content = parser.parse_non_stream("plain text, no channels")
    assert reasoning == ""
    assert content == "plain text, no channels"


def test_harmony_streaming_splits_marker_across_chunks():
    parser = ReasoningParser("gpt_oss")
    # Deliberately split inside the markers and bodies.
    chunks = [
        "<|chan",
        "nel|>analysis<|mess",
        "age|>think a<|e",
        "nd|><|start|>assistant<|channel|>fin",
        "al<|message|>Ans",
        "wer.",
    ]
    reasoning, content = _stream(parser, chunks)
    assert reasoning == "think a"
    assert content == "Answer."


# ---------------------------------------------------------------- think family
import pytest


@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax"])
def test_think_non_stream_splits_reasoning(name):
    parser = ReasoningParser(name, force_reasoning=False)
    reasoning, content = parser.parse_non_stream("<thinking>weigh options</thinking>final answer")
    assert reasoning == "weigh options"
    assert content == "final answer"


@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax"])
def test_think_implicit_only_closing_tag(name):
    # Implicit-think: template injected the opening <thinking>; output starts inside.
    parser = ReasoningParser(name, force_reasoning=True)
    reasoning, content = parser.parse_non_stream("reasoning here</thinking>the answer")
    assert reasoning == "reasoning here"
    assert content == "the answer"


@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax"])
def test_think_no_reasoning_passthrough(name):
    parser = ReasoningParser(name, force_reasoning=False)
    reasoning, content = parser.parse_non_stream("just an answer")
    assert reasoning == ""
    assert content == "just an answer"


@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax"])
def test_think_tool_call_after_close_routes_to_content(name):
    # Malformed Qwen stream: the closing  response is emitted but the tool-CSL
    # <tool_call> block stays in the reasoning channel. The tool_start_token
    # fallback must cut reasoning at <tool_call> and keep the block in content so
    # the tool-call parser sees it (previously it leaked as raw reasoning).
    stream = "I need to provide the content parameter.\n\n</thinking>\n\n<tool_call>\n<function=write>\n<parameter=path>\n/workspace/duck_hunt.py\n</parameter>\n</tool_call>"
    parser = ReasoningParser(name, force_reasoning=True)
    reasoning, content = parser.parse_non_stream(stream)
    assert "<tool_call>" not in reasoning and "<function=" not in reasoning
    assert content.startswith("<tool_call>")
    assert "<function=write>" in content
    assert "duck_hunt.py" in content


def test_think_streaming_tool_call_after_close_routes_to_content():
    # Streaming variant: feed the malformed stream token-by-token so the closer and
    # the <tool_call> are split across chunks, and the tool block must land in
    # content (not reasoning) once the parser sees the marker.
    chunks = [
        "I need to provide",
        " the content parameter.",
        "\n\n",
        "</thinking",
        ">",
        "\n\n",
        "<tool",
        "_call",
        ">",
        "\n<function=write>",
        "\n<parameter=path>\n/workspace/duck_hunt.py\n</parameter>",
        "\n</tool_call>",
    ]
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning_parts, content_parts = [], []
    for ch in chunks:
        r, c = parser.parse_stream_chunk(ch)
        reasoning_parts.append(r)
        content_parts.append(c)
    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts) + parser.flush()[1]
    assert "<tool_call>" in content and "<function=write>" in content
    assert "<tool_call>" not in reasoning and "<function=" not in reasoning
    assert reasoning.startswith("I need to provide")


# ------------------------------------------------ qwen3.6 short-markers family
def test_qwen36_short_response_marker_ends_reasoning():
    # Qwen3.6 "thinking-official" closes its reasoning with the single 8-byte BPE
    # control token (ids 248068/248069, hex 3c/2f/.../3e) instead of the 10/11-byte
    # <thinking>/</thinking> tags. The token decodes to the " response" spelling
    # (letters t-h-i-n-k, no "ing"), so `_first_marker` on the long markers never
    # matches and reasoning leaks. The short marker must end reasoning and the rest
    # must land in content.
    response_marker = bytes.fromhex("3c2f7468696e6b3e").decode("ascii")
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = parser.parse_non_stream("reason here" + response_marker + "the answer")
    assert reasoning == "reason here"
    assert content == "the answer"
    assert response_marker not in reasoning + content


def test_qwen36_short_response_marker_before_tool_block():
    # The live reproduction: reasoning text, `\n`, the 8-byte closer, `\n\n`, then a
    # Qwen tool-CSL block. Without the short-marker closer the block held in the
    # buffer flushed to reasoning; now reasoning ends at the marker and the tool
    # block stays in content for the tool-call parser.
    response_marker = bytes.fromhex("3c2f7468696e6b3e").decode("ascii")
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = _stream(
        parser,
        [
            "The user says continue.\n",
            response_marker,
            "\n\n",
            "<tool_call>",
            "\n<function=get_weather>",
            "\n<parameter=city>\nParis\n</parameter>",
            "\n</tool_call>",
        ],
    )
    assert reasoning.strip() == "The user says continue."
    assert content.startswith("\n\n<tool_call>")
    assert "<function=get_weather>" in content
    assert response_marker not in reasoning + content


def test_qwen36_short_response_marker_split_across_chunks():
    # Streaming reassembly: the 8-byte closer split mid-token must still be held and
    # reassembled, not streamed out as reasoning.
    response_marker = bytes.fromhex("3c2f7468696e6b3e").decode("ascii")
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = _stream(
        parser,
        ["reason", response_marker[:5], response_marker[5:], "\nanswer text"],
    )
    assert reasoning.strip() == "reason"
    assert content.strip() == "answer text"
    assert response_marker not in reasoning + content


# ------------------------------------------------- think-alias blocks
def test_think_alias_non_stream_after_primary_close_routes_to_reasoning():
    # Qwen3.6 wraps a stray "thought" section in this alias pair in addition to
    # the <thinking> protocol block. It must land in reasoning_content, never
    # leak as raw markup in the visible answer.
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = parser.parse_non_stream(
        "hidden thoughts</thinking>visible answer<thought>stray note</thought>tail"
    )
    assert reasoning == "hidden thoughtsstray note"
    assert content == "visible answertail"
    assert "<thinking>" not in reasoning + content
    assert "</thinking>" not in reasoning + content
    assert "<thought>" not in reasoning + content
    assert "</thought>" not in reasoning + content


def test_think_alias_streaming_after_primary_close_routes_to_reasoning():
    # Streaming variant with the markers split across arbitrary chunk boundaries
    # (including `<thought>` abutting `<` from `</thinking>`).
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = _stream(
        parser,
        [
            "hidden thoughts",
            "</thin",
            "king>visible answer<thought>",
            "stray note</",
            "thought>tail",
        ],
    )
    assert reasoning == "hidden thoughtsstray note"
    assert content == "visible answertail"
    assert "<" not in reasoning + content


def test_think_alias_nested_opener_is_noise():
    # An opener inside an already-open region is model noise: the nested
    # copy is stripped, the region boundary logic still applies to the rest.
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = _stream(
        parser,
        ["pre<thinking>inner<", "/thinking>x<", "thought>", "y</thought>tail"],
    )
    # The nested <thinking> opener is stripped as noise; the primary closer
    # still closes the block, so x lands in content, then <thought> reopens
    # for y until the alias closer ends reasoning before tail.
    assert reasoning == "preinnery"
    assert content == "xtail"
    assert "<" not in reasoning + content


def test_think_alias_unbalanced_opener_truncates_to_reasoning():
    # A lone opener with no closer at end-of-stream is truncated reasoning, not
    # liability in content (mirrors the primary <thinking> behaviour).
    parser = ReasoningParser("qwen3", force_reasoning=False)
    reasoning, content = parser.parse_non_stream("plain<thought>no end")
    assert reasoning == "no end"
    assert content == "plain"


# ------------------------------------------------------- answer-boundary token
@pytest.mark.parametrize("name", ["qwen3", "glm", "minimax"])
def test_think_answer_boundary_ends_reasoning(name):
    # The qwen3 fine-tune sometimes closes reasoning with the single BPE token
    # " response" (id 1965) instead of </thinking>. It is line-anchored; the
    # answer text after it must land in content, not reasoning_content.
    parser = ReasoningParser(name, force_reasoning=True)
    reasoning, content = parser.parse_non_stream(
        "The code compiled.\n response\n\nThe code compiles and the fixes are:\n1. done"
    )
    assert reasoning == "The code compiled."
    assert content == "The code compiles and the fixes are:\n1. done"
    assert "response" not in reasoning
    assert "response" not in content


def test_think_answer_boundary_streaming_split_across_chunks():
    # Streaming variant with the boundary token split across chunks, including the
    # leading newline arriving separately from " response".
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = _stream(
        parser,
        [
            "Looking at the context,",
            "\n\n",
            " response",
            "\n\nThe code compiles and I made these final fixes.",
        ],
    )
    assert reasoning.strip() == "Looking at the context,"
    assert content.lstrip() == "The code compiles and I made these final fixes."


def test_think_answer_boundary_midline_response_does_not_close():
    # A mid-sentence "response" (not anchored to a line start) is reasoning
    # prose, not the answer boundary; only the later line-anchored marker closes.
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = parser.parse_non_stream(
        "weigh the user response and our reply\n response\nfinal"
    )
    assert reasoning == "weigh the user response and our reply"
    assert content == "final"


def test_think_answer_boundary_before_tool_call_takes_precedence():
    # When the model emits the boundary AND then a tool block, reasoning ends at
    # the boundary and the tool block stays in content for the tool parser.
    stream = (
        "I need the weather.\n response\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n</tool_call>"
    )
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = parser.parse_non_stream(stream)
    assert reasoning == "I need the weather."
    assert content.startswith("<tool_call>")
    assert "get_weather" in content


# --------------------------------------------------- answer envelope stripping
def test_think_short_answer_envelope_stripped_from_content():
    # The live qwen3.6-35b-a3b-256k "thinking-official" checkpoint wraps its answer
    # in the SHORT <answer>/</answer> single-BPE pair (not <answer_prompt>), and the
    # parser must strip it the same way or the raw tags leak into the visible answer.
    answer_start = bytes.fromhex("3c616e737765723e").decode("ascii")
    answer_end = bytes.fromhex("3c2f616e737765723e").decode("ascii")
    short_resp = bytes.fromhex("3c2f7468696e6b3e").decode("ascii")
    text = (
        "Let me verify the spawn position.\n"
        + short_resp
        + "\n"
        + answer_start
        + "\nLet me check the current state of the spawn logic and rendering.\n"
        + answer_end
    )
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = parser.parse_non_stream(text)
    assert reasoning == "Let me verify the spawn position."
    assert content == "Let me check the current state of the spawn logic and rendering."
    assert answer_start not in reasoning + content
    assert answer_end not in reasoning + content


def test_think_short_answer_opener_alone_closes_reasoning():
    # When the model skips the closer entirely and jumps straight into the short
    # <answer> envelope, the opener must end reasoning by itself (mirror of the
    # <answer_prompt> boundary test).
    answer_start = bytes.fromhex("3c616e737765723e").decode("ascii")
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = parser.parse_non_stream(
        "weigh the options\n" + answer_start + "\nThe answer text."
    )
    assert reasoning == "weigh the options"
    assert content == "The answer text."


def test_think_short_answer_envelope_streaming_split_across_chunks():
    # Streaming variant with the answer envelope markers split at arbitrary chunk
    # boundaries; the enum holds the partial tags until they can be stripped whole.
    answer_start = bytes.fromhex("3c616e737765723e").decode("ascii")
    answer_end = bytes.fromhex("3c2f616e737765723e").decode("ascii")
    short_resp = bytes.fromhex("3c2f7468696e6b3e").decode("ascii")
    reasoning, content = _stream(
        parser := ReasoningParser("qwen3", force_reasoning=True),
        [
            "The user says continue.\n",
            short_resp,
            "\n",
            answer_start[:4],
            answer_start[4:],
            "\nThe answer.",
            "\n",
            answer_end[:5],
            answer_end[5:],
        ],
    )
    assert reasoning.strip() == "The user says continue."
    assert content.strip() == "The answer."
    assert "<" not in content


def test_think_answer_envelope_stripped_from_content():
    # The qwen3.6 "thinking-official" fine-tune wraps its answer in an
    # <answer_prompt> envelope and emits an empty <function_call> placeholder when
    # it declines its tool list. Both are framing and must not reach the client.
    text = (
        "Let me verify the spawn position.\n response\n\n"
        "<answer_prompt>\n<function_call>\n</function_call>\n"
        "The log confirms the diagnosis. Ducks spawn from row -5.\n"
    )
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = parser.parse_non_stream(text)
    assert reasoning == "Let me verify the spawn position."
    assert content == "The log confirms the diagnosis. Ducks spawn from row -5."
    assert "<answer_prompt>" not in reasoning + content
    assert "<function_call>" not in reasoning + content


def test_think_answer_prompt_alone_closes_reasoning():
    # The model can skip  response entirely and jump straight into the envelope;
    # the line-anchored <answer_prompt> opener must end reasoning by itself.
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = parser.parse_non_stream(
        "weigh the options\n<answer_prompt>\nThe answer text."
    )
    assert reasoning == "weigh the options"
    assert content == "The answer text."


def test_think_answer_envelope_streaming_split_across_chunks():
    # Streaming variant with every marker split at arbitrary chunk boundaries,
    # including the empty <function_call> placeholder arriving one piece at a time.
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = _stream(
        parser,
        [
            "The user says continue.\n",
            " resp",
            "onse\n\n<answer_pr",
            "ompt>\n<function",
            "_call>\n</function_c",
            "all>\nThe answer.",
        ],
    )
    assert reasoning.strip() == "The user says continue."
    assert content.lstrip() == "The answer."
    assert "<" not in content


def test_think_nonempty_function_call_survives_for_tool_parser():
    # A <function_call> block carrying a payload is real tool-call drift: only the
    # EMPTY placeholder collapses, so the full block stays in content for the tool
    # parser.
    text = (
        "need to run ls\n response\n\n"
        "<answer_prompt><function_call>{"
        '"name":"bash","arg":{"command":"ls"}}</function_call></answer_prompt>\n'
        "Waiting for result."
    )
    parser = ReasoningParser("qwen3", force_reasoning=True)
    reasoning, content = parser.parse_non_stream(text)
    assert reasoning == "need to run ls"
    assert "<answer_prompt>" not in content
    assert "<function_call>" in content
    assert '"command":"ls"' in content


# ----------------------------------------------------------------------- gemma
def test_gemma_thought_split():
    parser = ReasoningParser("gemma4", force_reasoning=True)
    reasoning, content = parser.parse_non_stream("my private thought<channel|>visible answer")
    assert reasoning == "my private thought"
    assert content == "visible answer"


# ------------------------------------------------------- build_reasoning_parser
from types import SimpleNamespace

from freetoken.server.reasoning_parser import build_reasoning_parser


def test_build_reasoning_parser_returns_none_when_unset():
    assert build_reasoning_parser(SimpleNamespace(reasoning_parser=None), True) is None


def test_build_reasoning_parser_builds_named_parser():
    parser = build_reasoning_parser(SimpleNamespace(reasoning_parser="gpt_oss"), False)
    assert parser is not None
    reasoning, content = parser.parse_non_stream(ANALYSIS_FINAL)
    assert reasoning == "The user says hi. Greet back."
