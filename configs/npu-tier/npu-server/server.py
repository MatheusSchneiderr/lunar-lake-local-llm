import json
import os
import queue
import re
import threading
import time
import uuid
from typing import List, Optional

import openvino_genai as ov_genai
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_PATH = os.environ["NPU_MODEL_PATH"]
MODEL_NAME = os.environ.get("NPU_MODEL_NAME", "npu-model")
NPU_PLATFORM = os.environ.get("NPU_PLATFORM", "4000")
MAX_PROMPT_LEN = int(os.environ.get("NPU_MAX_PROMPT_LEN", "4096"))
# tokenizer.encode(prompt) (used for our own compaction/overflow checks)
# doesn't always agree exactly with the token count the NPU pipeline's own
# generate() computes internally - observed a real mismatch in testing
# (our count said 8192, the pipeline's internal count was 8199, and it
# crashed anyway). A safety margin absorbs that discrepancy so our checks
# stay a genuine upper bound rather than an approximate one.
PROMPT_LEN_SAFETY_MARGIN = 64
EFFECTIVE_MAX_PROMPT_LEN = MAX_PROMPT_LEN - PROMPT_LEN_SAFETY_MARGIN
MIN_RESPONSE_LEN = int(os.environ.get("NPU_MIN_RESPONSE_LEN", "512"))
PORT = int(os.environ.get("NPU_SERVER_PORT", "8900"))

app = FastAPI()

pipe = ov_genai.LLMPipeline(
    MODEL_PATH,
    "NPU",
    NPU_PLATFORM=NPU_PLATFORM,
    MAX_PROMPT_LEN=MAX_PROMPT_LEN,
    MIN_RESPONSE_LEN=MIN_RESPONSE_LEN,
)
tokenizer = pipe.get_tokenizer()
generate_lock = threading.Lock()


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # OpenAI wire format: always a JSON-encoded string


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class ToolFunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ToolDef(BaseModel):
    type: str = "function"
    function: ToolFunctionDef


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    tools: Optional[List[ToolDef]] = None


# --- Qwen2.5's native tool-call convention: <tool_call>{"name": ..., "arguments": {...}}</tool_call> ---

# The model's own chat template instructs it to wrap tool calls in
# <tool_call>...</tool_call>, but empirically (verified via /debug/prompt +
# repeated real generations against the deployed int4 NPU checkpoint) this
# particular quantized Qwen2.5-Coder-7B-Instruct build never emits that
# literal special token - it reliably produces a well-formed
# {"name": ..., "arguments": {...}} JSON object with the right keys, just
# wrapped in a markdown ```json fence instead (consistent across greedy and
# sampled decoding). So both wrapper conventions are recognized; the fence
# form is only ever considered when the request actually carried `tools`,
# so the plain no-tools chat/inline path is completely unaffected - an
# ordinary fenced code example in a normal answer just won't parse as
# {"name","arguments"} JSON and falls back to being shown as content.
TAG_PAIR = ("<tool_call>", "</tool_call>")
FENCE_PAIR = ("```", "```")
_DELIMITER_PAIRS = [TAG_PAIR, FENCE_PAIR]


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return {}


def parse_tool_call_body(body: str) -> Optional[dict]:
    """Parse the text between an opener/closer pair. Returns
    {id, name, arguments (JSON string)} on success, or None if the body
    isn't well-formed - callers then fall back to showing the raw span
    instead of crashing or emitting a garbage tool call. Tries the body
    as-is first, then with an optional leading language-tag line (e.g. the
    "json" in ```json) stripped, to cover both wrapper conventions."""
    candidates = [body.strip()]
    stripped = body.strip()
    newline = stripped.find("\n")
    if newline != -1:
        first_line = stripped[:newline].strip()
        if first_line and first_line.isalpha():
            candidates.append(stripped[newline + 1 :].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        name = parsed.get("name")
        arguments = parsed.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            return {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "name": name,
                "arguments": json.dumps(arguments),
            }
    return None


def _find_balanced_json_span(text: str, start: int = 0) -> Optional[tuple]:
    """Find the first top-level {...} object in `text` at or after `start`,
    respecting string literals/escapes so braces inside strings don't
    confuse the depth count. Returns (start, end) (end exclusive) or None.
    This is delimiter-agnostic - it finds the JSON object regardless of
    whatever wrapper (if any) the model put around it."""
    n = len(text)
    i = text.find("{", start)
    while i != -1:
        depth = 0
        in_string = False
        escape = False
        j = i
        while j < n:
            c = text[j]
            if in_string:
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return (i, j + 1)
            j += 1
        i = text.find("{", i + 1)
    return None


def _delimiter_scan(text: str):
    """Scan `text` once for <tool_call>/```-wrapped spans (possibly several
    in sequence). Returns (events, found_any_tool_call)."""
    events = []
    found_any = False
    pos = 0
    while True:
        best = None
        for open_tag, close_tag in _DELIMITER_PAIRS:
            idx = text.find(open_tag, pos)
            if idx != -1 and (best is None or idx < best[0]):
                best = (idx, open_tag, close_tag)
        if best is None:
            if pos < len(text):
                events.append({"type": "content", "text": text[pos:]})
            return events, found_any

        start, open_tag, close_tag = best
        end = text.find(close_tag, start + len(open_tag))
        if end == -1:
            events.append({"type": "content", "text": text[pos:]})
            return events, found_any

        if start > pos:
            events.append({"type": "content", "text": text[pos:start]})
        body = text[start + len(open_tag) : end]
        close_end = end + len(close_tag)
        call = parse_tool_call_body(body)
        if call is None:
            events.append({"type": "content", "text": text[start:close_end]})
        else:
            events.append({"type": "tool_call", **call})
            found_any = True
        pos = close_end


def _tag_name_variants(snake_name: str):
    """Plausible XML-tag spellings a model might invent for a snake_case
    tool name: as-is, camelCase, PascalCase, and each of those with a
    trailing "Tool" suffix."""
    parts = snake_name.split("_")
    camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
    pascal = "".join(p.capitalize() for p in parts)
    variants = {snake_name, camel, pascal}
    variants |= {v + "Tool" for v in list(variants)}
    variants.add(snake_name + "_tool")
    return variants


def _extract_by_known_tag(text: str, known_names) -> Optional[dict]:
    """Last-resort fallback: some invented wrapper conventions use the
    tool's own name as the XML tag, with the JSON body being the raw
    arguments directly (no {"name","arguments"} envelope at all) - e.g.
    <insert_edit_into_file>{"filepath": ...}</insert_edit_into_file>. Since
    we know the finite set of real tool names from the request, search for
    each one (in several spelling variants) used as a tag, rather than
    trying to reverse-engineer an arbitrary tag name."""
    best = None  # (start, tool_name, args_dict, close_end)
    for name in known_names:
        for variant in _tag_name_variants(name):
            open_tag = f"<{variant}>"
            close_tag = f"</{variant}>"
            start = text.find(open_tag)
            if start == -1:
                continue
            end = text.find(close_tag, start + len(open_tag))
            if end == -1:
                continue

            inner = text[start + len(open_tag) : end].strip()
            try:
                parsed = json.loads(inner)
            except Exception:
                span = _find_balanced_json_span(text, start + len(open_tag))
                if not span or span[0] >= end:
                    continue
                try:
                    parsed = json.loads(text[span[0] : span[1]])
                except Exception:
                    continue
            if not isinstance(parsed, dict):
                continue

            if set(parsed.keys()) == {"name", "arguments"} and isinstance(parsed.get("arguments"), dict):
                args = parsed["arguments"]
            else:
                args = parsed

            close_end = end + len(close_tag)
            if best is None or start < best[0]:
                best = (start, name, args, close_end)

    if best is None:
        return None
    _, name, args, _ = best
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "name": name,
        "arguments": json.dumps(args),
    }


def extract_tool_calls_and_text(text: str, tools: Optional[List["ToolDef"]]):
    """Extract tool calls from a complete generated text. Recognizes (in
    order): <tool_call>/```-delimited spans (the documented and the
    observed-in-practice conventions), bare unwrapped JSON, a
    delimiter-agnostic scan for any balanced {"name":...,"arguments":...}
    JSON object anywhere in the text, and finally the tool's own name used
    as an XML tag around raw arguments - a small local model doesn't
    reliably stick to any one wrapper convention turn to turn, sometimes
    inventing its own. Falls back to showing the raw text as content if
    nothing recognizable is found, rather than ever crashing or dropping
    output. Only consulted when the request actually carried `tools` -
    otherwise the text is returned as plain content unchanged."""
    if not tools:
        return [{"type": "content", "text": text}] if text else []

    events, found = _delimiter_scan(text)
    if found:
        return events

    stripped = text.strip()
    if stripped:
        call = parse_tool_call_body(stripped)
        if call is not None:
            return [{"type": "tool_call", **call}]

    span = _find_balanced_json_span(text)
    if span:
        s, e = span
        call = parse_tool_call_body(text[s:e])
        if call is not None:
            events = []
            if text[:s]:
                events.append({"type": "content", "text": text[:s]})
            events.append({"type": "tool_call", **call})
            if text[e:]:
                events.append({"type": "content", "text": text[e:]})
            return events

    known_names = {t.function.name for t in tools}
    call = _extract_by_known_tag(text, known_names)
    if call is not None:
        return [{"type": "tool_call", **call}]

    return [{"type": "content", "text": text}] if text else []


class ToolCallScanner:
    """Accumulates a growing text buffer and only runs tool-call detection
    once generation ends. A small local model doesn't reliably stick to a
    single wrapper convention (observed: the documented <tool_call> tag,
    ```-fenced JSON, bare JSON, and ad hoc invented XML-ish tags, all from
    the same model across different turns), so there is no safe delimiter
    to watch for incrementally without risking a wrong guess mid-stream.
    Live token streaming is therefore only used when no tools are active
    (the vast majority of usage) - see extract_tool_calls_and_text for the
    actual detection logic, shared with the non-streaming path."""

    def __init__(self, tools: Optional[List["ToolDef"]]):
        self.buf = ""
        self.tools = tools

    def feed(self, new_text: str):
        self.buf += new_text
        if not self.tools:
            return [{"type": "content", "text": new_text}] if new_text else []
        return []

    def finish(self):
        if not self.tools:
            return []
        return extract_tool_calls_and_text(self.buf, self.tools)


def build_message(events) -> tuple[dict, str]:
    content = "".join(e["text"] for e in events if e["type"] == "content")
    tool_calls = [
        {
            "id": e["id"],
            "type": "function",
            "function": {"name": e["name"], "arguments": e["arguments"]},
        }
        for e in events
        if e["type"] == "tool_call"
    ]
    message = {"role": "assistant"}
    if tool_calls:
        message["tool_calls"] = tool_calls
        message["content"] = content if content else None
        finish_reason = "tool_calls"
    else:
        message["content"] = content
        finish_reason = "stop"
    return message, finish_reason


def message_to_template_dict(m: ChatMessage) -> dict:
    d = {"role": m.role}
    if m.content is not None:
        d["content"] = m.content
    if m.tool_calls:
        d["tool_calls"] = [
            {
                "function": {
                    "name": tc.function.name,
                    "arguments": _safe_json_loads(tc.function.arguments),
                }
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id is not None:
        d["tool_call_id"] = m.tool_call_id
    return d


# The model has no awareness of the real filesystem - if it isn't told a
# file's exact path (via context or the user's message), it tends to
# invent a plausible-looking placeholder (e.g. "path/to/file.ext") rather
# than admitting it doesn't know, which then fails at tool-execution time
# with a confusing "file does not exist" error. Nudge it to look the path
# up or ask instead, whenever any tool is available.
PATH_AWARENESS_REMINDER = (
    "If you don't already know a file's exact path from the conversation or "
    "provided context, use a search tool (e.g. file_search or grep_search) to "
    "find it first if one is available, or ask the user for the exact path. "
    "Never guess a path or invent a placeholder like path/to/file.ext."
)

# Observed real behavior: the model reliably makes ONE correct tool call
# (e.g. grep_search for an error message), then stops and asks the user to
# manually paste information it could have fetched itself with another
# available tool (e.g. read_file), even when explicitly asked to keep
# investigating. Nudge it to keep chaining tool calls on its own instead of
# handing the next obvious step back to the user.
KEEP_INVESTIGATING_REMINDER = (
    "When investigating something, use the available tools yourself to gather "
    "all the information you need - reading files, searching, running commands "
    "- instead of asking the user to look up or paste something you can "
    "retrieve with a tool. Keep making tool calls across multiple turns until "
    "you actually have enough information to answer, rather than stopping "
    "after the first one. Never end a response by only describing what you "
    "are about to do next (e.g. \"Let's read that file\" or \"I'll now check "
    "X\") without actually doing it - if you already know which tool to call "
    "next, call it immediately in this same response instead of narrating it."
)


def render_prompt(messages: List[ChatMessage], tools: Optional[List[ToolDef]] = None) -> str:
    tool_dicts = [t.model_dump(exclude_none=True) for t in tools] if tools else None
    msg_dicts = [message_to_template_dict(m) for m in messages]

    if tool_dicts:
        reminder = PATH_AWARENESS_REMINDER + "\n\n" + KEEP_INVESTIGATING_REMINDER
        if msg_dicts and msg_dicts[0].get("role") == "system":
            msg_dicts[0]["content"] = msg_dicts[0].get("content", "") + "\n\n" + reminder
        else:
            msg_dicts.insert(0, {"role": "system", "content": reminder})

    return tokenizer.apply_chat_template(
        msg_dicts,
        add_generation_prompt=True,
        tools=tool_dicts,
    )


# The model won't reliably choose to call file_search on its own before
# guessing a path (see PATH_AWARENESS_REMINDER above - prompting alone isn't
# enough on a model this size). So when it looks like the user is asking to
# edit/create a specific file but nothing in the conversation actually
# grounds that file's real path yet, and both a search tool and an edit
# tool are available, skip the model entirely for this turn and return a
# deterministic file_search call ourselves - forced, not requested. Once a
# tool result exists anywhere in the conversation, we stop forcing and let
# the model take over with real information in front of it.
FILENAME_RE = re.compile(r"\b[\w\-]+\.[A-Za-z]{1,10}\b")
ABS_PATH_RE = re.compile(r"(?<!\w)/[\w\-./]+\.[A-Za-z]{1,10}\b")


def _find_tool_def(tools: Optional[List[ToolDef]], name: str) -> Optional[ToolDef]:
    if not tools:
        return None
    for t in tools:
        if t.function.name == name:
            return t
    return None


def _has_tool_result(messages: List[ChatMessage]) -> bool:
    return any(m.role == "tool" for m in messages)


def _has_existing_path_grounding(messages: List[ChatMessage]) -> bool:
    for m in messages:
        content = m.content or ""
        if "<file>" in content or ABS_PATH_RE.search(content):
            return True
    return False


def maybe_force_file_search(
    messages: List[ChatMessage], tools: Optional[List[ToolDef]]
) -> Optional[dict]:
    if not _find_tool_def(tools, "file_search"):
        return None
    if not (_find_tool_def(tools, "insert_edit_into_file") or _find_tool_def(tools, "create_file")):
        return None
    if _has_tool_result(messages) or _has_existing_path_grounding(messages):
        return None

    last_user = next((m for m in reversed(messages) if m.role == "user"), None)
    if not last_user or not last_user.content:
        return None

    match = FILENAME_RE.search(last_user.content)
    if not match:
        return None

    filename = match.group(0)
    query = filename if "/" in filename else f"**/{filename}"
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "name": "file_search",
        "arguments": json.dumps({"query": query}),
    }


# If file_search (forced or model-initiated) comes back with more than one
# match, don't let the model guess which one is meant - the whole point of
# forcing a search was to replace guessing with certainty, and a guess
# between several real candidates is still a guess. Ask the user directly
# instead. codecompanion's file_search tool always renders its result to
# the model as "Searched files for `<query>`, N results\n```\n<paths>\n```"
# (see file_search.lua's `success` handler) - matched here regardless of
# whatever wrapper tag surrounds it.
FILE_SEARCH_RESULTS_RE = re.compile(
    r"Searched files for `[^`]*`,\s*(\d+)\s*results?\s*```\s*(.*?)```", re.DOTALL
)


def maybe_ask_to_disambiguate(messages: List[ChatMessage]) -> Optional[str]:
    if not messages:
        return None
    last = messages[-1]
    if last.role != "tool" or not last.content:
        return None

    match = FILE_SEARCH_RESULTS_RE.search(last.content)
    if not match:
        return None

    paths = [p.strip() for p in match.group(2).strip().splitlines() if p.strip()]
    if len(paths) <= 1:
        return None

    listing = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(paths))
    return (
        f"I found {len(paths)} files matching that name:\n\n{listing}\n\n"
        "Which one did you mean? Reply with the number or the full path."
    )


# Observed real behavior even with KEEP_INVESTIGATING_REMINDER in the
# prompt: after a tool result comes back, the model sometimes generates a
# genuinely empty response - no text, no tool call, nothing - instead of
# either continuing the investigation or giving an answer. Rather than
# returning that nothing to the user, retry once with an explicit nudge
# appended. Bounded to a single retry so this can never loop; if the retry
# is ALSO degenerate, give up and return the (empty) result as-is.
STALL_NUDGE_MESSAGE = ChatMessage(
    role="user",
    content=(
        "Continue investigating - call another tool if you need more "
        "information, or give your actual answer now. Don't just stop with "
        "nothing."
    ),
)


def is_degenerate_stall(message: dict, messages: List[ChatMessage]) -> bool:
    if message.get("tool_calls"):
        return False
    if (message.get("content") or "").strip():
        return False
    return bool(messages) and messages[-1].role == "tool"


# --- Context compaction: a long-running (especially tool-heavy) chat will
# eventually overflow MAX_PROMPT_LEN. Rather than just failing, try to keep
# the conversation going, cheapest/safest option first:
#   1. Deterministically truncate old, bulky tool-result payloads (file
#      content, diffs, search dumps) - no model call, nothing can go wrong.
#   2. If that alone isn't enough, ask the model to summarize a leading run
#      of plain dialogue turns - a real generation call, and this model
#      isn't perfectly reliable, so it's a last resort, not the first move.
#   3. If even that doesn't fit, fail with a clear message (see
#      chat_completions) rather than let the pipeline crash opaquely.

TOOL_RESULT_TRUNCATE_THRESHOLD = 400  # chars; only touch genuinely large payloads
KEEP_RECENT_TOOL_RESULTS = 1  # the most recent tool result stays at full fidelity
SUMMARY_MAX_NEW_TOKENS = 300
SUMMARY_TOOL_CONTENT_PREVIEW = 300  # chars of each tool call/result kept in the summarization prompt itself


def prompt_token_count(prompt: str) -> int:
    return int(tokenizer.encode(prompt).input_ids.shape[1])


def truncate_old_tool_results(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Replace older, large tool-result payloads with a short placeholder,
    oldest first, keeping the most recent KEEP_RECENT_TOOL_RESULTS at full
    fidelity. Returns a new list - never mutates the input."""
    tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
    truncatable = tool_indices[: max(0, len(tool_indices) - KEEP_RECENT_TOOL_RESULTS)]

    result = list(messages)
    for i in truncatable:
        content = result[i].content or ""
        if len(content) > TOOL_RESULT_TRUNCATE_THRESHOLD:
            result[i] = result[i].model_copy(
                update={
                    "content": f"[earlier tool output omitted to save context - was {len(content)} chars]"
                }
            )
    return result


def _find_summarizable_prefix_end(messages: List[ChatMessage], start: int) -> int:
    """Advance from `start` through complete conversation units - a plain
    user/assistant turn, or an assistant-with-tool_calls message together
    with ALL of its paired tool-result messages - never splitting a
    tool_call_id pairing, and never consuming the final message (the
    current ask)."""
    end = start
    n = len(messages)
    while end < n - 1:
        m = messages[end]
        if m.role in ("user", "assistant") and not m.tool_calls:
            end += 1
            continue
        if m.role == "assistant" and m.tool_calls:
            remaining_ids = {tc.id for tc in m.tool_calls}
            j = end + 1
            while j < n and messages[j].role == "tool" and messages[j].tool_call_id in remaining_ids:
                remaining_ids.discard(messages[j].tool_call_id)
                j += 1
            if remaining_ids or j >= n - 1:
                break  # a paired result is missing, or this unit would eat the final message
            end = j
            continue
        break
    return end


def _turn_to_transcript_line(m: ChatMessage) -> Optional[str]:
    if m.role in ("user", "assistant") and m.content:
        return f"{m.role}: {m.content}"
    if m.role == "assistant" and m.tool_calls:
        calls = ", ".join(
            f"{tc.function.name}({tc.function.arguments[:SUMMARY_TOOL_CONTENT_PREVIEW]})"
            for tc in m.tool_calls
        )
        return f"assistant called: {calls}"
    if m.role == "tool" and m.content:
        return f"tool result: {m.content[:SUMMARY_TOOL_CONTENT_PREVIEW]}"
    return None


def summarize_old_turns(messages: List[ChatMessage]) -> Optional[List[ChatMessage]]:
    """Last-resort compaction: ask the model to summarize a leading run of
    the conversation - plain dialogue and/or complete tool-call/result
    units - into one short message. Never splits a tool_call_id pairing
    and never touches the final message (the current ask). Returns None if
    there's nothing safe to summarize."""
    start = 1 if messages and messages[0].role == "system" else 0
    end = _find_summarizable_prefix_end(messages, start)

    if end - start < 2:
        return None

    lines = [_turn_to_transcript_line(m) for m in messages[start:end]]
    transcript = "\n".join(l for l in lines if l)
    if not transcript.strip():
        return None

    summary_prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "Summarize the key facts, decisions, and any specific file paths, "
                    "function/variable names, or values mentioned in this conversation "
                    "excerpt, in a few sentences:\n\n" + transcript
                ),
            }
        ],
        add_generation_prompt=True,
    )
    summary_config = ov_genai.GenerationConfig()
    summary_config.max_new_tokens = SUMMARY_MAX_NEW_TOKENS
    summary_config.do_sample = False

    with generate_lock:
        summary = str(pipe.generate(summary_prompt, summary_config)).strip()
    if not summary:
        return None

    summary_message = ChatMessage(
        role="system", content=f"[Earlier conversation summarized to save context]\n{summary}"
    )
    return messages[:start] + [summary_message] + messages[end:]


def compact_messages_if_needed(
    messages: List[ChatMessage], tools: Optional[List[ToolDef]]
) -> tuple[List[ChatMessage], str, int]:
    """Returns (messages, prompt, prompt_tokens) - compacting only as much
    as actually needed to fit EFFECTIVE_MAX_PROMPT_LEN, cheapest option
    first."""
    prompt = render_prompt(messages, tools)
    tokens = prompt_token_count(prompt)
    if tokens <= EFFECTIVE_MAX_PROMPT_LEN:
        return messages, prompt, tokens

    messages = truncate_old_tool_results(messages)
    prompt = render_prompt(messages, tools)
    tokens = prompt_token_count(prompt)
    if tokens <= EFFECTIVE_MAX_PROMPT_LEN:
        return messages, prompt, tokens

    summarized = summarize_old_turns(messages)
    if summarized is not None:
        messages = summarized
        prompt = render_prompt(messages, tools)
        tokens = prompt_token_count(prompt)

    return messages, prompt, tokens


def make_config(req: ChatRequest) -> ov_genai.GenerationConfig:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = req.max_tokens or 512
    config.do_sample = bool(req.temperature and req.temperature > 0)
    if config.do_sample:
        config.temperature = req.temperature
    return config


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    created = int(time.time())
    completion_id = f"chatcmpl-npu-{created}"

    disambiguation = maybe_ask_to_disambiguate(req.messages)
    if disambiguation is not None:
        if req.stream:
            return StreamingResponse(
                plain_text_stream(disambiguation, completion_id, created),
                media_type="text/event-stream",
            )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": disambiguation},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    forced_call = maybe_force_file_search(req.messages, req.tools) if req.tools else None

    if forced_call is not None:
        if req.stream:
            return StreamingResponse(
                forced_tool_call_stream(forced_call, completion_id, created),
                media_type="text/event-stream",
            )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": forced_call["id"],
                                "type": "function",
                                "function": {
                                    "name": forced_call["name"],
                                    "arguments": forced_call["arguments"],
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    config = make_config(req)

    # The NPU pipeline hard-crashes (RuntimeError) if the prompt exceeds
    # MAX_PROMPT_LEN. Try to keep the chat going via compaction first
    # (see compact_messages_if_needed); only if that's still not enough do
    # we return a clear message instead of letting the pipeline crash
    # opaquely or, in the streaming case, silently return nothing.
    messages, prompt, prompt_tokens = compact_messages_if_needed(req.messages, req.tools)
    if prompt_tokens > EFFECTIVE_MAX_PROMPT_LEN:
        overflow_message = (
            f"This conversation has grown too long for the local model to continue "
            f"({prompt_tokens} tokens needed, but the server is configured for a "
            f"{MAX_PROMPT_LEN}-token limit). Start a new chat to continue - "
            f"conversation history isn't trimmed automatically."
        )
        if req.stream:
            return StreamingResponse(
                plain_text_stream(overflow_message, completion_id, created),
                media_type="text/event-stream",
            )
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": overflow_message},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    if req.stream:
        return StreamingResponse(
            stream_completion(prompt, config, completion_id, created, req.tools, messages),
            media_type="text/event-stream",
        )

    with generate_lock:
        result = pipe.generate(prompt, config)

    events = extract_tool_calls_and_text(str(result), req.tools)
    message, finish_reason = build_message(events)

    if is_degenerate_stall(message, messages):
        # Bounded to a single retry: whether or not this second attempt is
        # itself degenerate, we use its result - if it's also empty, that's
        # observably identical to keeping the original (empty) message, so
        # there's nothing further worth checking for.
        retry_messages = messages + [STALL_NUDGE_MESSAGE]
        retry_prompt = render_prompt(retry_messages, req.tools)
        if prompt_token_count(retry_prompt) <= EFFECTIVE_MAX_PROMPT_LEN:
            with generate_lock:
                retry_result = pipe.generate(retry_prompt, config)
            retry_events = extract_tool_calls_and_text(str(retry_result), req.tools)
            message, finish_reason = build_message(retry_events)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def plain_text_stream(text, completion_id, created):
    def make_chunk(delta, finish_reason=None):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    yield f"data: {json.dumps(make_chunk({'content': text}))}\n\n"
    yield f"data: {json.dumps(make_chunk({}, 'stop'))}\n\n"
    yield "data: [DONE]\n\n"


def forced_tool_call_stream(forced_call, completion_id, created):
    def make_chunk(delta, finish_reason=None):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    delta_tool_call = {
        "index": 0,
        "id": forced_call["id"],
        "type": "function",
        "function": {"name": forced_call["name"], "arguments": forced_call["arguments"]},
    }
    yield f"data: {json.dumps(make_chunk({'tool_calls': [delta_tool_call]}))}\n\n"
    yield f"data: {json.dumps(make_chunk({}, 'tool_calls'))}\n\n"
    yield "data: [DONE]\n\n"


def _run_generate_to_events(prompt, config, tools):
    """Blocking (non-streaming) generation, returning extracted events -
    used both for a plain synchronous run and for the stall-retry below."""
    with generate_lock:
        result = pipe.generate(prompt, config)
    return extract_tool_calls_and_text(str(result), tools)


def stream_completion(prompt, config, completion_id, created, tools, messages=None):
    q: "queue.Queue" = queue.Queue()
    sentinel = object()

    def streamer(subword: str) -> bool:
        q.put(subword)
        return False  # False = keep generating

    def run():
        try:
            with generate_lock:
                pipe.generate(prompt, config, streamer)
        finally:
            q.put(sentinel)

    threading.Thread(target=run, daemon=True).start()

    def make_chunk(delta, finish_reason=None):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    scanner = ToolCallScanner(tools)

    while True:
        item = q.get()
        if item is sentinel:
            events = scanner.finish()
            break
        for ev in scanner.feed(item):
            # only ever "content" here - tool_calls only emerge at finish(),
            # since a tools-active stream is fully buffered until then
            if ev["text"]:
                yield f"data: {json.dumps(make_chunk({'content': ev['text']}))}\n\n"

    # Same bounded stall-retry as the non-streaming path. Safe to do here
    # (nothing has been yielded to the client yet) precisely because a
    # tools-active stream is fully buffered until this point - see
    # ToolCallScanner.
    if not events and tools and messages and messages[-1].role == "tool":
        retry_messages = messages + [STALL_NUDGE_MESSAGE]
        retry_prompt = render_prompt(retry_messages, tools)
        if prompt_token_count(retry_prompt) <= EFFECTIVE_MAX_PROMPT_LEN:
            events = _run_generate_to_events(retry_prompt, config, tools)

    any_tool_calls = False
    tool_call_index = 0
    for ev in events:
        if ev["type"] == "content":
            if ev["text"]:
                yield f"data: {json.dumps(make_chunk({'content': ev['text']}))}\n\n"
        else:  # tool_call - emitted whole (already fully buffered/parsed), not fragmented
            any_tool_calls = True
            delta_tool_call = {
                "index": tool_call_index,
                "id": ev["id"],
                "type": "function",
                "function": {"name": ev["name"], "arguments": ev["arguments"]},
            }
            tool_call_index += 1
            yield f"data: {json.dumps(make_chunk({'tool_calls': [delta_tool_call]}))}\n\n"

    final_finish_reason = "tool_calls" if any_tool_calls else "stop"
    yield f"data: {json.dumps(make_chunk({}, final_finish_reason))}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT)
