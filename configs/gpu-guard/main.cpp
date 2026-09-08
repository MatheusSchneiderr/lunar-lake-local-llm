// gpu-guard: a thin reverse proxy in front of gpu-server-hard (llama-server).
//
// Two jobs, and nothing else:
//   1. Pass every request through untouched, with real byte-for-byte
//      streaming (no buffering added), for the overwhelming majority of
//      requests - anything without a non-empty "tools" array.
//   2. For tool-bearing requests only: buffer the full completion, detect
//      two failure patterns this model exhibits under long tool-chains
//      (a silent stall, or a confident-but-wrong "I don't have file
//      access" refusal), and retry once with a nudge message if either
//      happens - exactly the same bounded-single-retry approach already
//      proven out in config/npu-server/server.py, ported to a case where
//      no Python model pipeline is involved, so a small always-resident
//      process makes sense instead of a heavy one.
//
// Deliberately no TLS, no compression, no auth - this only ever talks to
// 127.0.0.1, matching gpu-server-hard's own posture.

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <mutex>
#include <regex>
#include <string>
#include <thread>

using json = nlohmann::json;

namespace {

std::string env_or(const char *name, std::string fallback) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::string(v) : fallback;
}

int env_or_int(const char *name, int fallback) {
  const char *v = std::getenv(name);
  return (v && *v) ? std::atoi(v) : fallback;
}

// --- Detection: ported from is_degenerate_stall in npu-server/server.py ---
bool is_degenerate_stall(const json &message, const json &request_messages) {
  if (message.contains("tool_calls") && message["tool_calls"].is_array() &&
      !message["tool_calls"].empty()) {
    return false;
  }
  std::string content =
      message.value("content", std::string());
  // trim whitespace
  size_t start = content.find_first_not_of(" \t\r\n");
  if (start != std::string::npos) {
    return false; // has real content
  }
  if (request_messages.empty()) return false;
  const json &last = request_messages.back();
  return last.value("role", std::string()) == "tool";
}

// Refusal phrasing observed live in nvim (see plan file) - the model
// confidently claims no file/tool access despite tools being provided.
// Deliberately a plain substring/regex list, not exhaustive by design -
// easy to extend once more real phrasings are seen.
const std::vector<std::regex> &refusal_patterns() {
  static const std::vector<std::regex> patterns = {
      std::regex("don't have (direct )?access to", std::regex::icase),
      std::regex("do not have (direct )?access to", std::regex::icase),
      std::regex("i (can't|cannot) access", std::regex::icase),
      std::regex("unable to (read|access) files? directly", std::regex::icase),
      std::regex("please paste (the|your)", std::regex::icase),
      std::regex("could you paste", std::regex::icase),
      std::regex("share the (contents?|code) of", std::regex::icase),
      std::regex("i (don't|do not) have (file|tool) access", std::regex::icase),
  };
  return patterns;
}

bool is_false_refusal(const json &message) {
  if (message.contains("tool_calls") && message["tool_calls"].is_array() &&
      !message["tool_calls"].empty()) {
    return false; // it actually called a tool - not a refusal
  }
  std::string content = message.value("content", std::string());
  if (content.empty()) return false;
  for (const auto &re : refusal_patterns()) {
    if (std::regex_search(content, re)) return true;
  }
  return false;
}

const char *NUDGE_TEXT =
    "You do have working tools available right now and full access to the "
    "files/results they return - use them. Call another tool if you need "
    "more information, or give your actual answer now. Don't stop with "
    "nothing, and don't claim you lack file or tool access.";

json build_retry_body(json body) {
  json nudge = {{"role", "user"}, {"content", NUDGE_TEXT}};
  body["messages"].push_back(nudge);
  body["stream"] = false;
  body.erase("stream_options");
  return body;
}

struct UpstreamResult {
  bool ok = false;
  json response;
};

UpstreamResult call_upstream_blocking(httplib::Client &cli, const json &body) {
  UpstreamResult result;
  auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
  if (!res || res->status != 200) {
    return result;
  }
  try {
    result.response = json::parse(res->body);
    result.ok = true;
  } catch (...) {
    result.ok = false;
  }
  return result;
}

// --- The buffered, tool-aware path ---
void handle_tools_request(httplib::Client &cli, httplib::Response &res,
                           json body) {
  bool client_wants_stream = body.value("stream", false);
  json request_messages = body.value("messages", json::array());

  json upstream_body = body;
  upstream_body["stream"] = false;
  upstream_body.erase("stream_options");

  UpstreamResult first = call_upstream_blocking(cli, upstream_body);
  if (!first.ok) {
    res.status = 502;
    res.set_content(R"({"error":"gpu-guard: upstream request failed"})",
                     "application/json");
    return;
  }

  json final_response = first.response;
  try {
    json &message = final_response["choices"][0]["message"];
    if (is_degenerate_stall(message, request_messages) ||
        is_false_refusal(message)) {
      json retry_body = build_retry_body(body);
      UpstreamResult retry = call_upstream_blocking(cli, retry_body);
      if (retry.ok) {
        final_response = retry.response;
      }
      // If the retry itself failed to even come back, we fall through and
      // use the first (possibly stalled/refusing) response as-is - never
      // block the user on an upstream error, and never retry more than once.
    }
  } catch (...) {
    // Malformed upstream shape - just pass through whatever we got.
  }

  if (!client_wants_stream) {
    res.set_content(final_response.dump(), "application/json");
    return;
  }

  // Client asked for SSE - synthesize a minimal stream from the single
  // complete response we already have (same tradeoff the NPU server's
  // ToolCallScanner makes: tool-bearing turns are never token-streamed,
  // since detection/retry needs the complete message first anyway).
  std::string model = final_response.value("model", std::string("gpu-guard"));
  std::string id =
      final_response.value("id", std::string("chatcmpl-gpu-guard"));
  long created = final_response.value("created", (long)time(nullptr));

  auto make_chunk = [&](json delta, const char *finish_reason) {
    json chunk = {{"id", id},
                  {"object", "chat.completion.chunk"},
                  {"created", created},
                  {"model", model},
                  {"choices",
                   json::array({{{"index", 0},
                                 {"delta", delta},
                                 {"finish_reason", finish_reason ? json(finish_reason) : json()}}})}};
    return "data: " + chunk.dump() + "\n\n";
  };

  std::string body_out;
  const json &message = final_response["choices"][0]["message"];
  body_out += make_chunk({{"role", "assistant"}}, nullptr);

  std::string content = message.value("content", std::string());
  if (message.contains("reasoning_content") &&
      !message["reasoning_content"].is_null()) {
    body_out += make_chunk(
        {{"reasoning_content", message["reasoning_content"]}}, nullptr);
  }
  if (!content.empty()) {
    body_out += make_chunk({{"content", content}}, nullptr);
  }

  bool has_tool_calls = message.contains("tool_calls") &&
                        message["tool_calls"].is_array() &&
                        !message["tool_calls"].empty();
  if (has_tool_calls) {
    json tool_calls = message["tool_calls"];
    for (size_t i = 0; i < tool_calls.size(); ++i) {
      json tc = tool_calls[i];
      tc["index"] = (int)i;
      body_out += make_chunk({{"tool_calls", json::array({tc})}}, nullptr);
    }
  }

  const char *finish_reason = has_tool_calls ? "tool_calls" : "stop";
  body_out += make_chunk(json::object(), finish_reason);
  body_out += "data: [DONE]\n\n";

  res.set_content(body_out, "text/event-stream");
}

// --- The real-time streaming passthrough path (no tools involved) ---
// A small thread-safe queue bridges the upstream client callback (which
// runs on a background thread we spawn) and the server's chunked content
// provider (called repeatedly on httplib's own worker thread).
struct StreamBridge {
  std::mutex m;
  std::condition_variable cv;
  std::deque<std::string> chunks;
  bool done = false;
  bool upstream_failed = false;

  void push(const std::string &data) {
    std::lock_guard<std::mutex> lock(m);
    chunks.push_back(data);
    cv.notify_one();
  }

  void finish(bool failed) {
    std::lock_guard<std::mutex> lock(m);
    done = true;
    upstream_failed = failed;
    cv.notify_one();
  }

  // Returns false once there is nothing left and upstream is done.
  bool next(std::string &out) {
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&] { return !chunks.empty() || done; });
    if (!chunks.empty()) {
      out = std::move(chunks.front());
      chunks.pop_front();
      return true;
    }
    return false; // done and drained
  }
};

void stream_passthrough(const std::string &upstream_host, int upstream_port,
                         const std::string &raw_body, httplib::Response &res) {
  auto bridge = std::make_shared<StreamBridge>();

  std::thread([upstream_host, upstream_port, raw_body, bridge]() {
    httplib::Client cli(upstream_host, upstream_port);
    cli.set_read_timeout(3600, 0);
    cli.set_write_timeout(60, 0);

    httplib::Request req;
    req.method = "POST";
    req.path = "/v1/chat/completions";
    req.headers.emplace("Content-Type", "application/json");
    req.body = raw_body;
    req.content_receiver = [bridge](const char *data, size_t len, uint64_t,
                                     uint64_t) {
      bridge->push(std::string(data, len));
      return true;
    };

    httplib::Response upstream_res;
    httplib::Error error = httplib::Error::Success;
    bool ok = cli.send(req, upstream_res, error);
    bridge->finish(!ok || upstream_res.status != 200);
  }).detach();

  res.set_chunked_content_provider(
      "text/event-stream",
      [bridge](size_t /*offset*/, httplib::DataSink &sink) {
        std::string data;
        if (bridge->next(data)) {
          sink.write(data.data(), data.size());
          return true;
        }
        if (bridge->upstream_failed) {
          sink.write("data: {\"error\":\"gpu-guard: upstream failed\"}\n\n",
                     46);
        }
        sink.done();
        return true;
      });
}

void proxy_transparent(const std::string &upstream_host, int upstream_port,
                       const httplib::Request &req, httplib::Response &res) {
  httplib::Client cli(upstream_host, upstream_port);
  cli.set_read_timeout(3600, 0);
  auto upstream_res = cli.Get(req.path.c_str());
  if (!upstream_res) {
    res.status = 502;
    res.set_content(R"({"error":"gpu-guard: upstream unreachable"})",
                     "application/json");
    return;
  }
  res.status = upstream_res->status;
  res.set_content(upstream_res->body,
                  upstream_res->get_header_value("Content-Type").c_str());
}

}  // namespace

// Deterministic checks of the detection logic against fabricated inputs -
// exercised via `gpu-guard --selftest`, since the live model's behavior
// (especially the failure modes themselves) can't be reliably triggered
// on demand for a real integration test.
int run_selftest() {
  int failures = 0;
  auto check = [&](const char *name, bool got, bool want) {
    if (got != want) {
      std::cerr << "FAIL " << name << ": got " << got << " want " << want
                << std::endl;
      failures++;
    } else {
      std::cout << "PASS " << name << std::endl;
    }
  };

  json messages_ending_in_tool = json::array(
      {{{"role", "user"}, {"content", "do it"}},
       {{"role", "tool"}, {"content", "some result"}}});
  json messages_ending_in_user = json::array(
      {{{"role", "user"}, {"content", "do it"}}});

  check("stall: empty content after tool result -> true",
        is_degenerate_stall({{"role", "assistant"}, {"content", ""}},
                             messages_ending_in_tool),
        true);
  check("stall: whitespace-only content after tool result -> true",
        is_degenerate_stall({{"role", "assistant"}, {"content", "  \n\t"}},
                             messages_ending_in_tool),
        true);
  check("stall: real content after tool result -> false",
        is_degenerate_stall(
            {{"role", "assistant"}, {"content", "here is the answer"}},
            messages_ending_in_tool),
        false);
  check("stall: has tool_calls -> false regardless of content",
        is_degenerate_stall(
            {{"role", "assistant"},
             {"content", ""},
             {"tool_calls", json::array({{{"id", "x"}}})}},
            messages_ending_in_tool),
        false);
  check("stall: empty content but last message is user, not tool -> false",
        is_degenerate_stall({{"role", "assistant"}, {"content", ""}},
                             messages_ending_in_user),
        false);

  check("refusal: 'I don't have access to' -> true",
        is_false_refusal({{"role", "assistant"},
                          {"content", "I don't have access to the file."}}),
        true);
  check("refusal: 'please paste the contents' -> true",
        is_false_refusal(
            {{"role", "assistant"},
             {"content", "Could you please paste the file contents?"}}),
        true);
  check("refusal: normal answer -> false",
        is_false_refusal(
            {{"role", "assistant"}, {"content", "The bug is on line 42."}}),
        false);
  check("refusal: has tool_calls -> false even if phrasing matches",
        is_false_refusal(
            {{"role", "assistant"},
             {"content", "I don't have access to that yet, checking now."},
             {"tool_calls", json::array({{{"id", "x"}}})}}),
        false);

  json retry = build_retry_body(
      {{"messages", messages_ending_in_tool}, {"stream", true},
       {"stream_options", {{"include_usage", true}}}});
  check("retry: appends nudge message", retry["messages"].size() == 3, true);
  check("retry: nudge role is user",
        retry["messages"].back()["role"] == "user", true);
  check("retry: forces stream=false", retry["stream"] == false, true);
  check("retry: strips stream_options", !retry.contains("stream_options"),
        true);

  std::cout << (failures == 0 ? "All checks passed." : "Some checks FAILED.")
            << std::endl;
  return failures == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc > 1 && std::string(argv[1]) == "--selftest") {
    return run_selftest();
  }
  std::string listen_host = env_or("GPU_GUARD_HOST", "127.0.0.1");
  int listen_port = env_or_int("GPU_GUARD_PORT", 8899);
  std::string upstream_host = env_or("GPU_UPSTREAM_HOST", "127.0.0.1");
  int upstream_port = env_or_int("GPU_UPSTREAM_PORT", 8901);

  httplib::Server server;

  server.Get("/health", [&](const httplib::Request &req, httplib::Response &res) {
    proxy_transparent(upstream_host, upstream_port, req, res);
  });
  server.Get("/v1/models", [&](const httplib::Request &req, httplib::Response &res) {
    proxy_transparent(upstream_host, upstream_port, req, res);
  });

  server.Post("/v1/chat/completions", [&](const httplib::Request &req,
                                           httplib::Response &res) {
    json body;
    try {
      body = json::parse(req.body);
    } catch (...) {
      res.status = 400;
      res.set_content(R"({"error":"gpu-guard: invalid JSON body"})",
                       "application/json");
      return;
    }

    bool has_tools = body.contains("tools") && body["tools"].is_array() &&
                     !body["tools"].empty();

    if (!has_tools) {
      stream_passthrough(upstream_host, upstream_port, req.body, res);
      return;
    }

    httplib::Client cli(upstream_host, upstream_port);
    cli.set_read_timeout(3600, 0);
    cli.set_write_timeout(60, 0);
    handle_tools_request(cli, res, body);
  });

  std::cout << "gpu-guard listening on " << listen_host << ":" << listen_port
            << ", forwarding to " << upstream_host << ":" << upstream_port
            << std::endl;
  server.listen(listen_host.c_str(), listen_port);
  return 0;
}
