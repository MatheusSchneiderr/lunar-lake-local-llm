# Excerpt from a real nvf (NixOS + Neovim flake) configuration - only the
# parts relevant to codecompanion.nvim's local-LLM integration. This is NOT
# meant to be dropped in wholesale as a full nvim config; splice the
# `assistant.codecompanion-nvim`, `luaConfigRC`, and keymap pieces into
# your own nvf/nixvim/lazy.nvim setup, adjusted for however you manage
# plugins. See docs/05-nvim-integration.md for the full explanation of
# every gotcha referenced in these comments.
{ config, lib, pkgs, ... }:

{
  programs.nvf.settings.vim = {
    assistant.codecompanion-nvim = {
      enable = true;
      setupOpts = {
        # Only one adapter here: this project also runs an NPU tier
        # (Qwen2.5-Coder-7B), but the GPU model's tok/s made the NPU
        # fallback unnecessary in day-to-day nvim use - the NPU systemd
        # service still exists (see ../npu-tier), just isn't wired into
        # nvim. Must nest under `http` - this codecompanion version
        # organizes adapters by transport type (adapters.http.*,
        # adapters.acp.*), not a flat `adapters.*` table.
        #
        # api_key must be non-empty: codecompanion's get_schema() splits
        # this string on "." to walk a schema path, and an empty string
        # produces zero path segments, so the lookup loop never runs and
        # it degenerately returns the whole adapter table instead of nil -
        # which then crashes string substitution ("invalid replacement
        # value (a table)") when building the request. Any real string
        # avoids the bug since a local server needs no auth.
        adapters = lib.mkLuaInline ''
          {
            http = {
              -- Qwen3.6-35B-A3B (MoE) via llama.cpp's Vulkan backend on
              -- the Arc iGPU. Deliberately NOT openvino_genai -
              -- benchmarking showed OpenVINO's GPU-MoE inference path
              -- never completes a thinking-mode answer and destabilizes
              -- the GPU driver (see docs/03-gpu-tier-setup.md), while
              -- llama.cpp on the same hardware and quantization gets it
              -- right almost every time.
              gpu_hard = function()
                return require("codecompanion.adapters").extend("openai_compatible", {
                  env = { url = "http://127.0.0.1:8901", api_key = "not-needed" },
                  schema = {
                    model = {
                      default = "qwen3.6-35b-a3b-gpu",
                      -- context_window here is what powers the token-count
                      -- percentage below (adapters.shared.context_window
                      -- reads schema.model.choices[model].meta.context_window)
                      -- - keep in sync with gpu-server-hard's -c value.
                      choices = {
                        ["qwen3.6-35b-a3b-gpu"] = { meta = { context_window = 32768 } },
                      },
                    },
                    -- Toggleable per-chat in the settings block at the top
                    -- of the buffer, same mechanism as the built-in
                    -- gemini.lua adapter's thinkingLevel setting. Maps to
                    -- the top-level request field llama-server reads for
                    -- this: {"chat_template_kwargs": {"enable_thinking":
                    -- bool}} (mapping="body.chat_template_kwargs" merges
                    -- adapter.body straight into the request JSON - see
                    -- codecompanion's http.lua Client.merge_body).
                    --
                    -- Defaulted to false after a real benchmark: paired
                    -- thinking-on/off runs across 5 verifiable-answer
                    -- prompts (fixed seeds, 1536-token cap) found
                    -- near-identical tok/s (~39 vs ~41) and a small
                    -- draft-acceptance edge for thinking off (~87% vs
                    -- ~83%), but thinking-on completely failed to
                    -- converge on 2 of 5 prompt types (ran out the full
                    -- token budget spiraling in reasoning, 0/2 seeds
                    -- each), while thinking-off got 11/11 correct across
                    -- the whole matrix. See
                    -- docs/04-thinking-mode-and-preservation.md for the
                    -- full table.
                    --
                    -- Rule of thumb for when to flip it on: if you'd be
                    -- satisfied with the first reasonable answer, leave
                    -- it off. For tasks involving a tool-calling chain
                    -- (reading multiple files, then synthesizing a
                    -- result) leave it off too - thinking-under-a-tool-
                    -- chain was never actually benchmarked (only plain
                    -- single-turn prompts were), and tool-calling chains
                    -- are exactly where this model's degenerate-stall/
                    -- false-tool-refusal failure mode shows up (see
                    -- docs/08-troubleshooting-and-incidents.md). Instead,
                    -- work in two steps: (1) leave it off, let it gather
                    -- info and draft the result; (2) if the result looks
                    -- shallow or misses a relationship, flip it on just
                    -- for a follow-up message asking it to double-check/
                    -- reason through what it already produced.
                    enable_thinking = {
                      order = 1,
                      mapping = "body.chat_template_kwargs",
                      type = "boolean",
                      default = false,
                      desc = "Thinking mode - benchmarked worse convergence, no speed benefit. Toggle on when you specifically want visible deliberation.",
                    },
                  },
                  handlers = {
                    -- openai_compatible.lua uses the OLD flat handler
                    -- format (no handlers.response.* nesting), so the
                    -- reasoning-extraction hook is registered under its
                    -- old flat name (see codecompanion's
                    -- adapters/http/init.lua get_handler(): parse_meta
                    -- maps to "parse_message_meta" for old-format
                    -- adapters). llama-server already sends
                    -- reasoning_content as a non-standard field, which
                    -- openai.lua's own find_extra_fields() already
                    -- collects into data.extra generically - this just
                    -- moves it into data.output.reasoning so
                    -- show_reasoning (on by default) actually renders it,
                    -- mirroring the built-in deepseek adapter's approach.
                    parse_message_meta = function(self, data)
                      local reasoning_content = data.extra and data.extra.reasoning_content
                      if reasoning_content then
                        data.output.reasoning = { content = reasoning_content }
                        if data.output.content == "" then
                          data.output.content = nil
                        end
                      end
                      return data
                    end,
                  },
                })
              end,
            },
          }
        '';
        interactions = {
          chat.adapter = "gpu_hard";
          inline.adapter = "gpu_hard";
        };
        display.chat.token_count = lib.mkLuaInline ''
          function(tokens, adapter)
            local ok, shared = pcall(require, "codecompanion.adapters.shared")
            local window = ok and shared.context_window(adapter)
            if window and window > 0 then
              local pct = math.floor((tokens / window) * 100)
              return string.format(" (%d/%d tokens, %d%%)", tokens, window, pct)
            end
            return " (" .. tokens .. " tokens)"
          end
        '';
        # Required for enable_thinking (and any other schema field, e.g.
        # model choice) to actually be visible/editable - codecompanion
        # hides the settings block by default and has no per-chat keymap
        # to reveal it, only this global option. Renders as a YAML block
        # at the top of every chat buffer - edit the text directly
        # (e.g. change `enable_thinking: false` to `true`) before
        # submitting to change that turn's request.
        display.chat.show_settings = true;
      };
    };

    luaConfigRC.gpuServerWaitHelper = ''
      -- Start-and-wait-for-health helper: starts the systemd service, then
      -- polls /health before actually opening the chat, so you don't open
      -- an empty chat buffer against a server that's still cold-loading
      -- the model (~10-20s for a ~21GB GGUF onto the iGPU via Vulkan).
      _G.GpuStartAndOpen = function(open_fn)
        vim.fn.jobstart({ "systemctl", "--user", "start", "gpu-server-hard" }, { detach = true })

        vim.system(
          { "curl", "-s", "-m", "1", "-o", "/dev/null", "-w", "%{http_code}", "http://127.0.0.1:8901/health" },
          { text = true },
          function(res)
            vim.schedule(function()
              if res.code == 0 and res.stdout == "200" then
                open_fn()
              else
                vim.notify(
                  "GPU server is not up yet (cold start takes ~10-20s) - try again shortly",
                  vim.log.levels.WARN
                )
              end
            end)
          end
        )
      end
    '';

    keymaps = [
      {
        key = "<leader>ac";
        lua = true;
        action = ''
          function()
            GpuStartAndOpen(function()
              vim.cmd('CodeCompanionChat Toggle')
            end)
          end
        '';
        mode = "n";
        desc = "Toggle CodeCompanion chat (waits for local GPU model)";
      }
      {
        key = "<leader>ac";
        lua = true;
        action = ''
          function()
            GpuStartAndOpen(function()
              vim.cmd("'<,'>CodeCompanionChat Add")
            end)
          end
        '';
        mode = "v";
        desc = "Add visual selection to CodeCompanion chat (waits for local GPU model)";
      }
      {
        # The bare `CodeCompanion` command (not `CodeCompanionChat`) is the
        # real Inline Assistant: it applies the LLM's response directly to
        # the buffer as a diff (accept with `ga`, reject with `gr`), via
        # plain text completion - no tool-calling required.
        key = "<leader>ai";
        lua = true;
        action = ''
          function()
            GpuStartAndOpen(function()
              vim.cmd('CodeCompanion')
            end)
          end
        '';
        mode = "n";
        desc = "Inline-edit at cursor with CodeCompanion (waits for local GPU model)";
      }
      {
        key = "<leader>ai";
        lua = true;
        action = ''
          function()
            GpuStartAndOpen(function()
              vim.cmd("'<,'>CodeCompanion")
            end)
          end
        '';
        mode = "v";
        desc = "Inline-edit visual selection with CodeCompanion (waits for local GPU model)";
      }
    ];
  };
}
