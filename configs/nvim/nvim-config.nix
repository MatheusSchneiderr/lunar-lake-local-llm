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
              -- Qwen3.6-35B-A3B (MoE) via llama.cpp's SYCL backend on the
              -- Arc iGPU (docs/12) - replaced the original Vulkan backend
              -- entirely after North-Mini failed in production and a
              -- confirmed unfixed Vulkan coopmat crash bug motivated an
              -- alternative-engine search. Deliberately NOT openvino_genai -
              -- OpenVINO's own converted build of this model runs ~2x
              -- larger with no equivalently aggressive quantization
              -- available, and MLC-LLM's only real Intel-GPU path is
              -- Vulkan - the same backend this switch was meant to escape.
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
                        ["qwen3.6-35b-a3b-gpu"] = { meta = { context_window = 131072 } },
                      },
                    },
                    -- Single toggle in the per-chat settings block at the
                    -- top of the buffer - edit `thinking: true` there to
                    -- swap the ENTIRE sampling preset, not just the
                    -- enable_thinking flag. Mirrors the two full presets
                    -- set up for opencode's two model entries
                    -- (configs/opencode/opencode.json). Base numbers are
                    -- Qwen's own documented mode-specific presets, with one
                    -- deliberate deviation - see docs/13 for the full
                    -- runaway-thinking investigation and the presence_penalty
                    -- A/B behind these exact numbers:
                    --   thinking=true:  temp=0.6, top_p=0.95, top_k=20,
                    --                   min_p=0 (Qwen's "precise/coding"
                    --                   thinking preset) PLUS
                    --                   presence_penalty=1.0, added after a
                    --                   dedicated research pass found
                    --                   paraphrastic self-repetition (not
                    --                   literal/n-gram repetition)
                    --                   suppresses the </think> token's
                    --                   probability. Qwen's own coding
                    --                   preset sets presence_penalty=0 (no
                    --                   anti-repetition at all), which the
                    --                   same research flagged as a real gap
                    --                   for this exact failure mode.
                    --   thinking=false: temp=0.7, top_p=0.8, top_k=20,
                    --                   min_p=0, presence_penalty=1.5
                    --                   (Qwen's non-thinking preset -
                    --                   presence_penalty is the actual
                    --                   documented lever against literal-
                    --                   repetition/greedy loops in this
                    --                   mode, previously entirely unset)
                    -- This single boolean can't declaratively map to 5
                    -- different body fields at once (codecompanion's
                    -- schema `mapping` is one field -> one body path), so
                    -- it maps to a throwaway path (meta.thinking, never a
                    -- real API field) purely to carry the boolean through
                    -- to the form_parameters handler below, which reads
                    -- it, deletes the throwaway field, and injects the
                    -- whole matching preset - confirmed via codecompanion
                    -- source (adapters/http/init.lua's
                    -- map_schema_to_params -> http.lua's
                    -- Client.merge_body) that form_parameters receives
                    -- the schema-mapped params before the request is
                    -- built, so this ordering is real, not assumed.
                    --
                    -- Defaulted to false, per the same real production
                    -- incident that drove the presence_penalty work above:
                    -- a live task once generated 9800+ tokens with zero
                    -- output, stuck entirely inside <think>. Root cause
                    -- (docs/13) is paraphrastic self-repetition suppressing
                    -- the </think> token's probability - a documented,
                    -- model-family-wide Qwen3/3.6 issue, not specific to
                    -- this config. Rule of thumb: if you'd be satisfied
                    -- with the first reasonable answer, leave it off; flip
                    -- it on for a follow-up message when a result looks
                    -- shallow and you want it to double-check its own work.
                    thinking = {
                      order = 1,
                      mapping = "meta",
                      type = "boolean",
                      default = false,
                      desc = "Full Qwen3.6 sampling preset toggle: true = thinking mode (temp 0.6/top_p 0.95/presence_penalty 1.0), false = non-thinking mode (temp 0.7/top_p 0.8/presence_penalty 1.5, the fastest configuration found in this project).",
                    },
                  },
                  handlers = {
                    -- Reads the `thinking` toggle's mapped value (see
                    -- schema comment above), then replaces it with the
                    -- full matching sampling preset instead of a single
                    -- field - this is what makes one boolean apply Qwen's
                    -- whole mode-specific recommendation at once.
                    form_parameters = function(self, params, messages)
                      local thinking = params.meta and params.meta.thinking
                      params.meta = nil

                      local preset
                      if thinking then
                        preset = { temperature = 0.6, top_p = 0.95, top_k = 20, min_p = 0, presence_penalty = 1.0 }
                      else
                        preset = { temperature = 0.7, top_p = 0.8, top_k = 20, min_p = 0, presence_penalty = 1.5 }
                      end
                      for k, v in pairs(preset) do
                        params[k] = v
                      end

                      params.chat_template_kwargs = vim.tbl_deep_extend(
                        "force",
                        params.chat_template_kwargs or {},
                        { enable_thinking = thinking or false }
                      )

                      return params
                    end,
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
      -- the model (~10-15s for a ~10GB GGUF onto the iGPU via SYCL - the
      -- SYCL backend reports /health as ready noticeably later than
      -- Vulkan did, so don't assume a naive short sleep is enough).
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
