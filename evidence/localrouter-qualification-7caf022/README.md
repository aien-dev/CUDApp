# LocalRouter ↔ Omega exact-source qualification

**Disposition: partial compatibility demonstrated; not a drop-in/release qualification.** This is an evidence-only candidate, not a source implementation or a claim of speedup.

## Immutable inputs and probe topology

- LocalRouter upstream: [`jayleaton/localrouter` @ `7caf022157c7aba57d19452bed9002967da6a67a`](https://github.com/jayleaton/localrouter/tree/7caf022157c7aba57d19452bed9002967da6a67a), branch `qualification/localrouter-7caf022-20261023`.
- Built executable SHA-256: `8d02bcfc022398a99f2999ca4daa57b642fc49c664b2adb069534d790d31d66a`.
- Zig `0.17.0`; compiler SHA-256 `5165c9a2d17938a8173839b3adf81d025b558d38d9e9711237a6fa5b8d8bb3f7`.
- CUDApp integration context: open draft PR #28, source/base `e1f51071269614e25e8ef74aec4fbf4e318d0060`; issue #27 remains open. No PR was created or modified for this qualification.
- The isolated probe routed `127.0.0.1:18329` → LocalRouter `openai_server` → on-demand `socat` on `127.0.0.1:18330` → the already-running Omega service on `127.0.0.1:18091`. The probe used the actual server model ID `qwen3.6-35b-a3b-fp8`. Its config has a placeholder weights path and is explicitly a relay, not a LocalRouter-managed Omega process. No second 35B process or model download was started. The existing router on `18327` and Omega service on `18091` were left untouched.

## Compatibility matrix

| Area | Result | What the evidence establishes (and does not) |
| --- | --- | --- |
| Model discovery | **PASS, narrow** | `GET /v1/models` returned 200 and one model, `qwen3.6-35b-a3b-fp8`, with only `chat_completion`. It was initially unloaded. |
| Model-ID contract | **MISMATCH reproduced, then corrected** | `omega-live-35b` returned 400 (`server serves qwen3.6-35b-a3b-fp8`). With the exact backend ID, the chat probes passed. This is a name/config mismatch, not a transport failure. |
| Whole chat | **PASS** | Matched-ID `/v1/chat/completions` returned 200, assistant text `Omega`. |
| Streaming chat | **PASS** | Matched-ID SSE returned 200, content delta, final usage event, and `[DONE]`. |
| Multi-turn chat | **PASS** | Two-turn probe returned 200 and `SECOND`. This is a transport/format check, not semantic parity. |
| Function/tool calling | **PASS for chat protocol** | Forced `test_echo` returned 200 with `finish_reason=tool_calls`; the follow-up containing the tool result returned 200 and `DONE`. |
| MCP | **PASS for exercised chat paths** | `/mcp/tools/list` returned 200 with 17 tool definitions; MCP chat returned 200 with `isError:false`; MCP chat tool invocation returned 200 and a tool-call response. This does not qualify the listed image/audio/video/computer tools against live models/devices. |
| Unknown model | **PASS, expected rejection** | Unconfigured model returned 404 and named the one configured chat model. |
| Voice UI | **STATIC PAGE ONLY** | `GET /voice` returned 200 HTML (171,398 bytes). No microphone, speech, realtime, voice-session, or end-to-end UI workflow was exercised. |
| Image generation | **NOT AVAILABLE in this probe** | `qwen-image-2.1` returned 404; no configured `text_to_image` model. No image weights were downloaded or run. |
| Video generation | **NOT AVAILABLE in this probe** | Valid `minimax-h3` text-to-video request returned 404; no configured `text_to_video` model. No video weights were downloaded or run. |
| Speech synthesis | **NOT AVAILABLE in this probe** | `fishaudio/drama-3` returned 404; no configured `text_to_speech` model. |
| Transcription, realtime, computer control | **NOT QUALIFIED** | No valid transcription workload, realtime session, or paired computer helper was run. Do not infer support from route/tool listings. |
| Unload/reload | **FACADE RESPONSE ONLY** | LocalRouter's unload returned 200 with `state=unloaded`; a subsequent chat returned 200. Because the configured server is a `socat` relay, this does not unload/restart Omega or prove model memory release, child cleanup, or service lifecycle. Omega health stayed `ok` before and after. |
| CUDApp profile/tests | **BASELINE PASS, incompatible contract** | Existing CUDApp setup tests pass 9/9. But `tools/localrouter/setup.py` explicitly disables voice tools and expects tool calls to be rejected with HTTP 400; the matched-ID candidate instead returns a valid tool call with HTTP 200. The current PR's text-only smoke profile is not the requested seamless foundation swap. No Python tests or implementation were changed. |
| Upstream integration suite | **PASS** | Pinned-source `integration-test` completed all **41/41** tests under `quiet-build` with a 300-second ceiling. Earlier 120-second run stopped at test 35 because test 36 had started but had not completed; test 36 also passed in isolation. The separate `ClientGone` warning test passed 1/1 and emits that warning as part of its negative case. These are upstream test fixtures, not live multimodal Omega inference. |
| Speed / behavior preservation | **NOT MEASURED / NOT QUALIFIED** | No same-workload before/after throughput or latency comparison was run. The live model is Qwen3.6-35B; current LocalRouter README describes a different Qwen3.8-27B setup. No same-model golden corpus or behavior-equivalence evidence exists. No speedup claim is made. |

## Source and integration findings

- The pinned LocalRouter tree includes `/voice`, `/v1` audio/image/video/realtime routes, MCP, and computer-control routes, but its `openai_server` engine exposes a chat-completion capability. Actual models/capabilities must be configured and served; route presence and MCP definitions alone do not supply them.
- The live Omega server served model ID `qwen3.6-35b-a3b-fp8`. The mismatched alias was corrected only in the isolated probe config. The CUDApp branch has a `--model-id` option, but a fresh supervised Omega server with that alias was not started because it would require a second 35B load. Alias-on-launch and the actual CUDApp lifecycle remain unverified.
- The open CUDApp PR #28 is a draft and explicitly text-only; it is not evidence that commands, config, UI, APIs/MCP, voice, image/video workflows, or lifecycle remain unchanged.
- Canonical L458 review findings were checked before adding this evidence: L479 currently owns/resolves the Make override, overlay/skip-worktree byte-binding, and destination-symlink findings (#1–3); L457 remains the owner for dependency bootstrap and the STABLE recovery contract (#4–5). This candidate does not duplicate those repairs.

## Remaining gates

1. Run a supervised standalone Omega instance under an admitted hardware window; verify configured model alias, startup/readiness, load/unload/reload, memory accounting, and clean child/service exit without touching the existing service.
2. Wire or otherwise qualify the real audio/transcription, voice/realtime, image, video, MCP-tool, and computer-control workflows required by the acceptance bar. Use approved models/devices and existing shared admission/locks; no weights or GPU runs were added here.
3. Reconcile CUDApp's text-only profile and 400-on-tool-call expectation with actual LocalRouter tool behavior, while preserving the existing user-facing command/config/UI/API/MCP contracts. This candidate intentionally contains no new executable implementation or test logic.
4. Run a same-model, same-prompt, same-hardware before/after acceptance corpus and performance measurements. Until that evidence exists, neither behavioral parity nor speed improvement is established.
5. Resolve the still-owned L457 dependency-bootstrap and STABLE-recovery findings before relying on the canonical Omega deployment path; obtain independent review of this evidence branch.

## Raw receipts

`raw/` contains the probe config, request/response bodies and HTTP headers for the exercised routes, the matched/unmatched model outputs, MCP responses, lifecycle responses, and the complete 41-test log. `SHA256SUMS` binds each copied receipt. The qualification is limited to these pinned source/toolchain artifacts and the route topology above; it is not a release certification.
