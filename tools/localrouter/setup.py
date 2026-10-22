#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare and probe an isolated, text-only LocalRouter -> Omega candidate."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import socket
import sys
import time


MAX_BODY = 2 * 1024 * 1024
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def engine_of(cfg):
    command = cfg["tools"][0]["options"]["server"]
    return command[command.index("--engine") + 1]


def make_config(*, server, model_dir, model_id, engine, resident_bytes,
                working_bytes, data_dir, router_port, backend_port, reserve_bytes):
    cfg = {
        "host": "127.0.0.1", "port": router_port, "data_dir": data_dir,
        "reserve_bytes": reserve_bytes, "gpu_lock": "",
        "defaults": {"chat_completion": model_id},
        "voice": {"tools": False, "manage_models": False, "speculate": False,
                  "prime_chat": False},
        "tools": [{
            "id": model_id, "kind": "chat", "engine": "openai_server",
            "name": "Omega text candidate", "weights": model_dir,
            "resident_bytes": resident_bytes, "working_bytes": working_bytes,
            "keep_loaded": False, "idle_ttl_s": 300, "priority": 100,
            "load_timeout_s": 900, "generate_timeout_s": 180,
            "options": {
                "port": backend_port, "health": "/health", "stop_timeout_s": 30,
                "server": [server, "--engine", engine, "--model-dir", "{weights}",
                           "--bind", "127.0.0.1", "--port", "{port}",
                           "--model-id", "{id}", "--max-context", "4096"],
            },
        }],
    }
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    """Validate only this setup profile, not arbitrary LocalRouter configurations."""
    require(cfg.get("host") == "127.0.0.1", "candidate must bind 127.0.0.1")
    require(cfg.get("gpu_lock") == "", "LocalRouter must not hold a shared GPU lock in this text-only profile")
    tools = cfg.get("tools", [])
    require(len(tools) == 1, "candidate must contain exactly one text tool; mixed GPU scheduling is unqualified")
    tool = tools[0]
    require(tool.get("kind") == "chat" and tool.get("engine") == "openai_server", "expected one openai_server chat tool")
    require(MODEL_ID.fullmatch(tool.get("id", "")), "model id must be a simple name, not a path")
    require(cfg.get("defaults", {}).get("chat_completion") == tool["id"], "default model must match the backend model id")
    require(tool.get("keep_loaded") is False, "candidate must load on demand, not at daemon startup")
    voice = cfg.get("voice", {})
    require(all(voice.get(k) is False for k in ("tools", "manage_models", "speculate", "prime_chat")),
            "disable voice tool calling, model management, speculation and priming")
    opts = tool.get("options", {})
    command = opts.get("server", [])
    require(isinstance(command, list) and all(isinstance(x, str) for x in command), "server command must be an argv array")
    require(len(command) == 13, "unexpected server command; regenerate this candidate profile")
    engine = command[2]
    require(engine in ("stub", "omega"), "engine must be stub or omega")
    require(command[1:] == ["--engine", engine, "--model-dir", "{weights}", "--bind", "127.0.0.1",
                            "--port", "{port}", "--model-id", "{id}", "--max-context", "4096"],
            "server must use the inspected loopback, port and model-id contract")
    require("env" not in opts, "this profile must not override Omega's lock environment")
    for path in (command[0], tool.get("weights", ""), cfg.get("data_dir", "")):
        require(Path(path).is_absolute() and not any(c in path for c in "\x00\n\r{}"),
                "server, model and data paths must be absolute and contain no template braces or control characters")
    ports = (cfg.get("port"), opts.get("port"))
    require(all(type(p) is int and 1024 <= p <= 65535 and p != 18091 for p in ports),
            "use unprivileged candidate ports; live motor port 18091 is reserved")
    require(ports[0] != ports[1], "router and backend ports must differ")
    for key, value in (("resident_bytes", tool.get("resident_bytes")),
                       ("working_bytes", tool.get("working_bytes")),
                       ("reserve_bytes", cfg.get("reserve_bytes"))):
        require(type(value) is int and 0 <= value < 2**63, key + " must be a bounded nonnegative byte count")
    require(cfg["reserve_bytes"] > 0, "keep a nonzero OS reserve")
    require(engine == "stub" or tool["resident_bytes"] > 0, "Omega requires an explicit nonzero measured resident budget")


def check_files(cfg):
    validate_config(cfg)
    tool = cfg["tools"][0]
    server = Path(tool["options"]["server"][0])
    require(server.is_file() and os.access(server, os.X_OK), "server binary does not exist or is not executable: " + str(server))
    model = Path(tool["weights"])
    require(model.is_dir(), "model directory does not exist: " + str(model))
    if engine_of(cfg) == "omega":
        require((model / "tokenizer.json").is_file(), "Omega model directory needs tokenizer.json and its supported checkpoint")


def check_ports(cfg):
    for port in (cfg["port"], cfg["tools"][0]["options"]["port"]):
        try:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ValueError(f"candidate port {port} is unavailable; do not stop its owner") from exc


def check_completion(value, model):
    require(isinstance(value, dict) and value.get("model") == model and "error" not in value, "completion error or wrong model")
    choices = value.get("choices", [])
    require(len(choices) == 1, "expected one completion choice")
    choice = choices[0]
    msg = choice.get("message", {})
    require(msg.get("role") == "assistant" and not msg.get("tool_calls"), "expected a plain-text assistant response")
    text = msg.get("content")
    require(isinstance(text, str) and text.strip(), "completion has no text")
    require(choice.get("finish_reason") in ("stop", "length"), "completion has no successful finish reason")
    return text


def check_sse(lines, model):
    text = []
    finished = False
    data = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
            continue
        if line or not data:
            continue
        payload = "\n".join(data)
        data.clear()
        if payload == "[DONE]":
            require(finished and "".join(text).strip(), "SSE ended without text and a finish reason")
            return len("".join(text))
        event = json.loads(payload)
        require(isinstance(event, dict) and "error" not in event and event.get("model") == model,
                "SSE contains an error or wrong model")
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            require(not delta.get("tool_calls"), "unexpected tool call in text-only SSE")
            if delta.get("content") is not None:
                require(isinstance(delta["content"], str), "invalid content delta")
                text.append(delta["content"])
            if choice.get("finish_reason") is not None:
                require(choice["finish_reason"] in ("stop", "length"), "SSE failed finish reason")
                finished = True
    raise ValueError("SSE truncated: no complete [DONE] event")


def request(port, method, path, payload=None, *, timeout=180, stream_model=None):
    """Literal loopback only; no proxies, redirects or arbitrary upstream URLs."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        body = None if payload is None else json.dumps(payload).encode()
        conn.request(method, path, body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if stream_model is not None and response.status == 200:
            require(response.getheader("Content-Type", "").startswith("text/event-stream"), "expected SSE content type")
            def lines():
                total = 0
                deadline = time.monotonic() + timeout
                while True:
                    require(time.monotonic() < deadline, "SSE deadline exceeded")
                    if conn.sock:
                        conn.sock.settimeout(max(0.01, deadline - time.monotonic()))
                    raw = response.readline(65537)
                    if not raw:
                        return
                    total += len(raw)
                    require(len(raw) <= 65536 and total <= MAX_BODY, "SSE exceeds smoke-test size limit")
                    yield raw.decode("utf-8")
            return response.status, {"text_chars": check_sse(lines(), stream_model)}
        raw = response.read(MAX_BODY + 1)
        require(len(raw) <= MAX_BODY, "response exceeds smoke-test size limit")
        return response.status, json.loads(raw)
    finally:
        conn.close()


def exercise(cfg, timeout):
    """Explicitly sends short generation requests to an already running candidate."""
    validate_config(cfg)
    tool = cfg["tools"][0]
    model, port = tool["id"], cfg["port"]
    upstream = tool["options"]["port"]
    checks = []
    def expect(name, actual, expected):
        require(actual == expected, f"{name}: expected {expected}, got {actual}")
        checks.append(name)
    status, listed = request(port, "GET", "/v1/models", timeout=timeout)
    expect("router_models", status, 200)
    require(any(x.get("id") == model for x in listed.get("data", [])), "candidate model is not listed by router")
    messages = [{"role": "user", "content": "Reply with a short greeting."}]
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 64}
    status, reply = request(port, "POST", "/v1/chat/completions", payload, timeout=timeout)
    expect("whole_chat", status, 200)
    first = check_completion(reply, model)
    status, health = request(upstream, "GET", "/health", timeout=timeout)
    expect("backend_health", status, 200)
    require(health.get("model") == model and health.get("engine") == engine_of(cfg), "backend health model or engine mismatch")
    status, listed = request(upstream, "GET", "/v1/models", timeout=timeout)
    expect("backend_models", status, 200)
    require(any(x.get("id") == model for x in listed.get("data", [])), "backend model mismatch")
    second = dict(payload, messages=messages + [{"role": "assistant", "content": first},
                                               {"role": "user", "content": "Now reply with another short greeting."}])
    status, reply = request(port, "POST", "/v1/chat/completions", second, timeout=timeout)
    expect("second_turn", status, 200)
    check_completion(reply, model)
    status, _ = request(port, "POST", "/v1/chat/completions", dict(payload, stream=True),
                        timeout=timeout, stream_model=model)
    expect("streamed_chat", status, 200)
    function = {"type": "function", "function": {"name": "smoke_noop", "parameters": {"type": "object", "properties": {}}}}
    status, _ = request(port, "POST", "/v1/chat/completions", dict(payload, tools=[function]), timeout=timeout)
    expect("tool_call_rejected", status, 400)
    status, _ = request(port, "POST", "/v1/chat/completions", dict(payload, model="missing-smoke-model"), timeout=timeout)
    expect("unknown_model_rejected", status, 404)
    return {"http_protocol": "PASS", "engine": engine_of(cfg), "checks": checks,
            "gpu_qualification": "NOT_RUN", "lifecycle_qualification": "NOT_RUN",
            "cuda_free_provenance": "NOT_RUN",
            "note": "HTTP checks do not establish GPU correctness, memory budgets, lifecycle or deployed library provenance."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    generate = sub.add_parser("config", help="print candidate JSON; starts nothing")
    generate.add_argument("--server", required=True)
    generate.add_argument("--model-dir", required=True)
    generate.add_argument("--data-dir", required=True)
    generate.add_argument("--model-id", default="omega-candidate")
    generate.add_argument("--engine", choices=("omega", "stub"), default="omega")
    generate.add_argument("--resident-bytes", type=int, required=True)
    generate.add_argument("--working-bytes", type=int, required=True)
    generate.add_argument("--reserve-bytes", type=int, default=8 * 1024**3)
    generate.add_argument("--router-port", type=int, default=8192)
    generate.add_argument("--backend-port", type=int, default=18092)
    for name in ("check", "probe"):
        p = sub.add_parser(name, help="check setup without launching any process")
        p.add_argument("--config", type=Path, required=True)
        if name == "probe":
            p.add_argument("--exercise", action="store_true", help="send short generation requests to the running candidate")
            p.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)
    try:
        if args.action == "config":
            fields = vars(args).copy()
            fields.pop("action")
            cfg = make_config(**fields)
            check_files(cfg)
            print(json.dumps(cfg, indent=2))
        else:
            raw = args.config.read_bytes()
            cfg = json.loads(raw)
            validate_config(cfg)
            if args.action == "check":
                check_files(cfg)
                check_ports(cfg)
                result = {"setup": "PASS", "processes_started": 0, "gpu_qualification": "NOT_RUN"}
            elif args.exercise:
                require(1 <= args.timeout <= 1800, "timeout must be 1..1800 seconds")
                result = exercise(cfg, args.timeout)
            else:
                status, value = request(cfg["port"], "GET", "/v1/models", timeout=10)
                require(status == 200 and any(x.get("id") == cfg["tools"][0]["id"] for x in value.get("data", [])),
                        "candidate model not listed")
                result = {"router_listing": "PASS", "generation": "NOT_RUN", "gpu_qualification": "NOT_RUN"}
            result["config_sha256"] = hashlib.sha256(raw).hexdigest()
            print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, IndexError, AttributeError, OSError, http.client.HTTPException) as exc:
        print(f"localrouter setup: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
