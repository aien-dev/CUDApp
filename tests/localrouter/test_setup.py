# SPDX-License-Identifier: MIT
"""Host-only contract tests. No LocalRouter binary, model, or GPU is opened."""
import copy
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "tools/localrouter/setup.py"
spec = importlib.util.spec_from_file_location("localrouter_setup", SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.server = self.root / "aien-serve-http"
        self.server.write_text("#!/bin/sh\nexit 0\n")
        self.server.chmod(0o755)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "tokenizer.json").write_text("{}")

    def config(self, **extra):
        args = dict(server=str(self.server), model_dir=str(self.model),
                    model_id="omega-candidate", engine="omega",
                    resident_bytes=1024, working_bytes=512,
                    data_dir=str(self.root / "data"), router_port=8192,
                    backend_port=18092, reserve_bytes=8589934592)
        args.update(extra)
        return setup.make_config(**args)

    def test_candidate_preserves_omega_lock_without_router_shared_lock(self):
        cfg = self.config()
        setup.validate_config(cfg)
        self.assertEqual(cfg["gpu_lock"], "")
        self.assertEqual(len(cfg["tools"]), 1)
        self.assertFalse(cfg["tools"][0]["keep_loaded"])
        self.assertFalse(cfg["voice"]["tools"])
        self.assertFalse(cfg["voice"]["speculate"])
        self.assertNotIn("env", cfg["tools"][0]["options"])
        command = cfg["tools"][0]["options"]["server"]
        self.assertEqual(command[command.index("--port") + 1], "{port}")
        self.assertEqual(command[command.index("--model-id") + 1], "{id}")

    def test_rejects_live_ports_bad_budgets_and_unsafe_model_ids(self):
        for change in [dict(backend_port=18091), dict(router_port=18091),
                       dict(backend_port=8192), dict(resident_bytes=0),
                       dict(working_bytes=-1), dict(model_id="../live"),
                       dict(router_port=0), dict(reserve_bytes=0)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.config(**change)

    def test_refuses_mixed_workers_host_lock_and_remote_bind(self):
        for mutate in [lambda c: c.update(host="0.0.0.0"),
                       lambda c: c.update(gpu_lock="/tmp/aien-gb10.lock"),
                       lambda c: c["tools"].append(copy.deepcopy(c["tools"][0])),
                       lambda c: c["voice"].update(tools=True),
                       lambda c: c["tools"][0].update(keep_loaded=True)]:
            cfg = self.config()
            mutate(cfg)
            with self.assertRaises(ValueError):
                setup.validate_config(cfg)

    def test_missing_server_or_tokenizer_is_a_setup_failure(self):
        cfg = self.config()
        setup.check_files(cfg)
        (self.model / "tokenizer.json").unlink()
        with self.assertRaises(ValueError):
            setup.check_files(cfg)
        self.server.unlink()
        with self.assertRaises(ValueError):
            setup.check_files(cfg)

    def test_stub_explicitly_has_no_gpu_qualification(self):
        cfg = self.config(engine="stub", resident_bytes=0, working_bytes=0)
        (self.model / "tokenizer.json").unlink()
        setup.check_files(cfg)
        self.assertEqual(setup.engine_of(cfg), "stub")

    def test_valid_sse_and_usage_only_final_event(self):
        frames = [
            {"model": "omega-candidate", "choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
            {"model": "omega-candidate", "choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"model": "omega-candidate", "choices": [], "usage": {"completion_tokens": 1}},
        ]
        lines = [": keep-alive\n", "\n"]
        for frame in frames:
            lines.extend(["data: " + json.dumps(frame) + "\n", "\n"])
        lines.extend(["data: [DONE]\n", "\n"])
        self.assertEqual(setup.check_sse(iter(lines), "omega-candidate"), 5)

    def test_sse_rejects_error_truncation_wrong_model_and_empty_output(self):
        valid = {"model": "omega-candidate", "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}
        cases = [
            ["data: " + json.dumps(valid) + "\n", "\n"],
            ["data: {\"error\":{\"message\":\"fault\"}}\n", "\n", "data: [DONE]\n", "\n"],
            ["data: " + json.dumps(dict(valid, model="other")) + "\n", "\n", "data: [DONE]\n", "\n"],
            ["data: [DONE]\n", "\n"],
        ]
        for lines in cases:
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                setup.check_sse(iter(lines), "omega-candidate")

    def test_completion_rejects_other_model_and_tool_results(self):
        response = {"model": "omega-candidate", "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
        self.assertEqual(setup.check_completion(response, "omega-candidate"), "ok")
        with self.assertRaises(ValueError):
            setup.check_completion(dict(response, model="other"), "omega-candidate")
        response["choices"][0]["message"]["tool_calls"] = [{}]
        with self.assertRaises(ValueError):
            setup.check_completion(response, "omega-candidate")

    def test_http_probe_exercises_real_loopback_transport_but_never_claims_gpu(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def send_json(self, status, value):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self.send_json(200, {"model": "omega-candidate", "engine": "stub"})
                elif self.path == "/v1/models":
                    self.send_json(200, {"data": [{"id": "omega-candidate"}]})
                else:
                    self.send_json(404, {"error": "unknown path"})

            def do_POST(self):
                value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if value.get("model") != "omega-candidate":
                    self.send_json(404, {"error": "unknown model"})
                elif value.get("tools"):
                    self.send_json(400, {"error": "tools unsupported"})
                elif value.get("stream"):
                    event = {"model": "omega-candidate", "choices": [{"delta": {"content": "hello"}, "finish_reason": "stop"}]}
                    body = ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_json(200, {"model": "omega-candidate", "choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]})

        servers = []
        for _ in range(2):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
        cfg = self.config(engine="stub", resident_bytes=0, working_bytes=0,
                          router_port=servers[0].server_port,
                          backend_port=servers[1].server_port)
        result = setup.exercise(cfg, 5)
        self.assertEqual(result["http_protocol"], "PASS")
        self.assertEqual(result["gpu_qualification"], "NOT_RUN")
        self.assertEqual(len(result["checks"]), 8)
        with self.assertRaises(ValueError):
            setup.check_ports(cfg)


if __name__ == "__main__":
    unittest.main()
