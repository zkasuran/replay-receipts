# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran. See LICENSE in this directory.
"""Unit tests against a scripted fake Ollama server. No model needed."""
import copy
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import replayproof as rp  # noqa: E402


def steps(tokens, lps, runner=-5.0):
    return [
        {"token": t, "logprob": lp, "top_logprobs": [
            {"token": t, "logprob": lp}, {"token": "~" + t, "logprob": runner}]}
        for t, lp in zip(tokens, lps)
    ]


class Fake(BaseHTTPRequestHandler):
    script = {}

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "m", "digest": "d1"}, {"name": "m2", "digest": "d2"}]})
        else:
            self._send({"version": "9.9.9"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        lps = Fake.script[body["prompt"]]
        self._send({"response": "".join(s["token"] for s in lps), "logprobs": lps,
                    "prompt_eval_count": 3, "prompt_eval_cached_count": 0, "eval_count": len(lps)})


def serve():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


OPTS = {"seed": 0, "temperature": 0, "top_k": 1, "num_predict": 8, "num_ctx": 2048, "num_batch": 512}
TOK = ["a", "b", "c", "d"]


def receipt(script, model="m", opts=OPTS, prompt="p"):
    srv, url = serve()
    Fake.script = {prompt: script}
    try:
        return rp.make_receipt(url, model, prompt, opts)
    finally:
        srv.shutdown()


def test_identical_runs_have_same_result_id_and_accept():
    s = steps(TOK, [-0.1, -0.2, -0.3, -0.4])
    a, b = receipt(s), receipt(s)
    assert a["result"]["result_id"] == b["result"]["result_id"]
    v = rp.compare(a, b)
    assert v["verdict"] == "IDENTICAL" and v["accept"]


def test_numeric_drift_within_and_beyond_tolerance():
    a = receipt(steps(TOK, [-0.1, -0.2, -0.3, -0.4]))
    b = receipt(steps(TOK, [-0.1, -0.2 - 1e-9, -0.3, -0.4]))
    c = receipt(steps(TOK, [-0.1, -0.2 - 1e-3, -0.3, -0.4]))
    assert rp.compare(a, b)["verdict"] == "NUMERIC_DRIFT_OK"
    v = rp.compare(a, c)
    assert v["verdict"] == "NUMERIC_DRIFT_EXCEEDS_TOL" and not v["accept"]
    assert rp.compare(a, c, tol=1e-2)["accept"]


def test_divergence_is_never_accepted_even_when_margin_is_tiny():
    tie = [{"token": "c", "logprob": -0.70, "top_logprobs": [{"token": "c", "logprob": -0.70}, {"token": "z", "logprob": -0.72}]}]
    tie2 = [{"token": "z", "logprob": -0.71, "top_logprobs": [{"token": "z", "logprob": -0.71}, {"token": "c", "logprob": -0.73}]}]
    base = steps(["a", "b"], [-0.1, -0.2])
    v = rp.compare(receipt(base + tie), receipt(base + tie2))
    assert v["verdict"] == "DIVERGED" and v["first_divergence"] == 2 and not v["accept"]
    assert v["margin_a"] < 0.05 and v["margin_b"] < 0.05


def test_identical_output_is_accepted_even_if_cache_state_differs():
    s = steps(TOK, [-0.1] * 4)
    a, b = receipt(s), receipt(s)
    b["observed"]["prompt_eval_cached_count"] = 12
    b["receipt_id"] = rp.sha256(rp.canon({k: v for k, v in b.items() if k != "receipt_id"}))
    assert rp.compare(a, b)["verdict"] == "IDENTICAL"


def test_diverged_with_different_cache_state_is_not_comparable():
    a = receipt(steps(TOK, [-0.1] * 4))
    b = receipt(steps(["a", "b", "x", "d"], [-0.1] * 4))
    b["observed"]["prompt_eval_cached_count"] = 12
    b["receipt_id"] = rp.sha256(rp.canon({k: v for k, v in b.items() if k != "receipt_id"}))
    v = rp.compare(a, b)
    assert v["verdict"] == "NOT_COMPARABLE" and "cache" in v["why"]


def test_adjudicate_two_strike_rule():
    ref = receipt(steps(TOK, [-0.1] * 4))
    honest_rerun = receipt(steps(TOK, [-0.1] * 4))
    noisy = receipt(steps(["a", "b", "x", "d"], [-0.1] * 4))
    assert rp.adjudicate(ref, honest_rerun)["decision"] == "ACCEPT"
    assert rp.adjudicate(ref, noisy)["decision"] == "RERUN_NEEDED"
    assert rp.adjudicate(ref, noisy, honest_rerun)["decision"] == "ACCEPT_AFTER_RERUN"
    assert rp.adjudicate(ref, noisy, noisy)["decision"] == "REJECT"


def test_different_model_digest_prompt_or_pinned_is_not_comparable():
    s = steps(TOK, [-0.1] * 4)
    a = receipt(s)
    assert rp.compare(a, receipt(s, model="m2"))["verdict"] == "NOT_COMPARABLE"
    assert rp.compare(a, receipt(s, opts=dict(OPTS, seed=1)))["verdict"] == "NOT_COMPARABLE"
    srv, url = serve()
    Fake.script = {"other": s}
    other = rp.make_receipt(url, "m", "other", OPTS)
    srv.shutdown()
    assert rp.compare(a, other)["verdict"] == "NOT_COMPARABLE"


def test_free_options_may_differ():
    s = steps(TOK, [-0.1] * 4)
    a = receipt(s)
    b = receipt(s, opts=dict(OPTS, num_thread=1))
    assert b["request"]["free"] == {"num_thread": 1}
    assert rp.compare(a, b)["verdict"] == "IDENTICAL"


def test_num_batch_is_pinned():
    s = steps(TOK, [-0.1] * 4)
    v = rp.compare(receipt(s), receipt(s, opts=dict(OPTS, num_batch=64)))
    assert v["verdict"] == "NOT_COMPARABLE" and v["why"] == "pinned options differ"


def test_tampered_receipt_without_reseal_is_flagged():
    a = receipt(steps(TOK, [-0.1] * 4))
    t = copy.deepcopy(a)
    t["trace"][1]["token"] = "X"
    assert rp.compare(a, t)["verdict"] == "TAMPERED_RECEIPT"


def test_length_mismatch_is_rejected():
    a = receipt(steps(TOK, [-0.1] * 4))
    b = receipt(steps(TOK[:3], [-0.1] * 3))
    v = rp.compare(a, b)
    assert v["verdict"] == "LENGTH_MISMATCH" and not v["accept"]


def test_arweave_tags_carry_ids():
    a = receipt(steps(TOK, [-0.1] * 4))
    tags = {t["name"]: t["value"] for t in rp.arweave_tags(a)}
    assert tags["Receipt-Id"] == a["receipt_id"] and tags["Model-Digest"] == "d1"
