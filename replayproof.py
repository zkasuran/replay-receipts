# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran. See LICENSE in this directory.
"""replayproof: replay receipts for inference run on nodes you do not control.

A node that sells GPU time can return text. It cannot cheaply prove that the text
came from the model you paid for. replayproof records what a run produced down to
the logprobs, then compares two runs of the same request and says whether the gap
is honest numeric noise, a near-tie token flip, or something that looks like a
different model.

Stdlib only. Talks to any Ollama-compatible /api/generate endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import urllib.request

SCHEMA = "replayproof/1"
# Options that change the computation. Receipts that disagree here are not comparable.
PINNED = ("seed", "temperature", "top_k", "num_predict", "num_ctx", "num_batch")
# Options that did not change a single logprob in the measurements. Recorded, allowed to differ.
FREE = ("num_thread",)


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def _trace(logprobs: list) -> list:
    """Keep only what a verifier needs: token, chosen logprob, runner-up logprob."""
    out = []
    for step in logprobs:
        top = step.get("top_logprobs") or []
        runner = None
        for cand in top:
            if cand["token"] != step["token"]:
                runner = {"token": cand["token"], "logprob": cand["logprob"]}
                break
        out.append({"token": step["token"], "logprob": step["logprob"], "runner_up": runner})
    return out


def result_id(model_digest: str, prompt_sha: str, pinned: dict, trace: list) -> str:
    """Identity of the computation result. Equal ids mean bit-identical output."""
    return sha256(canon({"m": model_digest, "p": prompt_sha, "o": pinned, "t": trace}))


def make_receipt(endpoint: str, model: str, prompt: str, options: dict, timeout: float = 600.0) -> dict:
    endpoint = endpoint.rstrip("/")
    tags = _get(endpoint + "/api/tags", timeout)
    digest = next((m["digest"] for m in tags["models"] if m["name"] == model), None)
    if digest is None:
        raise SystemExit(f"model {model!r} not served at {endpoint}")
    version = _get(endpoint + "/api/version", timeout).get("version", "unknown")
    raw = _post(
        endpoint + "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 2,
            "options": options,
        },
        timeout,
    )
    trace = _trace(raw["logprobs"])
    prompt_sha = sha256(prompt.encode())
    pinned = {k: options.get(k) for k in PINNED}
    free = {k: options.get(k) for k in FREE if k in options}
    rid = result_id(digest, prompt_sha, pinned, trace)
    body = {
        "schema": SCHEMA,
        "model": {"name": model, "digest": digest},
        "server": {"name": "ollama", "version": version},
        "request": {"prompt_sha256": prompt_sha, "pinned": pinned, "free": free},
        "observed": {
            "prompt_eval_count": raw.get("prompt_eval_count"),
            "prompt_eval_cached_count": raw.get("prompt_eval_cached_count"),
            "eval_count": raw.get("eval_count"),
        },
        "env": {"machine": platform.machine(), "system": platform.system(), "gpu": None},
        "result": {
            "text_sha256": sha256(raw["response"].encode()),
            "n_tokens": len(trace),
            "result_id": rid,
        },
        "trace": trace,
    }
    body["receipt_id"] = sha256(canon(body))
    return body


def _margin(step: dict) -> float | None:
    if step.get("runner_up") is None:
        return None
    return step["logprob"] - step["runner_up"]["logprob"]


def compare(a: dict, b: dict, tol: float = 1e-6) -> dict:
    """Return a verdict. accept is True only for bit-identical output or drift within tol.

    A divergence is never accepted on the strength of a small margin. Measured data in
    this repo shows honest concurrency noise and a quantized KV cache both flip tokens
    at margins between 0.1 and 0.9 nats, so a margin cannot tell them apart.
    """
    for r in (a, b):
        if r.get("schema") != SCHEMA:
            return {"verdict": "NOT_COMPARABLE", "accept": False, "why": "unknown schema"}
        body = {k: v for k, v in r.items() if k != "receipt_id"}
        if sha256(canon(body)) != r.get("receipt_id"):
            return {"verdict": "TAMPERED_RECEIPT", "accept": False, "why": "receipt_id does not match bytes"}
    if a["model"]["digest"] != b["model"]["digest"]:
        return {"verdict": "NOT_COMPARABLE", "accept": False, "why": "model digest differs"}
    if a["request"]["prompt_sha256"] != b["request"]["prompt_sha256"]:
        return {"verdict": "NOT_COMPARABLE", "accept": False, "why": "prompt differs"}
    if a["request"]["pinned"] != b["request"]["pinned"]:
        return {"verdict": "NOT_COMPARABLE", "accept": False, "why": "pinned options differ"}

    ta, tb = a["trace"], b["trace"]
    if a["result"]["result_id"] == b["result"]["result_id"]:
        return {"verdict": "IDENTICAL", "accept": True, "n_tokens": len(ta), "max_abs_dlogprob": 0.0}
    ca, cb = a["observed"]["prompt_eval_cached_count"], b["observed"]["prompt_eval_cached_count"]
    if ca != cb:
        return {"verdict": "NOT_COMPARABLE", "accept": False, "why": "prompt cache state differs",
                "cached_a": ca, "cached_b": cb}

    n = min(len(ta), len(tb))
    first = next((i for i in range(n) if ta[i]["token"] != tb[i]["token"]), None)
    if first is None and len(ta) == len(tb):
        drift = max(abs(ta[i]["logprob"] - tb[i]["logprob"]) for i in range(n))
        ok = drift <= tol
        return {
            "verdict": "NUMERIC_DRIFT_OK" if ok else "NUMERIC_DRIFT_EXCEEDS_TOL",
            "accept": ok,
            "n_tokens": n,
            "max_abs_dlogprob": drift,
            "tol": tol,
        }
    if first is None:
        return {"verdict": "LENGTH_MISMATCH", "accept": False, "first_divergence": n,
                "len_a": len(ta), "len_b": len(tb)}
    return {
        "verdict": "DIVERGED",
        "accept": False,
        "first_divergence": first,
        "margin_a": _margin(ta[first]),
        "margin_b": _margin(tb[first]),
        "tokens": [ta[first]["token"], tb[first]["token"]],
    }


def adjudicate(reference: dict, claim: dict, rerun: dict | None = None, tol: float = 1e-6) -> dict:
    """Two-strike rule. Replay noise from a busy node earns one exclusive rerun, not a verdict.

    reference: the verifier's own cold, sequential run.
    claim:     the receipt the node returned.
    rerun:     the receipt the node returned when asked to run the same request alone.
    """
    first = compare(reference, claim, tol)
    if first["accept"]:
        return {"decision": "ACCEPT", "first": first}
    if first["verdict"] in ("TAMPERED_RECEIPT", "NOT_COMPARABLE") and first.get("why") != "prompt cache state differs":
        return {"decision": "REJECT", "first": first}
    if rerun is None:
        return {"decision": "RERUN_NEEDED", "first": first}
    second = compare(reference, rerun, tol)
    return {"decision": "ACCEPT_AFTER_RERUN" if second["accept"] else "REJECT", "first": first, "second": second}


def arweave_tags(receipt: dict) -> list:
    return [
        {"name": "App-Name", "value": "replayproof"},
        {"name": "App-Version", "value": "1"},
        {"name": "Content-Type", "value": "application/json"},
        {"name": "Receipt-Id", "value": receipt["receipt_id"]},
        {"name": "Result-Id", "value": receipt["result"]["result_id"]},
        {"name": "Model-Digest", "value": receipt["model"]["digest"]},
    ]


def _parse_opts(items: list[str]) -> dict:
    opts = {}
    for it in items:
        k, _, v = it.partition("=")
        try:
            opts[k] = int(v)
        except ValueError:
            try:
                opts[k] = float(v)
            except ValueError:
                opts[k] = v
    return opts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="replayproof")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a prompt and write a receipt")
    r.add_argument("--endpoint", default="http://127.0.0.1:11434")
    r.add_argument("--model", required=True)
    r.add_argument("--prompt", required=True)
    r.add_argument("--opt", action="append", default=[], help="key=value, repeatable")
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare", help="compare two receipts")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--tol", type=float, default=1e-6)
    t = sub.add_parser("tags", help="print Arweave tags for a receipt")
    t.add_argument("receipt")
    args = p.parse_args(argv)

    if args.cmd == "run":
        opts = {"seed": 0, "temperature": 0, "top_k": 1, "num_predict": 64, "num_ctx": 2048, "num_batch": 512}
        opts.update(_parse_opts(args.opt))
        rec = make_receipt(args.endpoint, args.model, args.prompt, opts)
        with open(args.out, "w") as f:
            json.dump(rec, f, sort_keys=True, indent=1)
        print(rec["receipt_id"], rec["result"]["result_id"])
        return 0
    if args.cmd == "compare":
        a = json.load(open(args.a))
        b = json.load(open(args.b))
        v = compare(a, b, args.tol)
        print(json.dumps(v, sort_keys=True))
        if v["verdict"] in ("NOT_COMPARABLE", "TAMPERED_RECEIPT"):
            return 2
        return 0 if v["accept"] else 1
    if args.cmd == "tags":
        print(json.dumps(arweave_tags(json.load(open(args.receipt))), indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
