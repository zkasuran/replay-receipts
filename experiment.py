# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran. See LICENSE in this directory.
"""Run the replayproof experiments against local Ollama servers and save raw receipts.

    python3 experiment.py <stage> ...

Stages: base threads batch ctx restart cache parallel cheats
Each stage writes raw/<stage>/<label>__p<i>.json and is safe to re-run (skips files that exist).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import replayproof as rp

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
MAIN = "http://127.0.0.1:11999"
PAR = "http://127.0.0.1:11998"
SMALL = "gemma3:1b"
BIG = "gemma3:4b"
Q8 = "gemma3:1b-it-q8_0"
NPRED = 96

PROMPTS = [
    "Explain in two sentences why a GPU rental node cannot be trusted to run the model you paid for.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List five risks of storing model weights on a single cloud provider and one mitigation for each.",
    "Summarize the idea of federated learning for a high school student in one paragraph.",
    "What is the capital of Australia and why is it not Sydney?",
    "Translate to French: The network has no single owner, so nobody can switch it off.",
    "Give a step by step plan to verify that a downloaded file matches a published SHA-256 hash.",
    "Describe what a Merkle tree is and one place it is used in a blockchain.",
]


def opts(**kw):
    o = {"seed": 0, "temperature": 0, "top_k": 1, "num_predict": NPRED, "num_ctx": 2048, "num_batch": 512}
    o.update(kw)
    return o


def save(stage: str, label: str, i: int, rec: dict):
    d = os.path.join(RAW, stage)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{label}__p{i}.json"), "w") as f:
        json.dump(rec, f, sort_keys=True)


def exists(stage, label, i):
    return os.path.exists(os.path.join(RAW, stage, f"{label}__p{i}.json"))


def run_one(stage, label, i, endpoint, model, o):
    if exists(stage, label, i):
        return
    rec = rp.make_receipt(endpoint, model, PROMPTS[i], o)
    save(stage, label, i, rec)
    print(f"{stage}/{label} p{i} {rec['result']['result_id'][:10]} cached={rec['observed']['prompt_eval_cached_count']}", flush=True)


def server_pid(port: int):
    out = subprocess.run(["ss", "-ltnp", f"sport = :{port}"], capture_output=True, text=True).stdout
    if "pid=" in out:
        return int(out.split("pid=")[1].split(",")[0])
    return None


def stop_server(port: int):
    pid = server_pid(port)
    if pid:
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            if not server_pid(port):
                return
            time.sleep(0.2)


def start_server(port: int, parallel: int, extra=None):
    env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_NUM_PARALLEL=str(parallel))
    env.update(extra or {})
    log = open(f"/tmp/dai-ollama-{port}.log", "a")
    subprocess.Popen(["ollama", "serve"], env=env, stdout=log, stderr=log, start_new_session=True)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/version", timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    raise SystemExit(f"server on {port} did not start")


def stage_base():
    for rep in range(5):
        for i in range(len(PROMPTS)):
            run_one("base", f"r{rep}", i, MAIN, SMALL, opts())


def stage_threads():
    for t in (1, 4, 11, 22):
        for i in range(len(PROMPTS)):
            run_one("threads", f"t{t}", i, MAIN, SMALL, opts(num_thread=t))


def stage_batch():
    for b in (64, 128, 256, 512):
        for i in range(len(PROMPTS)):
            run_one("batch", f"b{b}", i, MAIN, SMALL, opts(num_batch=b))


def stage_ctx():
    for i in range(len(PROMPTS)):
        run_one("ctx", "c4096", i, MAIN, SMALL, opts(num_ctx=4096))


def stage_restart():
    if not all(exists("restart", "rs", i) for i in range(len(PROMPTS))):
        stop_server(11999)
        start_server(11999, 1)
    for i in range(len(PROMPTS)):
        run_one("restart", "rs", i, MAIN, SMALL, opts())


def stage_cache():
    """Warm run: send the first half of the prompt first so its KV prefix is cached."""
    for i, p in enumerate(PROMPTS):
        if exists("cache", "warm", i):
            continue
        words = p.split(" ")
        prefix = " ".join(words[: max(2, len(words) // 2)])
        rp.make_receipt(MAIN, SMALL, prefix, opts(num_predict=1))
        run_one("cache", "warm", i, MAIN, SMALL, opts())


def stage_parallel():
    stop_server(11998)
    start_server(11998, 4)
    todo = [i for i in range(len(PROMPTS)) if not exists("parallel", "p4", i)]
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda i: run_one("parallel", "p4", i, PAR, SMALL, opts()), todo))
    stop_server(11998)


def stage_cheats():
    for i in range(len(PROMPTS)):
        run_one("cheats", "h4b_a", i, MAIN, BIG, opts())
        run_one("cheats", "h4b_b", i, MAIN, BIG, opts(num_thread=4))
    for i in range(len(PROMPTS)):
        run_one("cheats", "c_q8", i, MAIN, Q8, opts())


def stage_kv():
    """Same weights, same digest, different server settings a node can flip to fit more jobs."""
    KV = "http://127.0.0.1:11997"
    for label, extra in (
        ("fa_f16", {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "f16"}),
        ("fa_q8", {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}),
        ("fa_q4", {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q4_0"}),
    ):
        if all(exists("kv", label, i) for i in range(len(PROMPTS))):
            continue
        stop_server(11997)
        start_server(11997, 1, extra)
        for i in range(len(PROMPTS)):
            run_one("kv", label, i, KV, SMALL, opts())
        stop_server(11997)


def unload(endpoint, model):
    req = urllib.request.Request(endpoint + "/api/generate", data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=120).read()
    time.sleep(0.5)


def stage_cold():
    """Cache-empty reference: unload the runner before every request."""
    for label in ("cold_a", "cold_b"):
        for i in range(len(PROMPTS)):
            if not exists("cold", label, i):
                unload(MAIN, SMALL)
                run_one("cold", label, i, MAIN, SMALL, opts())


def stage_warm2():
    """Same warm protocol as stage_cache but from an unloaded runner, twice."""
    for label in ("warm_a", "warm_b"):
        for i, p in enumerate(PROMPTS):
            if exists("warm2", label, i):
                continue
            words = p.split(" ")
            prefix = " ".join(words[: max(2, len(words) // 2)])
            unload(MAIN, SMALL)
            rp.make_receipt(MAIN, SMALL, prefix, opts(num_predict=1))
            run_one("warm2", label, i, MAIN, SMALL, opts())


def stage_parallel2():
    """Repeat the concurrent burst, and run the same server with sequential requests."""
    stop_server(11998)
    start_server(11998, 4)
    todo = [i for i in range(len(PROMPTS)) if not exists("parallel2", "p4_again", i)]
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda i: run_one("parallel2", "p4_again", i, PAR, SMALL, opts()), todo))
    for i in range(len(PROMPTS)):
        unload(PAR, SMALL)
        run_one("parallel2", "p4_seq", i, PAR, SMALL, opts())
    stop_server(11998)


def long_prompt(i):
    notes = " ".join(f"Note {k}: node {k} reported hash {(k * 7919 + i * 104729) % 4294967296:08x} for shard {(k + i) % 5} at epoch {k * 3 + i}." for k in range(14))
    return ("Read the following notes and then answer the question at the end. " + notes +
            " Question: which shard appears most often and what does that say about load balance?")


def stage_chunk():
    """Prompts of about 420 tokens, so num_batch below 420 forces chunked prompt evaluation."""
    for b in (512, 256, 128, 64):
        for i in range(len(PROMPTS)):
            if exists("chunk", f"b{b}", i):
                continue
            unload(MAIN, SMALL)
            rec = rp.make_receipt(MAIN, SMALL, long_prompt(i), opts(num_batch=b, num_predict=64))
            save("chunk", f"b{b}", i, rec)
            print(f"chunk/b{b} p{i} tokens={rec['observed']['prompt_eval_count']} {rec['result']['result_id'][:10]}", flush=True)
    for i in range(len(PROMPTS)):
        if not exists("chunk", "b64_again", i):
            unload(MAIN, SMALL)
            save("chunk", "b64_again", i, rp.make_receipt(MAIN, SMALL, long_prompt(i), opts(num_batch=64, num_predict=64)))


def stage_hw():
    """Different silicon: the integrated Intel Arc GPU through Vulkan, against the CPU reference."""
    HW = "http://127.0.0.1:11996"
    stop_server(11996)
    start_server(11996, 1, {"OLLAMA_IGPU_ENABLE": "1", "OLLAMA_VULKAN": "1"})
    for label in ("vulkan_a", "vulkan_b"):
        for i in range(len(PROMPTS)):
            if not exists("hw", label, i):
                unload(HW, SMALL)
                run_one("hw", label, i, HW, SMALL, opts())


STAGES = {
    "chunk": stage_chunk,
    "hw": stage_hw,
    "cold": stage_cold, "warm2": stage_warm2, "parallel2": stage_parallel2,
    "kv": stage_kv,
    "base": stage_base, "threads": stage_threads, "batch": stage_batch, "ctx": stage_ctx,
    "restart": stage_restart, "cache": stage_cache, "parallel": stage_parallel, "cheats": stage_cheats,
}

if __name__ == "__main__":
    for s in sys.argv[1:] or list(STAGES):
        print("== stage", s, flush=True)
        STAGES[s]()
