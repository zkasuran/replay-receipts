# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran. See LICENSE in this directory.
"""Bind the evidence to bytes: hashes of code, results and every raw receipt, plus versions."""
import glob, hashlib, json, os, platform, subprocess, urllib.request

H = os.path.dirname(os.path.abspath(__file__))
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
tags = json.load(urllib.request.urlopen("http://127.0.0.1:11999/api/tags"))
ver = json.load(urllib.request.urlopen("http://127.0.0.1:11999/api/version"))["version"]
cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
raw = sorted(glob.glob(os.path.join(H, "raw", "*", "*.json")))
m = {
    "code": {f: sha(os.path.join(H, f)) for f in ("replayproof.py", "experiment.py", "summarize.py", "run_all.sh", "LICENSE")},
    "tests": sha(os.path.join(H, "tests", "test_replayproof.py")),
    "results_json": sha(os.path.join(H, "results.json")),
    "results_txt": sha(os.path.join(H, "results.txt")),
    "raw_receipts": {"count": len(raw), "tree_sha256": hashlib.sha256("".join(f"{os.path.relpath(p, H)} {sha(p)}\n" for p in raw).encode()).hexdigest()},
    "environment": {"ollama": ver, "python": platform.python_version(), "cpu": cpu, "kernel": platform.release(),
                    "models": {t["name"]: {"digest": t["digest"], "quant": t["details"]["quantization_level"]} for t in tags["models"]}},
    "not_verified": ["Nosana GPU nodes", "NVIDIA or AMD GPUs", "Arweave upload", "models above 4B parameters",
                     "more than 8 prompts", "temperature above 0", "cross-machine CPU runs"],
}
json.dump(m, open(os.path.join(H, "MANIFEST.json"), "w"), indent=1, sort_keys=True)
print(json.dumps(m["raw_receipts"]), m["code"]["replayproof.py"][:12])
