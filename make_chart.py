# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran. See LICENSE in this directory.
"""Draw the cover chart from results.json: outputs that differed from the reference, out of 8."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = os.path.dirname(os.path.abspath(__file__))
E = json.load(open(os.path.join(H, "results.json")))["experiments"]


def differed(*names):
    n = tot = 0
    for k in names:
        v = E[k]["verdicts"]
        tot = max(tot, E[k]["n"])
        n = max(n, E[k]["n"] - v.get("IDENTICAL", 0))
    return n


rows = [
    ("Thread count 1 to 22", differed("threads_t1_vs_cold_a", "threads_t4_vs_cold_a", "threads_t11_vs_cold_a", "threads_t22_vs_cold_a"), "same"),
    ("Runner restart", differed("restart_rs_vs_cold_a"), "same"),
    ("Flash attention, f16 KV", differed("flashattn_f16kv_vs_cold_a"), "same"),
    ("Same GPU, run twice", differed("gpu_vulkan_b_vs_gpu_vulkan_a (same device, cold)"), "same"),
    ("Prompt chunks of 256", differed("chunk_num_batch256_vs_512 (430-token prompts)"), "honest"),
    ("Prompt chunks of 64", differed("chunk_num_batch64_vs_512 (430-token prompts)"), "honest"),
    ("Warm prefix cache", differed("warm_a_vs_cold_a (cache state differs)"), "honest"),
    ("Concurrent requests", differed("parallel4_burst_vs_cold_a"), "honest"),
    ("Arc GPU vs CPU", differed("gpu_vulkan_a_vs_cpu_cold_a (different silicon)"), "honest"),
    ("Quantized KV cache (q8)", differed("kv_q8_vs_cold_a"), "cheat"),
    ("Q8 weights, forged digest", differed("cheat_q8_posing_as_q4"), "cheat"),
    ("1B model posing as 4B", differed("cheat_1b_posing_as_4b"), "cheat"),
]
col = {"same": "#00ff88", "honest": "#ffb020", "cheat": "#ff5470"}
fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor("#0b0f14")
ax.set_facecolor("#0b0f14")
y = list(range(len(rows)))[::-1]
ax.barh(y, [r[1] for r in rows], color=[col[r[2]] for r in rows], height=0.62)
for yy, r in zip(y, rows):
    ax.text(r[1] + 0.12, yy, f"{r[1]} of 8", va="center", color="white", fontsize=14)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], color="white", fontsize=14)
ax.set_xlim(0, 9.6)
ax.set_xticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
fig.text(0.04, 0.93, "Replaying LLM runs: how many of 8 outputs changed", color="white", fontsize=22, weight="bold")
fig.text(0.04, 0.885, "green: never moved a bit   amber: honest conditions that did   red: a lying node", color="#9aa7b4", fontsize=13)
fig.subplots_adjust(left=0.27, right=0.97, top=0.84, bottom=0.04)
out = os.path.join(H, "cover.png")
fig.savefig(out, facecolor=fig.get_facecolor())
print(out, [(r[0], r[1]) for r in rows])
