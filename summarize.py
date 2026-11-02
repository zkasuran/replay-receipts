# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran. See LICENSE in this directory.
"""Compare every raw receipt against a reference and write results.json plus a text table."""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

import replayproof as rp

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")


def load(stage, label, i):
    p = os.path.join(RAW, stage, f"{label}__p{i}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def reseal(r):
    r = dict(r)
    r.pop("receipt_id", None)
    r["receipt_id"] = rp.sha256(rp.canon(r))
    return r


def forge_as(r, honest):
    """What a lying node does: run another model, then write the honest model's name and digest."""
    f = json.loads(json.dumps(r))
    f["model"] = dict(honest["model"])
    f["request"]["pinned"] = dict(honest["request"]["pinned"])
    f["result"]["result_id"] = rp.result_id(
        f["model"]["digest"], f["request"]["prompt_sha256"], f["request"]["pinned"], f["trace"]
    )
    return reseal(f)


def align_pinned(r, ref):
    f = json.loads(json.dumps(r))
    f["request"]["pinned"] = dict(ref["request"]["pinned"])
    f["result"]["result_id"] = rp.result_id(
        f["model"]["digest"], f["request"]["prompt_sha256"], f["request"]["pinned"], f["trace"]
    )
    return reseal(f)


def labels(stage):
    return sorted({os.path.basename(p).split("__")[0] for p in glob.glob(os.path.join(RAW, stage, "*.json"))})


def n_prompts():
    import experiment
    return len(experiment.PROMPTS)


def run(tol=1e-6):
    N = n_prompts()
    out = {"tol": tol, "experiments": {}, "adjudication": {}, "detail": []}

    def add(name, kind, pairs):
        c = collections.Counter()
        drifts, margins = [], []
        for i, a, b in pairs:
            if a is None or b is None:
                continue
            v = rp.compare(a, b, tol)
            c[v["verdict"]] += 1
            if "max_abs_dlogprob" in v:
                drifts.append(v["max_abs_dlogprob"])
            if "margin_a" in v:
                margins.append([v["margin_a"], v["margin_b"], v["first_divergence"]])
            out["detail"].append({"exp": name, "prompt": i, **v})
        out["experiments"][name] = {
            "kind": kind, "n": sum(c.values()), "verdicts": dict(c),
            "max_drift": max(drifts) if drifts else None, "divergence_margins": margins,
        }

    L = lambda st, lab: [load(st, lab, i) for i in range(N)]
    ref = L("cold", "cold_a")  # cache-empty reference, runner unloaded before the request
    pr = lambda x, y: [(i, x[i], y[i]) for i in range(N)]
    # honest, same conditions, repeated
    add("cold_b_vs_cold_a", "honest", pr(ref, L("cold", "cold_b")))
    for lab in labels("base"):
        add(f"repeat_{lab}_vs_cold_a (server stays up)", "honest", pr(ref, L("base", lab)))
    # honest, one knob changed
    for stage in ("threads", "restart"):
        for lab in labels(stage):
            add(f"{stage}_{lab}_vs_cold_a", "honest", pr(ref, L(stage, lab)))
    for lab in labels("batch"):  # prompts under 35 tokens: always one chunk, so this is a weak null
        add(f"batch_{lab}_vs_cold_a (short prompts, one chunk)", "honest-batch-short",
            [(i, ref[i], align_pinned(load("batch", lab, i), ref[i])) for i in range(N)])
    for lab in labels("ctx"):
        add(f"ctx_{lab}_vs_cold_a (pinned diff aligned)", "honest-ctx",
            [(i, ref[i], align_pinned(load("ctx", lab, i), ref[i]) if load("ctx", lab, i) else None) for i in range(N)])
    # prompt prefill chunking: ~430 token prompts, num_batch smaller than the prompt forces chunks
    c512 = L("chunk", "b512")
    for b in (256, 128, 64):
        add(f"chunk_num_batch{b}_vs_512 (430-token prompts)", "honest-chunk", [(i, c512[i], align_pinned(load("chunk", f"b{b}", i), c512[i])) for i in range(N)])
    add("chunk_num_batch64_vs_64_again", "honest", pr(L("chunk", "b64"), L("chunk", "b64_again")))
    # prefix cache
    wa, wb = L("warm2", "warm_a"), L("warm2", "warm_b")
    add("warm_b_vs_warm_a (same cache state)", "honest", pr(wa, wb))
    add("warm_a_vs_cold_a (cache state differs)", "honest-cache", pr(ref, wa))
    add("warm_orig_vs_warm_a (same cache state)", "honest", pr(wa, L("cache", "warm")))
    # concurrency
    p1, p2, ps = L("parallel", "p4"), L("parallel2", "p4_again"), L("parallel2", "p4_seq")
    add("parallel4_burst_vs_cold_a", "honest-conc", pr(ref, p1))
    add("parallel4_burst_vs_burst_again", "honest-conc", pr(p1, p2))
    add("parallel4_server_sequential_vs_cold_a", "honest", pr(ref, ps))
    # flash attention and KV cache precision (same weights, same digest)
    fa = L("kv", "fa_f16")
    add("flashattn_f16kv_vs_cold_a", "honest-setting", pr(ref, fa))
    add("kv_q8_vs_cold_a", "cheat-setting", pr(ref, L("kv", "fa_q8")))
    add("kv_q4_vs_cold_a", "cheat-setting", pr(ref, L("kv", "fa_q4")))
    add("kv_q8_vs_flashattn_f16", "cheat-setting", pr(fa, L("kv", "fa_q8")))
    add("kv_q4_vs_flashattn_f16", "cheat-setting", pr(fa, L("kv", "fa_q4")))
    # big model honest pair
    ha, hb = L("cheats", "h4b_a"), L("cheats", "h4b_b")
    add("honest_4b_threadsdefault_vs_threads4", "honest", pr(ha, hb))
    # different silicon: integrated Intel Arc GPU (Vulkan) vs the CPU reference
    va, vb = L("hw", "vulkan_a"), L("hw", "vulkan_b")
    add("gpu_vulkan_b_vs_gpu_vulkan_a (same device, cold)", "honest", pr(va, vb))
    add("gpu_vulkan_a_vs_cpu_cold_a (different silicon)", "honest-hw", pr(ref, va))
    # model substitution, forged to carry the honest digest
    q8 = L("cheats", "c_q8")
    add("cheat_1b_posing_as_4b", "cheat", [(i, ha[i], forge_as(ref[i], ha[i])) for i in range(N)])
    add("cheat_q8_posing_as_q4", "cheat", [(i, ref[i], forge_as(q8[i], ref[i])) for i in range(N)])
    add("unforged_q8_vs_q4", "cheat-unforged", pr(ref, q8))
    # two-strike adjudication: honest nodes under noise get one exclusive rerun, cheaters do not escape
    def adj(name, kind, triples):
        c = collections.Counter(rp.adjudicate(r, cl, rr)["decision"] for r, cl, rr in triples)
        out["adjudication"][name] = {"kind": kind, "n": sum(c.values()), "decisions": dict(c)}
    adj("honest_burst_then_solo_rerun", "honest", [(ref[i], p1[i], ps[i]) for i in range(N)])
    adj("honest_warm_cache_then_cold_rerun", "honest", [(ref[i], wa[i], L("cold", "cold_b")[i]) for i in range(N)])
    adj("honest_gpu_node_vs_cpu_reference_with_rerun", "honest-hw", [(ref[i], va[i], vb[i]) for i in range(N)])
    adj("honest_gpu_node_vs_gpu_reference", "honest", [(va[i], vb[i], vb[i]) for i in range(N)])
    adj("honest_threads1", "honest", [(ref[i], L("threads", "t1")[i], L("threads", "t1")[i]) for i in range(N)])
    for nm, lab in (("cheat_kv_q8", "fa_q8"), ("cheat_kv_q4", "fa_q4")):
        k = L("kv", lab)
        adj(nm, "cheat", [(ref[i], k[i], k[i]) for i in range(N)])
    fq = [forge_as(q8[i], ref[i]) for i in range(N)]
    adj("cheat_weights_q8_forged_digest", "cheat", [(ref[i], fq[i], fq[i]) for i in range(N)])
    f1 = [forge_as(ref[i], ha[i]) for i in range(N)]
    adj("cheat_1b_posing_as_4b", "cheat", [(ha[i], f1[i], f1[i]) for i in range(N)])
    # tampered receipt bytes
    tam = []
    for i in range(N):
        if ref[i]:
            t = json.loads(json.dumps(ref[i]))
            t["trace"][3]["token"] = "X"
            tam.append((i, ref[i], t))
    add("tampered_trace_without_reseal", "tamper", tam)
    return out


def table(out):
    lines = []
    for name, e in out["experiments"].items():
        v = ", ".join(f"{k}={n}" for k, n in sorted(e["verdicts"].items()))
        md = "" if e["max_drift"] is None else f" max_dlogprob={e['max_drift']:.3g}"
        lines.append(f"{name:55s} n={e['n']:2d}  {v}{md}")
    lines.append("")
    for name, e in out["adjudication"].items():
        lines.append(f"ADJ {name:45s} n={e['n']:2d}  " + ", ".join(f"{k}={n}" for k, n in sorted(e["decisions"].items())))
    return "\n".join(lines)


if __name__ == "__main__":
    tol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
    o = run(tol)
    json.dump(o, open(os.path.join(HERE, "results.json"), "w"), indent=1, sort_keys=True)
    print(table(o))
