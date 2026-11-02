# What Replaying LLM Runs Tells You About an Untrusted GPU Node

Rent a GPU from a stranger and you get text back. You do not get proof that the text came from the model you paid for. A node can swap a 4B model for a 1B one or quantize the KV cache to fit more jobs on the card. The answer still looks like an answer.

Decentralized compute networks such as Nosana have this problem by design. So I ran the cheapest check I could think of: replay the same request somewhere else and compare the logprobs. Then I spent a few days finding out when that comparison can be trusted. It is trustworthy less often than I expected.

## The receipt

`replayproof` is a small stdlib-only Python tool. For each run it records the model digest, a hash of the prompt and the pinned sampling options. At every step it stores the chosen token with the runner-up logprob. It hashes the result into a `result_id`. Two runs with the same `result_id` produced bit-identical output.

A receipt is 2 to 12 KB. The ar.io docs list a free tier of 105 KiB per Arweave data item through Turbo, so anchoring a receipt permanently costs nothing. The tool prints the Arweave tags for it. I have not uploaded anything yet.

## Setup

- Ollama 0.35.1 on an Intel Core Ultra 9 185H (22 threads) and its Arc iGPU through Vulkan
- gemma3:1b, gemma3:4b and gemma3:1b-it-q8_0
- 8 prompts, seed 0, temperature 0, 96 new tokens, cold runner unless stated

Everything below is 8 of 8 prompts unless a count says otherwise. All 288 raw receipts and a hash manifest sit next to the code.

## What never moved a bit

These matched the reference on every prompt:

- thread count of 1, 4, 11 and 22
- a runner restart
- `num_ctx` 2048 against 4096
- flash attention with an f16 KV cache
- five repeats on a warm server
- a server set to four parallel slots, when requests arrive one at a time
- the same GPU run twice

## What did move bits

- **Prefill chunking.** With 430-token prompts, `num_batch` 64 against 512 changed the output on 8 of 8 prompts (5 diverged, 3 drifted by up to 0.37 nats). With 128 it was 8 of 8 again. With 256 it was 6 of 8. My first batch test used prompts under 35 tokens, which always fit in one chunk. It showed nothing. A null result from a test that cannot fail is not a result.
- **A warm prefix cache.** After a request that cached the first half of the prompt, 4 of 8 outputs differed from the cold run. The receipt shows `prompt_eval_cached_count`, so a verifier can see it.
- **Concurrent requests.** Eight requests in a burst against four slots: all 8 differed from the sequential run. Repeating the same burst gave 8 of 8 different results again. The same server answering the same prompts one at a time matched exactly.
- **Different silicon.** The Arc GPU against the CPU: 8 of 8 differed. The GPU against itself: 8 of 8 identical.

This matches what Horace He and Thinking Machines argued in "Defeating Nondeterminism in LLM Inference". Floating-point order depends on batch size and on how a request is sliced up, which covers chunked prefill and prefix caching. My thread-count runs fit that account: threads never changed a bit and batch shape did.

## What a cheating node looks like

I built four cheats and wrote receipts as a lying node would: honest model name and digest on top of someone else's tokens.

- KV cache quantized to q8_0: 8 of 8 caught
- KV cache quantized to q4_0: 8 of 8 caught
- Q8_0 weights posing as Q4_K_M: 8 of 8 caught
- gemma3:1b posing as gemma3:4b: 8 of 8 caught

An unforged substitute fails earlier because the model digest differs. Editing a receipt without recomputing its id is flagged as tampering on 8 of 8.

## The design that failed

My first version accepted a divergence when the winning token had a thin lead over the runner-up, on the theory that honest numeric noise flips near-ties and a different model does not. The data says no. At the first diverging token, honest burst noise had margins from 0.11 to 0.62 nats. Honest GPU against CPU ran from 0.04 to 0.37. The q8 KV cheat ran from 0.09 to 0.73. Two of its eight cases fell under my 0.25 cutoff and would have passed.

So the final rule has no margin. A run is accepted only if it is bit-identical (or within 1e-6 of it) to a reference produced under the same conditions.

## What replaces the margin: one exclusive rerun

If a claim does not match, the verifier asks the node to run the request again alone. That is a two-strike rule. The same data lets me score it.

- Honest node under burst load, then solo rerun: 8 of 8 accepted after the rerun
- Honest node with a warm cache, then a cold rerun: 4 accepted, 4 accepted after the rerun
- All four cheats, 32 cases: 32 rejected

The same table shows the main limit. An honest GPU node checked against a CPU reference was rejected 8 of 8, even after a rerun. A reference has to come from the same hardware class as the node it checks. On a network that sells many GPU models this means one reference pool per model. That is a cost.

## What I did not prove

- I did not run on Nosana. These are local runs on one machine.
- I did not test NVIDIA or AMD GPUs. Whether two cards of the same model match bit for bit is unknown to me.
- A cheater who notices the challenge and runs the real model for it passes the rerun. The fix is spot checks the node cannot tell apart from normal work. I have not built or tested that.
- Eight prompts, two sizes of one model family, temperature 0 only.
- Prior work gets further on some axes. TOPLOC from Prime Intellect and Together AI (Ong et al., ICML 2025) hashes intermediate activations and reports no false positives or negatives on its tests. Its abstract claims robustness across GPU types, which my exact-match rule does not have. It also stores 258 bytes per 32 tokens against my kilobytes. Replay is the simple baseline and the numbers above show where it stops.

## Next step

Claim the Nosana credit, run the same eight prompts on one Nosana GPU market twice and once more on a different market. Then see which of the lines above carry over.

## Reproduce

```
git clone https://github.com/zkasuran/replay-receipts
cd replay-receipts
./run_all.sh      # 43 minutes on my 22-thread laptop
cat results.txt
```

Disclosure: Claude (Anthropic) wrote the code and the first draft of this post. Every number above comes from `results.txt` and `MANIFEST.json` in the repo.
