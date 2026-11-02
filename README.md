# replay-receipts

The tool inside is called `replayproof`.

Replay receipts for LLM inference run on nodes you do not control. Written for the HackerNoon Decentralize AI Hackathon.

Files:

- `replayproof.py` makes a receipt (`run`), compares two (`compare`) and prints Arweave tags (`tags`). Python 3.11, stdlib only.
- `experiment.py` and `summarize.py` run and score the experiments. `run_all.sh` does both.
- `tests/` has 12 unit tests against a scripted fake server. No model needed.
- `raw/` holds the 288 receipts behind `results.txt`. `MANIFEST.json` binds code, results and receipts to SHA-256 hashes and lists what was not verified.
- `ARTICLE.md` is the post.

## Quick start

```
git clone https://github.com/zkasuran/replay-receipts && cd replay-receipts
ollama pull gemma3:1b && ollama pull gemma3:4b && ollama pull gemma3:1b-it-q8_0
python3 -m pytest -q tests
./run_all.sh          # ports 11996-11999, about 43 minutes, needs Ollama 0.35.1
```

One receipt by hand:

```
python3 replayproof.py run --endpoint http://127.0.0.1:11434 --model gemma3:1b --prompt "Hello" --out a.json
python3 replayproof.py compare a.json b.json   # exit 0 accept, 1 reject, 2 not comparable
```

## Rules

Accept only a bit-identical result (or drift of at most 1e-6) against a reference run under the same pinned options, cache state and hardware class. Never accept a divergence because the margin is small. Details and measurements are in `ARTICLE.md`.

## Permissions and teardown

Needs only localhost HTTP to Ollama. Nothing is uploaded. Teardown: stop the `ollama serve` processes on ports 11996 to 11999 and delete `raw/`.

## Licence

Source-available, no derivatives. See `LICENSE`.
