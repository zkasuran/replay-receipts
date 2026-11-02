#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Regenerates every receipt and the results table. Needs ollama 0.35.1 on PATH and the
# models gemma3:1b, gemma3:4b and gemma3:1b-it-q8_0 already pulled. Uses ports 11996-11999.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pytest -q tests
if ! curl -s -m 2 127.0.0.1:11999/api/version >/dev/null; then
  (OLLAMA_HOST=127.0.0.1:11999 OLLAMA_NUM_PARALLEL=1 nohup ollama serve >/tmp/dai-ollama.log 2>&1 &)
  sleep 6
fi
python3 experiment.py base threads batch ctx cold warm2 cache restart parallel parallel2 kv cheats chunk hw
python3 summarize.py | tee results.txt
sha256sum replayproof.py experiment.py summarize.py > code.sha256
