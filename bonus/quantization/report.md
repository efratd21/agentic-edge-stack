# Bonus 2 — Quantization Performance Report

Model: **llama3.2:3b** · Engine: **Ollama** (local) · Hardware: **CPU-only**
(12 cores, WSL2 VM with ~3.8 GB RAM on a 7.7 GB host) — peak memory below is
**RAM**, not VRAM.

Method: 2 timed runs × 3 prompts per quant,
`temperature=0, seed=0, num_predict=150, num_thread=10` (identical for every
quant); the cold-load warm-up run is discarded; exactly one model is resident
at a time (`keep_alive=0` between levels). TPS comes from Ollama's own
`eval_count / eval_duration` timings; peak RAM is the runner process's `VmHWM`.
Harness: [`profile.py`](profile.py). Verbatim generations: [`raw_outputs/`](raw_outputs/).

| Quant level | Bits (approx) | Disk size | Peak RAM | TPS (mean ± sd) |
|---|---|---|---|---|
| `q3_K_M` | ~3-bit | 1.7 GB | 2,169 MB | 6.6 ± 0.9 |
| `q4_K_M` | ~4-bit | 2.0 GB | 2,496 MB | 6.6 ± 0.3 |
| `q8_0`   | 8-bit  | 3.4 GB | — | **did not load** — exceeds this machine's RAM |

## Speed vs. quality trade-off

**Speed: quantizing below 4-bit bought nothing here.** q3_K_M and q4_K_M both
generate at ~6.6 TPS. CPU inference is memory-bandwidth-bound (every generated
token streams the full weight file through the CPU), so q3's smaller weights
*should* be faster — but K-quant 3-bit dequantization costs more compute per
weight, and on this CPU the two effects cancel out. The only real q3 win is
memory: ~330 MB less resident RAM.

**Quality: the drop from q4 to q3 is visible immediately** (see
[`raw_outputs/`](raw_outputs/), same prompts, greedy decoding):

- *Factual (capital of France):* q4 is accurate (Eiffel Tower, Notre-Dame,
  Louvre). **q3 hallucinates** — it claims Paris is the birthplace of Monet,
  Renoir *and van Gogh* (van Gogh was Dutch; Renoir was born in Limoges).
- *Reasoning ("all but 9 run away"):* the correct answer is 9. **q3 answers 8**,
  flatly wrong. q4 reasons messily (it first restates 17) but lands on 9.
- *Domain question (ReAct):* both struggle — a 3B-model knowledge limit rather
  than a quantization artifact — but q4 stays coherent while q3 explicitly
  answers about a different concept (reactive RL agents).

**The 8-bit level is a capacity lesson, not a data point.** q8_0 needs ~3.4 GB
resident; this machine's WSL VM has ~3.8 GB total, so the load attempt
swap-thrashed and timed out (tried twice). On edge hardware the first
quantization question is not "how fast" but *"does it fit at all"* — which is
exactly why 4-bit variants are the default deployment choice.

**Bottom line:** on this hardware `q4_K_M` strictly dominates — same speed as
q3, no observed hallucinations on the test set, and it fits comfortably in
RAM, which q8_0 does not. That is why `llama3.2:3b` (= q4_K_M) serves as the
agent's brain in Parts 1–4.

## Reproduce

```bash
# pull the quant variants (network required once)
ollama pull llama3.2:3b-instruct-q3_K_M
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q8_0   # only if you have >4 GB free RAM

python bonus/quantization/profile.py     # writes this report + raw_outputs/
```
