"""Bonus 2 — quantization performance profiling.

Compares one model (llama3.2:3b) across quantization levels on two metrics the
assignment asks for — generation throughput (tokens/sec) and peak memory — plus
a saved sample of each level's output so the speed-vs-quality trade-off can be
judged.

Fair-comparison controls (so the numbers mean something):
  * identical prompts, identical options (temperature=0, fixed seed, fixed
    num_predict) across every quant;
  * a warm-up generation per model is discarded (it pays the one-time cold load);
  * TPS is averaged over several runs;
  * each model is unloaded before the next is loaded (keep_alive=0), so the peak
    memory of one level never bleeds into another's.

TPS is read straight from Ollama's own timings (`eval_count`, `eval_duration`),
not estimated. This machine is CPU-only, so "peak memory" is peak RAM (VmHWM of
the Ollama runner process); on a GPU box the same table column would be VRAM.

Run (after the quant tags are pulled):
    python bonus/quantization/profile.py
"""

from __future__ import annotations

import glob
import statistics
import sys
import time
from pathlib import Path

# Make `src` importable whether this is run as a script or a module.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ollama import Client  # noqa: E402

from src.config import settings  # noqa: E402

# Quant levels to compare — low / mid / high precision of the same 3B model.
# Ordered lightest-first: q8_0 (~3.2GB resident) is at this machine's RAM
# ceiling, so it runs last — the lighter levels' results are already saved
# if the heaviest one struggles.
MODELS = [
    "llama3.2:3b-instruct-q3_K_M",
    "llama3.2:3b-instruct-q4_K_M",
    "llama3.2:3b-instruct-q8_0",
]

# One prompt set, reused for every quant. Mixes a factual, a reasoning, and an
# in-domain (AI-agents) question so the quality sample is representative.
PROMPTS = [
    "In two sentences, what is the capital of France and why is it famous?",
    "A farmer has 17 sheep. All but 9 run away. How many are left? Explain briefly.",
    "In 3-4 sentences, explain the ReAct pattern used by AI agents.",
]

NUM_PREDICT = 150          # cap generated tokens so runs are comparable
N_TIMED_RUNS = 2           # TPS is averaged over this many runs (after warm-up)
NUM_THREADS = 10           # leave 2 of the 12 cores free so the machine stays usable
OPTIONS = {
    "temperature": 0,
    "seed": 0,
    "num_predict": NUM_PREDICT,
    "num_thread": NUM_THREADS,  # identical for every quant → fair comparison
}

OUT_DIR = Path(__file__).parent
RAW_DIR = OUT_DIR / "raw_outputs"
REPORT = OUT_DIR / "report.md"


# --------------------------------------------------------------------------- #
# Memory: peak RSS (VmHWM) of the Ollama runner subprocess holding the model.
# Only one model is loaded at a time, so the runner is unambiguous.
# --------------------------------------------------------------------------- #
def _runner_pids() -> list[int]:
    pids = []
    for proc in glob.glob("/proc/[0-9]*"):
        try:
            cmd = Path(proc, "cmdline").read_bytes().decode("utf-8", "ignore")
        except OSError:
            continue
        cmd = cmd.replace("\x00", " ")
        # Ollama's model-serving subprocess: named `llama-server` in current
        # versions ("runner" in older ones), always pointed at a model blob.
        if ("llama-server" in cmd or "runner" in cmd) and ("blobs" in cmd or ".gguf" in cmd):
            pids.append(int(proc.rsplit("/", 1)[-1]))
    return pids


def _vmhwm_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024  # kB -> MB
    except OSError:
        pass
    return 0.0


def peak_ram_mb() -> float:
    """Peak resident memory of the loaded model's runner process, in MB."""
    return max((_vmhwm_mb(p) for p in _runner_pids()), default=0.0)


def unload_all(client: Client) -> None:
    """Unload every currently-resident model (keep_alive=0), so the next model
    loads into a clean memory state. Only touches models that are actually
    loaded — checking ps() first avoids loading a model just to unload it."""
    try:
        loaded = client.ps().models
    except Exception:
        return
    for m in loaded:
        try:
            client.generate(model=m.model, prompt="", keep_alive=0)
        except Exception:
            pass
    if loaded:
        time.sleep(1)  # give the runner a moment to release memory


# --------------------------------------------------------------------------- #
# Profiling one model
# --------------------------------------------------------------------------- #
def _disk_size_gb(client: Client, model: str) -> float:
    for m in client.list().models:
        if m.model == model:
            return (m.size or 0) / 1e9
    return 0.0


def _tps(resp) -> float:
    # Ollama returns nanoseconds; guard against a zero duration.
    ec = getattr(resp, "eval_count", 0) or 0
    ed = getattr(resp, "eval_duration", 0) or 0
    return (ec / (ed / 1e9)) if ed else 0.0


def profile_model(client: Client, model: str) -> dict:
    print(f"\n=== {model} ===")

    # Ensure a clean load: unload everything, then warm up (discarded).
    unload_all(client)
    print("  warm-up (cold load, discarded) …")
    client.generate(model=model, prompt="Hello", options=OPTIONS, keep_alive="5m")

    # Timed runs — average TPS across the prompt set, N times each.
    tps_samples: list[float] = []
    for run in range(N_TIMED_RUNS):
        for prompt in PROMPTS:
            r = client.generate(model=model, prompt=prompt, options=OPTIONS, keep_alive="5m")
            tps_samples.append(_tps(r))
        print(f"  timed run {run + 1}/{N_TIMED_RUNS} done")

    peak = peak_ram_mb()  # while the model is still resident

    # One clean generation per prompt, saved verbatim for the quality judgement.
    outputs = []
    for prompt in PROMPTS:
        r = client.generate(model=model, prompt=prompt, options=OPTIONS, keep_alive="5m")
        outputs.append((prompt, (r.response or "").strip()))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe = model.replace(":", "_").replace("/", "_")
    (RAW_DIR / f"{safe}.txt").write_text(
        "\n\n".join(f"PROMPT: {p}\nOUTPUT: {o}" for p, o in outputs),
        encoding="utf-8",
    )

    # Unload so the next model starts from a clean slate.
    try:
        client.generate(model=model, prompt="", keep_alive=0)
    except Exception:
        pass

    result = {
        "model": model,
        "disk_gb": _disk_size_gb(client, model),
        "peak_ram_mb": peak,
        "tps_mean": statistics.mean(tps_samples) if tps_samples else 0.0,
        "tps_stdev": statistics.stdev(tps_samples) if len(tps_samples) > 1 else 0.0,
    }
    print(f"  TPS={result['tps_mean']:.1f}  peak RAM={peak:.0f} MB  disk={result['disk_gb']:.1f} GB")
    return result


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(results: list[dict]) -> None:
    lines = [
        "# Bonus 2 — Quantization Performance Report",
        "",
        f"Model: **llama3.2:3b** · Engine: **Ollama {settings.ollama_host}** · "
        "Hardware: **CPU-only** (peak memory is RAM, not VRAM).",
        "",
        f"Method: {N_TIMED_RUNS} timed runs × {len(PROMPTS)} prompts per quant, "
        f"`temperature=0, seed=0, num_predict={NUM_PREDICT}, num_thread={NUM_THREADS}` "
        "(identical for every quant); warm-up discarded; one model resident at a "
        "time. TPS from Ollama's `eval_count / eval_duration`.",
        "",
        "| Quant level | Bits (approx) | Disk size | Peak RAM | TPS (mean ± sd) |",
        "|---|---|---|---|---|",
    ]
    bits = {"q8_0": "8-bit", "q4_K_M": "~4-bit", "q3_K_M": "~3-bit"}
    for r in results:
        tag = r["model"].split("-", 1)[-1] if "-" in r["model"] else r["model"]
        b = next((v for k, v in bits.items() if k in r["model"]), "—")
        if r.get("error"):
            lines.append(f"| `{tag}` | {b} | {r['disk_gb']:.1f} GB | — | — ({r['error']}) |")
        else:
            lines.append(
                f"| `{tag}` | {b} | {r['disk_gb']:.1f} GB | {r['peak_ram_mb']:.0f} MB | "
                f"{r['tps_mean']:.1f} ± {r['tps_stdev']:.1f} |"
            )
    lines += [
        "",
        "## Speed vs. quality trade-off",
        "",
        "_(Fill in from the raw outputs in `raw_outputs/` after running.)_ "
        "Lower-bit levels use less disk and RAM and typically generate faster on "
        "CPU (less memory bandwidth per token), at the cost of output quality: "
        "expect the 3-bit level to be the fastest and lightest but the most prone "
        "to errors/incoherence, while 8-bit is the heaviest and slowest but "
        "closest to full precision. See side-by-side generations in "
        "[`raw_outputs/`](raw_outputs/).",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT}")


def main() -> None:
    client = Client(host=settings.ollama_host)
    available = {m.model for m in client.list().models}
    missing = [m for m in MODELS if m not in available]
    if missing:
        print(f"Missing quant tags (pull them first): {missing}", file=sys.stderr)
        raise SystemExit(1)

    t0 = time.time()
    results: list[dict] = []
    for m in MODELS:
        try:
            results.append(profile_model(client, m))
        except Exception as exc:
            # A quant that cannot even load is a *finding*, not a crash: record it.
            # (q8_0 is 3.4GB resident vs ~3.8GB total WSL RAM on this machine.)
            print(f"  FAILED to profile: {type(exc).__name__}: {exc}")
            results.append({
                "model": m,
                "disk_gb": _disk_size_gb(client, m),
                "peak_ram_mb": None,
                "tps_mean": None,
                "tps_stdev": None,
                "error": "did not load — exceeds available RAM on this machine",
            })
        write_report(results)  # rewrite after every model → partial run still yields a report
    print(f"Done in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
