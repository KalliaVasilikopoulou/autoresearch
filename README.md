# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069) and [this tweet](https://x.com/karpathy/status/2031135152349524125).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Multi-Agent NN Parameter Optimization (NEW)

This repo now includes a **multi-agent framework** for intelligent hyperparameter optimization using explainability feedback. Instead of blindly trying random changes, a team of AI agents work together to understand *why* models succeed or fail, then make informed decisions.

### The Multi-Agent Team

```
┌─────────────────────────────────────────────────────────┐
│ ORCHESTRATOR (Team Leader)                              │
│ Coordinates parallel execution of all agents            │
└──────────────┬──────────────┬──────────────┬────────────┘
               │              │              │
     ┌─────────▼──┐  ┌────────▼──────┐  ┌───▼──────────┐
     │ Agent 1    │  │ Agent 2       │  │ Agent 3      │
     │ TRAINING   │  │ XAI           │  │ ANALYST      │
     │ SPECIALIST │  │ SPECIALIST    │  │              │
     └────────────┘  └───────────────┘  └──────────────┘
```

**Agent 1: Training Specialist**
- Trains models on specific hyperparameters
- Validates and measures performance (val_bpb)
- Adjusts hyperparameters based on Agent 3 summaries
- Uses heuristic rules by default (no API cost)
- Optional: Can consult Claude for smarter suggestions

**Agent 2: XAI Specialist**
- Analyzes each trained model using explainability techniques
- Fast method (default):
  - **Ablation**: Turn off attention heads, measure impact
  - **Partial Dependence**: Vary hyperparameters, see effect
- Generates statistical reports (+ optional Claude prose interpretation)
- Flags important insights like "Learning rate matters 70% more than batch size"

**Agent 3: Report Analyst**
- Waits for batch of 3 new reports from Agent 2
- Aggregates across all past reports
- Finds consistent patterns: "Which hyperparameters always matter?"
- Creates hierarchical summaries that preserve all historical context
- Sends strategically important insights to Agent 1

**Orchestrator**
- Manages parallel execution
- Ensures agents run efficiently without deadlock
- Tracks state: models, reports, hyperparameters
- Implements stopping conditions (accuracy threshold, cost limit)

### Workflow

```
1. Agent 1 trains model with hyperparameters
2. Agent 2 analyzes: "Which attention heads matter? Which hyperparams?"
   → Generates statistical report
3. Agent 3 waits for batch of 3 reports
   → Creates summary: "Learning rate consistency → try 2x higher next time"
4. Agent 1 reads summary
   → Adjusts hyperparameters intelligently based on patterns
5. Repeat until accuracy threshold reached or cost limit hit
```

### Zero-Cost by Default

**All features are optional and disabled by default to avoid API costs:**

```yaml
agents:
  agent1:
    use_llm: false  # Statistical heuristics, no LLM cost
  agent2:
    use_llm: false  # Statistical reports, no LLM cost
  agent3:
    use_llm: false  # Pattern tables, no LLM cost
```

Cost breakdown if you enable LLM:
- Agent 2 report: $0.05-0.10 per model
- Agent 3 summary: $0.10-0.20 per batch (3 models)
- Agent 1 suggestions: $0.02-0.05 per decision

**Total default cost: $0.00 (all statistical)**
**Total with all LLMs enabled: ~$0.20-0.35 per model** (negligible vs GPU cost)

### Configuration

Edit `agents_config.yaml` to customize:

```yaml
# Stop when accuracy reaches this threshold
accuracy_threshold: 0.97

# Analyze only top-K attention heads (fast)
ablation_k: 10

# Batch size for report aggregation
batch_size: 3

# Enable Claude for intelligent analysis (optional)
use_llm: false
```

### Running Multi-Agent Optimization

```bash
# Start the multi-agent loop
python agents/orchestrator.py --config agents_config.yaml

# Monitor reports
ls -la reports/agent2_reports/  # Individual analyses
ls -la reports/agent3_summaries/  # Strategic summaries

# Check state
cat state/metadata.json  # Track all models and hyperparameters
```

### Key Design Decisions

1. **LLM is Optional**: All XAI analysis uses pure statistical methods by default. Claude can enhance reports, but isn't required.
2. **Hierarchical Memory**: Agent 3 summaries include condensed history—no information is lost, patterns accumulate.
3. **Incremental Analysis**: Agent 2 only re-analyzes what changed, not the entire model each iteration.
4. **Heuristic + Optional Claude**: Agent 1 uses statistical rules first ("LR matters most → try 2x"), Claude confirms or overrides.
5. **Stuck Detection**: Automatically triggers radical changes if model patterns repeat too much.

### Example Multi-Agent Run Output

```
============================================================
[Orchestrator] Iteration 1
============================================================

[Orchestrator] Phase 1: Agent 1 Training
[Agent 1] Deciding next hyperparameters (iteration 0)...
[Agent 1] Next hyperparams: {'n_layer': 12, 'n_embd': 512, 'learning_rate': 0.001}
[Agent 1] Starting training with: ...
[Orchestrator] Saved model: model_0000, val_bpb: 0.998400

[Orchestrator] Phase 2: Agent 2 XAI Analysis
[Agent 2] Analyzing model model_0000...
[Agent 2] Report saved: reports/agent2_reports/report_0000.md

[Orchestrator] Phase 3: Agent 3 Report Aggregation
(waiting for batch of 3...)

[Orchestrator] Iteration 1 complete
```



## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Multi-Agent System: Hardware & Subscriptions

### Minimum Requirements

**For single-agent baseline (original autoresearch):**
- NVIDIA GPU: H100 recommended, but A100, A6000, or similar work
- RAM: 80GB+ for H100, 40GB+ for smaller GPUs
- Cost: ~$2-3 per hour on cloud GPU

**For multi-agent optimization:**
Same as above. Multi-agent system **runs on same GPU**, just adds analysis overhead.

### Optional: Claude API (For Enhanced Reports)

**Cost if you enable LLM features:**
- ~$0.20-0.35 per model trained (~5 minutes per model)
- ~2-3 USD for 100 models overnight
- Completely optional—system works with $0 cost if disabled

**Setup:**
```bash
export ANTHROPIC_API_KEY=sk-ant-...  # From Claude console
```

### Simple Breakdown

| Scenario | GPU Cost | API Cost | Total Cost |
|----------|----------|----------|-----------|
| 100 models overnight (8h) | $16-24 | $0 | $16-24 |
| 100 models with Claude analysis | $16-24 | $2-3 | $18-27 |
| 1000 models overnight (80h) | $160-240 | $20-30 | $180-270 |

The GPU dominates the cost. LLM analysis is negligible but **improves decision quality**.

### Easy Start (No API)

```bash
# Everything works without Claude API
# Agent 1 uses heuristics
# Agent 2 generates pure statistical reports
# Agent 3 creates pattern tables
# Total cost: just GPU

python agents/orchestrator.py --config agents_config.yaml
```

### Advanced (With Claude)

```bash
# Edit agents_config.yaml
agent1:
  use_llm: true  # Claude suggests next hyperparams
agent2:
  use_llm: true  # Claude interprets XAI results
agent3:
  use_llm: true  # Claude writes strategic summaries

python agents/orchestrator.py --config agents_config.yaml
```



## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
