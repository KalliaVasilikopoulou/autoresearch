# Multi-Agent NN Parameter Optimization: Complete Guide

## Overview

This guide explains the **multi-agent framework** for intelligent neural network optimization. Instead of random hyperparameter search, a team of specialized AI agents works together to understand model behavior and make informed decisions.

## The Four Agents

### Agent 1: Training Specialist
**Job:** Train models and decide next hyperparameters

**How it works:**
1. Receives summary from Agent 3 (or starts with defaults)
2. Adjusts hyperparameters using heuristic rules
3. Trains model for 5 minutes using `train.py`
4. Saves checkpoint and metrics
5. Repeats

**Heuristic Rules (Default - No LLM Cost):**
```
- If "learning rate" mentioned as important in summary → try 2x or 0.5x
- If "depth/layers" mentioned as important → try +1 or +2 layers
- If model stuck → try radical changes (random depth)
- Otherwise → small random exploration in early iterations
```

**Optional: Claude Enhancement**
```python
if use_llm and summary_exists:
    claude_suggestion = ask_claude(
        "Summary insights + our heuristic suggestion",
        "Should we adjust? Respond with JSON"
    )
    new_hyperparams = claude_suggestion or heuristic_suggestion
```

**Files:**
- `agents/agent1_training_specialist.py`
- `model_hyperparams.yaml` (where hyperparams are stored)

---

### Agent 2: XAI Specialist
**Job:** Analyze trained model and understand what matters

**How it works:**
1. Loads trained model checkpoint
2. Runs XAI analysis (fast method by default):
   - **Ablation**: Turn off each top-K attention head, measure impact
   - **Partial Dependence**: Vary hyperparams, estimate effect
3. Generates statistical report with tables
4. Optional: Sends to Claude for interpretive insights
5. Saves report to `reports/agent2_reports/report_XXXX.md`

**Statistical Report Example:**
```markdown
## Attention Head Ablation
| Head | Impact (bpb drop) |
|------|------------------|
| L3_H2 | 0.015420 |
| L5_H7 | 0.008310 |
| L2_H1 | 0.002140 |

## Hyperparameter Importance
| Parameter | Estimated Impact |
|-----------|------------------|
| learning_rate | 0.065000 |
| n_layers | 0.012500 |
| batch_size | 0.003200 |

## Detected Signals
- ✓ Model showing new patterns
- Flagged opportunities: Consider pruning unused heads
```

**Optional: Claude Interpretation**
```python
if use_llm:
    insights = ask_claude(
        "Statistical Report",
        "Please interpret and provide recommendations"
    )
    # Adds "## Claude Insights" section to report
```

**Why Fast Methods?**
- **Ablation**: ~5 minutes per model (testing ~10 attention heads)
- **Partial Dependence**: Heuristic estimates (no retraining needed)
- **Thorough methods**: Would add 30+ minutes per model

**Files:**
- `agents/agent2_xai_specialist.py`
- `agents/xai_methods/fast_methods.py` (ablation, partial dependence)
- `agents/xai_methods/thorough_methods.py` (placeholder for later)

---

### Agent 3: Report Analyst
**Job:** Find patterns across all past reports

**How it works:**
1. Waits for batch of 3 new reports from Agent 2
2. When batch complete:
   - Reads new batch of 3 reports
   - Reads previous summary (if exists)
   - Aggregates into strategic summary
3. Optional: Sends to Claude for narrative
4. Saves to `reports/agent3_summaries/summary_XXXX.md`

**Statistical Summary Example:**
```markdown
# Summary Report #0

## This Batch
Analyzed 3 new model reports.
- Stuck models detected: 0/3
- Most frequently important heads: [L3_H2, L5_H7, L4_H1]

## Consistent Patterns (All History)
- Attention heads show consistent importance ranking
- Model depth is consistently important hyperparameter
- Learning rate variations have measurable impact

## Recommendations for Agent 1
- Focus on varying learning rate and model depth
- Consider pruning consistently unused attention heads
- Maintain architectural stability if models not stuck

## Previous Summary (Condensed)
[Historical patterns from past models...]
```

**Hierarchical Memory:**
Each summary includes condensed version of previous summary. No information lost!

**Optional: Claude Narrative**
```python
if use_llm:
    narrative = ask_claude(
        "Statistical summary",
        "What patterns emerge? What should Agent 1 focus on?"
    )
    # Adds "## Strategic Narrative" section
```

**Files:**
- `agents/agent3_report_analyst.py`

---

### Orchestrator: Team Leader
**Job:** Coordinate all agents and manage state

**How it works:**
1. Initialize all agents
2. Loop:
   - Agent 1 trains model
   - Agent 2 analyzes (parallel)
   - Agent 3 aggregates when batch ready (parallel)
   - Check stopping conditions
   - Repeat or exit

**Stopping Conditions:**
- Accuracy threshold reached (val_bpb < threshold)
- API cost exceeded (if using Claude)
- Max iterations reached (safety limit)

**Files:**
- `agents/orchestrator.py`

---

## State Management

**StateManager tracks everything:**

```python
metadata.json:
{
  "models": {
    "model_0000": {
      "iteration": 0,
      "hyperparams": {...},
      "val_bpb": 0.998,
      "report_id": "report_0000",
      "timestamp": 1234567890
    },
    ...
  },
  "latest_summary": "summary_0000",
  "iteration": 5
}

report_tracking.json:
{
  "reports": {
    "report_0000": {
      "report_num": 0,
      "model_id": "model_0000",
      "timestamp": 1234567890
    },
    ...
  },
  "batch_count": 3,
  "latest_summary_covers_up_to": 0
}
```

**Files:**
- `state/state_manager.py`
- `state/metadata.json` (auto-generated)
- `state/report_tracking.json` (auto-generated)
- `state/models/model_XXXX.pt` (model checkpoints)

---

## Configuration

**agents_config.yaml:**

```yaml
agent1:
  use_llm: false                    # No LLM by default
  accuracy_threshold: 0.97          # Stop when reached
  cost_limit_usd: 50.0              # Stop when exceeded
  training_budget_seconds: 300      # 5 min per model

agent2:
  xai_method: "fast"                # fast or thorough
  use_llm: false                    # No LLM by default
  ablation_k: 10                    # Top-K heads to ablate
  incremental: true                 # Only re-analyze changes

agent3:
  batch_size: 3                     # Reports per summary
  use_llm: false                    # No LLM by default
  preserve_history: true            # Keep all past summaries

orchestrator:
  parallel: true                    # Run agents in parallel
  poll_interval_seconds: 5          # Check every 5s
  max_iterations: 100               # Safety limit
```

---

## Enabling LLM (Claude)

**Cost:** $0.20-0.35 per model (compared to $2-3 GPU cost—negligible)

### Step 1: Get Claude API Key

1. Go to [https://console.anthropic.com](https://console.anthropic.com)
2. Create account and get API key
3. Set environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### Step 2: Enable in Config

Edit `agents_config.yaml`:

```yaml
agent1:
  use_llm: true  # Claude suggests next hyperparams

agent2:
  use_llm: true  # Claude interprets XAI results

agent3:
  use_llm: true  # Claude writes strategic narrative
```

### Step 3: Run

```bash
python agents/orchestrator.py --config agents_config.yaml
```

---

## Running the System

### Start Multi-Agent Loop

```bash
# Run multi-agent optimization
python agents/orchestrator.py --config agents_config.yaml

# Or with custom settings
python agents/orchestrator.py --config my_config.yaml --iterations 50
```

### Monitor Progress

```bash
# Watch reports being generated
watch 'ls -lh reports/agent2_reports/ | tail -5'
watch 'ls -lh reports/agent3_summaries/ | tail -5'

# Check state
cat state/metadata.json | jq '.iteration'

# View latest summary
tail -50 reports/agent3_summaries/summary_*.md
```

### Analyze Results

```bash
# Best model found
cat state/metadata.json | jq '.models | max_by(.val_bpb)'

# Total API cost (if enabled)
grep -r "cost_usd" state/

# Model progression
jq '.models | .[] | [.iteration, .val_bpb]' state/metadata.json
```

---

## Claude API Calls (Simple Explanation)

### Where Claude is Used

| Agent | Purpose | Example |
|-------|---------|---------|
| **Agent 2** | Interpret XAI results | "Here are ablation results. What does this mean?" |
| **Agent 3** | Summarize reports | "Summarize 3 reports. What patterns emerge?" |
| **Agent 1** | Suggest hyperparams | "Given summary, should we adjust learning rate?" |

### What Actually Happens

```python
# 1. Build prompt
prompt = f"""
Statistical Report:
{ablation_results}
{partial_dependence}

Please interpret and recommend next steps.
"""

# 2. Call Claude API
response = claude.messages.create(
    model="claude-opus-4-7",
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}]
)

# 3. Use response
insight = response.content[0].text
report += f"\n\n## Claude Insights\n{insight}"

# 4. Save report
with open("reports/agent2_reports/report_0000.md", "w") as f:
    f.write(report)
```

### Cost Breakdown

Each API call costs roughly:

```
Tokens in:  ~500-1000 tokens
Tokens out: ~500-1000 tokens
Model:      claude-opus-4-7 @ $15/M in, $75/M out

Cost per call ≈ (750 in + 750 out) / 1M tokens = ~$0.06
Per model: Agent 2 ($0.06) + Agent 1 ($0.03) = ~$0.09
Per batch: Agent 3 ($0.15) for 3 models = $0.05 per model
Total per model: ~$0.20 (with all enabled)
```

### Error Handling

If API call fails:
- Agent uses fallback (statistical methods only)
- Report still generated, just without Claude insights
- No crash, system continues

---

## File Structure

```
autoresearch/
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py                # Team coordinator
│   ├── agent1_training_specialist.py  # Trains models
│   ├── agent2_xai_specialist.py       # Analyzes models
│   ├── agent3_report_analyst.py       # Aggregates reports
│   └── xai_methods/
│       ├── __init__.py
│       ├── fast_methods.py            # Ablation, partial dependence
│       └── thorough_methods.py        # Future: circuits, SVCCA
│
├── state/
│   ├── __init__.py
│   ├── state_manager.py               # Tracks everything
│   ├── models/                        # Model checkpoints
│   ├── metadata.json                  # Model info
│   └── report_tracking.json           # Report info
│
├── reports/
│   ├── agent2_reports/                # Individual analyses
│   │   ├── report_0000.md
│   │   ├── report_0001.md
│   │   └── ...
│   └── agent3_summaries/              # Aggregated summaries
│       ├── summary_0000.md
│       └── ...
│
├── agents_config.yaml                 # Configuration
├── model_hyperparams.yaml             # Current hyperparams
├── train.py                           # GPT training (modified by Agent 1)
├── prepare.py                         # Data prep (read-only)
├── program.md                         # Original agent instructions
└── README.md
```

---

## Troubleshooting

### Claude API Errors

**Problem:** `ANTHROPIC_API_KEY not set`
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Problem:** Rate limited
- Wait 60 seconds
- Reduce report batch size
- Disable Claude temporarily

### Training Errors

**Problem:** `Model checkpoint not found`
- Make sure Agent 1 successfully trained
- Check `state/models/` directory

**Problem:** Training timeout
- Increase `training_budget_seconds` in config
- Or reduce model size

### Memory Issues

**Problem:** CUDA out of memory
- Reduce `n_embd` in model config
- Reduce `DEVICE_BATCH_SIZE` in train.py
- Switch to smaller GPU

---

## Next Steps

1. **Start simple:** Run with all LLM disabled (0 cost)
2. **Monitor:** Check reports being generated
3. **Enable gradually:** Try Agent 2 LLM first, then others
4. **Tune config:** Adjust batch size, ablation_k, etc.
5. **Scale:** Run overnight for 100+ models

---

## References

- **XAI Methods:** fast_methods.py, thorough_methods.py
- **Heuristic Rules:** agent1_training_specialist.py
- **State Tracking:** state/state_manager.py
- **Configuration:** agents_config.yaml

