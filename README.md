# claude-scope-tracker

> A Claude Code hook that learns from your session history to predict token cost before you run a task — and warns you when something looks likely to blow up mid-execution.

## why this exists

Claude Code sessions sometimes hit token limits mid-task: context fills up, output gets truncated, agent loops break. By the time you notice, you've already burned the tokens and the work is half-done.

This tool quietly logs every task you run, learns what kinds of prompts in what kinds of repos tend to cost what, and starts warning you before high-cost tasks begin. It doesn't ask Claude to estimate itself (Claude is bad at this). It watches what actually happens in your sessions and builds a model from your real usage.

Everything stays local. No network calls, no telemetry. Your data lives in `~/.scope-tracker/`.

## how it works

Three hooks get installed into Claude Code:

| Hook | Purpose |
|---|---|
| `UserPromptSubmit` | Extracts features from prompt + repo, records task start, predicts total tokens, warns if high |
| `PostToolUse` | Counts tool calls per task |
| `Stop` | Reads the session transcript, computes actual tokens used, retrains the predictor on milestones |

A task = one user prompt + everything Claude does until it stops responding. Each task is one row in a local SQLite DB.

## install

```bash
pip install claude-scope-tracker
scope-tracker install
```

That writes hook entries into `~/.claude/settings.json` (backing up your existing file first) and creates `~/.scope-tracker/`.

Then use Claude Code normally. **Predictions stay silent until 20 tasks have been logged with usage data.** Before that, the tool is just observing.

## commands

```bash
scope-tracker install      # add hooks to ~/.claude/settings.json
scope-tracker uninstall    # remove hooks (keeps local data)
scope-tracker stats        # show your task history and model state
scope-tracker reset        # wipe local data
```

Example `stats` output once you've got data:

```
data dir:     ~/.scope-tracker
model:        trained
tasks logged: 47
  completed:  43

token usage across completed tasks:
  mean:    38,200
  min:      4,100
  max:    142,800

last 10 tasks:
  when         actual      pred    err  prompt
  05-15 14:22   28,400   31,000   +9%  fix the failing auth tests in user_service
  05-15 13:50  112,300   85,000  -24%  refactor the entire payments module to use stripe v3
  ...
```

## the model

Right now it's a gradient-boosted regressor over ten cheap hand-crafted features (prompt length, verb tier, file references, repo size, dominant language, etc.). It's deliberately simple. The point of v0 is to find out whether token cost is at all predictable from these signals. If it is, the model gets fancier. If it isn't, no model would help.

The warning fires only when:

1. You have ≥ 20 completed tasks logged
2. The prediction is above `max(50,000 tokens, 2 × your historical mean)`

The warning text goes into Claude's context as a notice for the user — Claude reads it and surfaces it naturally in its reply, often with a scope-down suggestion. It doesn't block your prompt.

## what this is NOT

- **Not a billing tool.** It estimates total tokens for context-limit purposes, not cost.
- **Not a guarantee.** v0 is rough. Treat warnings as nudges, not hard limits.
- **Not Claude estimating itself.** Self-estimation by LLMs is poorly calibrated. This learns from observed outcomes instead.

## roadmap (maybe)

- Prediction intervals, not point estimates
- Per-repo personalization (hierarchical model)
- Mid-task budget tracking — warn when consumption crosses % of predicted
- Pre-`PreCompact` warning hook
- Cross-session learning across teams (opt-in, anonymized)
- A "scope down" agent that auto-proposes smaller versions of big tasks

## design notes

A few things worth knowing if you want to hack on this:

- **Feature stability matters.** `FEATURE_NAMES` in `core.py` defines the vector order. If you add features, retrain — old models become invalid.
- **Transcript schema is permissive.** Claude Code's transcript format is mostly stable but undocumented; the parser silently tolerates missing fields rather than crashing.
- **Hooks must never crash.** Every handler swallows its own exceptions and exits 0. A broken hook would silently break Claude Code.
- **Cache reads are excluded from the "total" metric.** Cache reads count toward context but aren't "new" tokens. The total tracks input + output + cache creation.

## license

MIT — see `LICENSE`.
