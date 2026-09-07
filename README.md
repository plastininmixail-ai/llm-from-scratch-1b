# LLM from Scratch — 1B Model

GPT-style decoder-only transformer trained from scratch with custom multi-agent framework.

## 🎯 Goal

Build a 1B-parameter reasoning-capable language model from scratch, with:
- Custom BPE tokenizer
- Custom training framework (no HuggingFace Trainer)
- Multi-agent orchestration (Architect / Data Engineer / Trainer / Curator)
- Stage 0: Pretrain on FineWeb-Edu
- Stage 1: Instruction following (planned)

## 🏗️ Architecture

- **Parameters**: ~1.07B
- **Layers**: 20
- **d_model**: 2048
- **Heads**: 32 (head_dim=64)
- **Context**: 1024 tokens
- **Vocab**: 32K BPE
- **Norm**: RMSNorm
- **Position**: RoPE
- **Activation**: GELU

## 📦 Repo Structure

```
llm-from-scratch/
├── agents/              # Multi-agent orchestration
│   ├── architect/       # Model design
│   ├── data_engineer/   # Corpus preparation
│   ├── trainer/         # Training loop
│   ├── curator/         # Checkpoint management
│   ├── orchestrator/    # Pipeline coordination
│   └── common/          # Base classes & protocol
├── framework/           # Custom ML framework
│   ├── model/           # GPT + RMSNorm + RoPE
│   ├── data/            # BinaryDataset + tokenization
│   ├── training/        # Trainer with AdamW + cosine
│   ├── inference/       # Text generation
│   └── evaluation/      # Perplexity metrics
├── configs/             # Training configs (YAML)
├── scripts/             # 30+ training/eval/upload scripts
├── data/                # Corpus (gitignored, 30GB+)
├── checkpoints/         # Model weights (gitignored)
└── logs/                # Training logs (gitignored)
```

## 🚀 Quick Start

### Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch pyyaml requests python-dotenv pynacl huggingface_hub
```

### Train

```bash
# Pretrain from scratch (1B model)
python -m scripts.train_v1 \
  --config configs/v1_fineweb.yaml \
  --warm-start checkpoints/v0_1b_1000/best.pt

# Or run pipeline end-to-end
python -m scripts.test_pipeline
```

### Generate

```bash
python -m scripts.sample \
  --checkpoint checkpoints/v1_fineweb/best.pt \
  --prompt "The capital of France is" \
  --max-tokens 50 \
  --n-samples 3
```

## 🤖 Multi-Agent System

```
Orchestrator
   │
   ├─→ Architect (model design)
   ├─→ Data Engineer (corpus)
   ├─→ Trainer (training loop)
   └─→ Curator (artifacts)
```

Each agent is a Python class in `agents/<name>/agent.py` with:
- `execute(task) -> AgentResult`
- Message passing via `logs/mailbox.jsonl`
- Logging to `logs/<role>/agent.log`

## 📊 Training Status

| Stage | Model | Steps | Loss |
|-------|-------|-------|------|
| Stage 0 (v0) | 113M | 400 | 4.08 |
| Stage 0.5 (v1) | 1B | running | 4.70 (step 80) |
| Stage 1 | 1B + SFT | planned | — |

## 🛠️ Tech Stack

- **PyTorch 2.x** (CPU-only training)
- **Python 3.12**
- **Custom BPE** tokenizer (byte-level)
- **No HuggingFace** — everything custom
- **CPU-only** — 8 threads, ~3.5 min/step on 1B model

## 📜 License

Apache 2.0

## 🤝 Links

- **GitHub**: https://github.com/plastininmixail-ai/llm-from-scratch-1b
- **HuggingFace**: https://huggingface.co/mixailplastinin/llm-from-scratch-1b
- **Telegram bot**: @Gopcaninebot
