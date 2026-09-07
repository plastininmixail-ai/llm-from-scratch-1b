"""Train 500M Bridge Bot.

3 phases:
1. Pre-training на corpus
2. SFT на ChatML данных
3. DPO на preference данных
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path("C:/Users/mixai/Desktop/llm-from-scratch")


def run_phase(phase_name: str, cmd: list[str]):
    print(f"\n{'='*70}")
    print(f"🚀 Phase: {phase_name}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"❌ Phase {phase_name} failed with code {result.returncode}")
        sys.exit(1)
    print(f"✅ Phase {phase_name} completed")


def main():
    # Phase 1: Pre-training
    run_phase("Pre-training 500M", [
        "python", "-m", "training.train",
        "--config", "configs/xxxlarge.yaml",
        "--tokenizer", "tokenizer/vocab.json",
        "--corpus", "data/raw/chat_xlarge_corpus.txt",
        "--max-steps", "5000",
        "--batch-size", "1",
        "--seq-len", "256",
        "--lr", "2e-4",
        "--warmup", "500",
        "--out-dir", "checkpoints/chat-xxxlarge-pretrain",
    ])

    # Phase 2: SFT
    run_phase("SFT 500M", [
        "python", "-m", "training.train",
        "--config", "configs/xxxlarge.yaml",
        "--tokenizer", "tokenizer/vocab.json",
        "--corpus", "data/sft_chatml.jsonl",
        "--max-steps", "2000",
        "--batch-size", "1",
        "--seq-len", "256",
        "--lr", "5e-6",
        "--warmup", "200",
        "--base", "checkpoints/chat-xxxlarge-pretrain/best.pt",
        "--reset-step",
        "--out-dir", "checkpoints/chat-xxxlarge-sft",
    ])

    # Phase 3: DPO
    run_phase("DPO 500M", [
        "python", "-m", "training.dpo_train",
        "--base", "checkpoints/chat-xxxlarge-sft/best.pt",
        "--data", "data/dpo_dataset.jsonl",
        "--max-pairs", "1000",
        "--lr", "1e-6",
        "--beta", "0.1",
    ])

    print("\n" + "="*70)
    print("🎉 All 3 phases completed!")
    print("="*70)


if __name__ == "__main__":
    main()
