"""GRPO (Group Relative Policy Optimization) — скелет для будущего использования.

НЕ полная реализация — только структура и контракты.
Будет реализована когда дойдём до Stage 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    """GRPO hyperparameters."""
    group_size: int = 4          # G сэмплов на промпт
    kl_coef: float = 0.04        # β для KL divergence
    clip_range: float = 0.2      # PPO-style clipping
    learning_rate: float = 1e-6
    epochs_per_step: int = 1
    max_new_tokens: int = 512


@dataclass
class RewardResult:
    """Результат reward function."""
    score: float                 # 0.0 - 1.0
    correct: bool
    partial_credit: float = 0.0  # для дробных rewards


class RewardFunction:
    """Базовый класс reward function. Должен быть subclassed."""

    def compute(self, prompt: str, response: str, ground_truth: str) -> RewardResult:
        """Вычислить reward для пары (prompt, response, ground_truth)."""
        raise NotImplementedError


class MathRewardFunction(RewardFunction):
    """Reward для математических задач."""

    def compute(self, prompt: str, response: str, ground_truth: str) -> RewardResult:
        # Извлечь финальный ответ из response (после #### или QED)
        import re

        # GSM8K format: ответ после "####"
        match = re.search(r"####\s*([-+]?\d*\.?\d+)", response)
        if match:
            predicted = match.group(1).strip()
        else:
            # Попробовать последнее число
            numbers = re.findall(r"[-+]?\d*\.?\d+", response)
            predicted = numbers[-1] if numbers else ""

        gt = str(ground_truth).strip()

        correct = predicted == gt
        score = 1.0 if correct else 0.0

        # Partial credit: формат правильный но ответ нет
        if not correct and re.search(r"####\s*[-+]?\d*\.?\d+", response):
            score = 0.1

        return RewardResult(
            score=score,
            correct=correct,
            partial_credit=score,
        )


class GRPOTrainer:
    """GRPO trainer — скелет, будет реализован позже."""

    def __init__(self, config: GRPOConfig, reward_fn: RewardFunction):
        self.config = config
        self.reward_fn = reward_fn
        self.step = 0

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """Group-relative advantage: (r - mean) / std."""
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        if len(rewards_t) < 2:
            return torch.zeros_like(rewards_t)

        mean = rewards_t.mean()
        std = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean) / std
        return advantages

    def grpo_loss(
        self,
        old_logprobs: torch.Tensor,
        new_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        ref_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        """GRPO loss function.

        L = -E[min(ratio * A, clip(ratio) * A)] + β * KL(new || ref)

        Args:
            old_logprobs: log π_old(a|s)
            new_logprobs: log π_θ(a|s)
            advantages: A(s, a) — group-relative
            ref_logprobs: log π_ref(a|s) — reference model
        """
        ratio = torch.exp(new_logprobs - old_logprobs)

        # PPO-style clipping
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        # KL penalty (k3 estimator)
        kl = (new_logprobs - ref_logprobs).mean()
        kl_penalty = self.config.kl_coef * kl

        return policy_loss + kl_penalty

    def step_update(self, group_responses, group_rewards):
        """Один GRPO update step."""
        advantages = self.compute_advantages(group_rewards)
        # TODO: forward pass, compute log probs, apply loss
        raise NotImplementedError("GRPO step_update not implemented yet")


if __name__ == "__main__":
    # Пример использования reward function
    rf = MathRewardFunction()
    result = rf.compute(
        prompt="If 2+2=?",
        response="Let me think. 2+2 = 4. #### 4",
        ground_truth="4"
    )
    print(f"Reward: {result.score}, Correct: {result.correct}")
