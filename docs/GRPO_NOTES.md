# GRPO (Group Relative Policy Optimization)

Источник: `refs/reasoning-from-scratch/ch06/`

## Что это

Метод RL для reasoning LLM. Вместо обычного PPO использует **группы** сэмплов для оценки преимущества.

## Ключевые идеи

```
Для каждого промпта:
1. Сгенерировать G ответов (группа)
2. Вычислить reward для каждого (правильность)
3. Преимущество = (reward - mean(rewards)) / std(rewards)
4. Loss = -log(prob) * advantage (без value function)
```

## Преимущества перед PPO

| PPO | GRPO |
|-----|------|
| Нужен critic (value network) | Не нужен critic |
| Двойной forward pass | Одинарный |
| Медленно | Быстрее |
| Сложно настраивать | Проще |

## Для нашего проекта

**Когда применять**: Stage 2 (после SFT на Stage 1)
- Базовый чекпойнт: v1_fineweb + SFT
- Reward function: проверка ответа на математике (GSM8K)
- G = 4-8 сэмплов на промпт

## Данные для GRPO

- GSM8K train (7,473 задач)
- MATH train (7,500 задач)
- MetaMathQA (уже добавлено)

## Что нужно реализовать

1. `framework/rl/grpo.py` — GRPO алгоритм
2. `framework/rl/reward.py` — reward function (проверка ответа)
3. `framework/rl/prompts.py` — форматирование промптов с CoT
4. `scripts/train_grpo.py` — training loop

## Гиперпараметры (для 1B модели)

```yaml
learning_rate: 1e-6
group_size: 4
kl_coef: 0.04
clip_range: 0.2
batch_size: 2  # промптов
epochs_per_step: 1
```

## Ресурсы

- Paper: DeepSeekMath (https://arxiv.org/abs/2402.03300)
- Implementation: refs/reasoning-from-scratch/ch06/
