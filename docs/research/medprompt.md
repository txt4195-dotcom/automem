# Medprompt — Dynamic Few-Shot Prompting

> **Paper**: "Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine"
> **Authors**: Harsha Nori, Yin Tat Lee, Sheng Zhang, et al. (Microsoft Research)
> **Date**: November 2023
> **arXiv**: https://arxiv.org/abs/2311.16452
> **Code**: https://github.com/microsoft/promptbase

## Core Idea

Generalist LLM (GPT-4) + systematic prompting > domain-specific fine-tuned models.
No training, no weight updates — prompting alone으로 MedQA SOTA 달성 (90.2%).

## Pipeline (3 Components)

### 1. Dynamic Few-Shot Selection (kNN)

고정 few-shot이 아니라, **query마다 가장 관련 있는 example을 동적으로 선택**.

```
[Offline Prep]
Training set 전체 → text-embedding-ada-002 → embedding DB

[Per Query]
Test question → embed → kNN(k=5) → 가장 유사한 training examples 5개 선택
→ 이 5개를 few-shot example로 prompt에 주입
```

- Embedding model: OpenAI text-embedding-ada-002
- k = 5 (default)
- Distance metric: cosine similarity in embedding space

### 2. Self-Generated Chain of Thought

인간 전문가가 아닌 **GPT-4 자신이 CoT 설명을 생성**.

```
[Offline Prep]
For each training example:
  1. GPT-4에게 문제 + 정답을 주고 CoT 설명 생성
  2. GPT-4의 설명이 ground truth와 일치하는지 검증
  3. 불일치하면 필터링 (=잘못된 추론 제거)

→ 검증된 (question, CoT, answer) 트리플릿이 few-shot DB가 됨
```

**핵심 발견**: GPT-4가 생성한 CoT > 인간 전문가가 작성한 CoT (더 상세하고 효과적)

### 3. Choice Shuffle Ensembling

LLM의 positional bias (A번 선호 등)를 제거.

```
[Per Query]
1. 선택지 순서를 랜덤으로 shuffle
2. 같은 질문을 5~15회 반복 실행 (각각 다른 shuffle)
3. Majority vote로 최종 답 선정
```

## Ablation Results (MedQA)

```
Zero-shot baseline:          81.7%
+ Random few-shot:           83.9%  (+2.2)
+ Chain of Thought:          87.3%  (+3.4)  ← 가장 큰 점프
+ kNN dynamic selection:     88.4%  (+1.1)  ← random → kNN
+ Choice shuffle ensemble:   90.2%  (+1.8)
                             ─────
총 error rate 27% 감소 (vs MedPaLM 2)
```

## Key Findings

1. **Domain-agnostic**: 의학에서 만든 방법이 법학, 회계, 심리학, 전기공학에도 그대로 적용
2. **GPT-4 CoT > Human expert CoT**: 자가 생성 설명이 더 효과적
3. **kNN > Random**: 관련 example이 무관한 example보다 낫다 (정량적 입증)
4. **Ensemble이 bias 제거**: positional bias 상쇄로 안정적 성능

## Code Implementation (promptbase repo)

```python
# embed_problems.py — offline prep
embed_batch(questions)  # ada-002로 training set 전체 embed

# MMLU.py — inference
run_cot_without_rank(
    test_problem,
    examples=dev_cot_results,   # CoT가 붙은 training examples
    mode="knn",                  # kNN selection
    num_examples=5,              # k=5
    num_repeat=5,                # 5회 반복 (shuffle)
    max_thread=50                # 병렬
)

# eval.py — aggregation
eval_answers()  # majority vote (가장 많이 나온 답 선택)
```

## Follow-up Paper

> **Paper**: "From Medprompt to o1: Exploration of Run-Time Strategies for Medical Challenge Problems and Beyond"
> **Date**: November 2024
> **arXiv**: https://arxiv.org/html/2411.03590v1

Key findings:
- o1-preview는 prompting 없이도 Medprompt+GPT-4를 대부분 이김
- **Few-shot prompting이 o1 성능을 오히려 저해** — reasoning-native 모델에서는 in-context learning이 방해
- Cost-accuracy Pareto frontier: GPT-4o (저렴) vs o1 (고성능)

## References

- [Microsoft Research Blog: The Power of Prompting](https://www.microsoft.com/en-us/research/blog/the-power-of-prompting/)
- [arXiv: Original Paper](https://arxiv.org/abs/2311.16452)
- [arXiv: Follow-up Paper](https://arxiv.org/html/2411.03590v1)
- [GitHub: promptbase](https://github.com/microsoft/promptbase)
