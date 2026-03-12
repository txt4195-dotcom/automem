# Situation Categories — 조합 기반 파생 컨텐츠 생성

> 원칙 × 카테고리 조합 → situation + reasoning → embed → Qdrant

## 핵심 아이디어

원칙 하나에서 situation을 "자유롭게 5개 생성"하면 nano가 비슷한 것만 만든다.
대신 **상황 카테고리 축을 미리 정의**하고, 축 값의 조합으로 생성하면:
- coverage가 체계적으로 보장됨
- 빠진 영역을 명시적으로 볼 수 있음
- pairwise coverage로 조합 폭발 없이 효율적

## 카테고리 축 설계

### 연구 기반

| 연구 | 기여 | 적용 |
|------|------|------|
| **TUNA** (Google, 2025) | 유저 행동 6 modes | Axis 1: Intent |
| **Plutchik's Wheel** | 8 기본 감정, 28 dyad 조합 | Axis 2: Emotion |
| **Russell Circumplex** | valence × arousal 2차원 | Axis 2를 2D로 매핑 가능 |
| **Pairwise Testing** | 조합 폭발 해결 | 생성 전략: all-pairs coverage |

### 5개 축

```
Axis 1: Intent (유저가 뭘 원하는가)
  ├── information_seeking  — 정보 찾기, 사실 확인
  ├── decision_support     — 결정 도움, 비교, 추천
  ├── emotional_support    — 감정 토로, 위로, 공감 요청
  ├── procedural_guidance  — 방법/절차 안내, how-to
  ├── reflection           — 자기 성찰, 회고, 의미 부여
  └── creation             — 아이디어 생성, 콘텐츠 작성

Axis 2: Emotional Valence (감정 방향)
  ├── positive_high    — 흥분, 기대, 열정 (high arousal, positive valence)
  ├── positive_low     — 만족, 평온, 감사 (low arousal, positive valence)
  ├── neutral          — 중립, 사무적
  ├── negative_low     — 무기력, 지침, 실망 (low arousal, negative valence)
  └── negative_high    — 분노, 좌절, 불안 (high arousal, negative valence)

Axis 3: Conversation Phase (대화 어디쯤인가)
  ├── opening          — 첫 질문, 탐색, 관계 형성
  ├── deepening        — 후속 질문, 구체화, 파고들기
  ├── turning          — 방향 전환, 새 주제, 기존 답변 불만
  ├── repeating        — 같은 질문 반복, 재확인, 불만족 신호
  └── closing          — 마무리, 요약 요청, 다음 단계

Axis 4: Stakes (얼마나 중요한 상황인가)
  ├── casual           — 가벼운 호기심, 잡담
  ├── moderate         — 실질적 관심, 학습 목적
  └── critical         — 큰 결정, 돈/건강/관계 관련, 긴급

Axis 5: Prior Knowledge (사전 지식 수준)
  ├── novice           — 처음 접함, 기초 설명 필요
  ├── intermediate     — 어느 정도 알지만 깊이 부족
  └── expert           — 잘 알고 있음, 고급 논의 원함
```

### 축 선정 이유

| 축 | 왜 필요한가 | 빠지면 뭐가 안 되나 |
|----|------------|-------------------|
| **Intent** | 같은 원칙이라도 정보 요청 vs 감정 토로는 대응이 완전히 다름 | 모든 situation이 "질문-답변" 패턴에만 치우침 |
| **Emotion** | 감정 상태가 원칙 적용 방식을 결정 | "분노한 사용자"와 "궁금한 사용자"에 같은 대응을 생성 |
| **Phase** | 첫 질문과 반복 질문은 의미가 다름 | 대화 흐름 무시, 모든 상황이 독립적으로 보임 |
| **Stakes** | 가벼운 호기심과 인생 결정은 다른 수준의 주의 필요 | 모든 situation이 같은 무게감 |
| **Prior Knowledge** | 전문가에게 기초 설명하면 짜증 | 대응 수준 단일화 |

## 조합 전략: Pairwise Coverage

### 왜 전수 조합을 하면 안 되나

```
6 × 5 × 5 × 3 × 3 = 1,350 조합 (per principle!)
원칙 100개 × 1,350 = 135,000 situations
→ 비용 폭발, 대부분 쓸모없는 조합 포함
```

### Pairwise로 줄이기

소프트웨어 테스팅의 경험칙:
**"결함의 90%+ 는 2개 파라미터의 상호작용에서 발생"**

Pairwise = 임의의 2개 축 조합이 모두 한 번 이상 나타나도록 최소 세트 생성.

```
5개 축의 pairwise coverage ≈ 30~35 조합 (per principle)
→ 1,350개 대신 ~30개로 모든 쌍 커버
→ 원칙 100개 × 30 = 3,000 situations (관리 가능)
```

### 생성 흐름

```
Phase 1: 유효 조합 필터링
  원칙마다 모든 pairwise 조합을 생성하되,
  의미 없는 조합은 미리 제거:
    × "casual" + "critical" (모순)
    × "novice" + "expert반론" (맥락 불일치)
    × "opening" + "repeating" (시간순 모순)

Phase 2: nano 생성 (per valid combination)
  input:
    원칙: "반복 질문은 불만족 신호"
    조합: intent=emotional_support, emotion=negative_high,
          phase=repeating, stakes=moderate, knowledge=intermediate

  nano(reasoning_effort=high):
    "이 원칙이 [이 조합의 구체적 상황]에서 어떻게 적용되는가?
     situation, reasoning, guidance를 생성하라."

  output:
    situation: "중급 투자자가 같은 종목에 대해 3번째 질문.
               이전 답변이 도움이 안 됐다는 좌절감 표출.
               '아 진짜 이거 왜 이래' 같은 감정적 표현."
    reasoning: "반복은 이전 답변의 실패 신호. 감정이 negative_high이므로
               기술적 답변 전에 감정 인정이 먼저. 중급자이므로 기초 설명은
               오히려 무시당한다는 느낌을 줄 수 있음."
    guidance: "감정 인정 → 이전 답변의 어떤 부분이 부족했는지 확인 →
              중급 수준에 맞는 새로운 각도의 설명"

Phase 3: Embed & Store
  embed(situation + reasoning) → Qdrant
  FalkorDB: (:Situation)-[:DERIVED_FROM]->(:Principle)
            (:Situation {axes: "emotional_support,negative_high,repeating,moderate,intermediate"})
```

## 카테고리 coverage 시각화 (dashboard)

```
Principle: "반복 질문은 불만족 신호"

Intent        ■■■■□□  4/6 covered
Emotion       ■■■□□   3/5 covered
Phase         ■■■■■   5/5 covered  ✓
Stakes        ■■■     3/3 covered  ✓
Knowledge     ■■□     2/3 covered

Total: 28/35 pairwise combos covered (80%)
Missing pairs: [information_seeking × novice], [creation × critical], ...
→ "Generate missing" 버튼 → nano가 빈 조합 채움
```

## Plutchik 조합 — 복합 감정 생성

기본 5개 감정 카테고리 대신, Plutchik의 8 기본 → 28 dyad를 활용하면
더 미세한 감정 상황을 생성할 수 있다:

```
기본 8: joy, trust, fear, surprise, sadness, disgust, anger, anticipation

Primary dyads (인접 조합):
  joy + trust       = love          → "신뢰하며 기대하는 상태"
  trust + fear      = submission    → "불안하지만 따르는 상태"
  fear + surprise   = awe           → "압도된 상태"
  sadness + disgust = remorse       → "후회하는 상태"
  anger + anticipation = aggressiveness → "공격적으로 요구하는 상태"
```

이건 Axis 2 (Emotional Valence)를 확장하는 방향 — 처음에는 5단계로 시작하고,
필요하면 Plutchik dyad로 세분화.

## TUNA 매핑 — Intent 축 상세

Google의 TUNA (2025)에서 검증된 6 modes를 우리 Axis 1에 매핑:

```
TUNA Mode               → Our Intent Value         → 예시
Information Seeking      → information_seeking      "이게 뭐야?"
Info Processing/Synth    → reflection               "이걸 종합하면 뭐지?"
Procedural Guidance      → procedural_guidance      "어떻게 해?"
Content Creation         → creation                 "이걸로 뭔가 만들어줘"
Social Interaction       → emotional_support        "힘들다..."
Meta-Conversation        → (별도 처리)              "너 이전에 뭐라 했지?"
```

Meta-Conversation (대화 자체에 대한 대화)은 별도 축으로 분리하거나,
conversation phase에서 "turning"으로 흡수 가능.

## Pairwise 생성 도구

Python `allpairspy` 라이브러리로 pairwise 조합 자동 생성:

```python
from allpairspy import AllPairs

parameters = [
    # Intent
    ["information_seeking", "decision_support", "emotional_support",
     "procedural_guidance", "reflection", "creation"],
    # Emotion
    ["positive_high", "positive_low", "neutral", "negative_low", "negative_high"],
    # Phase
    ["opening", "deepening", "turning", "repeating", "closing"],
    # Stakes
    ["casual", "moderate", "critical"],
    # Knowledge
    ["novice", "intermediate", "expert"],
]

for i, pair in enumerate(AllPairs(parameters)):
    intent, emotion, phase, stakes, knowledge = pair
    print(f"{i+1}: {intent} × {emotion} × {phase} × {stakes} × {knowledge}")

# 예상 출력: ~30-35 조합
```

## 도메인별 축 확장

기본 5축은 도메인 무관. 도메인별로 추가 축을 넣을 수 있다:

```
자기계발 도메인:
  + Axis 6: Life Area (career, relationship, health, finance, spirituality)

투자/경마 도메인:
  + Axis 6: Market Condition (bull, bear, volatile, stable)
  + Axis 7: Time Pressure (none, moderate, urgent)

커뮤니케이션 도메인:
  + Axis 6: Relationship Type (stranger, colleague, friend, authority)
```

## 실행 순서

```
Step 1: 자기계발 도메인 파일럿
  - 원칙 10개 선택 (책에서)
  - 기본 5축 pairwise → ~30 조합/원칙
  - nano 생성 → 300 situations
  - embed → Qdrant → 시뮬레이션

Step 2: Coverage 분석 (dashboard)
  - 어떤 조합이 hit 많은지
  - 어떤 조합이 miss인지
  - pairwise로 부족한 3-way 상호작용 발견

Step 3: 확장
  - miss 기반 추가 생성
  - 도메인별 축 추가
  - 다른 도메인으로 확대
```

## References

- [TUNA: Taxonomy of User Needs and Actions](https://arxiv.org/abs/2510.06124) — Google Research, 2025. 1193개 대화 분석 기반 유저 행동 분류
- [Plutchik's Wheel of Emotions](https://www.6seconds.org/2025/02/06/plutchik-wheel-emotions/) — 8 기본 감정 → 28 dyad 조합
- [Russell's Circumplex Model](https://psu.pb.unizin.org/psych425/chapter/circumplex-models/) — valence × arousal 2차원 감정 공간
- [Pairwise Testing](https://www.pairwise.org/) — 조합 폭발 해결, 90%+ 결함 검출
- [NIST Combinatorial Testing](https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=910001) — 이론적 근거
- [offline-prep-strategy.md](./offline-prep-strategy.md) — situation + reasoning 생성 파이프라인
- [reasoning-in-vectors.md](./reasoning-in-vectors.md) — embed(content + reasoning) 전략
- [conversation-axes.md](./conversation-axes.md) — 기존 대화 축 분류 (비교 참고)
