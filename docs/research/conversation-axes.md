# Conversation Axes — Multi-Dimensional Retrieval for Agent Few-Shot

> 대화의 숨은 축을 분류하고, 축별로 적절한 지혜를 검색하는 구조 설계를 위한 리서치

## Problem Statement

### Medprompt과 우리 상황의 차이

Medprompt에서 kNN이 잘 된 이유:
- **질문과 답이 같은 도메인** (MedQA → MedQA training set)
- **표면 텍스트가 유사** ("두통+발열" → "두통+발열 감별진단")
- **단일 축** (의학 지식)

우리 상황:
- **표면과 숨은 뜻이 다름** ("이거 또 안 돼" → 기술 문제 + 좌절 + 신뢰 하락)
- **다중 축이 동시 존재** (감정, 관계, 의도, 맥락, 표면 도메인)
- **검색해야 할 지혜와 query의 도메인이 다름** ("반복 질문" → "불만족 신호 대응법")

### Semantic Gap

```
Medprompt:  query ──cosine──→ answer     (같은 공간)
우리:       상황  ──???───→ 지혜         (다른 공간)
```

단순 embedding cosine은 **같은 도메인 내 유사성**에 강하지만,
**상황 → 적절한 대응 지혜**처럼 cross-domain bridging에는 약하다.

---

## Research: 대화의 축(Dimension)은 무엇인가?

### 1. ISO 24617-2 / DIT++ — 대화 행위 차원 (Dialogue Act Dimensions)

국제 표준(ISO 24617-2)은 DIT++ 분류 체계를 기반으로 대화를 **9~10개 독립 차원**으로 분석한다.
하나의 발화가 **동시에 여러 차원에서** 기능을 가질 수 있다 (multidimensional annotation).

| Dimension | 설명 | 예시 |
|-----------|------|------|
| **Task** | 과제 수행 관련 행위 | 정보 요청, 제안, 지시 |
| **Auto-Feedback** | 자기 이해 상태 표현 | "아 그렇구나", "잠깐 뭐라고?" |
| **Allo-Feedback** | 상대 이해 상태 확인 | "이해했어?", "따라오고 있지?" |
| **Turn Management** | 발화권 관리 | 끼어들기, 양보, 유지 |
| **Time Management** | 시간 관리 | "잠깐만", 멈춤, 서두름 |
| **Discourse Structuring** | 담화 구조화 | 주제 전환, 요약, 되돌아가기 |
| **Own Communication Management** | 자기 발화 수정 | 말 고침, 재구성 |
| **Partner Communication Management** | 상대 발화 수정 | 교정, 명확화 요청 |
| **Social Obligations Management** | 사회적 의무 | 인사, 사과, 감사, 작별 |
| **Contact Management** | 채널/연결 관리 | "들려?", 연결 확인 |

**핵심 발견**: 하나의 "이거 또 안 돼"가 Task(기술 문제 보고) + Auto-Feedback(좌절 표현) + Social(기대 위반 항의)를 동시에 수행한다. ISO 표준이 이미 이 다차원성을 공식화했다.

**출처**: [ISO 24617-2:2020](https://www.iso.org/standard/76443.html), [DIT++ Annotation Guide](https://aclanthology.org/2020.lrec-1.69.pdf)

### 2. Grice의 협력 원칙 + 함축 (Conversational Implicature)

Paul Grice의 4가지 대화 격률:

| Maxim | 원칙 | 위반 시 함축 |
|-------|------|-------------|
| **Quantity** | 필요한 만큼만 | 너무 적게 = 숨기는 게 있음, 너무 많이 = 강조/변명 |
| **Quality** | 진실만 | 의도적 위반 = 아이러니, 과장 |
| **Relation** | 관련된 것만 | 갑자기 주제 전환 = 회피, 불편 |
| **Manner** | 명확하게 | 모호하게 = 의도적 거리두기 |

**핵심 발견**: 사람이 격률을 **의도적으로 위반**(flouting)할 때 숨은 의미가 생긴다.
"괜찮아"가 Manner 위반(짧고 건조)이면 "괜찮지 않다"는 함축.
AI agent가 이 함축을 놓치면 표면만 읽게 된다.

**출처**: [Cooperative Principle (Wikipedia)](https://en.wikipedia.org/wiki/Cooperative_principle), [Stanford Encyclopedia — Implicature](https://plato.stanford.edu/entries/implicature/)

### 3. Brown & Levinson — 체면(Face) 이론

대화는 항상 **체면 관리**가 깔려있다:

| Face Type | 욕구 | 위협 예시 |
|-----------|------|----------|
| **Positive Face** | 인정받고 싶음 | 비판, 무시, 불동의 |
| **Negative Face** | 방해받지 않고 싶음 | 명령, 요청, 제안 |

**핵심 발견**: "이거 또 안 돼"는 positive face 위협 (= "너 또 실패했잖아").
Agent가 바로 해결책을 제시하면 negative face도 위협 (= "이렇게 해").
→ 먼저 positive face 회복 (공감) → 그 다음 해결책.

이 순서를 아는 게 "지혜"이고, 이걸 상황에 맞게 찾아주는 게 dynamic few-shot의 목표.

**출처**: [Brown & Levinson Politeness Theory](https://www.academypublication.com/issues2/tpls/vol06/01/07.pdf)

### 4. 감정 분류 (Emotion Taxonomy)

NLP 연구에서 대화 감정 인식(Dialogic Emotion Analysis)은 활발한 분야:

| 체계 | 감정 수 | 분류 |
|------|---------|------|
| **Ekman 6** | 6 | anger, disgust, fear, joy, sadness, surprise |
| **Plutchik 8** | 8 | + trust, anticipation |
| **GoEmotions** | 27 | fine-grained (Google, Reddit 기반) |
| **Valence-Arousal** | 연속 | 긍정/부정 × 흥분/차분 (2D 공간) |

**핵심 발견**: 맥락(context)을 포함하면 감정 인식 정확도가 44% → 57%로 점프.
dialogue act 정보까지 넣으면 62%. **축을 조합하면 각각의 정확도가 올라간다.**

**출처**: [Emotion-LLaMA (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/c7f43ada17acc234f568dc66da527418-Paper-Conference.pdf), [Dialogic Emotion Analysis Survey](https://www.sciencedirect.com/science/article/pii/S0031320324005454)

### 5. 의도 분류 (Intent Classification)

대화에서 화자의 의도를 분류하는 연구:

| 수준 | 예시 |
|------|------|
| **표면 의도** | 질문, 요청, 보고, 확인 |
| **대화 전략** | 설득, 협상, 회피, 위로 |
| **깊은 욕구** | 인정, 통제, 안전, 소속 |

**핵심 발견**: LLM의 zero-shot 의도 분류 성능이 이미 상당히 높다.
nano 수준으로도 "이 발화의 의도는 X" 분류가 가능하다는 뜻.

**출처**: [Intent Classification for Dialogue Utterances](https://sentic.net/intent-classification-for-dialogue-utterances.pdf)

---

## Research: Semantic Gap을 어떻게 메우는가?

### 1. Query Rewriting / Decomposition (RAG 분야)

RAG에서 query와 knowledge 사이의 gap을 메우는 기법들:

| 기법 | 방식 | 적용 |
|------|------|------|
| **Multi-Query Rewriting** | 하나의 query를 여러 관점으로 재작성 | DMQR-RAG (2025) |
| **Step-Back Prompting** | 구체적 query → 추상적 query로 변환 | "이거 안 돼" → "반복 실패 상황 대응" |
| **Typed-RAG** | query를 type별로 분해 (fact/opinion/cause) | Multi-aspect decomposition |
| **MaFeRw** | 검색 결과 피드백으로 query 재작성 | Multi-aspect feedback rewriting |
| **MA-RAG** | Multi-agent가 query 분해 + 각각 검색 | Collaborative chain-of-thought |

**핵심 발견**: 우리가 하려는 "축별 분류 → 축별 검색"은 Typed-RAG / MA-RAG와 구조적으로 동일하다.
단, 우리의 "type"은 대화 축(감정/관계/의도/...)이고, 그들의 "type"은 정보 유형(fact/opinion/cause).

**출처**: [DMQR-RAG (OpenReview 2025)](https://openreview.net/forum?id=lz936bYmb3), [Typed-RAG (2025)](https://arxiv.org/html/2503.15879v2), [MA-RAG (2025)](https://arxiv.org/pdf/2505.20096)

### 2. Application-Aware RAG (RAG+)

RAG+는 **지식 + 적용 사례**를 dual corpus로 구성:
- Knowledge corpus: 원리, 법칙, 지혜
- Application corpus: 그 지혜가 적용된 구체적 사례

Query가 들어오면 지식과 사례를 **동시에** 검색해서 LLM에 준다.
→ "원칙"만 주는 것보다 "원칙 + 그걸 적용한 사례"가 더 효과적.

**핵심 발견**: 우리의 wisdom store에도 이 dual 구조가 필요할 수 있다.
"공감 먼저" (원칙) + "사용자가 3번 실패해서 화났을 때 공감으로 시작해서 성공한 대화" (사례).

**출처**: [RAG+ (2025)](https://arxiv.org/html/2506.11555v4)

---

## 제안: 대화 축 분류 체계 (Conversation Axis Taxonomy)

위 리서치를 종합해서, **agent가 대화에서 지혜를 검색할 때** 사용할 축 체계:

### Tier 1: 핵심 축 (항상 분류)

| 축 | ID | 분류 기준 | 예시 값 |
|---|---|---|---|
| **감정** | `emotion` | 상대의 감정 상태 | frustrated, confused, excited, neutral |
| **의도** | `intent` | 상대가 진짜 원하는 것 | solve, vent, validate, learn, delegate |
| **관계** | `relation` | 나와 상대의 현재 역학 | trust_building, trust_declining, established, new |
| **체면** | `face` | 어떤 face가 위협받는가 | positive_threatened, negative_threatened, none |
| **도메인** | `domain` | 표면 주제 | tech, personal, business, creative |

### Tier 2: 맥락 축 (있으면 분류)

| 축 | ID | 분류 기준 | 예시 값 |
|---|---|---|---|
| **반복성** | `repetition` | 같은 주제/문제의 반복 횟수 | first, recurring, escalating |
| **긴급도** | `urgency` | 시간 압박 | urgent, normal, reflective |
| **Grice 위반** | `implicature` | 격률 위반으로 인한 숨은 뜻 | irony, understatement, evasion, none |
| **대화 단계** | `phase` | 대화 전체에서의 위치 | opening, exploring, deciding, closing |
| **에너지** | `energy` | 상대의 참여도/에너지 | high, medium, low, disengaged |

### 사용법

```
1. Subagent (Axis Classifier)
   Input: 대화 최근 N턴
   Output: { emotion: "frustrated", intent: "validate", relation: "trust_declining",
             face: "positive_threatened", domain: "tech" }
   Model: nano (reasoning_effort=low) — 분류만 하면 되니까

2. Axis-aware Retrieval
   각 축의 값을 query로 변환:
   - emotion:frustrated → "좌절한 상대를 대할 때"
   - intent:validate   → "인정/확인을 원하는 상대에게"
   - face:positive_threatened → "체면이 손상된 상대에게"

   각각 Qdrant kNN 검색 → 축별 wisdom 후보

3. Merge + Select
   축별 후보를 합치고, 가장 많이 겹치는 wisdom 우선
   + FalkorDB evidence/promotion status 확인
   → 최종 k개 few-shot
```

---

## 우리가 부딪힌 문제들 (Problems Encountered)

### P1: 단순 cosine으로는 cross-domain 검색이 안 된다

Medprompt 환경(같은 도메인, 같은 형태)에서는 cosine이 잘 작동하지만,
"사용자가 3번째 같은 질문 반복" → "반복 질문은 불만족 신호" 는 표면 텍스트가 달라서 hit하지 않는다.

**해결 방향**: 축 분류 → 축별 query 생성 → 축 공간에서 검색.
"반복 질문"이 아니라 "frustrated + recurring + validate"로 검색.

### P2: 숨은 도메인이 여러 개다

하나의 발화에 감정/관계/의도/체면이 동시에 존재한다.
하나의 query로는 하나의 축만 잡힌다.

**해결 방향**: 미리 축을 정의하고, subagent가 분류만 한다 (생성이 아님).
축이 10개여도 분류는 싸다 (nano, classification task).

### P3: 설계가 복잡해진다

축 추출 → 축별 검색 → merge → few-shot 선별...
파이프라인이 길어지면 구현도 디버깅도 어렵다.

**해결 방향**: 최소 버전부터 시작.
- v0: wisdom을 넣고 상황 그대로 recall (지금 가능)
- v1: 축 분류 추가 (subagent 1개)
- v2: 축별 검색 분리
- v3: evidence 기반 promotion + 선별

### P4: Wisdom 저장 시 "어떤 축에서 유효한가"가 태그되어야 한다

"공감 먼저 하라"는 emotion:frustrated 축에서 유효하고,
"해결책을 빨리 줘라"는 intent:solve + urgency:urgent 축에서 유효하다.

저장할 때 이 메타데이터가 없으면 검색할 때 noise가 생긴다.

**해결 방향 A**: 저장 시 subagent가 자동 태깅 (축 + 값).
**해결 방향 B**: wisdom 자체에 "situation: ..." prefix를 넣어서 embedding에 반영.
**해결 방향 C**: 둘 다 (태그로 필터 + embedding으로 유사도).

### P5: 테스트가 가능해야 한다

Medprompt처럼 **넣고 → 검색 → 맞는지 확인** 루프가 돌아야 한다.
구조 없이 데이터만 넣으면 "뭐가 잘 되고 뭐가 안 되는지" 측정 불가.

**해결 방향**: 최소한의 curating 구조 (축 분류 + 태깅)를 먼저 만들고,
소량 데이터 (10~20개)로 hit/miss 테스트.

---

## Open Questions

1. **축 개수**: Tier 1 (5개) + Tier 2 (5개) = 10개. 너무 많은가? 처음에 Tier 1만?
2. **Wisdom 포맷**: "이럴 때 이렇게" pair? 문장 하나? 문단? situation + guidance 구조?
3. **축별 검색 vs 합쳐서 검색**: 축마다 따로 Qdrant 쳐야 하나, 축 값을 조합해서 1회 검색?
4. **분류 모델**: nano로 충분한가? 축이 많아지면 정확도 떨어지나?
5. **RAG+ dual corpus**: 원칙 + 적용 사례를 분리 저장? 하나로?
6. **측정 지표**: hit/miss 외에 few-shot이 응답 품질에 실제로 기여했는지 어떻게 측정?

---

## Connection to Existing Design

| 기존 설계 | 이 리서치에서의 위치 |
|-----------|-------------------|
| knowledge-graph-design.md — Concept node | Concept이 축별로 분류되어야 의미 있는 검색 가능 |
| knowledge-graph-design.md — Crystallize | 축별 evidence 축적 → 축별 promotion |
| curation-strategy.md — Curator interface | Curator = Axis Classifier + Axis-aware Retriever |
| curation-strategy.md — 5 retrieval modes | 각 mode가 특정 축 조합에 매핑될 수 있음 |
| AutoMem recall — auto_decompose | 현재는 텍스트 키워드 분해. 축 기반 분해로 확장 가능 |
| AutoMem enrichment — entity extraction | entity 외에 축 태깅도 enrichment에 추가 가능 |

---

## References

- [ISO 24617-2:2020 — Dialogue Acts](https://www.iso.org/standard/76443.html)
- [DIT++ Annotation Standard (LREC 2020)](https://aclanthology.org/2020.lrec-1.69.pdf)
- [ISO 24617-2 Semantically-based Standard](https://people.ict.usc.edu/~traum/Papers/Buntetal-ISO24617-2.pdf)
- [Grice — Cooperative Principle](https://en.wikipedia.org/wiki/Cooperative_principle)
- [Stanford Encyclopedia — Implicature](https://plato.stanford.edu/entries/implicature/)
- [Brown & Levinson — Politeness Theory](https://www.academypublication.com/issues2/tpls/vol06/01/07.pdf)
- [Emotion-LLaMA (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/c7f43ada17acc234f568dc66da527418-Paper-Conference.pdf)
- [Dialogic Emotion Analysis Survey](https://www.sciencedirect.com/science/article/pii/S0031320324005454)
- [Intent Classification for Dialogue](https://sentic.net/intent-classification-for-dialogue-utterances.pdf)
- [DMQR-RAG — Diverse Multi-Query Rewriting (2025)](https://openreview.net/forum?id=lz936bYmb3)
- [Typed-RAG — Multi-Aspect Decomposition (2025)](https://arxiv.org/html/2503.15879v2)
- [MA-RAG — Multi-Agent RAG (2025)](https://arxiv.org/pdf/2505.20096)
- [RAG+ — Application-Aware Reasoning (2025)](https://arxiv.org/html/2506.11555v4)
- [MaFeRw — Multi-Aspect Feedback Rewriting](https://arxiv.org/html/2408.17072v1)
- [Medprompt](./medprompt.md) — Dynamic few-shot selection 원본
- [Knowledge Graph Design](./knowledge-graph-design.md) — 3-layer architecture
