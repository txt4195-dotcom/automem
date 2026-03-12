# Offline Prep Strategy — Self-Generated Situation Variants

> 원칙 하나 → nano가 구체적 상황 여러 개 생성 → Qdrant에 embed → 검색 시 hit 확률 극대화

## Core Idea

Medprompt의 **Self-Generated Chain of Thought** (ablation에서 가장 큰 점프: +3.4%)를
agent wisdom 검색에 적용한다.

```
Medprompt:   training example → GPT-4가 CoT 생성 → 검증 → embed → DB
우리:        원칙/지혜 → nano가 situation variants 생성 → 검증 → embed → Qdrant
```

## Key Insight: 추론을 벡터 안에 넣는다

**복잡성을 검색 시점(online)이 아니라 저장 시점(offline)에 밀어넣는다.**

```
❌ 검색 시점에 역추론 (복잡, 비쌈):
  유저 발화 → "뭘 찾아야 하지?" 역추론 → query 생성 → 검색
  매 턴 LLM 호출 필요

✅ 저장 시점에 추론 포함 (단순, 쌈):
  원칙 → nano가 "이건 어떤 상황에서 필요한가?" 추론 → 추론 포함해서 embed
  검색 시 유저 발화 그대로 → cosine → hit
  매 턴 LLM 호출 불필요
```

원칙: "군중이 탐욕일 때 조심하라"

```
❌ 원칙만 embed:
  "군중이 탐욕일 때 조심하라" → [0.12, 0.45, ...]
  "비트코인 들어가도 돼?" → [0.78, 0.23, ...]
  cosine = 낮음 → miss

✅ 추론 포함해서 embed:
  "사용자가 FOMO를 느끼며 매수 타이밍을 물을 때,
   군중 심리가 탐욕 쪽이면 역행해야 한다.
   이 원칙은 감정적 투자 결정을 막기 위한 것이다.
   '들어가도 돼?', '나만 안 사고 있는 거 아냐', '다 오른다는데'
   같은 표현이 나오면 이 원칙이 필요하다." → [0.71, 0.28, ...]
  "비트코인 들어가도 돼?" → [0.78, 0.23, ...]
  cosine = 높음 → hit!
```

**추론 내용(왜 필요한지, 어떤 상황에서 나오는지, 어떤 표현이 트리거인지)이
벡터 공간에 이미 들어가 있으면 검색 시점에 역추론이 필요 없다.**

이는 Medprompt에서 CoT를 offline에서 생성해두고 inference time에는
단순 kNN만 사용한 것과 정확히 같은 원리다.

## Problem This Solves

단순 원칙 ("공감 먼저 하라")을 embed하면, 실제 대화 상황 ("사용자가 3번째 같은 에러 보고")과
cosine distance가 멀다. 표면 텍스트가 완전히 다르기 때문.

```
원칙 embedding:  "공감 먼저 하라"           ──→ [0.12, 0.45, ...]
실제 query:      "사용자가 같은 에러 3번째"  ──→ [0.78, 0.23, ...]
                                              cosine = 낮음 → miss
```

nano가 미리 구체적 상황 + 추론을 만들어두면:

```
situation+reasoning embedding:
  "사용자가 같은 문제를 반복해서 보고하는 상황.
   반복은 이전 답변이 도움 안 됐다는 불만족 신호.
   기술 문제가 아니라 소통 실패로 봐야 한다.
   '또 안 돼', '아까 말했잖아' 같은 표현이 트리거."  ──→ [0.71, 0.28, ...]
실제 query: "사용자가 같은 에러 3번째"                   ──→ [0.78, 0.23, ...]
                                                          cosine = 높음 → hit!
```

**추상적 원칙 1개 → 구체적 상황+추론 N개**로 벡터 공간에 퍼뜨리면 semantic gap이 줄어든다.

## Pipeline

### Phase 1: Situation Generation (offline, batch)

**모델 설정:**
- **nano (reasoning_effort=high)** — 추론 깊이가 벡터 커버리지를 결정
- **글자 제한: 넉넉히** — reasoning이 길수록 벡터에 더 많은 의미가 들어감
  - situation: ~200자
  - reasoning: ~500자+ (implicit belief, 숨은 감정, 트리거 표현 포함)
  - guidance: ~200자
  - enrichment 요약(500자 압축)과 다른 맥락 — 여기선 풍부함이 벡터 품질

```
Input:  원칙/지혜 하나
        "반복 질문은 불만족 신호다. 해결책보다 인정이 먼저."

nano prompt (reasoning_effort=high):
  "이 원칙이 적용되는 구체적 대화 상황을 5개 생성하라.
   각 상황은 agent가 실제로 마주칠 수 있는 장면이어야 한다.
   중요: reasoning에는 '왜 이 원칙이 이 상황에서 필요한지',
   '어떤 숨은 의도/감정이 깔려있는지', '어떤 implicit belief가 있는지',
   '어떤 표현이 트리거인지'를 깊이 있게 풀어서 작성하라.
   이 추론이 검색 벡터에 포함된다. 추론이 풍부할수록 더 많은
   표면 발화가 벡터 검색에 걸린다. 글자 수를 아끼지 마라.
   포맷: situation | reasoning | guidance"

Output:
  1. situation: "사용자가 같은 에러를 3번째 보고한다"
     reasoning: "반복 = 이전 답변이 도움 안 됐다는 신호. 기술 문제가 아니라 소통 실패."
     guidance: "이전 시도를 인정하고 ('여러 번 시도하셨네요'), 다른 각도로 접근"

  2. situation: "사용자가 같은 기능을 다른 말로 다시 요청한다"
     reasoning: "표현을 바꿔 재시도 = 내 답변을 이해 못 했거나, 원하는 게 아니었음"
     guidance: "먼저 확인 ('혹시 이런 걸 원하시는 건가요?'), 내 이해가 맞는지 검증"

  3. situation: "사용자가 '아까 말했잖아'라고 한다"
     reasoning: "명시적 불만 표현. positive face 위협 상태. 즉시 인정 필요."
     guidance: "사과 + 이전 내용 요약 + 구체적 행동 제시"

  4. situation: "사용자가 한 줄짜리 답만 보낸다 (이전에는 길게 썼는데)"
     reasoning: "참여도 하락 = disengagement. Grice Quantity 위반 — 의도적으로 짧게."
     guidance: "열린 질문으로 다시 끌어들이기, 강요하지 않기"

  5. situation: "사용자가 다른 도구/서비스를 언급하며 비교한다"
     reasoning: "대안 언급 = 현재 경험 불만족. 이탈 신호."
     guidance: "방어하지 않기. 차이점을 인정하고 우리 강점을 자연스럽게"
```

### Phase 2: Verification — 역검증 루프 (Medprompt-style)

Medprompt에서 CoT가 정답과 일치하는지 검증했듯이,
생성된 situation이 원래 원칙과 정합성이 있는지 + 실제로 벡터 검색에 걸리는지 검증.

```
검증 1: 논리 정합성 (nano, reasoning_effort=medium)
  - 이 situation에서 이 guidance를 따르면 원칙에 부합하는가?
  - situation이 현실적인가? (hallucinated scenario 아닌가?)
  - reasoning이 situation → guidance를 논리적으로 연결하는가?
  - 불합격 → 재생성

검증 2: 역검증 — cosine 사전 체크
  miss 원인 추론 → 가상 situation 생성 → embed → 원래 쿼리와 cosine 계산
  - cosine >= threshold → ✅ 원인 맞음, 이 situation이 구멍을 메운다 → 저장
  - cosine < threshold → ❌ 원인 틀림 → 다른 원인 탐색 → 재추론

검증 3: 역추론 — 결과에서 원인으로
  생성된 situation에서 역으로 "이 상황의 원인/원칙은 뭐지?" 추론시키기
  → 원래 원칙과 일치하면 ✅
  → 다른 원칙이 나오면 reasoning 품질 문제 → 재생성
```

**생성 전에 검증이 가능하다.** 가상으로 만들어보고 cosine 찍어보면 되니까.
실제로 저장하기 전에 "이게 진짜 구멍을 메우는지" 확인할 수 있다.

### Phase 3: Embed & Store

```
각 situation variant:
  ├─ Qdrant: situation + reasoning 합쳐서 embed (검색용)
  │     embed 대상 텍스트 =
  │       situation + reasoning + 트리거 표현 목록
  │       (guidance는 embed에서 제외 — 검색이 아니라 주입용이므로)
  │
  ├─ Qdrant payload: {
  │     situation: "...",
  │     reasoning: "...",
  │     guidance: "...",
  │     trigger_expressions: ["또 안 돼", "아까 말했잖아"],
  │     source_principle_id: "uuid-of-original",
  │     axes: ["emotion:frustrated", "repetition:recurring"],
  │     variant_index: 1,
  │     verified: true
  │  }
  └─ FalkorDB: (:Situation)-[:DERIVED_FROM]->(:Wisdom)
               (:Situation)-[:APPLIES_ON {axis: "emotion", value: "frustrated"}]->(:Axis)
```

**핵심: situation + reasoning(추론)을 합쳐서 embed.**

추론 내용이 벡터에 포함되면:
- "왜 이게 필요한지" (숨은 의도)가 벡터 공간에 반영됨
- "어떤 표현이 트리거인지"가 벡터 공간에 반영됨
- 유저의 표면 발화와 cosine이 가까워짐
- **검색 시점에 역추론(축 분류, query 변환)이 불필요해질 수 있음**

guidance는 embed에서 제외 — "이렇게 대응하라"는 검색 매칭에 도움이 안 되고,
결과 주입 시에만 필요하므로 payload에 저장.

### Phase 4: Retrieval (online, per turn)

추론이 벡터 안에 있으므로 online은 극도로 단순:

```
V8 대화 턴
  ↓
  유저 발화 (최근 N턴) 그대로 embed
  (축 분류 불필요 — 추론이 이미 벡터 안에 있으니까)
  ↓
  Qdrant kNN (situation+reasoning embedding space)
  ↓
  hit: situation variant #1 (cosine 0.87)
       situation variant #3 (cosine 0.82)
  → 추론이 벡터에 포함되어 있어서 숨은 의도까지 매칭됨
  ↓
  payload에서 guidance 추출
  ↓
  context로 prompt에 주입 (example이 아님 — reasoning model 방해 방지):
  "관련 데이터:
   [상황 패턴] 반복 보고는 불만족 신호. 소통 실패로 봐야 함.
   [대응 원칙] 이전 시도 인정 → 다른 각도로 접근
   [출처] principle:반복질문_불만족신호 (evidence: 5, sources: 3)"
```

**축 분류기(Axis Classifier)는 옵션이 된다:**
- 추론이 벡터에 잘 들어가 있으면 → cosine만으로 충분 → 축 분류 불필요
- cosine으로 부족한 경우에만 → 축 분류 추가해서 검색 보강
- 실험해봐야 어느 쪽인지 알 수 있음

## Cost Structure

### nano reasoning_effort 설정표

| 작업 | reasoning_effort | 이유 |
|------|-----------------|------|
| **Situation+reasoning 생성** | **high** | 추론 깊이 = 벡터 커버리지. 핵심 투자. |
| 논리 정합성 검증 | medium | 맞는지만 판단 |
| 역추론 검증 | medium | 원칙 역추적만 |
| cosine 사전 체크 | (LLM 불필요) | embed + 계산만 |
| enrichment 요약 | low | 압축이 목적 (기존) |
| node scoring | medium | 판단 (기존) |
| 축 분류 (optional) | low | 라우팅용 (기존) |

### 비용 테이블

| 단계 | 시점 | 비용 | 빈도 |
|------|------|------|------|
| Situation + reasoning generation | offline | nano(high) × N variants per principle | 원칙 추가 시 1회 |
| Verification (논리 + 역검증) | offline | nano(medium) × N + embed × N | 생성 직후 1회 |
| Embed (situation+reasoning) | offline | embedding API × N | 생성 직후 1회 |
| kNN search | online | Qdrant API (무료) | 매 턴 |
| Axis classification (optional) | online | nano(low) | 필요시만 |
| **Total per turn** | **online** | **embed 1회 + Qdrant 1회** | **LLM 호출 0~1회** |

**비싼 건 offline에서 다 하고, online은 거의 공짜.**
추론을 벡터에 넣었기 때문에 online에서 LLM 호출이 최소화된다.

## Wisdom → Situation 비율

- 원칙 1개 → situation 3~7개 (도메인에 따라)
- 원칙 100개 → situation 300~700개
- Qdrant에 700개 벡터 = 아무 문제 없음

더 많은 situation = 더 넓은 coverage = 더 높은 hit 확률.
단, 중복/유사 situation이 많으면 검색 시 같은 원칙만 k개 다 차지 → diversity 필요.

## Diversity Control

검색 결과에서 같은 원칙의 variant가 여러 개 hit하면 1개만 남기고 다른 원칙으로 교체.

```python
def select_diverse_fewshot(hits, k=5):
    selected = []
    seen_principles = set()
    for hit in sorted(hits, key=lambda h: h.score, reverse=True):
        principle_id = hit.payload["source_principle_id"]
        if principle_id in seen_principles:
            continue
        seen_principles.add(principle_id)
        selected.append(hit)
        if len(selected) >= k:
            break
    return selected
```

## Two-Store Architecture: Qdrant가 찾고, FalkorDB가 연결한다

### 왜 이 구조가 필요한가

핵심 문제: 유저가 실제로 닥친 문제 상황이 DB에 없으면 검색이 안 된다.
모델이 바로 원인을 추론할 수 있으면 recall이 필요 없는데, 대부분 못 한다.
그래서 recall이 필요한 건데, 검색할 데이터가 없다 — 이게 현실.

**해결: 원칙을 먼저 넣고, 거기서 파생된 상황들을 생성해서 벡터 DB에 깔아놓는다.**

```
Principle (원본, FalkorDB)
  │
  ├─[:DERIVED_FROM]── Situation A (생성, Qdrant + FalkorDB)
  ├─[:DERIVED_FROM]── Situation B (생성, Qdrant + FalkorDB)
  └─[:DERIVED_FROM]── Situation C (생성, Qdrant + FalkorDB)
```

- **Qdrant**: 생성된 situation+reasoning을 embed → 벡터 검색으로 잡힘
- **FalkorDB**: situation → principle DERIVED_FROM edge → 원칙으로 데려감

```
유저 상황 → embed → Qdrant cosine hit: Situation A
  → FalkorDB: Situation A -[:DERIVED_FROM]-> Principle "공감 먼저"
  → guidance + principle을 context로 주입
```

### 생성 데이터의 장점

자연 발생 데이터와 달리, 생성된 situation은:
- **원칙과의 관계가 100% 확실** (DERIVED_FROM edge)
- **coverage를 의도적으로 설계** 가능 (빠진 상황을 채울 수 있음)
- **검증이 가능** (원칙에 부합하는지 체크)

### 기존 AutoMem 구조와의 호환

AutoMem에 이미 DERIVED_FROM edge type이 있다:
```python
DERIVED_FROM    # Derived knowledge (transformation, confidence)
```

기존: `(:Memory)-[:DERIVED_FROM]->(:Memory)` — 자연 발생 지식 간 파생
추가: `(:Situation)-[:DERIVED_FROM]->(:Principle)` — 생성 상황 → 원칙

기존 knowledge-graph-design.md의 구조와 비교:
```
기존:    Memory(자연 발생) -[:MENTIONS]-> Concept(추출)      방향: 데이터 → 개념
여기:    Situation(생성)   -[:DERIVED_FROM]-> Principle(원본)  방향: 개념 → 데이터
```
방향이 반대지만 그래프 구조는 동일. 양쪽이 공존하면서 서로 보강.

### 원칙 자료는 풍부하다

원칙/지혜의 소스는 의외로 많다:

| 소스 | 예시 | 양 |
|------|------|---|
| 책 | 커뮤니케이션, 심리학, 투자, 개발 방법론 | 책 1권당 원칙 20~50개 |
| SOUL-COMMON.md | 이미 정리된 에이전트 철학 | ~20개 |
| IDENTITY-COMMON.md | 행동 원칙 | ~10개 |
| 커뮤니티 delta에서 추출된 Pattern | Crystallize 결과 | 자동 증가 |
| 디버깅 경험 (AutoMem memories) | kind:insight, kind:pattern | ~144개 존재 |
| 사용자 교정 (kind:correction) | 창한의 피드백 | 축적 중 |

**원칙은 이미 많다. 부족한 건 상황 데이터.**
원칙 하나당 situation 5개만 생성해도:

```
원칙 100개 × situation 5개 = 500개 벡터
  → Qdrant에서 어떤 실제 상황이 와도 가까운 situation이 하나는 걸림
  → edge를 따라 원칙으로 감
  → context로 주입
```

## Incremental Growth

```
Phase 1: 수동 시드
  SOUL-COMMON.md에서 원칙 10개 추출
  → 원칙당 situation 5개 생성 (nano)
  → 50개 벡터 + 10개 원칙 노드 + 50개 edge
  → 테스트: 실제 대화 상황으로 recall → hit/miss 측정

Phase 2: 확장
  책/자료에서 원칙 추가 (도메인별)
  → hit/miss 분석 → miss된 상황 유형 파악
  → 해당 유형에 대한 원칙 + situation 추가

Phase 3: 자동화
  커뮤니티 delta에서 Pattern 승격 시 (Crystallize)
  → 자동으로 situation 생성 + DERIVED_FROM edge
  → 벡터 coverage가 자동으로 확대

Phase 4: 피드백 루프
  실제 사용에서 hit한 situation에 evidence++
  miss된 상황 → 새로운 situation 생성 트리거
  → 점점 더 정확해짐
```

## Connection to Other Docs

| 문서 | 연결 |
|------|------|
| [conversation-axes.md](./conversation-axes.md) | 축 분류 체계 → Axis Classifier의 출력 형태 |
| [knowledge-graph-design.md](./knowledge-graph-design.md) | Concept/Principle node → Wisdom node의 원본 |
| [medprompt.md](./medprompt.md) | Self-Generated CoT → Situation generation의 이론적 근거 |

## Development Workflow — Simulation-First (그물전법)

### 핵심: 런타임 전에 DB를 완성시킨다

자연 발생 데이터를 기다리는 게 아니라, 오프라인에서 벡터 공간을 의도적으로 채운다.

```
Phase A: Seed (씨뿌리기)
  원칙 수집 (책, SOUL-COMMON.md, 경험 등)
  → 원칙별 situation+reasoning 생성 (nano)
  → Qdrant에 embed + FalkorDB에 DERIVED_FROM edge

Phase B: Simulate (시뮬레이션)
  실제 대화처럼 쿼리 던져보기
  → "비트코인 들어가도 돼?" → cosine hit? miss?
  → "뭘 해도 의욕이 없어" → cosine hit? miss?
  → hit rate 측정

Phase C: Fill Gaps (빈틈 채우기)
  miss된 쿼리 분석
  → 어떤 원칙이 필요한데 없는지
  → 있는 원칙인데 situation coverage가 부족한지
  → 해당 situation 추가 생성 → 재시뮬레이션

Phase D: Iterate
  Phase B-C 반복 → hit rate 수렴
  → DB 완성 → 런타임에 붙이기
```

### 그물전법 (Net Strategy)

```
원칙 100개 × situation 5개 = 500개 벡터
  → 각 벡터에 reasoning(숨은 의도, 트리거 표현)이 포함
  → 현실의 표면 발화가 이 500개 중 하나와 cosine이 높을 확률 ↑
  → 원칙이 많을수록 그물이 촘촘해짐
  → miss → 새 원칙 or 새 situation 추가 → 그물 보강
```

**coverage를 의도적으로 설계할 수 있다.** 이게 자연 발생 데이터만 있는 시스템과의 결정적 차이.

### 검색 흐름 (최종)

```
대화 턴
  ↓
  유저 발화 embed (LLM 호출 없음)
  ↓
  Qdrant cosine → situation+reasoning hit (벡터 검색만)
  ↓
  FalkorDB: DERIVED_FROM → 원칙 역추적 + evidence/breadth scoring
  ↓
  diversity filter (같은 원칙 중복 제거)
  ↓
  context로 주입 (example 아님 — reasoning model 방해 방지)
```

Qdrant = recall (찾기), FalkorDB = precision/ranking (고르기).

### AutoMem과의 관계

```
AutoMem (기존):
  자연 발생 데이터 안에서 enrichment, 연결, 패턴 탐지
  → 있는 노드 안에서만 작동

Offline Prep (이것):
  원칙에서 상황을 생성 → 벡터 공간 확장
  → 없는 데이터를 만들어냄

Crystallize (순환):
  AutoMem에서 Pattern 승격 → 새 원칙
  → Offline Prep이 situation 생성
  → 다시 AutoMem이 enrichment
  → self-reinforcing loop
```

## Three-Phase Cost Architecture — 사전/실시간/사후

### 원칙: 미리 알 수 있는 건 전부 사전에, 나머지는 사후에. 실시간은 검색만.

```
┌─────────────────────────────────────────────────────────┐
│ 사전 (Offline Prep) — 비싸지만 1회성                     │
│                                                          │
│ 미리 알 수 있는 것:                                      │
│   원칙/지혜 (책, 연구, SOUL)     → situation 생성         │
│   유저 프로필 축 (48개)           → 축별 situation 생성    │
│   감정/의도/패턴 유형 (유한)      → 유형별 situation 생성  │
│   트리거 표현 ("들어가도 돼?")    → reasoning에 포함       │
│   도메인 목록 (경마, 금융 등)     → collection routing     │
│                                                          │
│ 전부 embed → Qdrant에 깔아놓기                           │
│ FalkorDB에 DERIVED_FROM / MAPS_TO_AXIS edge              │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 실시간 (Runtime) — 거의 공짜                             │
│                                                          │
│ embed(유저 발화) + Qdrant cosine + FalkorDB scoring      │
│ LLM 호출: 0회                                            │
│ 지연: embed API 1회 + vector search                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ 사후 (Background Post-processing) — 비동기, 급하지 않음  │
│                                                          │
│ 대화 세션 종료 후 백그라운드 worker:                      │
│   miss 감지 → cosine top-1 < threshold였던 턴 수집       │
│   miss 원인 분류 → 원칙 부재? situation 부족?            │
│   새 situation 생성 트리거 (miss → fill gaps)            │
│   evidence++ (hit한 situation/principle)                  │
│   유저 프로필 Beta [a,b] 업데이트 (feedback)             │
│   새 패턴 발견 → Crystallize 입력                        │
│                                                          │
│ 기존 AutoMem enrichment worker와 같은 구조               │
│ (store → background job → PRECEDED_BY, SIMILAR_TO 등)    │
└─────────────────────────────────────────────────────────┘
```

### 미리 알 수 있는 것 vs 없는 것

| 미리 알 수 있다 (사전 생성) | 미리 알 수 없다 (런타임/사후) |
|---------------------------|----------------------------|
| 원칙, 지혜, 패턴 (유한, 안정적) | 구체적 맥락 (비트코인 vs 이더리움) |
| 유저 프로필 축 48개 (정의됨) | 이 유저의 현재 감정 상태 |
| 감정/의도/패턴 유형 (연구에 있음) | 실시간 시장/커뮤니티 데이터 |
| 트리거 표현 목록 (열거 가능) | 이 대화만의 고유 맥락 |
| 도메인 목록 (알고 있음) | 새로운 도메인의 출현 |

**미리 알 수 있는 것이 의외로 많다.** 이것들을 전부 situation+reasoning으로 변환해서
벡터 공간에 깔면, 런타임은 "이 중에서 뭐가 맞지?"만 찾으면 된다.

## Reasoning-in-Vectors — 원칙의 일반화

Offline Prep에서 situation+reasoning을 embed하는 것과 같은 원칙이
**AutoMem 전체 embedding에 적용**되어야 한다.

→ 상세 설계: [reasoning-in-vectors.md](./reasoning-in-vectors.md)

```
Offline Prep (Level 2):
  principle → nano가 situation+reasoning 생성 → embed(situation+reasoning)
  "없는 데이터를 만들어서" 벡터 공간을 채운다

AutoMem Enrichment (Level 1):
  content → nano가 reasoning 생성 → embed(content+reasoning)
  "있는 데이터를 풍부하게 해서" 벡터 품질을 올린다

같은 메커니즘, 같은 Qdrant, 같은 원칙:
  추론 과정이 벡터에 들어가면 검색 품질이 올라간다.
```

### 벡터 품질 계층 (전체 그림)

| Level | 무엇을 embed | 검색 가능 범위 | 구현 위치 |
|-------|-------------|--------------|----------|
| **0** | content만 | 표면 텍스트 매칭 | 현재 AutoMem |
| **1** | content + reasoning | 의도/맥락 매칭 | [reasoning-in-vectors.md](./reasoning-in-vectors.md) |
| **2** | situation + reasoning (생성) | 가상 상황까지 매칭 | 이 문서 (Offline Prep) |

Level 0→1: enrichment pipeline 변경 (reasoning generation + re-embed)
Level 1→2: Offline Prep batch process (situation generation)
둘 다 같은 Qdrant collection에 공존.

## Open Questions

1. **생성 모델**: nano로 충분한가? situation 품질이 떨어지면 더 큰 모델 필요?
2. **검증 자동화 수준**: 초기에는 사람 검증? 양이 많아지면 자동?
3. **situation 업데이트**: 원칙이 바뀌면 하위 situation도 재생성?
4. **다국어**: 한국어 situation + 영어 situation 둘 다 필요?
5. **negative examples**: "이렇게 하지 마라"도 situation으로 만들면 few-shot에 도움?
