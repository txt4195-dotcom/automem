# Knowledge Graph Design — AutoMem

> Community delta → Knowledge Graph → Dynamic Few-Shot for Agents

## Motivation

커뮤니티에서 들어오는 delta들을 아무생각없이 밀어넣고,
cycle이 반복될수록 공통된 원칙/통찰/패턴이 드러나게 하고 싶다.

최종 목표: V8Claw agent가 판단할 때 knowledge graph에서
dynamic few-shot example을 뽑아서 context로 사용.

## Architecture (3 Layers)

```
Layer 1: Ingest (cheap, fast)
  community delta → POST /memory → Memory node + embedding
  태그: source:community:<name>
  구조화 없이 dump

Layer 2: Enrich (async, background)
  enrichment pipeline이 Memory에서 entity 추출
  → MERGE (:Concept) + [:MENTIONS] edge
  → Concept node도 embed → Qdrant에 저장

Layer 3: Crystallize (periodic batch)
  축적된 Concept node들을 분석
  → co-occurrence, frequency, trend 감지
  → Pattern/Insight/Principle 승격
  → k-means clustering으로 유사 concept 병합
```

## Concept Node Schema

```cypher
(:Concept {
  name: "canonical_name",           -- nano가 생성한 canonical form
  description: "human readable",    -- 설명
  evidence: 5,                      -- 관찰 횟수
  counter_evidence: 1,              -- 반박 관찰 횟수
  sources: ["community:A", "community:B"],  -- 출처 breadth
  first_seen: "2026-03-01",
  last_seen: "2026-03-13",
  status: "concept"                 -- concept | pattern | insight | principle
})

(:Memory)-[:MENTIONS {context: "..."}]->(:Concept)
(:Concept)-[:CO_OCCURS {count: 3, pmi: 0.8}]->(:Concept)
(:Concept)-[:EVOLVED_INTO]->(:Concept)  -- 개념 진화 추적
```

## Promotion Rules (concept → pattern → principle)

### Signal Axes

| 축 | 의미 | noise vs signal |
|---|---|---|
| **frequency** | 몇 번 관찰 | 1 = noise, 5+ = signal |
| **breadth** | 몇 개 source에서 | 1 = local, 3+ = cross-domain |
| **recency** | 최근에도 나오나 | 옛날만 = dead, 지금도 = alive |

### Promotion Thresholds (initial heuristic)

```python
is_pattern = (
    evidence >= 3
    and len(sources) >= 1
    and last_seen > 7_days_ago
)

is_insight = (
    evidence >= 5
    and len(sources) >= 2
    and last_seen > 14_days_ago
)

is_principle = (
    evidence >= 10
    and len(sources) >= 3
    and last_seen > 30_days_ago
)
```

데이터 쌓이면서 threshold 조정.

## Entity Resolution (같은 concept 판단)

### Problem

```
"비 온 날 A 말 1착"
"우천 시 A 말 성적 좋음"
"A 말이 비 오면 잘 달린다"
→ 같은 concept인가?
```

### Approach: Medprompt-style Dynamic Matching

Write-time (enrichment):
1. 새 delta에서 concept 추출
2. Concept description을 embed
3. kNN으로 기존 Concept 중 유사한 k개 검색
4. 검색 결과를 few-shot example로 nano에게 제공
5. nano 판단: 기존 concept과 같으면 ID 반환, 아니면 새 concept 생성

```
nano prompt:
  "기존 concept들:
   1. horse:A:wet_track (evidence: 3) - 'A말 우천 강세'
   2. horse:B:distance (evidence: 2) - 'B말 장거리 선호'

   새 텍스트: '비 온 날 A 말 1착'
   → 기존 concept ID 반환 or 새 concept 이름 + 설명 생성"
```

### Batch-time (Crystallize): k-means Merge

```
1. 전체 Concept embedding → k-means clustering
2. 같은 cluster 내 concept들 → merge 후보
3. nano 확인: "이것들 같은 건가?"
4. 병합 → evidence 합산, edge 통합
```

Write-time은 빠르게 (false negative OK — 나중에 merge),
Batch-time은 정확하게 (false positive 방지 — 잘못 합치면 오염).

## Agent Integration (V8Claw Dynamic Few-Shot)

최종 소비자는 V8Claw agent:

```
Agent가 판단해야 하는 순간
  ↓ query embed
  ↓ Knowledge graph에서 kNN → 관련 Pattern/Insight/Principle k개
  ↓ few-shot example로 prompt에 주입
  ↓ LLM 판단

현재: agent → system prompt (고정) → LLM
목표: agent → system prompt + dynamic few-shot (from KG) → LLM
```

Knowledge graph는 agent의 **경험 기반 few-shot example 저장소**.
Cycle이 돌수록 concept → pattern → principle 승격되고,
agent 판단 시 더 정제된 example이 context에 들어감.

## Connection to Existing Systems

| Component | AutoMem 현재 | Knowledge Graph 추가 |
|-----------|-------------|---------------------|
| Entity extraction | spaCy/regex → tags | + MERGE (:Concept) node |
| Embedding | Memory content → Qdrant | + Concept description → Qdrant |
| kNN | recall에서 memory 검색 | + concept matching에도 사용 |
| Pattern detection | EXEMPLIFIES edge (3+ memories) | + frequency/breadth/recency 기반 |
| Consolidation | decay, creative, cluster | + Crystallize (concept merge) |
| Agent context | system prompt only | + dynamic few-shot from KG |

## Storage Architecture — Collection per Domain

### 원칙: 모든 도메인을 하나에 때려넣으면 noise

경마 질문에 엔터테인먼트 메모리가 올라오면 noise.
domain별 collection으로 격리하되, query 시점에 어디를 볼지 선택.

### Qdrant Collections

```
memories_horseracing     ← 경마 도메인
memories_entertainment   ← 엔터테인먼트
memories_finance         ← 금융
memories_shared          ← cross-domain principle/pattern
```

### FalkorDB Graphs (scale 시)

```python
db.select_graph("kg_horseracing")
db.select_graph("kg_entertainment")
db.select_graph("kg_finance")
db.select_graph("kg_shared")
```

현재는 single graph (`memories`)로 충분. 양이 커지면 분리.

### Collection Router

query 시점에 어느 collection을 볼지 결정:

```python
def select_collections(query, agent_domain, intent) -> List[str]:
    # 기본: agent의 domain + shared
    collections = [f"memories_{agent_domain}", "memories_shared"]

    # cross-domain 의도가 있으면 추가
    if intent == "cross_domain":
        collections = ALL_COLLECTIONS

    return collections
```

### Crystallize → shared collection 승격

```
concept이 2+ domain에서 발견됨
  → kg_shared graph에 (:Principle) node 생성
  → memories_shared collection에 embed
  → 모든 agent가 shared에서 검색 가능
```

domain-specific 검색은 빠르고 정확 (noise 없음).
cross-domain principle은 shared에서만 검색.

### Dynamic Few-Shot Flow (최종)

```
Agent 판단 순간
  ↓ query embed
  ↓ Collection router → [memories_horseracing, memories_shared]
  ↓ 각 collection에서 kNN → 후보 k개씩
  ↓ merge + FalkorDB에서 evidence/breadth 확인
  ↓ 최종 few-shot example 선택
  ↓ prompt에 주입 → LLM 판단
```

## System Architecture — Three-Layer Loop

기존 3 Layer (Ingest/Enrich/Crystallize)는 **자연 발생 데이터 안에서의** 파이프라인.
여기에 **데이터를 생성하는** Offline Prep 레이어가 추가되면서 자기 강화 루프가 완성된다.

```
┌─────────────────────────────────────────────────────────┐
│ Layer 0: AutoMem Infrastructure                         │
│   Ingest → Enrich → 자연 발생 데이터의 연결/패턴 탐지   │
│   (있는 노드 안에서만 작동)                              │
└──────────────────────┬──────────────────────────────────┘
                       │ Pattern 승격 (evidence >= threshold)
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Offline Prep — Situation Generation             │
│   원칙 (책, SOUL, 승격된 Pattern)                        │
│   → nano가 situation + reasoning 생성                    │
│   → Qdrant embed (situation+reasoning 합쳐서)            │
│   → FalkorDB DERIVED_FROM edge                           │
│   (없는 데이터를 만들어냄 — 그물전법)                     │
└──────────────────────┬──────────────────────────────────┘
                       │ 실제 사용 → hit → evidence++
                       │          → miss → 새 situation 생성 트리거
                       ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: Crystallize — Pattern Promotion                 │
│   축적된 Concept → frequency/breadth/recency 분석        │
│   → concept → pattern → insight → principle 승격         │
│   → 승격된 principle이 Layer 1의 입력이 됨               │
│   (자기 강화 루프 완성)                                   │
└─────────────────────────────────────────────────────────┘

순환:
  Layer 0 (관찰) → Layer 2 (승격) → Layer 1 (생성) → Layer 0 (관찰) → ...
```

### 각 Layer의 역할

| Layer | 하는 일 | 입력 | 출력 |
|-------|---------|------|------|
| **0: AutoMem** | 자연 데이터 연결, enrichment | community delta, 대화 메모리 | Concept, SIMILAR_TO, EXEMPLIFIES |
| **1: Offline Prep** | 원칙에서 상황 생성 | Principle node, 책, SOUL | Situation+Reasoning 벡터, DERIVED_FROM |
| **2: Crystallize** | 패턴 승격, 개념 병합 | 축적된 Concept/Pattern | Principle (→ Layer 1 입력) |

### Agent 검색 흐름 (Layer 횡단)

```
V8 대화 턴
  ↓ 유저 발화 embed
  ↓ Qdrant cosine → Layer 1이 생성한 situation hit (벡터 검색만)
  ↓ FalkorDB DERIVED_FROM → 원칙 역추적
  ↓ evidence/breadth scoring (Layer 2가 축적한 점수)
  ↓ context로 주입 (reasoning model에 데이터로 제공)
```

## Reasoning-in-Vectors — Qdrant 벡터 품질 전환

Knowledge Graph의 모든 노드(Memory, Situation, Concept)가 Qdrant에 embed될 때,
**content만이 아니라 content + reasoning**을 embed해야 한다.

→ 상세 설계: [reasoning-in-vectors.md](./reasoning-in-vectors.md)

### Agent 검색 흐름에 미치는 영향

```
Before (content만 embed):
  유저 발화 → cosine → surface text 매칭 → miss 많음 → 축 분류기 필요

After (content + reasoning embed):
  유저 발화 → cosine → 의도/맥락까지 매칭 → hit rate ↑ → 축 분류기 optional
```

벡터 품질이 올라가면 FalkorDB의 역할이 더 명확해진다:
- **Qdrant**: "뭘 찾을까" (semantic search — reasoning 덕분에 정확도 향상)
- **FalkorDB**: "찾은 것들 중에 뭘 고를까" (evidence, breadth, user alignment scoring)

### Layer별 reasoning 생성

| Layer | 노드 타입 | reasoning 생성 시점 | reasoning 내용 |
|-------|----------|-------------------|---------------|
| **0: AutoMem** | Memory | enrichment time (async) | "이 기억이 왜 중요한가, 어떤 상황에서 필요한가" |
| **1: Offline Prep** | Situation | generation time (batch) | "어떤 숨은 의도/감정/트리거가 있는가" |
| **2: Crystallize** | Concept/Pattern | promotion time | "이 패턴이 무엇을 의미하는가, 어디에 적용되는가" |

## Open Questions

1. Collection 생성 시점 — agent 등록 시 자동? 수동?
2. nano의 canonical naming 일관성 — 얼마나 안정적인지 실험 필요
3. Crystallize 주기 — 매일? 매주? evidence 변화량 기반?
4. shared collection 승격 threshold — sources >= 2? >= 3?
5. Agent prompt에 몇 개의 example이 최적인지 (Medprompt은 k=5)
6. Collection 간 embedding 호환성 — 같은 model 써야 cross-search 가능
7. reasoning 길이 최적값 — 벡터 품질 vs embedding 모델의 토큰 제한 트레이드오프

## References

- [Medprompt](./medprompt.md) — Dynamic few-shot selection 원본
- [Reasoning-in-Vectors](./reasoning-in-vectors.md) — Qdrant embedding 전략 전환
- [Offline Prep Strategy](./offline-prep-strategy.md) — Situation+reasoning 생성 파이프라인
- [Named Property Migration](../plans/) — FalkorDB schema 개선 (prerequisite)
- Enrichment pipeline: `automem/enrichment/runtime_orchestration.py`
- Entity extraction: `automem/enrichment/` (spaCy + regex)
