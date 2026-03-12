# Reasoning-in-Vectors — AutoMem Embedding 전략 전환

> content만 embed하지 않는다. content + reasoning을 embed한다.
> 이것이 Qdrant의 진짜 사용법이다.

## 발견

현재 AutoMem enrichment pipeline은 nano를 불러서 56차원 stance scoring을 한다.
nano는 텍스트를 읽고 "이게 어떤 가치관/의도/맥락을 담고 있는지" 추론한다.
그런데 그 추론 결과를 **숫자(lor)로만 압축**해서 FalkorDB에 넣고,
**추론 텍스트 자체는 버린다.**

Qdrant 벡터는 여전히 원래 content만 반영한다.

```
현재:
  "비 온 날 A말이 1착했다" → embed("비 온 날 A말이 1착했다") → Qdrant
  nano는 이걸 읽고 "우천 강세 패턴이다"라고 이해했지만
  → lor_* 숫자만 FalkorDB에 → 벡터에는 반영 안 됨

있어야 하는 것:
  "비 온 날 A말이 1착했다" → nano reasoning →
  "이 기록은 A말의 우천 트랙 강세 패턴을 보여준다.
   습한 노면에서 그립력이 좋은 말이 유리하다.
   '비', '우천', '습', '노면 상태' 같은 키워드와 관련.
   비 예보가 있는 날 A말 출전 여부를 확인할 때 필요하다."
  → embed(content + reasoning) → Qdrant
```

## 왜 이게 중요한가 — Medprompt CoT 원리

Medprompt에서 가장 큰 성능 점프(+3.4%)가 Self-Generated CoT에서 나왔다.
같은 원리: **추론 과정이 벡터에 들어가면 검색 품질이 비약적으로 올라간다.**

```
content만 embed:
  표면 텍스트 → [surface vector]
  검색: 표면 텍스트가 비슷한 것만 hit

content + reasoning embed:
  표면 텍스트 + 왜 중요한지 + 어떤 상황에서 쓰이는지 + 숨은 의도/맥락
  → [rich vector]
  검색: 의도가 비슷한 것도 hit, 맥락이 비슷한 것도 hit
```

같은 embedding model, 같은 Qdrant, 같은 cosine — 입력 텍스트만 바꿔도 검색 품질이 확 달라진다.

## 구현 전략

### 현재 enrichment flow

```
POST /memory → content 저장
  → enqueue_embedding(memory_id, content)     ← content만 embed
  → enqueue_enrichment(memory_id)
      → entity extraction (spaCy/regex)
      → temporal links (PRECEDED_BY)
      → semantic neighbors (SIMILAR_TO)
      → stance scoring (nano → lor numbers)   ← 추론 텍스트 버림
      → summary generation (nano)
      → Qdrant payload sync (metadata만)      ← 벡터 자체는 안 바꿈
```

### 바뀌어야 하는 flow

```
POST /memory → content 저장
  → enqueue_embedding(memory_id, content)     ← 1차: content만 (빠른 검색 가능하게)
  → enqueue_enrichment(memory_id)
      → entity extraction (spaCy/regex)
      → temporal links (PRECEDED_BY)
      → semantic neighbors (SIMILAR_TO)        ← 1차 벡터 기준으로 연결
      → ★ reasoning generation (nano, HIGH)    ← 새로 추가
      │     "이 텍스트는 무엇에 대한 것인가?"
      │     "왜 중요한가?"
      │     "어떤 상황에서 다시 찾게 될까?"
      │     "숨은 의도/맥락/overstory는?"
      │     "어떤 표현/키워드가 이것과 관련될까?"
      │     → reasoning text → FalkorDB m.reasoning 필드에 저장
      │
      → stance scoring (nano → lor numbers)    ← 기존 유지 (필터링용)
      → summary generation
      → ★ RE-EMBED (content + reasoning)       ← 새로 추가
      │     embed_text = f"{content}\n\n---\n\n{reasoning}"
      │     → Qdrant 벡터 교체 (upsert)
      │
      → Qdrant payload sync (metadata + reasoning)
```

핵심 변경: **enrichment 끝에 re-embed 단계 추가.**
1차 embed(content만)는 즉시 검색 가능하게 하고,
enrichment 완료 후 reasoning이 포함된 rich vector로 교체.

### nano 설정 변경

```
현재 stance scoring:
  model: gpt-4.1-nano
  reasoning_effort: (없음 — 기본값)
  max_tokens: 1500
  temperature: 0.0
  output: JSON (숫자)

추가할 reasoning generation:
  model: gpt-4.1-nano
  reasoning_effort: high          ← 핵심: 추론 깊이 = 벡터 품질
  max_tokens: 2000+               ← 넉넉히: 추론이 길수록 벡터에 더 많은 의미
  temperature: 0.0
  output: free text (reasoning)
```

**reasoning_effort=high + max_tokens 넉넉히.**
추론이 길수록 벡터에 더 많은 의미 차원이 들어간다.
500자 추론보다 1000자 추론이 더 많은 surface query에 걸린다.

### reasoning generation prompt (초안)

```
system: "You analyze memories/texts and generate rich reasoning about them.
Your reasoning will be embedded alongside the content for vector search.
The richer your reasoning, the more search queries will match this memory.

Include:
- What this text is fundamentally about (overstory)
- Why this would be important to recall later
- What situations or questions would make this relevant
- Hidden intentions, emotions, or context behind the text
- Related keywords, expressions, and trigger phrases
- What domain knowledge connects to this

Write naturally, not as a list. Think deeply about the text."

user: "---
{content}
---

Generate reasoning for vector search enrichment."
```

### embed text 구성

```python
def build_rich_embed_text(content: str, reasoning: str | None) -> str:
    """Build the text to embed in Qdrant.

    content + reasoning = rich vector.
    If reasoning is not yet generated, content alone is used (1차 embed).
    """
    if not reasoning:
        return content
    return f"{content}\n\n---\n\n{reasoning}"
```

reasoning이 없으면 content만 (기존 동작), 있으면 합쳐서 embed.
이 함수가 embedding pipeline의 유일한 입구가 되어야 한다.

## 기존 데이터 마이그레이션

~350개 기존 메모리에 대해:

```
Phase 1: reasoning 생성 (batch)
  기존 메모리 content → nano(reasoning_effort=high) → reasoning text
  → FalkorDB m.reasoning 필드에 저장

Phase 2: re-embed (batch)
  content + reasoning → new embedding → Qdrant upsert
  기존 벡터를 rich vector로 교체

Phase 3: semantic neighbors 재계산
  벡터가 바뀌었으므로 SIMILAR_TO edge 재계산 필요
  → reenrich_batch.py 확장 or 별도 스크립트
```

reembed_embeddings.py 스크립트가 이미 있다 — 이걸 확장하면 된다.

## Offline Prep과의 통합

[offline-prep-strategy.md](./offline-prep-strategy.md)에서 설계한 situation+reasoning 생성은
이미 reasoning-in-vectors 원칙을 따르고 있다.

```
Offline Prep:  principle → nano가 situation+reasoning 생성 → embed(situation+reasoning)
AutoMem 전체:  content → nano가 reasoning 생성 → embed(content+reasoning)
```

같은 원칙, 같은 메커니즘. Offline Prep은 "없는 데이터를 만드는" 것이고,
여기서 설계한 건 "있는 데이터를 풍부하게 만드는" 것.

둘 다 최종적으로 Qdrant에 들어가는 벡터의 품질을 높인다.

## 벡터 품질 계층

```
Level 0: content만 embed (현재)
  → 표면 텍스트 매칭만 가능
  → "비 온 날 A말" 검색 → "비 온 날 A말" hit (당연)
  → "우천에 강한 말?" 검색 → miss (표면이 다름)

Level 1: content + reasoning embed (이 설계)
  → 의도/맥락 매칭 가능
  → "우천에 강한 말?" 검색 → hit (reasoning에 "우천 트랙 강세" 포함)
  → "비 예보 있는데 어떤 말?" 검색 → hit (reasoning에 "비 예보" 트리거 포함)

Level 2: content + reasoning + situation variants (Offline Prep)
  → 가상 상황까지 매칭 가능
  → 원칙에서 파생된 다양한 표면 상황이 벡터 공간에 퍼져 있음
  → 어떤 각도의 질문이든 하나는 걸림
```

Level 0 → Level 1은 enrichment pipeline 변경으로 달성.
Level 1 → Level 2는 Offline Prep (별도 배치 프로세스).
둘 다 같은 Qdrant collection에 공존.

## stance scoring과의 관계

stance scoring(lor)은 여전히 유용하다 — 다른 역할:
- **lor (FalkorDB)**: user profile alignment 필터링, 구조화된 수치 비교
- **reasoning (Qdrant vector)**: semantic search 품질 향상

같은 nano 호출에서 둘 다 뽑을 수도 있다:
```
nano에게 한 번에:
  1. reasoning text (벡터용)
  2. stance dimensions (lor용)
→ 하나의 API call로 두 가지 결과
```

현재는 stance scoring만 별도로 하지만, reasoning generation을 추가하면서
두 작업을 하나의 call로 합치는 것도 고려할 수 있다.
다만 reasoning_effort가 다를 수 있으므로 (reasoning=high, scoring=medium) 분리가 나을 수도.

## 비용 영향

| 작업 | 현재 | 변경 후 |
|------|------|---------|
| store time | embed × 1 | embed × 1 (동일) |
| enrichment | nano scoring × 7 calls | + nano reasoning × 1 call + re-embed × 1 |
| 추가 비용 per memory | 0 | nano(high) 1회 + embed API 1회 |

nano(high) 1회 추가 — 메모리 하나당 ~$0.001 수준.
350개 기존 메모리 전체 retroactive = ~$0.35.

**벡터 검색 품질 향상 대비 비용 거의 무시할 수준.**

## 구조적 전환: Store-Heavy → Recall-Heavy

현재 agent들이 `store_memory`를 자주 호출하는 이유:
- recall 품질이 낮아서 "나중에 찾으려면 지금 명시적으로 태그/메타데이터 붙여서 저장"
- content만 embed되니까 나중에 의도로 검색하면 miss → 미리 구조화해서 넣어야 함

reasoning-in-vectors가 작동하면:
- 대화 내용이 이미 reasoning과 함께 embed → 의도/맥락으로 검색 가능
- "이건 중요하니까 따로 저장" 하지 않아도 recall에서 알아서 걸림
- **store가 줄고, recall 단에서 대부분 해결**

```
Before:
  agent → "이거 중요하다" → store(content + tags + importance) → recall 시 태그로 찾기
  agent → 저장 안 한 대화 → recall → miss → "아 이걸 저장했어야..."

After:
  대화 내용 → 자동 reasoning enrichment → rich vector → Qdrant
  agent → recall → 의도/맥락으로 hit → "저장할 필요가 없었네"
```

패러다임: "뭘 저장할까" → "이미 있는 것 중에 뭘 찾을까"

이건 Offline Prep의 그물전법과 같은 방향:
- Offline Prep은 "없는 데이터를 만들어서" 벡터 공간을 채운다
- Reasoning enrichment는 "있는 데이터를 풍부하게 해서" 벡터 품질을 올린다
- 둘 다 결과적으로 recall 시점의 hit rate를 올린다
- store의 부담이 recall 인프라로 옮겨간다

## Embedding 환경 (확인됨)

- **Provider**: OpenAI `text-embedding-3-large` (`EMBEDDING_PROVIDER=openai`, `VECTOR_SIZE=3072`)
- **Input limit**: 8191 tokens → content(2000자) + reasoning(1000자+) = 여유 충분
- Voyage-4는 미사용. `.env`에서 `EMBEDDING_PROVIDER=openai` 명시.

## Open Questions

1. reasoning generation과 stance scoring을 하나의 nano call로 합칠 것인가?
   - 합치면: API call 수 줄어듦, 일관성
   - 분리하면: reasoning_effort 다르게 설정 가능, 각각 실패해도 독립적
2. reasoning 길이 최적값 — 500자? 1000자? 실험 필요
3. re-embed 시 1차 벡터(content only)와 2차 벡터(content+reasoning) 공존할 필요?
   - 아니면 2차가 1차를 완전히 교체?
4. SIMILAR_TO edge 재계산 비용 — 벡터가 바뀌면 기존 연결이 부정확해짐

## References

- [offline-prep-strategy.md](./offline-prep-strategy.md) — situation+reasoning embed 원칙 (Level 2)
- [medprompt.md](./medprompt.md) — Self-Generated CoT +3.4% jump → reasoning-in-vectors의 이론적 근거
- `automem/enrichment/runtime_orchestration.py` — 현재 enrichment pipeline
- `automem/embedding/runtime_pipeline.py` — 현재 embedding pipeline
- `automem/utils/resonance_scorer.py` — nano stance scoring (reasoning 텍스트를 버리는 곳)
