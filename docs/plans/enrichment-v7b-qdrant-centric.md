# Enrichment v7b: Full Reasoning in Qdrant + Stance Evidence

## Context

v7 커밋 (4fcb13e)에서 6Q battery + scenario 분리까지 완료됐지만, **Qdrant embed text를 over-strip**한 설계 오류 발견:

**원래 의도**: store/enrichment/stance scoring 시 생성되는 **모든 중간 추론 단계**를 Qdrant에 넣어서 벡터 검색 surface를 최대화. FalkorDB는 그래프 관계 + lor 수치 + summary만.

**현재 문제**:
1. `build_rich_embed_text(content, words)` — words 필드만 추출해서 임베딩. enrichment 설명(overstory 분석, inference chain, frame 분석 등)이 벡터에서 빠짐.
2. `enrichment_json`을 FalkorDB에 저장 중 — Qdrant에 들어가면 FalkorDB에 불필요.
3. `score_stance()`가 lor 수치만 반환하고 evidence(인용, 방향, 강도)를 버림 — 이것도 Qdrant에 넣어야 함.

### 수정 후 데이터 흐름

```
content
  → generate_reasoning() [6Q battery]
  → enrichment_json (전체 분석) + words (관련 단어)
  → Qdrant vector = embed(content + enrichment descriptions + words)
  → Qdrant payload에 enrichment_json 저장
  → FalkorDB: reasoning(words) 필드만 (backward compat), enrichment_json 제거

  → score_stance(content, debug=True)
  → lor values → FalkorDB (그래프 쿼리용)
  → stance evidence (quotes, directions) → Qdrant payload

  → [별도 단계] generate_scenarios(content, enrichment_json)
  → 시나리오들 → 별도 Memory 노드 + DERIVED_FROM
```

### 저장소 역할 분리

| 데이터 | Qdrant | FalkorDB |
|--------|--------|----------|
| content | vector + payload | node.content |
| enrichment descriptions | **vector** (embed text) | ❌ |
| enrichment_json | **payload** | ❌ (제거) |
| words | **vector** (embed text) | node.reasoning (compat) |
| lor values | ❌ | **node.lor_*** |
| stance evidence | **payload** | ❌ |
| summary | payload | node.summary |
| tags, tag_prefixes | payload | node |
| graph relationships | ❌ | edges |

---

## Phase 1: Enrichment text를 full로 복원

**File:** `automem/utils/reasoning_generator.py`

### 1a. `_flatten_enrichment()` 신규 추가
enrichment_json → 사람 읽기용이 아닌, 벡터 검색용 flat text.
각 섹션의 **설명 + words**를 모두 포함.

```python
def _flatten_enrichment(data: Dict[str, Any]) -> str:
    """Flatten enrichment JSON into text for vector embedding.

    Includes all descriptions and words — maximizes search surface.
    """
    parts = []
    for key in ENRICHMENT_KEYS:
        val = data.get(key)
        if isinstance(val, dict):
            # Include all text fields (not just words)
            for field in ("narrative", "chain", "frame", "conditions", "description", "words"):
                text = val.get(field, "")
                if text:
                    parts.append(str(text))
            # Also include any nested lists (bridges has list of dicts)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    parts.extend(str(v) for v in item.values() if v)
                elif isinstance(item, str):
                    parts.append(item)
        elif isinstance(val, str):
            parts.append(val)
    return " ".join(parts)
```

### 1b. `build_rich_embed_text()` 변경
```python
def build_rich_embed_text(
    content: str,
    enrichment_text: Optional[str] = None,  # flattened enrichment descriptions
    words: Optional[str] = None,
) -> str:
    parts = [content]
    if enrichment_text:
        parts.append(enrichment_text)
    if words:
        parts.append(words)
    return "\n\n".join(parts)
```

### 1c. `_extract_all_words()` 유지 (현재 그대로)
words만 따로 추출하는 기능은 여전히 필요 (FalkorDB reasoning 필드 호환용).

### 1d. `generate_reasoning()` 반환값 확장
현재: `(enrichment_json, words_text)` → 유지
호출자가 `_flatten_enrichment(enrichment_json)`으로 enrichment_text 생성.

---

## Phase 2: Orchestration — Qdrant 중심으로 전환

**File:** `automem/enrichment/runtime_orchestration.py`

### 2a. enrichment_json을 FalkorDB에서 제거
- 현재 (line ~274): `graph.query("SET m.reasoning = $reasoning, m.enrichment_json = $ej")`
- 변경: `graph.query("SET m.reasoning = $reasoning")` — enrichment_json 제거
- `metadata.enrichment.reasoning_json`도 제거

### 2b. re-embed에 full enrichment text 사용
- 현재: `build_rich_embed_text(content, reasoning)` (reasoning = words만)
- 변경:
```python
from automem.utils.reasoning_generator import _flatten_enrichment, build_rich_embed_text
enrichment_text = _flatten_enrichment(enrichment_json)
rich_text = build_rich_embed_text(content, enrichment_text, reasoning)
re_embed_fn(memory_id, rich_text)
```

### 2c. Qdrant payload에 enrichment_json 추가
enrichment 완료 시 `qdrant_client.set_payload()`에 `enrichment_json` 추가:
```python
qdrant_client.set_payload(
    collection_name=collection_name,
    points=[memory_id],
    payload={
        "tags": tags,
        "tag_prefixes": tag_prefixes,
        "metadata": metadata,
        "enrichment": enrichment_json,  # 신규: 구조화된 분석 결과
    },
)
```

---

## Phase 3: Stance scoring evidence → Qdrant

**File:** `automem/utils/resonance_scorer.py`

### 3a. `score_stance()` — 항상 evidence 캡처
현재: `debug=True`일 때만 `_debug` dict에 evidence 포함.
변경: 항상 evidence를 수집하고, `_evidence` 키로 반환.

```python
def score_stance(content: str) -> Optional[Dict[str, Any]]:
    """Returns {concept: lor, ..., "_evidence": {concept: {e, d, s, c, lor}}}"""
```

lor values는 기존대로 float. `_evidence`는 Qdrant payload 저장용.

### 3b. Orchestration에서 evidence를 Qdrant에 저장
`enrich_memory()`의 stance scoring 단계 후:
```python
lor_values = score_stance(content)
if lor_values is not None:
    save_lor_to_graph(graph, memory_id, lor_values)  # FalkorDB: lor 수치만
    evidence = lor_values.pop("_evidence", None)
    if evidence:
        # Qdrant payload에 stance evidence 추가
        qdrant_client.set_payload(
            collection_name=collection_name,
            points=[memory_id],
            payload={"stance_evidence": evidence},
        )
```

### 3c. Stance evidence도 embed text에 포함 (선택적)
evidence의 "e" (원문 인용)은 이미 content에 있으므로 중복.
하지만 "d" (pole 방향)은 새로운 검색 surface.
→ stance evidence를 flatten해서 embed text에 추가할지는 **향후 A/B 테스트로 결정**.

---

## Phase 4: Scenario 생성 (이미 구현됨, 수정 불필요)

`scenario_generator.py` + `runtime_bindings.py`의 `_store_scenario()` — v7에서 이미 완료.
각 시나리오 → 별도 Memory 노드 + DERIVED_FROM + 독립 Qdrant 벡터.

---

## Phase 5: FalkorDB 그래프 노드 전개 (후속 — 이번 scope 밖)

enrichment_json에서 추출 → 그래프 노드 MERGE:
- `(:Overstory {name})` ← `HAS_OVERSTORY`
- `(:Frame {name})` ← `HAS_FRAME`
- `(:Domain {name})` ← `BRIDGES_TO`
- `(:Bias {name})` ← `ACTIVATES_BIAS`

---

## 수정 파일 목록

| File | Action |
|------|--------|
| `automem/utils/reasoning_generator.py` | `_flatten_enrichment()` 추가, `build_rich_embed_text()` 3-param으로 변경 |
| `automem/enrichment/runtime_orchestration.py` | FalkorDB에서 enrichment_json 제거, Qdrant payload에 추가, re-embed에 full text |
| `automem/utils/resonance_scorer.py` | `score_stance()` 항상 evidence 반환 |
| `automem/enrichment/runtime_bindings.py` | `_re_embed()` 시그니처 조정 (enrichment_text 전달) |

---

## Verification

1. **Unit test**: `pytest tests/` — 기존 테스트 통과
2. **수동 테스트**:
   - Qdrant vector가 content + enrichment descriptions + words로 구성되는지
   - Qdrant payload에 `enrichment` (JSON) + `stance_evidence` 있는지
   - FalkorDB에 `enrichment_json` 필드가 더 이상 저장되지 않는지
   - lor 값은 FalkorDB에 정상 저장되는지
3. **Recall A/B**: `make lab-test` — enrichment text 포함 전후 recall 비교
