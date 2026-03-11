# AutoMem v2: Edge-Centric Bayesian Memory

> 관계가 본체다. 대상이 아니라.
> 이게 안 되면 나머지 다 의미 없다. 이게 최소 MVP.

## 한 문장

메모리의 가치는 메모리 자체가 아니라 **관계에서** 나온다. 관계의 weight는 **observer의 ideology에 따라 다르고**, **evidence로 갱신**되고, **행동으로 검증**된다. 시스템이 유저보다 유저를 더 잘 알게 되는 구조.

## 왜 v2인가

v1은 작동하는 것처럼 보이지만 실제로는:
- creative association — dead code (embedding이 graph에 없음)
- clustering — dead code (같은 이유)
- decay — 시간 기반 고정 공식 (evidence 없음)
- pattern detection — count 기반 (3개면 패턴?)
- scoring weights — 환경변수 고정 (모든 observer 동일)
- importance — node에 직접 찍음 (관계와 무관)
- classification — argmax 하나 (분포가 아님)

이걸 고치는 게 아니라, **기반을 바꿔야** 함.

## 핵심 구조

### 1. Edge가 본체

```
v1: node.importance = 0.8 (직접 부여)
v2: node.importance = f(connected edges)  (파생)
```

메모리 하나가 혼자 중요한 게 아니야. 강한 관계가 많으면 중요하고, 관계가 약해지면 덜 중요해지고, 고립되면 사라져도 되는 거.

### 2. Node — Type Distribution

```python
# v1: argmax 하나
classify("Redis 캐시 문제 해결") → ("insight", 0.9, ...)

# v2: 전체 타입에 대한 확률 분포
classify("Redis 캐시 문제 해결") → {
    "insight": 0.85,
    "pattern": 0.6,
    "decision": 0.3,
    "preference": 0.05,
    "correction": 0.02,
    ...
}
```

node.type은 단일 라벨이 아니라 **전체 타입에 대한 확률 분포**. "이 메모리는 insight인가 pattern인가"가 이분법이 아니라 **동시에 둘 다**일 수 있고, evidence가 들어오면 분포가 바뀜.

### 3. Edge Scoring — gpt-nano 배치 평가

edge 생성 시 gpt-nano에게 **한 번에** 평가시킨다:

```
프롬프트:
"메모리 A: 'Redis 캐시 문제 해결'
관련 메모리들:
  B: 'Docker 네트워크 설정' (cosine: 0.85)
  C: '팀 회고 회의록' (cosine: 0.82)
  D: 'PostgreSQL 인덱스 튜닝' (cosine: 0.81)

각 관계를 8개 sub-layer별로 0~1 평가해:

[CULTURE] collectivism, hierarchy, uncertainty_avoid, time_orientation, indulgence, context_comm
[IDEOLOGY] liberalism, progressivism, market, growth_orient, tribalism, dataism, feminism, environmentalism, identity_politics, romanticism, nihilism, populism, anarchism, transhumanism, achievement, stoicism, consumerism
[RELIGION] religiosity, afterlife, sacred_boundary
[BELIEF] empiricism, humanism, rationalism, agency, existential, moral_care
[SEX] same_attraction, other_attraction, gender_identity, sexuality_openness
[META] self_awareness, bias_recognition, feedback_quality, dunning_kruger, confirmation_bias, anchoring, availability, loss_aversion
[COGNITIVE] openness, conscientiousness, extraversion, agreeableness, neuroticism, IQ, EQ, SQ, narcissism, machiavellianism, psychopathy, expertise
[PROCESSING] temporal, abstraction, risk, action, depth, novelty, pragmatic, autonomy, curiosity, structure, emotional_valence"

→ 응답 (JSON, sub-layer별 객체):
A-B: {
  CULTURE:    {collectivism: 0.3, hierarchy: 0.2, ...},
  IDEOLOGY:   {liberalism: 0.5, dataism: 0.8, ...},
  RELIGION:   {religiosity: 0.0, ...},
  BELIEF:     {empiricism: 0.9, rationalism: 0.8, ...},
  SEX:        {same_attraction: 0.0, ...},
  META:       {self_awareness: 0.4, confirmation_bias: 0.3, ...},
  COGNITIVE:  {openness: 0.7, IQ: 0.9, ...},
  PROCESSING: {pragmatic: 0.8, depth: 0.7, ...}
}
```

**API call 1번으로 edge N개 × 8 sub-layer 평가 끝.** 비용 거의 없음.

응답의 각 sub-layer가 **그대로** 해당 edge type의 properties로 저장:
```cypher
// nano 응답의 CULTURE → SCORE_CULTURE edge
CREATE (a)-[:SCORE_CULTURE {collectivism: 0.3, α_coll: 1, β_coll: 1,
                            hierarchy: 0.2, α_hier: 1, β_hier: 1, ...}]->(b)
// nano 응답의 IDEOLOGY → SCORE_IDEOLOGY edge
CREATE (a)-[:SCORE_IDEOLOGY {liberalism: 0.5, α_lib: 1, β_lib: 1,
                             dataism: 0.8, α_data: 1, β_data: 1, ...}]->(b)
// ... 8개 edge type
```

현재 enrichment 흐름에 끼워넣기:
```
현재: store → embedding → Qdrant similar 검색 → SIMILAR_TO edge 생성 → 끝
v2:   store → embedding → Qdrant similar 검색 → nano 배치 평가 → base edge + 8 score edges 저장
                                                  ↑ 여기 하나 추가
```

cosine은 "누가 비슷한가"만 줌. nano는 **"어떻게 비슷한가"**를 67차원 × 8 sub-layer로 설명. 같은 0.85라도 "기술적으로 비슷한 0.85"인지 "사회적으로 비슷한 0.85"인지가 다르니까.

### 4. Observer — Vector 표현 (계층적)

observer는 discrete label이 아니라 **연속 벡터**, 차원은 **계층 구조**:

```
Layer 0: Ideology (overstory)
  → "뭘 중요하게 보는가"
  → 모든 하위 판단의 방향을 결정
  → 본인이 모름. 행동에서 추론해야 함.

Layer 1: Meta-cognition
  → "자기 자신을 얼마나 아는가"
  → 피드백의 신뢰도를 결정

Layer 2: Cognitive (MBTI + Intelligence)
  → "어떻게 처리하는가"
  → ideology 안에서의 인지 스타일

Layer 3: Processing
  → "어떤 형태를 선호하는가"
  → 구체적 행동 패턴
```

**Overstory가 0이면 하위 레이어가 아무리 높아도 의미 없음.**
이슬람 observer한테 "돼지고기 레시피" ↔ "저녁 메뉴" 관계는 — MBTI가 뭐든 상관없이 weight가 0.

### Observer Dimensions (v1) — 67차원, 8 sub-layers

FalkorDB 구조: **sub-layer = edge type, dimension = edge property**.
관계 하나당 base edge 1 + score edge 8 = **9 edges**.

```
관계 A-B 저장 형태:
(A)-[:SIMILAR_TO     {cosine: 0.85}]->(B)                     // base
(A)-[:SCORE_CULTURE  {collectivism: 0.5, α_coll: 1, β_coll: 1, ...}]->(B)
(A)-[:SCORE_IDEOLOGY {liberalism: 0.7, α_lib: 1, β_lib: 1, ...}]->(B)
(A)-[:SCORE_RELIGION {religiosity: 0.2, ...}]->(B)
(A)-[:SCORE_BELIEF   {empiricism: 0.8, ...}]->(B)
(A)-[:SCORE_SEX      {same_attraction: 0.1, ...}]->(B)
(A)-[:SCORE_META     {self_awareness: 0.6, ...}]->(B)
(A)-[:SCORE_COGNITIVE {openness: 0.8, ...}]->(B)
(A)-[:SCORE_PROCESSING {temporal: 0.3, ...}]->(B)
```

**recall 시 계층적 fetch**: overstory 5개 edge 먼저 → gate 통과하면 나머지 3개.
**α/β는 score 옆에**: 같은 edge property로 bayesian state 동거.

```
═══════════════════════════════════════════════════════════════
 Layer 0: Overstory — "뭘 중요하게 보는가"
 본인이 모름. 행동에서 추론. 0이면 하위 전체 차단.
═══════════════════════════════════════════════════════════════

── SCORE_CULTURE (6) ──────────────────────  Edge type: SCORE_CULTURE
 Hofstede 6 cultural dimensions + Hall
 국가별 통계 데이터 존재 → cold start prior 직접 사용 가능

 1. collectivism      집단주의 ↔ 개인주의           (Hofstede IDV)
 2. hierarchy         위계 ↔ 수평                    (Hofstede PDI: power distance)
 3. uncertainty_avoid 불확실성 회피 ↔ 수용           (Hofstede UAI)
 4. time_orientation  장기 ↔ 단기                    (Hofstede LTO)
 5. indulgence        향유 ↔ 절제                    (Hofstede IVR)
 6. context_comm      고맥락(암시) ↔ 저맥락(직접)    (Hall)

── SCORE_IDEOLOGY (17) ────────────────────  Edge type: SCORE_IDEOLOGY
 Harari "imagined orders" + 사회운동 + 현대 사상

 7. liberalism        자유주의 ↔ 권위주의            (자유주의 휴머니즘)
 8. progressivism     진보 ↔ 보수
 9. market            자유시장 ↔ 규제/복지            (자본주의 ↔ 사회주의 휴머니즘)
10. growth_orient     성장 ↔ 지속가능                 (진화적 휴머니즘)
11. tribalism         부족주의/민족주의 강도           (Harari: 제국 vs 민족)
12. dataism           인간중심 ↔ 데이터/알고리즘       (Harari, Homo Deus)
13. feminism          가부장제 ↔ 성평등
14. environmentalism  인간중심 ↔ 생태중심
15. identity_politics 보편주의 ↔ 정체성/교차성
16. romanticism       감성·자연·개인감정 지향          (반계몽주의, 예술적 자아)
17. nihilism          의미 부정 ↔ 의미 구축            (니체, 실존적 허무)
18. populism          엘리트 불신 ↔ 전문가 신뢰        (대중주의)
19. anarchism         자율·탈권력 ↔ 제도·질서           (무정부주의)
20. transhumanism     인간 한계 수용 ↔ 기술적 초월      (Bostrom, 특이점주의)
21. achievement       자기착취·성과주의 ↔ 여유·존재      (Byung-Chul Han, 피로사회)
22. stoicism          감정 절제·수용 ↔ 감정 표현·저항    (스토아 철학, 현대 스토이시즘)
23. consumerism       소비 = 정체성 ↔ 미니멀리즘         (소비주의)

── SCORE_RELIGION (3) ─────────────────────  Edge type: SCORE_RELIGION

24. religiosity       세속 ↔ 신앙
25. afterlife         현세 ↔ 내세
26. sacred_boundary   성역 유무 (halal, kosher, 계율 등)

── SCORE_BELIEF (6) ───────────────────────  Edge type: SCORE_BELIEF

27. empiricism        경험·데이터 ↔ 원칙·교리
28. humanism          인본 ↔ 기술/초월                 (테크노 휴머니즘)
29. rationalism       이성 ↔ 직관/감성
30. agency            자기결정 ↔ 운명/환경
31. existential       의미추구 ↔ 실용추구
32. moral_care        care/fairness ↔ loyalty/authority/purity  (Haidt MFT)

── SCORE_SEX (4) ──────────────────────────  Edge type: SCORE_SEX
 성적 지향은 단일 축이 아니라 다차원.
 Kinsey scale 확장 — 끌림/정체성/태도 분리.

33. same_attraction   동성 끌림 강도 (0=없음, 1=강함)
34. other_attraction  이성 끌림 강도 (0=없음, 1=강함)
35. gender_identity   시스젠더 ↔ 논바이너리/트랜스
36. sexuality_openness 성적 보수 ↔ 성적 개방

═══════════════════════════════════════════════════════════════
 Layer 1: Meta-cognition — "자기 자신을 얼마나 아는가"
 피드백의 신뢰도를 결정. learning rate 역할.
═══════════════════════════════════════════════════════════════

── SCORE_META (8) ─────────────────────────  Edge type: SCORE_META
 자기인식 (4) + 인지편향 (4, Kahneman)

37. self_awareness    자기 인식 수준
38. bias_recognition  편향 인식 능력
39. feedback_quality  피드백 일관성 (시스템이 추정)
40. dunning_kruger    자기 능력 과대/과소 평가 경향

 # 인지 편향 — Kahneman, Thinking Fast and Slow
 # "이 사람이 얼마나 이 편향에 취약한가" (0=면역, 1=매우 취약)
 # 메타인지의 일부: 편향을 아는 것 ≠ 편향이 없는 것
41. confirmation_bias 확증편향 — 기존 믿음 강화하는 정보만 수용
42. anchoring         앵커링 — 처음 본 숫자/정보에 고정
43. availability      가용성 편향 — 쉽게 떠오르는 것이 중요하다고 착각
44. loss_aversion     손실회피 — 같은 크기의 손실이 이득보다 2배 크게 느껴짐

═══════════════════════════════════════════════════════════════
 Layer 2: Cognitive — "어떻게 처리하는가"
 성격 특성 + 지능 유형 + 어두운 면
═══════════════════════════════════════════════════════════════

── SCORE_COGNITIVE (12) ───────────────────  Edge type: SCORE_COGNITIVE

 # Big Five (OCEAN) — Costa & McCrae
 # MBTI 대체. 50% 유전, 높은 test-retest 신뢰도.
 # 과학적으로 가장 검증된 성격 모델.
45. openness          경험에 대한 개방성 (호기심, 상상력, 미적 감수성)
46. conscientiousness 성실성 (계획적, 체계적, 자기절제)
47. extraversion      외향성 (사교적, 활동적, 자극추구)
48. agreeableness     친화성 (이타적, 협조적, 신뢰)
49. neuroticism       신경성 (불안, 적대감, 감정 불안정)

 # Intelligence types
50. IQ                논리·분석·추상
51. EQ                감정 인식·공감
52. SQ                사회적 맥락·관계 파악

 # Dark Triad — Paulhus & Williams (2002)
 # 반사회적 스펙트럼. 소시오패스/사이코패스 연속체.
 # 높다고 나쁜 건 아님 — CEO, 외과의, 변호사에서 높은 경향.
53. narcissism        자기애 (과대 자아, 특권 의식, 칭찬 욕구)
54. machiavellianism  마키아벨리즘 (조종, 전략적 기만, 냉소)
55. psychopathy       사이코패시 (공감 결여, 충동성, 무정함)

 # Expertise
56. expertise         초보 ↔ 전문가 (해상도가 다름)

═══════════════════════════════════════════════════════════════
 Layer 3: Processing — "어떤 형태를 선호하는가"
 구체적 행동 패턴. 가장 관측 가능한 층.
═══════════════════════════════════════════════════════════════

── SCORE_PROCESSING (11) ──────────────────  Edge type: SCORE_PROCESSING

57. temporal          최신 중시 ↔ 시간 무관
58. abstraction       구체적 ↔ 추상적
59. risk              안전 ↔ 모험
60. action            숙고 ↔ 실행
61. depth             깊이 ↔ 넓이
62. novelty           익숙한 것 ↔ 새로운 것
63. pragmatic         이론 ↔ 실용
64. autonomy          독립 ↔ 협업
65. curiosity         목적 지향 ↔ 탐색 지향
66. structure         자유 ↔ 체계
67. emotional_valence 긍정 편향 ↔ 위협 민감
```

**67차원, 8 sub-layers.** (Overstory: Culture 6 + Ideology 17 + Religion 3 + Belief 6 + Sex 4 = 36 | Meta 8 | Cognitive 12 | Processing 11)

각 sub-layer = FalkorDB edge type. 각 dimension = edge property.
α/β도 같은 edge에 property로 동거 → dimension당 3개 property (score, α, β).

### 왜 Processing에서 3개 뺐나

v0의 `social_identity`, `domain_tech/social/creative`를 제거:
- `social_identity`(내집단/외집단) → `tribalism`(ideology)과 중복
- `domain_*` 3개 → 이건 node의 topic이지 observer의 processing style이 아님. P5 위반(이름이 약속한 범위 밖의 일).

**edge.scores와 observer.vector가 같은 67차원 공간.**
sub-layer 구조도 동일 — observer.vector도 8개 sub-vector로 분리 저장.

### 5. Recall — 계층적 연산

단순 dot product가 아니라 **계층적 곱셈**:

```python
# overstory가 0이면 전체가 0 (게이트 역할)
ideology_score = dot(edge.ideology_scores, observer.ideology_vector)

if ideology_score < threshold:
    weight = 0  # overstory에서 차단
else:
    meta_factor = observer.meta_cognition_score  # 피드백 신뢰도
    cognitive_score = dot(edge.cognitive_scores, observer.cognitive_vector)
    processing_score = dot(edge.processing_scores, observer.processing_vector)
    weight = ideology_score × cognitive_score × processing_score
```

**LLM이 "이해"를 하고 (edge scoring), 수학이 "개인화"를 한다 (계층적 연산).**

### 6. 3-Level Bayesian Update

전부 같은 메커니즘 — Beta(α, β), evidence 들어오면 갱신, posterior가 다음 prior.

```
Level 1: edge.scores      ← 피드백으로 축별 α/β 갱신
Level 2: observer.vector   ← 피드백 패턴으로 유저 취향 갱신
Level 3: node.type_dist    ← "insight 아니라 pattern이었네" 갱신
```

```
edge A-B 생성 시:
  nano 평가: {IQ: 0.7, EQ: 0.3, ...}  ← prior
  α=1, β=1 (각 축마다)               ← 확신 없음

recall에서 A-B 나옴 → 피드백 좋음:
  IQ축: α=2, β=1 → E[weight] = 0.67   ← "이 축 평가 맞았네"

또 피드백 좋음:
  IQ축: α=3, β=1 → E[weight] = 0.75   ← 더 확신

피드백 나쁨:
  IQ축: α=3, β=2 → E[weight] = 0.60   ← 조금 내려감
```

α, β가 쌓일수록 한 번의 피드백이 weight를 덜 흔들어. 증거가 많을수록 안정적. Beta-Binomial의 자연스러운 성질.

### Meta-cognition의 역할 — 피드백 learning rate

메타인지가 높은 유저의 피드백은 신뢰도가 높음:
```python
feedback_weight = observer.meta_cognition_score  # 0.0 ~ 1.0

# 메타인지 높음: 피드백 한 번에 α += 1.0 (빠르게 학습)
# 메타인지 낮음: 피드백 한 번에 α += 0.3 (천천히, 노이즈 줄임)
alpha += feedback_weight * 1.0
```

메타인지 점수 자체도 시스템이 추정 — **피드백의 일관성**으로. 자주 바뀌면 낮고, 일관되면 높고.

## 피드백 채널 — 3가지

### 1. Store 역추론 (암묵적, 공짜) — MVP 핵심

유저가 "이거 기억해"라고 할 때 — 그 내용의 edge.scores를 보면 이 사람이 뭘 중시하는지가 드러남.

```
유저가 기억시킨 것의 edge.scores 분석
→ "이 사람은 이런 축을 중시하는구나"
→ observer.vector 역추론 갱신
```

**추가 비용 없이 자동으로 쌓이는 피드백.**

중요: **ideology를 본인한테 물어서 초기화하면 안 됨.** "나는 자유주의야"라고 말해도 행동은 권위주의일 수 있음. store 행동에서 역추론하는 게 더 정확.

### 2. Recall 피드백 (명시적)

```
POST /memory { feedback: { useful: [...], irrelevant: [...] } }
```

에이전트가 줘야 함. 정확하지만 구조 필요.

### 3. 프로필 Prior (cold start)

유저 프로필 → 통계적 prior → 초기 observer vector.

```
"한국인, 30대, 개발자"

→ 한국인: 자유주의, 인본주의, romantic consumerism
  → liberalism: 0.65, humanism: 0.7, consumerism: 0.6, ...

→ 30대: 디지털 네이티브, 경험 축적기
  → risk: 0.6, temporal: 0.7, ...

→ 개발자: 분석적, 실용 지향
  → IQ: 0.75, pragmatic: 0.85, depth: 0.65, ...
```

이건 **개인이 아니라 집단의 통계적 평균**. 여기서 시작해서 store/recall 행동으로 개인 쪽으로 수렴.

prior가 틀려도 괜찮음 — Beta(α=1, β=1)로 시작하니까 첫 몇 번의 피드백이 크게 움직여서 빠르게 보정.

## 양방향 흐름

```
Forward (recall):
  observer.vector × edge.scores → weight → recall 순위
  피드백 → observer.vector 갱신

Reverse (store):
  유저가 "이거 기억해" → 그 내용의 edge.scores 분석
  → "이 사람은 이런 축을 중시하는구나" → observer.vector 역추론
```

**DB(basement)는 그대로 있고, observer vector만 움직이는 구조.**

## 전체 흐름

```
1. 가입/시작
   프로필 (한국인, 30대, 개발자) → 통계적 prior → 초기 observer vector

2. store
   내용 → embedding → Qdrant similar 검색 → nano 배치 평가 → edge.scores 저장
   + 저장 행위 자체가 observer.vector 역추론 신호

3. recall
   계층적 연산: ideology gate → cognitive × processing → 개인화된 순위

4. 피드백 (explicit or implicit)
   "유용했다" → edge α/β 갱신 + observer.vector 이동
   "쓸모없다" → 반대 방향
   메타인지 점수가 learning rate 결정

5. 반복 → observer vector가 점점 그 사람에 수렴
   → 시스템이 유저보다 유저를 더 잘 아는 상태
```

## Surprise

피드백 없이도 계산 가능한 유일한 signal.

새 메모리가 들어올 때, 기존 embedding 분포에서 얼마나 벗어나는가.
surprise가 높으면 = 기존 belief를 많이 바꿈 = 주목할 가치.

하지만 "놀라운 = 좋은"은 아님. surprise는 attention trigger이지 quality signal이 아님.

## Feedback Attribution — observer vs edge

피드백이 들어왔을 때 "누가 틀렸는가":

```
edge score:  "이 관계가 pragmatic하다"  ← nano가 평가 (신념 기반, 대체로 맞음)
observer:    "이 사람이 pragmatic을 좋아한다" ← 추측 (cold start or 적은 evidence)
```

**α+β가 곧 확신도**이므로, 확신 낮은 쪽이 더 많이 움직임:
```python
edge_lr = 1 / (edge.alpha + edge.beta)    # evidence 많으면 → 거의 안 움직임
obs_lr  = 1 / (obs.alpha + obs.beta)      # evidence 적으면 → 크게 움직임
```

edge score 초기값은 nano confidence에 비례:
- nano가 0.9 → α=5, β=1 (확신 높음) → 피드백에 거의 안 흔들림
- nano가 0.5 → α=2, β=2 (애매) → 피드백에 잘 움직임

### 집단 피드백 — Intersubjective Signal

비슷한 observer vector를 가진 집단이 같은 edge에 반복 negative:
- 1명 negative → observer 문제 (개인 취향)
- N명(비슷한 vector) negative → edge 문제 (점수가 틀렸다)

```python
similar_observers = find_near(current_obs.vector, threshold=0.8)
neg_count = count_negative(edge, similar_observers)

if neg_count >= k:  # 집단 signal
    edge_update_ratio = 0.8   # edge 많이 갱신
else:               # 개인 signal
    obs_update_ratio = 0.8    # observer 많이 갱신
```

sub-layer별 독립 판단 가능: SCORE_IDEOLOGY에서 가까운 집단이 negative → SCORE_IDEOLOGY edge만 갱신.

### Dimension Evolution — 신념은 변하지 않는다, 새로 태어난다

집단 피드백이 기존 차원으로 설명 안 될 때 — **기존 score가 틀린 게 아니라 새 차원이 필요한 거.**

봉건제가 자본주의로 "변한" 게 아니라, 자본주의라는 새 imagined order가 태어나서 대체한 것처럼 (Harari).

```
detection:
  1. 비슷한 observer 집단이 반복 negative
  2. 기존 차원 α/β 갱신으로 residual이 줄지 않음
  3. → "기존 67차원에 없는 새 variance 발견"

response:
  1. nano한테 물어봄: "이 패턴을 설명하는 새 축이 뭘까?"
  2. nano 제안: "post_growth (탈성장)" 같은 새 차원
  3. SCORE_IDEOLOGY edge에 새 property 추가 (schema-less)
  4. 기존 edge들에 nano가 소급 평가
  5. observer vector도 1차원 확장 (α=1, β=1)
```

**67차원은 seed.** 유저가 많아질수록 차원도 성장.
- 1명: residual이 개인 취향인지 새 차원인지 구분 불가
- 100명: 비슷한 observer 20명이 같은 패턴 → 통계적으로 유의미
- 1000명: 클러스터 간 차이가 자동으로 새 차원 후보 생성

FalkorDB schema-less → 기술적으로 가능. multi-observer → 통계적으로 가능. nano → 의미론적으로 가능.

MVP에서는 수동 (우리가 차원 추가). 유저 scale-up 시 자동화 가능한 아키텍처.

## FalkorDB 기술 검증

- **edge에 weight 저장**: 가능 — edge property로 임의 값 저장 (`r.weight = 0.8`, `r.alpha = 3`)
- **edge type 동적 생성**: 가능 — schema-less, 런타임에 아무 type이나 만들 수 있음
- **edge property 인덱스**: 가능 — `CREATE INDEX FOR ()-[r:SIMILAR_TO]-() ON (r.weight)`
- **multi-edge**: 지원 — 같은 두 노드 사이에 같은 type의 edge 여러 개 가능
- **다른 type multi-edge**: 검증 필요 — 같은 두 노드 사이에 SCORE_CULTURE + SCORE_IDEOLOGY + ... 8개 type이 공존하는 시나리오

**observer는 edge를 복제하지 않고 연산으로 풀기** — edge.scores(불변) × observer.vector(가변) = weight.

### FalkorDB 저장 구조

```cypher
// 관계 A-B 에 대해 9개 edge 생성
CREATE (a)-[:SIMILAR_TO      {cosine: 0.85}]->(b)
CREATE (a)-[:SCORE_CULTURE   {collectivism: 0.3, hierarchy: 0.2, ...}]->(b)
CREATE (a)-[:SCORE_IDEOLOGY  {liberalism: 0.5, tribalism: 0.3, ...}]->(b)
CREATE (a)-[:SCORE_RELIGION  {religiosity: 0.0, ...}]->(b)
CREATE (a)-[:SCORE_BELIEF    {empiricism: 0.9, ...}]->(b)
CREATE (a)-[:SCORE_SEX       {same_attraction: 0.0, ...}]->(b)
CREATE (a)-[:SCORE_META      {self_awareness: 0.4, ...}]->(b)
CREATE (a)-[:SCORE_COGNITIVE {openness: 0.7, IQ: 0.9, ...}]->(b)
CREATE (a)-[:SCORE_PROCESSING {pragmatic: 0.8, depth: 0.7, ...}]->(b)

// 계층적 recall: overstory만 먼저 fetch
MATCH (a)-[c:SCORE_CULTURE]->(b),
      (a)-[i:SCORE_IDEOLOGY]->(b),
      (a)-[r:SCORE_RELIGION]->(b),
      (a)-[bl:SCORE_BELIEF]->(b),
      (a)-[s:SCORE_SEX]->(b)
WHERE a.id = $memory_id
RETURN c, i, r, bl, s
// → overstory gate 계산 → 통과하면 META, COGNITIVE, PROCESSING fetch
```

## Appetite

이건 AutoMem의 기반을 바꾸는 거. remarkable 하려면 이게 MVP.

- **Must-have**: 계층적 edge scoring + observer vector(ideology overstory) + store 역추론 + 3-level bayesian update
- **Nice-to-have**: surprise 계산, explicit recall 피드백, creative/clustering 재구현, 메타인지 자동 추정

## Rabbit Holes

- "완벽한 피드백 채널"을 기다리면 영원히 못 시작 — store 역추론으로 시작
- observer별 weight를 저장하려고 하면 폭발 — 연산으로 풀어야
- observer 차원 설계에 과몰입 — 35개로 시작하고 죽는 축은 알아서 죽음
- ideology를 유저한테 직접 물으면 안 됨 — 행동에서 추론해야
- 기존 v1 데이터 마이그레이션 — edge에 scores 추가는 가능, 하지만 nano 재평가 필요
- FalkorDB가 edge property 쿼리에 얼마나 효율적인지 — 벤치마크 필요
- nano의 평가 품질 — 축 설명이 명확해야 일관된 점수가 나옴

## Dependencies

- [x] observer dimensions 확정 — v1: 67차원, 8 sub-layers
- [x] FalkorDB 저장 구조 확정 — sub-layer = edge type, dimension = edge property
- [ ] 다른 type multi-edge 성능 벤치마크 (같은 두 노드 사이 9개 edge)
- [ ] nano edge scoring 프롬프트 설계 (8 sub-layer별 JSON 응답)
- [ ] observer vector 초기화 (프로필 → 통계적 prior, 8 sub-vector)
- [ ] store 역추론 로직 설계 (edge.scores → observer.vector 갱신)
- [ ] 계층적 recall 연산 (overstory 5 edge types gate → 나머지 3)
- [ ] 메타인지 추정 (피드백 일관성 → learning rate)
- [ ] edge α/β 갱신 API
- [ ] consolidation creative/clustering이 Qdrant에서 vector 가져오게 수정 (현재 dead code)

## Related

### Overstory 이론적 기반
- **Harari, Sapiens** — "imagined order"가 우리의 overstory. 종교/자본주의/자유주의/인권 등 상상의 질서가 개인의 세계관 렌즈를 결정. 본인은 자각 못 함. 인류 분류 체계가 overstory 차원에 직접 매핑.
- **Harari, Homo Deus** — dataism (인간 → 데이터 처리 시스템), transhumanism, techno-humanism. 미래 ideology 차원의 원천.
- **Harari, 21 Lessons** — 어떤 narrative도 완전하지 않다. 모든 ideology는 현실의 일부만 보여주는 필터. observer vector = 현재 belief이지 정답이 아님.
- **Hofstede, Culture's Consequences** — 6차원 국가별 문화 벡터. 117,000명 IBM 직원 조사 기반. cold start prior에 직접 사용 가능 (국적 → SCORE_CULTURE 초기값). PDI, IDV, MAS, UAI, LTO, IVR.
- **Hall, Beyond Culture** — 고맥락/저맥락 소통. context_comm 차원의 원천.
- **Haidt, The Righteous Mind** — Moral Foundations Theory. 도덕 판단의 6개 기반. care/fairness ↔ loyalty/authority/purity. moral_care 차원.
- **Byung-Chul Han, 피로사회(The Burnout Society)** — 성과주의 사회, 자기착취. achievement 차원의 원천. "할 수 있다"의 폭력.

### 성격·인지 기반
- **Costa & McCrae, Big Five (OCEAN)** — 가장 검증된 성격 모델. 50% 유전, 높은 test-retest 신뢰도. MBTI 대체. SCORE_COGNITIVE의 5개 차원.
- **Paulhus & Williams, Dark Triad (2002)** — narcissism, machiavellianism, psychopathy. 반사회적 연속체. SCORE_COGNITIVE의 3개 차원.
- **Kahneman, Thinking Fast and Slow** — System 1/2, 인지 편향들. confirmation_bias, anchoring, availability, loss_aversion. SCORE_META의 4개 편향 차원.
- **Kinsey Scale** — 성적 지향의 연속성. same/other_attraction 이중 축으로 확장. SCORE_SEX의 원천.

### 시스템 설계 기반
- SOUL-COMMON.md: "epistemic vs aleatory uncertainty", Spiegelhalter
- docs/memory-strategy.md: importance scoring, forgetting policy
- Bayesian surprise (Itti & Baldi, 2009)
- PageRank — node importance를 edge에서 파생하는 대표적 사례
- Collaborative filtering / matrix factorization — observer vector × edge scores = weight
- Netflix cold start — 인구통계 prior → 시청 행동으로 개인화 수렴
