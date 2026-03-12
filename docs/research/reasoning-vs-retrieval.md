# Reasoning vs Retrieval — CoT 모델에게 Few-Shot을 어떻게 줘야 하는가

> 상위 모델은 이미 숨은 축을 추론한다. 우리가 제공해야 하는 건 추론이 아니라 데이터다.

## 핵심 긴장

```
상위 모델 (o1, Claude thinking, gpt-4o CoT):
  "비트코인 들어가도 돼?"
  → CoT: "FOMO 느끼고 있고, 군중 심리에 휩쓸리려 하고..."
  → 숨은 축 추론은 이미 한다

그러면 우리가 축 분류기를 따로 만드는 이유가 뭐지?
→ 추론은 모델이 하지만, 추론할 재료(데이터)는 모델이 없다
```

## 모델이 하는 것 vs 못 하는 것

| 능력 | 상위 모델 | 우리 시스템 |
|------|----------|-----------|
| 숨은 의도/감정 추론 | ✅ CoT가 이미 함 | 축 분류기 (보조) |
| 일반 상식/원칙 | ✅ 학습 데이터에 있음 | 불필요 |
| **커뮤니티 최신 delta** | ❌ 학습 컷오프 | ✅ Qdrant에 있음 |
| **저자 책 특정 passage** | ❌ 학습에 없을 수 있음 | ✅ vector DB에 있음 |
| **유저 취향 (user lens)** | ❌ 세션 간 기억 없음 | ✅ FalkorDB에 있음 |
| **축적된 패턴/evidence** | ❌ | ✅ KG에 있음 |
| **군중 감정 분포 (지금)** | ❌ | ✅ delta 집계 가능 |

**모델은 추론은 하지만 네 데이터를 못 본다.**
추론 결과를 가지고 네 데이터를 검색하는 파이프라인이 필요한 이유.

## Medprompt 후속 논문의 발견

> "From Medprompt to o1: Exploration of Run-Time Strategies" (Microsoft, Nov 2024)

```
핵심 발견:
  - o1-preview는 prompting 없이도 Medprompt+GPT-4를 이김
  - few-shot prompting이 o1 성능을 오히려 저해
  - reasoning-native 모델에서는 in-context learning이 방해
```

**왜 방해되는가:**

reasoning model은 자체적으로 사고 사슬을 구축한다.
거기에 "이 예시를 따라해"라는 few-shot을 넣으면
모델의 자유로운 추론을 **특정 패턴으로 제한**한다.

```
❌ Few-shot as EXAMPLE (추론 방해):
  "비슷한 상황: 사용자가 FOMO 상태일 때 이렇게 응대했음. 너도 이 패턴을 따라."
  → 모델: 내 추론 대신 이 패턴을 모방해야 하나? → 추론 품질 하락

✅ Few-shot as CONTEXT (추론 재료):
  "참고 데이터: 현재 커뮤니티 70%가 상승 전망. Buffett은 이런 상황에서 X라고 했음."
  → 모델: 이 데이터를 바탕으로 내가 추론하면... → 추론 품질 유지 + 데이터 보강
```

## Example vs Context — 주입 방식의 차이

### ❌ Example 방식 (reasoning model에 해로움)

```
system prompt에:
  "다음은 비슷한 상황에서의 대응 예시입니다. 이 패턴을 참고하세요:

   상황: 사용자가 FOMO를 표현
   대응: 먼저 감정을 인정하고, 데이터를 보여주고, 결정을 유보하게 유도

   위 패턴에 따라 응답하세요."
```

모델에게 **어떻게 응답하라고 지시**하는 것. reasoning model의 자체 판단을 억제.

### ✅ Context 방식 (reasoning model과 시너지)

```
system prompt에:
  "현재 상황 관련 데이터:

   [커뮤니티] 최근 3일 암호화폐 커뮤니티 감정: 상승 70%, 하락 15%, 관망 15%
   [원칙] Buffett: '군중이 탐욕적일 때 두려워하라'
   [패턴] 지난 6개월 유사 상황 3회 관찰. 2회는 이후 조정 발생.
   [유저] 이 유저의 과거 투자 패턴: 충동 매수 후 후회 3회 기록.

   위 데이터를 참고하여 응답하세요."
```

모델에게 **무엇을 보고 판단하라고 데이터를 주는 것**. 추론은 모델이 자유롭게.

## 우리 시스템에서의 적용

### 검색 파이프라인은 동일

```
유저 발화 → 축 분류 → 축별 동시 recall → 결과 merge
```

이 부분은 바뀌지 않는다. 달라지는 건 **결과를 prompt에 넣는 방식**.

### 주입 포맷 변경

```
Before (example 방식):
  "비슷한 상황 예시:
   상황: ...
   추론: ...
   대응: ..."

After (context 방식):
  "관련 데이터:
   [source: 저자 책 3장] '멈춤은 후퇴가 아니다'
   [source: 커뮤니티 delta 2026-03-12] XX 실험 결과 공개
   [source: 유저 히스토리] 이 유저는 과거에 ...
   [source: KG pattern] 이 패턴은 5회 관찰, 3개 소스에서 확인"
```

### 모델 티어별 전략

| 모델 | 추론 능력 | 주입 방식 | 이유 |
|------|----------|----------|------|
| **상위** (o1, Claude thinking) | 강함 | context (데이터만) | 추론은 모델에 위임, 예시가 방해됨 |
| **중위** (gpt-4o, Sonnet) | 중간 | context + light guidance | 데이터 + "이런 관점도 고려" 힌트 |
| **하위** (nano, Haiku) | 약함 | example (패턴 포함) | 추론력이 부족하니 패턴 제시 필요 |

**같은 검색 결과를, 모델 티어에 따라 다른 포맷으로 주입.**

## Axis Classifier의 역할 재정의

상위 모델이 이미 축 추론을 하니까, 축 분류기(nano)의 역할은:

```
상위 모델 직접 추론:
  비용 높음, 느림, 하지만 정확
  → 매 턴 상위 모델에게 "이 상황의 숨은 축이 뭐야?"라고 물으면 비쌈

nano 축 분류기:
  비용 낮음, 빠름, 약간 덜 정확
  → 검색 라우팅용으로는 충분
  → 정확한 추론은 상위 모델이 최종 응답에서 하면 됨
```

즉, nano는 **정밀 추론이 아니라 검색 라우팅**을 위한 것.
"대충 어느 방향으로 검색할지"만 알면 되니까 nano면 충분.

```
nano (싸고 빠름): "이 질문은 crowd_sentiment + intent:decide 축이다"
  → 검색 실행
  → 결과를 상위 모델에 context로 전달
상위 모델 (비싸고 정확): 데이터를 보고 자유롭게 추론 → 최종 응답
```

## 정리

```
우리가 만드는 것:
  ✅ 데이터를 모으고 (community delta, 책, 경험)
  ✅ 축 기반으로 빠르게 찾아서 (nano 분류 + Qdrant kNN)
  ✅ 모델이 추론할 재료로 제공 (context, not example)

우리가 만들지 않는 것:
  ❌ 추론 자체 (모델이 함)
  ❌ "이렇게 응답하라" 패턴 (모델의 추론을 방해)
  ❌ 상위 모델 대체 (보완)
```

## Connection to Other Docs

| 문서 | 관계 |
|------|------|
| [medprompt.md](./medprompt.md) | 후속 논문(o1 vs Medprompt) 발견이 이 문서의 근거 |
| [conversation-axes.md](./conversation-axes.md) | 축 분류 체계. 분류기는 라우팅용, 정밀 추론은 상위 모델 |
| [offline-prep-strategy.md](./offline-prep-strategy.md) | situation variant 저장. 주입 시 example이 아니라 context로 |
| [use-cases-dynamic-fewshot.md](./use-cases-dynamic-fewshot.md) | 4개 use case에서 context 주입 방식 적용 |
| [knowledge-graph-design.md](./knowledge-graph-design.md) | KG가 제공하는 건 "추론 재료" (evidence, 관계, 패턴) |

## References

- [From Medprompt to o1 (Microsoft, Nov 2024)](https://arxiv.org/html/2411.03590v1) — reasoning model에서 few-shot이 해로운 이유
- [Medprompt 원본](./medprompt.md) — Self-Generated CoT의 원래 맥락
- [RAG+ (2025)](https://arxiv.org/html/2506.11555v4) — application-aware reasoning, knowledge + examples dual corpus
