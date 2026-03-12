# Use Cases — Dynamic Few-Shot Retrieval

> 표면 질문 ≠ 진짜 필요한 지식. 숨은 축을 찾아야 맞는 답이 나온다.

## 공통 패턴

4개 use case 모두 같은 구조:

```
Surface Query (표면)
  ↓
  Axis Classification (숨은 축 분류)
  ↓
  축에 따라 검색 대상 + 전략이 달라짐
  ↓
  Few-shot으로 주입 → Agent 응답
```

단순 cosine(표면 query → 벡터 DB)으로는 4개 다 실패한다.
축을 먼저 찾아야 어디서 뭘 검색할지가 정해진다.

---

## Use Case 1: 저자 분신 AI — 메타인지

### 시나리오

저자가 쓴 책들을 전부 vector DB에 저장.
유저가 책 내용과 전혀 달라 보이는 질문을 함.
숨은 도메인 속 패턴을 찾아서 책 내용을 연결 → "너는 지금 이런 상태다" (메타인지 제공).

### 구체 예시

```
유저: "요즘 뭘 해도 의욕이 없어"

표면: 감정 토로, 일상 대화
숨은 축:
  emotion: depleted
  intent: understand_self (자기 이해)
  phase: stuck (정체)

축별 검색:
  → 저자 책 corpus에서 "의욕 상실" "정체기" "번아웃" 관련 passage
  → hit: 저자 책 3장 "멈춤은 후퇴가 아니라 재충전이다"
  → hit: 저자 책 7장 "의욕이 없을 때는 몸이 먼저 알고 있다"

Agent 응답:
  "저자가 3장에서 이런 말을 했어요: '멈춤은 후퇴가 아니다.'
   지금 상태가 정체처럼 느껴지겠지만, 이건 재충전 신호일 수 있어요.
   7장에서는 '의욕보다 몸의 신호를 먼저 읽으라'고 했는데..."
```

### 핵심

- **메타인지**: "너는 지금 이런 상태다"를 유저 대신 인식해줌
- 유저는 "의욕 없다"만 말했지만, agent가 숨은 패턴(정체기)을 찾아서 저자의 관점으로 비춰줌
- 표면 키워드("의욕")가 아니라 상태 축(depleted + stuck)으로 검색해야 맞는 passage가 나옴

### 검색 구조

| 축 | 값 | 검색 대상 | 검색 전략 |
|---|---|---|---|
| emotion | depleted | 저자 책 corpus | 감정 상태 관련 passage |
| intent | understand_self | 저자 책 corpus | 자기 인식/메타인지 passage |
| phase | stuck | 저자 책 corpus | 정체기/전환점 passage |

---

## Use Case 2: 커뮤니티 큐레이션 — 취향 매칭

### 시나리오

유저가 "뭐 재밌는거 없냐"고 물음.
커뮤니티 delta에서 유저의 취향(user lens)에 맞을 만한 것을 찾아서 알려줌.

### 구체 예시

```
유저: "뭐 재밌는거 없냐"

표면: 추천 요청
숨은 축:
  intent: discover (새로운 것 탐색)
  energy: low (능동적 탐색 의지 낮음 — "없냐"는 수동적 표현)
  taste: user lens [a,b] (기존 취향 프로파일)

축별 검색:
  → community delta (최근 7일)에서:
    - user lens로 stance scoring → 이 유저가 좋아할 확률 높은 delta
    - energy:low이므로 → 가볍게 소비 가능한 것 우선 (긴 분석 글 < 짧은 흥미 거리)

  → hit: "어제 XX 커뮤니티에서 이런 실험 결과 올라왔는데"
  → hit: "YY가 새 프로젝트 공개했음, 3분이면 볼 수 있는 데모"

Agent 응답:
  "이거 좋아할 것 같은데 — YY가 어제 공개한 데모, 3분이면 됨.
   그리고 XX 커뮤니티에서 재밌는 실험 결과 올라왔어."
```

### 핵심

- "재밌는거"로 cosine 치면 "재미" 관련 메모리만 나옴 → noise
- **user lens (Beta [a,b])로 취향 필터링**이 진짜 검색 축
- energy 축이 결과의 **포맷/깊이**를 결정 (가벼운 것 vs 깊은 분석)

### 검색 구조

| 축 | 값 | 검색 대상 | 검색 전략 |
|---|---|---|---|
| intent | discover | community delta (recent) | 최신순 + 신규성 |
| taste | user lens [a,b] | community delta | stance scoring으로 취향 매칭 |
| energy | low | (필터) | 소비 비용 낮은 것 우선 |

---

## Use Case 3: 시장 예측 — 군중심리 역행

### 시나리오

유저가 코인 시장 예측을 물어봄.
커뮤니티 delta에서 개미들의 탐욕/공포 집단의식 트렌드를 읽고,
군중과 다르게 할 방법을 알려줌.

### 구체 예시

```
유저: "비트코인 지금 들어가도 돼?"

표면: 매수 타이밍 질문
숨은 축:
  crowd_sentiment: 현재 커뮤니티 감정 분포
  intent: decide (매수/매도 의사결정)
  domain: crypto

축별 검색:
  → community delta (최근 3일) 감정 집계:
    - "올라간다" 70%, "떨어진다" 15%, "모름" 15%
    - 탐욕 지수 높음 → 군중이 낙관적

  → wisdom corpus에서 contrarian 원칙 검색:
    - hit: "군중이 탐욕일 때 공포하고, 공포일 때 탐욕하라" (Buffett)
    - hit: "합의가 강할수록 반대 포지션의 기대값이 높다"
    - hit: "개미가 전부 같은 방향이면 그게 신호다"

Agent 응답:
  "지금 커뮤니티 70%가 상승 전망이야.
   이럴 때 Buffett 원칙 — '군중이 탐욕일 때 조심하라.'
   합의가 이렇게 강하면 반대 시나리오를 먼저 점검하는 게 낫다.
   들어갈 거면 분할 매수로 리스크 분산."
```

### 핵심

- "비트코인 들어가도 돼?"로 cosine 치면 비트코인 가격 관련 메모리만 나옴
- 진짜 필요한 건 **지금 군중이 어디를 보는가** (delta 집계) + **그때 어떻게 해야 하는가** (원칙)
- 두 단계 검색: (1) delta에서 군중 상태 파악, (2) 그 상태에 맞는 원칙 검색

### 검색 구조

| 축 | 값 | 검색 대상 | 검색 전략 |
|---|---|---|---|
| crowd_sentiment | greedy (70% bullish) | community delta (recent) | 감정 분포 집계 |
| intent | decide | wisdom corpus | contrarian 원칙 검색 |
| domain | crypto | (필터) | 도메인 필터 |

### 특이점: 2단계 검색

```
1단계: community delta → 군중 상태 파악 (분석)
2단계: 군중 상태 → wisdom에서 대응 원칙 검색 (retrieval)
```

1단계 출력이 2단계 입력이 된다. 다른 use case는 1단계만으로 충분하지만
이 경우는 **분석 → 검색** 파이프라인.

---

## Use Case 4: 개발 디버깅 — 패턴 인식 → 근본 원인

### 시나리오

에러에 에러가 꼬리를 물고 있는 상황.
상황 자체는 특수하지만, 그런 상황에서 나타나는 **패턴**을 찾고,
그 패턴을 막을 수 있는 근본 원인을 찾아줌.

### 구체 예시

```
유저: "CORS 에러 고쳤더니 이번엔 인증 에러, 인증 고쳤더니 타임아웃..."

표면: 기술 에러 보고
숨은 축:
  pattern: cascade (연쇄 실패)
  emotion: frustrated (점점 지침)
  meta_pattern: whack-a-mole (증상 치료 반복)

축별 검색:
  → 개발 경험 KG에서 cascade 패턴 검색:
    - hit: "연쇄 에러는 공통 원인이 있다 — 증상이 아니라 원인을 찾아라"
    - hit: "3개 이상 에러가 연쇄하면 멈추고 아키텍처를 봐라"

  → situation variants에서:
    - hit: "한 에러를 고치면 다른 에러가 나타나는 상황"
      reasoning: "whack-a-mole 패턴 = 증상만 치료 중.
                  공통 의존성(인증, 네트워크, 설정)을 먼저 점검."
      guidance: "증상 나열 → 공통점 찾기 → 의존성 그래프 그리기 → root cause"

Agent 응답:
  "지금 whack-a-mole 패턴이야 — 하나 고치면 다른 데서 터지는 거.
   CORS → 인증 → 타임아웃 이 셋의 공통 원인이 있을 거야.
   멈추고 이 세 에러가 공유하는 의존성을 먼저 봐.
   네트워크 레이어? 설정 파일? 프록시?"
```

### 핵심

- 개별 에러("CORS", "인증", "타임아웃")로 검색하면 각각의 해결법만 나옴
- 진짜 필요한 건 **"연쇄 에러" 자체의 메타 패턴** 인식
- 상황은 매번 다르지만 패턴은 같다 → 패턴 수준의 wisdom이 필요

### 검색 구조

| 축 | 값 | 검색 대상 | 검색 전략 |
|---|---|---|---|
| pattern | cascade / whack-a-mole | 개발 경험 KG | 메타 패턴 검색 |
| emotion | frustrated | wisdom corpus | 감정 대응 + 디버깅 태도 |
| domain | dev | (필터) | 도메인 필터 |

---

## Cross-Case Analysis

### 공통 구조

```
모든 use case:
  1. 표면 query로는 답을 못 찾음 (semantic gap)
  2. 숨은 축을 분류해야 진짜 검색 방향이 보임
  3. 축에 따라 검색 대상(collection)과 전략이 달라짐
  4. 결과를 few-shot으로 주입하면 agent가 더 나은 응답 생성
```

### 축 → 검색 대상 매핑

| 축 | 검색 대상 |
|---|---|
| emotion / face / relation | wisdom corpus (대인관계 원칙) |
| taste (user lens) | community delta + stance scoring |
| crowd_sentiment | community delta 집계 (분석) → wisdom (원칙) |
| pattern (meta) | 경험 KG (패턴 노드) |
| domain | collection filter |
| intent | 검색 전략 결정 (discover vs solve vs decide) |
| energy | 결과 포맷 결정 (길이, 깊이) |

### 축이 하는 3가지 역할

```
1. 어디서 찾을지 (collection routing)
   → taste → community delta
   → pattern → 경험 KG
   → metacognition → 저자 책 corpus

2. 어떻게 찾을지 (search strategy)
   → crowd_sentiment → 집계 먼저 → 원칙 검색
   → taste → stance scoring
   → pattern → 메타 패턴 매칭

3. 어떻게 보여줄지 (response format)
   → energy:low → 짧게
   → intent:understand_self → 메타인지 프레임
   → intent:decide → 판단 근거 + 옵션
```

---

## Offline Prep — Use Case별 Situation Variants

각 use case에서 필요한 wisdom 유형과 situation variant 예시:

### 저자 분신 AI

```
원칙: "멈춤은 후퇴가 아니라 재충전이다"
situations:
  - "유저가 번아웃 증상을 말한다"
  - "유저가 생산성이 떨어졌다고 한다"
  - "유저가 의미 없다고 느끼는 상태를 표현한다"
  - "유저가 쉬고 싶다고 하면서도 죄책감을 느낀다"
```

### 커뮤니티 큐레이션

```
원칙: "추천은 취향 매칭이지 인기순이 아니다"
situations:
  - "유저가 뭐 재밌는거 없냐고 한다"
  - "유저가 요즘 볼 게 없다고 한다"
  - "유저가 특정 분야 최신 소식을 물어본다"
  - "유저가 지루하다고 한다"
```

### 시장 역행

```
원칙: "군중이 탐욕일 때 조심하라"
situations:
  - "유저가 지금 사도 되냐고 묻는다"
  - "유저가 모두 오른다고 하니까 나도 들어가야 하냐고 한다"
  - "유저가 FOMO를 표현한다"
  - "커뮤니티 전체가 한 방향으로 몰려 있다"
```

### 디버깅 패턴

```
원칙: "연쇄 에러는 증상이 아니라 원인을 찾아라"
situations:
  - "하나 고치면 다른 에러가 나타난다"
  - "에러 메시지가 매번 다르지만 같은 곳에서 터진다"
  - "Quick fix를 3번 이상 했는데 문제가 안 끝난다"
  - "에러가 겉보기에 서로 관련 없어 보이는데 동시에 발생한다"
```

---

## References

- [conversation-axes.md](./conversation-axes.md) — 축 분류 체계, 리서치 근거
- [offline-prep-strategy.md](./offline-prep-strategy.md) — situation variant 생성 파이프라인
- [knowledge-graph-design.md](./knowledge-graph-design.md) — KG 아키텍처, Collection Router
- [medprompt.md](./medprompt.md) — Dynamic few-shot selection 원본 논문
