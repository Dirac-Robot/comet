# ☄️ CoMeT — Cognitive Memory Tree

**Lossless Hierarchical Memory System for LLM Agents**

CoMeT는 긴 대화를 구조화된 메모리 트리로 압축하는 시스템이다. 핵심 철학은 **"요약은 인덱스, 원본은 보존"** — 단순 요약이 유실하는 구체적 수치·고유명사·맥락을 원본 데이터로 온전히 유지하면서, 에이전트가 필요한 깊이에서만 정보를 열어볼 수 있도록 한다.

---

## 1. 설계 철학

### 1.1 기존 방식의 한계

| 방식 | 문제점 |
|------|--------|
| **Full Context Injection** | 100K+ 토큰을 매 질문마다 주입 → 비용 폭발, 레이턴시 증가, context 길이 제한 |
| **Naive Summarization** | 구체적 수치, 전문 용어, 고유명사가 유실 → 정밀 질의 시 정확도 50% 이하 |
| **Chunk-based RAG** | 의미 단위가 아닌 물리적 단위로 분할 → cross-topic 연관을 포착 못 함 |

### 1.2 CoMeT의 접근

- **논리적 분할**: 토큰 수가 아닌 인지 부하(topic shift, 정보 밀도)에 따라 압축 시점 결정
- **무손실 구조**: summary는 검색용 인덱스일 뿐, 원본 raw data는 항상 보존
- **동적 해상도(Dynamic Resolution)**: depth 0(summary) → depth 1(metadata) → depth 2(raw) 단계별 접근
- **Lazy Loading**: 요약으로 충분하면 raw를 열지 않음 → 토큰 절약

---

## 2. 시스템 아키텍처

```
User Input
    │
    ▼
┌─────────────┐    SLM (fast)     ┌───────────┐
│   Sensor    │ ───────────────▶ │ L1 Buffer │
│ (gpt-4o-m) │   entity/intent   └─────┬─────┘
└─────────────┘                         │ cognitive load trigger
                                        ▼
                                  ┌───────────┐
                                  │ Compacter │  LLM (slow, gpt-4o)
                                  └─────┬─────┘
                                        │ summary + trigger + recall_mode + tags
                                        ▼
                            ┌───────────┴───────────┐
                            │                       │
                      ┌───────────┐         ┌──────────────┐
                      │   Store   │         │  VectorIndex │  ChromaDB
                      │  (JSON)   │         │  (3 colls)   │  summary / trigger / raw
                      └───────────┘         └──────┬───────┘
                                                   │ semantic search
                                                   ▼
                                            ┌─────────────┐
                                            │  Retriever  │  RRF fusion
                                            └──────┬──────┘
                                                   │
                                            ┌──────┴──────┐
                                            │ Consolidator│  dedup + cross-link
                                            └─────────────┘
```

### 2.1 주요 컴포넌트

| 컴포넌트 | 파일 | 역할 | 사용 모델 |
|----------|------|------|-----------|
| **CognitiveSensor** | `sensor.py` | 각 턴에서 핵심 정보 추출, 인지 부하 판단 | SLM (gpt-4o-mini) |
| **MemoryCompacter** | `compacter.py` | L1 버퍼 → MemoryNode 구조화, auto-linking | Main LLM (gpt-4o) |
| **MemoryStore** | `storage.py` | JSON 기반 KV 스토어, 노드/raw 데이터 관리 | - |
| **VectorIndex** | `vector_index.py` | ChromaDB 3-collection 벡터 스토어 | text-embedding-3-small |
| **Retriever** | `retriever.py` | 쿼리 분석 + RRF 기반 score fusion | SLM (gpt-4o-mini) |
| **Consolidator** | `consolidator.py` | 세션 간 dedup, cross-link, tag 정규화 | - |
| **CoMeT** | `orchestrator.py` | 전체 파이프라인 오케스트레이션 + 에이전트 도구 제공 | - |

---

## 3. 핵심 메커니즘

### 3.1 Dual-Speed Layer

CoMeT는 두 개의 속도 레이어로 동작한다:

**Fast Layer (Sensor, SLM)**
```
Turn Input → L1Extraction(core_content, entities, intent) → L1 Buffer
          → CognitiveLoad(logic_flow, load_level, reasoning)
```

- 매 턴마다 핵심 정보를 1-2문장으로 추출
- `logic_flow`: MAINTAIN(맥락 유지) / BROKEN(주제 전환)
- `load_level`: 1~5 (정보 밀도/복잡도)
- 트리거 조건: `BROKEN` OR `load_level >= threshold` OR `buffer >= max`

**Slow Layer (Compacter, Main LLM)**
```
L1 Buffer → CompactedResult(summary, trigger, recall_mode, topic_tags) → MemoryNode
```

- L1 버퍼가 트리거되면 Main LLM이 구조화
- summary: 검색용 1줄 요약 (수치/날짜 제외)
- trigger: 이 정보가 필요한 상황 서술 ("내가 ~정보가 필요할 때")
  - 모호한 trigger 금지: "관련 정보를 찾을 때" 등은 무효
  - 수치/날짜/비율이 있으면 반드시 trigger에 포함
- recall_mode: passive/active/both 자동 분류
- topic_tags: 기존 태그 재사용 우선 → 자동 양방향 링킹
- compaction_reason: 압축 트리거 원인 기록 (내부 메타데이터, LLM에 비노출)

### 3.2 MemoryNode 스키마

```python
class MemoryNode(BaseModel):
    node_id: str          # e.g. "mem_20260209_135636_14d204"
    depth_level: int      # 0=summary, 1=metadata, 2=raw
    recall_mode: str      # passive | active | both
    topic_tags: list[str] # 자동 링킹 기준
    summary: str          # 검색용 요약
    trigger: str          # 리트리벌 트리거
    content_key: str      # raw 데이터 포인터
    raw_location: str     # raw 파일 경로
    links: list[str]      # 연관 노드 ID 목록
    created_at: datetime
    compaction_reason: str | None  # topic_shift | high_load | buffer_overflow | forced | external
```

### 3.3 Recall Mode

| 모드 | 동작 | 예시 |
|------|------|------|
| `passive` | 항상 context window에 포함 | 사용자 신원, 영구 선호 |
| `active` | semantic search로만 접근 | 구체적 사실, 결정, 이벤트 |
| `both` | 항상 context + 검색 가능 | 핵심 제약 + 세부사항 |

### 3.4 Dual-Path RAG Retrieval

일반 RAG는 "내용이 무엇인가(WHAT)"만 검색한다. CoMeT는 **WHAT + WHEN** 두 경로로 검색한다:

```
Query
  │
  ├─ QueryAnalyzer (SLM)
  │     ├─ semantic_query: "WHAT을 찾는가"
  │     ├─ search_intent: "WHEN/WHY 필요한가"
  │     └─ risk_level: low/medium/high (요약만으로 답변 시 정확도 위험도)
  │
  ├─ Summary Collection ← semantic_query   (WHAT 매칭)
  ├─ Trigger Collection ← search_intent    (WHEN 매칭)
  └─ Raw Collection     ← both queries     (fallback)
          │
          ▼
  ScoreFusion (RRF)
          │
          ▼
  Top-K Results + Auto-linked Nodes
  + Risk Signal (high → raw 확인 강제, low → 요약 응답 허용)
```

**ScoreFusion 공식:**
```
combined_score = RRF_score × 0.6 + similarity_score × 0.4

RRF_score = Σ weight_i × (1 / (k + rank_i + 1))
  - weight_summary = α × scale
  - weight_trigger = (1-α) × scale
  - weight_raw = raw_search_weight (default 0.2)
```

### 3.5 Topic-Aware Auto-Linking

Compacter가 새 노드를 생성할 때, 기존 노드들의 topic_tags와 비교하여 겹치는 태그가 있으면 양방향 링크를 자동 생성한다. 이를 통해:

- 하나의 검색 결과에서 연관 노드들이 자동으로 따라옴 (retriever가 links 순회)
- cross-topic 질의(예: "A 얘기랑 B 얘기의 공통점은?")에 높은 정확도

### 3.6 Lazy Loading

`retrieve_memory` 도구는 **summary, trigger, tags, links만 반환**하고 raw content는 제외한다. 에이전트가 요약만으로 답변 가능하면 `read_memory_node`를 호출하지 않아 토큰을 절약한다.

```
Agent 질문 수신
  │
  ├─ retrieve_memory(summary_query, trigger_query)
  │     → 노드 요약 + trigger만 반환
  │
  ├─ 요약으로 충분? → 바로 답변 (read 0회)
  │
  └─ 상세 정보 필요? → read_memory_node(node_id)
        → raw 데이터 반환
```

### 3.7 Consolidation Pipeline

세션 종료 시 `close_session()` 또는 수동 `consolidate()` 호출로 다음 3단계를 실행:

1. **Dedup**: VectorIndex에서 유사도 > 0.32인 노드 쌍 탐지 → 오래된 노드에 병합
2. **Trigger 재생성**: 병합된 노드의 trigger를 SLM이 재생성 (두 trigger의 정보 통합)
3. **Cross-Link**: 유사도 > 0.15인 비중복 노드 간 양방향 링크 생성
4. **Tag Normalization**: 대소문자/부분 문자열 기반 태그 통합

---

## 4. Agent Tool Interface

CoMeT는 LangChain 호환 도구 4종을 제공한다:

| 도구 | 설명 | 토큰 비용 |
|------|------|----------|
| `get_memory_index()` | 모든 노드의 ID, summary, trigger 목록 | 중간 |
| `read_memory_node(node_id)` | 특정 노드의 raw 데이터 전체 반환 | 높음 |
| `search_memory(tag)` | topic tag로 노드 검색 | 낮음 |
| `retrieve_memory(summary_query, trigger_query)` | 의미 검색 → summary/trigger만 반환 (lazy) | 중간 |

**Lazy Loading 전략의 에이전트 가이드:**
1. `retrieve_memory`로 관련 노드 검색
2. 반환된 요약으로 답변 시도
3. 요약으로 부족할 때만 `read_memory_node`로 원본 확인

---

## 5. 벤치마크

### 5.1 실험 환경

- **데이터**: 8개 실제 대화 (conversations.json에서 선별)
- **규모**: 119 turns, 100,752 chars
- **모델**: gpt-4.1 (답변 생성), gpt-4o-mini (sensor/query analyzer)
- **질문**: 20개 (10 single-topic + 10 cross-conversation complex)
- **벤치마크**: `benchmark_real.py` + `measure_cost.py`

### 5.2 4-Mode 비교 결과

| 지표 | Full Context | Naive Summary | CoMeT Session | CoMeT RAG |
|------|---|---|---|---|
| **정확도** | **19/20** | 10/20 | 18/20 | **19/20** |
| **평균 레이턴시** | 25.7s | 2.7s | 9.1s | 11.3s |
| **평균 토큰/질문** | 54,219 | 4,107 | **2,903** | 9,642 |
| **총 토큰 (20Q)** | 1,084,398 | 82,152 | **58,072** | 192,846 |
| **비용 (20Q)** | **$2.2717** | $0.1819 | **$0.1677** | $0.4350 |

### 5.3 분석

#### 정확도 vs 비용 효율

```
Full Context:   ████████████████████ 19/20  $2.27   ← 가장 정확하지만 13.5배 비쌈
CoMeT RAG:      ████████████████████ 19/20  $0.44   ← 동일 정확도, 5.2× 저렴
CoMeT Session:  ██████████████████   18/20  $0.17   ← 가성비 최강, 13.5× 저렴
Naive Summary:  ██████████           10/20  $0.18   ← 저렴하지만 50% 정확도
```

#### 토큰 절감 효과

| 비교 | 토큰 절감 배율 | 비용 절감 배율 | 정확도 차이 |
|------|--------------|--------------|------------|
| **CoMeT Session vs Full Context** | **18.7×** | **13.5×** | -1문제 |
| **CoMeT RAG vs Full Context** | **5.6×** | **5.2×** | 동일 |
| **CoMeT Session vs Naive Summary** | 1.4× (더 작음) | 1.1× | **+8문제** |

#### Lazy Loading 효과

20개 질문 중 RAG 도구 사용 패턴:

| 패턴 | 질문 수 | 설명 |
|------|---------|------|
| summary만으로 답변 (read=0) | **6/20** | 요약으로 충분한 경우 |
| read 1회 | 11/20 | 특정 사실 확인 필요 |
| read 2~3회 | 3/20 | cross-topic 심층 탐색 |

총 tool calls: retrieve=23, read=16 (평균 1.9회/질문)

#### 레이턴시 특성

- **Naive Summary**: 2.7s — 매우 빠르지만 정확도 미달
- **CoMeT Session**: 9.1s — context window 크기가 작아 LLM 처리 빠름
- **CoMeT RAG**: 11.3s — embedding search + tool roundtrip 오버헤드
- **Full Context**: 25.7s — 54K 토큰을 매번 처리

### 5.4 실패 케이스 분석

모든 모드에서 공통 실패한 Q18 (haystack 실험 수치):
- 질문: "haystack 실험에서 문장 수 10→20→30으로 늘렸을 때 전통 RAG 정확도 변화"
- 원인: 구체적 수치 변화가 원본 대화에서도 명시적이지 않거나, compaction 시 수치 범위가 trigger에 반영되지 않음

CoMeT Session에서 추가 실패한 Q03:
- 페르소나 일관성 테스트의 구체적 수치 — context window의 summary에 수치가 생략됨
- RAG에서는 `read_memory_node`로 원본 확인하여 해결

---

## 6. 최적화 기록

### 6.1 인지 부하 기반 Compaction

초기에는 고정 턴 수로 분할했으나, 이는 의미 단위가 아닌 물리적 단위였다. 현재는:
- `logic_flow == BROKEN` (주제 전환 감지)
- `load_level >= threshold` (정보 밀도 임계치)
- `buffer_size >= max_l1_buffer` (최대 버퍼 오버플로우)

이 세 가지 조건의 OR로 compaction 시점을 결정한다. 최소 버퍼 크기(`min_l1_buffer=3`)도 설정하여 너무 잦은 compaction을 방지한다.

### 6.2 Triple-Path Embedding

초기에는 summary/trigger 2-path였으나, raw content embedding을 추가:
- raw embedding은 summary/trigger가 포착하지 못한 구체적 키워드 매칭에 유효
- fusion weight에서 raw는 0.2 비중 (summary/trigger가 주력)

### 6.3 Summary → Trigger 분리

기존 RAG가 "내용(WHAT)"만 검색하던 것을, "언제 필요한가(WHEN)"를 분리하여 검색 정확도 향상:
- trigger는 LLM 관점에서 작성 ("내가 ~정보가 필요할 때")
- 사용자가 명시적으로 요청하지 않아도 맥락적으로 연관된 정보를 검색

### 6.4 Lazy Loading RAG

기존에는 `retrieve_memory`가 raw content를 포함하여 반환 → 대부분 불필요한 raw를 읽음.
현재는 summary/trigger/tags/links만 반환하고, 에이전트가 판단하여 필요시에만 `read_memory_node` 호출.

**효과**: 30% 질문에서 read 없이 답변 완성, 총 토큰 사용량 감소

### 6.5 벤치마크 캐싱

`benchmark_real.py`에 `.bench_cache` 고정 경로 캐싱 도입:
- 첫 실행 시 ingestion 결과를 `.bench_cache/`에 저장
- `--phase2-only` 옵션으로 이후 실행 시 ingestion 스킵
- 실행 시간: ~30분 → ~3분 (10배 단축)

---

## 7. 압축 통계

119 turns (100,752 chars) → 33 MemoryNodes

| 지표 | 값 |
|------|---|
| **입력 턴** | 119 |
| **생성 노드** | 33 |
| **구조적 압축률** | 72.3% (119 → 33) |
| **원본 문자 수** | 100,752 chars |
| **CoMeT context window** | ~2,500 chars per question |
| **문자 압축률** | ~97.5% |
| **평균 노드당 턴** | ~3.6 |
| **누적 링크 수** | topic tag 기반 자동 생성 |

---

## 8. Configuration

[ato](https://github.com/Dirac-Robot/ato) 기반 scope 설정:

```python
@scope.observe(default=True)
def default(config):
    config.slm_model = 'gpt-4o-mini'          # Sensor, QueryAnalyzer
    config.main_model = 'gpt-4o'              # Compacter
    config.compacting.load_threshold = 3       # CognitiveLoad 임계치
    config.compacting.max_l1_buffer = 5        # L1 버퍼 최대 크기
    config.compacting.min_l1_buffer = 3        # 최소 compaction 단위

    config.retrieval.embedding_model = 'text-embedding-3-small'
    config.retrieval.vector_backend = 'chroma'
    config.retrieval.vector_db_path = './memory_store/vectors'
    config.retrieval.top_k = 5
    config.retrieval.fusion_alpha = 0.6        # summary vs trigger 가중치
    config.retrieval.rrf_k = 5                 # RRF k 파라미터
    config.retrieval.raw_search_weight = 0.2   # raw embedding 가중치
```

---

## 9. 프로젝트 구조

```
comet/
├── orchestrator.py      # CoMeT 메인 클래스 — 전체 파이프라인 오케스트레이션
├── sensor.py            # CognitiveSensor — L1 추출 + 인지 부하 판단 (SLM)
├── compacter.py         # MemoryCompacter — L1→MemoryNode 구조화 (Main LLM)
├── storage.py           # MemoryStore — JSON KV 스토어 + 노드/raw 관리
├── schemas.py           # MemoryNode, CognitiveLoad, L1Memory, RetrievalResult
├── config.py            # ato scope 설정
├── vector_index.py      # VectorIndex — ChromaDB triple-collection 벡터 스토어
├── retriever.py         # Retriever — QueryAnalyzer + ScoreFusion + 링크 순회
├── consolidator.py      # Consolidator — dedup + cross-link + tag 정규화
└── templates/
    ├── compacting.txt      # 메모리 구조화 프롬프트
    ├── cognitive_load.txt  # 인지 부하 판단 프롬프트
    ├── l1_extraction.txt   # L1 정보 추출 프롬프트
    └── query_analysis.txt  # 쿼리 분해 프롬프트

benchmark_real.py           # 119턴 리얼 벤치마크 (Phase 1 Session + Phase 2 RAG)
measure_cost.py             # 4모드 토큰/레이턴시/비용 측정
```

---

## 10. 결론

CoMeT는 **"요약은 인덱스, 원본은 보존"** 원칙 하에:

1. **Full Context와 동일한 정확도(19/20)를 5.2배 저렴하게** 달성 (CoMeT RAG)
2. **18.7배 토큰 절감으로 93% 정확도** 유지 (CoMeT Session Memory)
3. **Naive Summarization 대비 정확도 2배** — 단순 요약의 정보 유실 문제 해결
4. **Lazy Loading으로 30% 질문에서 raw 접근 불필요** — 추가 토큰 절약

핵심 차별점은 인지 부하 기반 논리적 분할, summary+trigger dual-path 검색, topic tag 기반 자동 링킹의 조합이며, 이를 통해 cross-topic 복합 질의에서도 높은 정확도를 유지한다.
