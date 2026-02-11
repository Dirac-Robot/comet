"""Benchmark Real: Real ChatGPT conversations + cross-topic complex questions.

Replaces synthetic benchmark data with actual user/assistant turns from
multiple real conversations, then evaluates CoMeT with harder questions
that require cross-conversation recall.

Conversations used (119 turns total):
  [C1] KTO BCO DPO 비교 — ML training hybrid strategies
  [C2] 알고리즘 신뢰성 판단 — RAG selection algorithm reliability
  [C3] B200 활용 아이디어 — GPU hardware & project ideas
  [C4] 조직 레포 clone 문제 — Git SSH troubleshooting
  [C5] 직장 내 괴롭힘 기준 — Workplace law & harassment criteria
  [C6] WritingBench 8.3 점수 의미 — LLM benchmark scoring
  [C7] LLM 툴로 AI 만들기 — LLM engineering solo builds
  [C8] LLM 벤치마크 라이브러리 — Benchmark framework selection
"""
import json
import os
import shutil
import sys
import tempfile
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from loguru import logger

from comet import CoMeT, scope
from comet.vector_index import VectorIndex

sys.setrecursionlimit(10000)


# ─── Real conversation extraction ─────────────────────────────

def extract_turns(json_path: str, selected_indices: list[int]) -> list[str]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    def get_main_thread(mapping):
        root_id = None
        for nid, node in mapping.items():
            if node.get('parent') is None:
                root_id = nid
                break
        if not root_id:
            return []
        def walk(node_id):
            node = mapping[node_id]
            msg = node.get('message')
            children = node.get('children', [])
            result = []
            if msg:
                content = msg.get('content', {})
                content_type = content.get('content_type', '')
                role = msg.get('author', {}).get('role', '')
                hidden = msg.get('metadata', {}).get('is_visually_hidden_from_conversation', False)
                if content_type == 'text' and role in ('user', 'assistant') and not hidden:
                    parts = content.get('parts', [])
                    text = ''.join(str(p) for p in parts if isinstance(p, str)).strip()
                    if text:
                        result.append((role, text))
            if children:
                result.extend(walk(children[0]))
            return result
        return walk(root_id)

    all_turns = []
    for idx in selected_indices:
        conv = data[idx]
        title = conv.get('title', '(no title)')
        thread = get_main_thread(conv.get('mapping', {}))
        for role, text in thread:
            if role == 'assistant' and len(text) > 2000:
                text = text[:2000]
            prefix = '[User]' if role == 'user' else '[Assistant]'
            all_turns.append(f'{prefix} {text}')

    return all_turns


# Conversations to use (indices in conversations.json)
SELECTED_CONVS = [7, 14, 32, 33, 49, 56, 62, 64]


# ─── Complex cross-conversation questions ─────────────────────
# These are designed to feel like natural recall:
# "아 맞다 이게 뭐였지?" / "흠 그러고보니 이거랑 이걸 결합해보면.."

QUESTIONS = [
    # ── Single-topic recall (within one conversation) ──
    {
        'q': '아 맞다, DPO보다 KTO를 먼저 추천한 이유가 뭐였지? 기하 리워드랑 관련 있었던 거 같은데.',
        'expected': ['unpaired', '스칼라', 'KTO'],
        'topic': 'c1_kto_reason',
    },
    {
        'q': 'GRPO를 주엔진으로 쓰고 DPO를 가드레일로 쓰는 구조 얘기했었는데, 왜 이 구조가 노블티 있다고 했지?',
        'expected': ['역할 분해', '가드레일', '주엔진'],
        'topic': 'c1_hybrid_novelty',
    },
    {
        'q': '페르소나 일관성 테스트에서 밸런스 게임 4개 질문에 리샘플 6개씩 했을 때 결과가 어땠어?',
        'expected': ['거의 전부 같은', '동일', '수렴'],
        'topic': 'c2_persona_consistency',
    },
    {
        'q': 'RAG에서 20개 중 19개가 노이즈일 때 알고리즘 신뢰성 판단하는 기준이 뭐였지?',
        'expected': ['exclusion', '배제', 'retention'],
        'topic': 'c2_rag_reliability',
    },
    {
        'q': 'B200 4대로 할 수 있는 재밌는 프로젝트 얘기했었는데 뭐 나왔었어?',
        'expected': ['B200', '4대'],
        'topic': 'c3_b200_ideas',
    },
    {
        'q': 'Git clone이 안 됐던 이유가 뭐였더라? organization 레포에서.',
        'expected': ['SSH', 'clone', 'permission', '권한'],
        'topic': 'c4_git_clone',
    },
    {
        'q': '직장 상사한테 공개 자리에서 지적하는 게 괴롭힘인지 아닌지 논의했었는데, 핵심 기준이 뭐였지?',
        'expected': ['업무', '적정성', '인격', '모욕'],
        'topic': 'c5_harassment_criteria',
    },
    {
        'q': 'WritingBench에서 8.3점이 나왔는데 리더보드랑 비교해서 어느 정도 수준이었어?',
        'expected': ['8.3', 'WritingBench', '100점'],
        'topic': 'c6_writingbench',
    },
    {
        'q': '데이터 있고 LLM 툴만으로 AI 만드는 게 쉽냐는 질문에 결론이 어땠지?',
        'expected': ['쉽', '어렵', '엔지니어', 'MVP'],
        'topic': 'c7_llm_tools_easy',
    },
    {
        'q': 'LLM 벤치마크 통합 프레임워크 추천 받았을 때 사실상 표준이라고 한 게 뭐였지?',
        'expected': ['lm-evaluation-harness', 'EleutherAI', 'harness'],
        'topic': 'c8_benchmark_lib',
    },
    # ── Cross-conversation complex questions ──
    {
        'q': '아 그러고보니, 기하 리워드로 GRPO 쓴다는 얘기랑 RAG 신뢰성 판단 얘기 둘 다 "배제 기반 판단"이 핵심이었잖아. 두 맥락에서 각각 어떤 식으로 배제를 썼었지?',
        'expected': ['exclusion', 'GRPO', 'veto', '배제', '가드레일'],
        'topic': 'cross_c1c2_exclusion',
    },
    {
        'q': '밸런스 게임 페르소나 테스트로 모델 안정성 확인하고, WritingBench로 점수 재는 거, 두 가지 다 모델 평가 방법이긴 한데 측정하는 게 다르잖아. 차이가 뭐였지?',
        'expected': ['페르소나', '의사결정', '글쓰기', 'writing', '일관성'],
        'topic': 'cross_c2c6_eval_diff',
    },
    {
        'q': '1인 빌드로 페르소나 모델을 밑바닥부터 만드는 데 기간이 얼마나 걸린다 그랬지? 그리고 그거랑 "LLM 툴만으로 AI 만들기 쉽다" 얘기는 어떻게 관련 있어?',
        'expected': ['3', '5개월', '6', '10주', '데이터', '파이프라인'],
        'topic': 'cross_c2c7_solo_build',
    },
    {
        'q': 'B200 같은 고성능 GPU가 있으면 GRPO+DPO 하이브리드 학습은 실용적이야? 리소스 측면에서 두 대화 연결해서 생각해봐.',
        'expected': ['B200', 'GRPO', 'DPO', '학습', 'GPU'],
        'topic': 'cross_c1c3_gpu_training',
    },
    {
        'q': '벤치마크 프레임워크로 Hydra 추천 받았었잖아. 근데 GCRI라는 커스텀 러너도 있었는데, 왜 기존 프레임워크에 직접 통합하면 안 되고 브릿지를 만들어야 한다 그랬지?',
        'expected': ['태스크 타입', '실행 모델', 'Hydra', '브릿지', '오케스트레이터'],
        'topic': 'cross_c8_hydra_bridge',
    },
    {
        'q': '직장 내 괴롭힘 기준 논의에서 "공개적 지적"이 합당한 경우의 조건이랑, RAG에서 노이즈를 공개적으로 배제하는 로직이랑 비유적으로 비슷한 점이 있지 않아? 각각의 판단 기준을 비교해봐.',
        'expected': ['업무', '적정', '근거', 'exclusion', '배제'],
        'topic': 'cross_c5c2_analogy',
    },
    {
        'q': '상용 LLM이 맥락은 잘 맞추는데 말투가 개판이고, 파인튜닝 모델은 말투만 잘 따라한다 그랬는데, 이 두 가지를 SFT 데이터 만들 때 어떻게 결합한다고 했었지?',
        'expected': ['플래너', '리얼라이저', '역할 분리', '상용', '파인튜닝'],
        'topic': 'cross_c1_sft_combine',
    },
    {
        'q': '아 맞다, haystack 실험에서 문장 수 10→20→30으로 늘렸을 때 전통 RAG는 noise reduction이 어떻게 변했고, ERAS(E)는 어떻게 됐었지?',
        'expected': ['86', '94', '96', '1.0', '전부'],
        'topic': 'c2_haystack_scaling',
    },
    {
        'q': '페르소나 모델 1인 빌드에서 현실적 기간 3~5개월이라고 했는데, 이때 병목이 모델이 아니라 뭐라고 했지?',
        'expected': ['시스템', '파이프라인', '데이터'],
        'topic': 'c2_bottleneck',
    },
    {
        'q': 'lm-evaluation-harness가 벤치마크 표준이라고 했으면서, 실제로 GCRI 같은 에이전트를 평가할 땐 왜 부적합하다고 했어? 그러면 inspect-evals는 어딨어?',
        'expected': ['에이전트', '태스크', '실행', 'inspect', '통합'],
        'topic': 'cross_c6c8_eval_limit',
    },
]


def check_answer(answer: str, expected: list[str]) -> bool:
    a = answer.lower()
    return any(kw.lower() in a for kw in expected)


@scope
def main(config):
    phase2_only = '--phase2-only' in sys.argv
    if phase2_only:
        sys.argv.remove('--phase2-only')

    cache_dir = os.path.join(os.path.dirname(__file__), '.bench_cache')
    os.makedirs(cache_dir, exist_ok=True)
    config.storage.base_path = f'{cache_dir}/store'
    config.storage.raw_path = f'{cache_dir}/store/raw'
    config.retrieval.vector_db_path = f'{cache_dir}/vectors'

    logger.info('Extracting real conversations...')
    turns = extract_turns('comet/conversations.json', SELECTED_CONVS)
    n_turns = len(turns)
    n_questions = len(QUESTIONS)
    full_context = '\n'.join(f'[Turn {i+1}] {t}' for i, t in enumerate(turns))
    full_chars = len(full_context)

    logger.info(f'Extracted {n_turns} turns ({full_chars:,} chars) from {len(SELECTED_CONVS)} conversations')

    llm = ChatOpenAI(model='gpt-4.1')
    slm = ChatOpenAI(model=config.slm_model)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Session Memory Benchmark
    # ═══════════════════════════════════════════════════════════════
    print('=' * 70)
    print(f'PHASE 1: Session Memory ({n_turns} real turns, {n_questions} questions)')
    print('=' * 70)

    if not phase2_only:
        # ── 1A: Full Context Injection ────────────────────────────────
        print('\n--- [1A] Full Context Injection ---')
        t0 = time.time()
        full_results = []
        for i, q in enumerate(QUESTIONS, 1):
            a = llm.invoke(
                f'아래 대화 기록을 보고 질문에 답해줘. 기록에 없으면 "정보 없음"이라고 해.\n\n'
                f'## 대화 기록\n{full_context}\n\n## 질문\n{q["q"]}'
            ).content
            hit = check_answer(a, q['expected'])
            full_results.append(hit)
            print(f'  Q{i:02d} {"✅" if hit else "❌"} {q["q"][:55]}')
        full_time = time.time()-t0
        print(f'  Context: {full_chars:,} chars | Accuracy: {sum(full_results)}/{n_questions}')
        print(f'  Time: {full_time:.1f}s')

        # ── 1B: Naive Summary ─────────────────────────────────────────
        print('\n--- [1B] Naive Summary ---')
        t0 = time.time()
        naive_summary = llm.invoke(
            '다음 대화 기록을 핵심 내용 위주로 요약해줘. '
            '구체적인 숫자, 전문 용어, 수식, 고유명사, 약어를 반드시 원문 그대로 포함해서 요약해.\n\n'
            f'{full_context}'
        ).content
        naive_chars = len(naive_summary)
        naive_results = []
        for i, q in enumerate(QUESTIONS, 1):
            a = llm.invoke(
                f'아래 요약본만 보고 질문에 답해줘. 요약에 없는 내용은 "정보 없음"이라고 해.\n\n'
                f'## 요약\n{naive_summary}\n\n## 질문\n{q["q"]}'
            ).content
            hit = check_answer(a, q['expected'])
            naive_results.append(hit)
            print(f'  Q{i:02d} {"✅" if hit else "❌"} {q["q"][:55]}')
        naive_time = time.time()-t0
        print(f'  Context: {naive_chars:,} chars ({naive_chars/full_chars*100:.1f}%) | Accuracy: {sum(naive_results)}/{n_questions}')
        print(f'  Time: {naive_time:.1f}s')

    # ── 1C: CoMeT Session Memory ──────────────────────────────────
    cache_index = os.path.join(cache_dir, 'store', 'index.json')
    ingestion_cached = os.path.exists(cache_index)

    print('\n--- [1C] CoMeT Session Memory ---')
    t0 = time.time()
    memo = CoMeT(config)

    if not ingestion_cached:
        logger.info('Ingesting turns (first run, will be cached)...')
        for content in turns:
            memo.add(content)
        memo.force_compact()
    else:
        logger.info(f'Using cached ingestion from {cache_dir}')

    nodes = memo.list_memories()
    comet_context = memo.get_context_window(max_nodes=50)
    comet_chars = len(comet_context)

    print(f'  Nodes: {len(nodes)} | VectorIndex: {memo._vector_index.count}')

    tools = memo.get_tools()

    if not phase2_only:
        agent = create_react_agent(llm, tools)
        sys_prompt = (
            'You are a memory retrieval agent. '
            'Use get_memory_index or retrieve_memory to find relevant nodes. '
            'The results contain summaries and triggers — try to answer from these first. '
            'Only call read_memory_node if the summaries lack the specific details needed. '
            'Follow linked nodes if the current node does not fully answer. '
            'Answer in Korean, preserving original English technical terms as-is.'
        )

        comet_results = []
        for i, q in enumerate(QUESTIONS, 1):
            response = agent.invoke({
                'messages': [
                    {'role': 'system', 'content': sys_prompt},
                    {'role': 'user', 'content': q['q']},
                ]
            })
            a = response['messages'][-1].content
            hit = check_answer(a, q['expected'])
            comet_results.append(hit)

            reads = sum(
                1 for m in response['messages']
                if hasattr(m, 'tool_calls') and m.tool_calls
                for tc in m.tool_calls if tc['name'] == 'read_memory_node'
            )
            print(f'  Q{i:02d} {"✅" if hit else "❌"} reads={reads} {q["q"][:50]}')
        comet_time = time.time()-t0
        print(f'  Context: {comet_chars:,} chars ({comet_chars/full_chars*100:.1f}%) | Accuracy: {sum(comet_results)}/{n_questions}')
        print(f'  Time: {comet_time:.1f}s')

    if not phase2_only:
        # ── Phase 1 Summary ───────────────────────────────────────────
        print('\n' + '=' * 70)
        print('PHASE 1 RESULTS: Session Memory')
        print('=' * 70)
        print(f'{"Method":<25} {"Chars":>8} {"Ratio":>8} {"Accuracy":>10} {"Time":>8}')
        print('-' * 70)
        print(f'{"Full Context":<25} {full_chars:>8,} {"100%":>8} {f"{sum(full_results)}/{n_questions}":>10} {f"{full_time:.0f}s":>8}')
        print(f'{"Naive Summary":<25} {naive_chars:>8,} {f"{naive_chars/full_chars*100:.1f}%":>8} {f"{sum(naive_results)}/{n_questions}":>10} {f"{naive_time:.0f}s":>8}')
        print(f'{"CoMeT":<25} {comet_chars:>8,} {f"{comet_chars/full_chars*100:.1f}%":>8} {f"{sum(comet_results)}/{n_questions}":>10} {f"{comet_time:.0f}s":>8}')

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: RAG Retrieval Benchmark
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print(f'PHASE 2: RAG Retrieval Benchmark ({n_questions} queries)')
    print('=' * 70)

    from openai import OpenAI as RawOpenAI
    raw_openai = RawOpenAI()

    if not phase2_only:
        # ── 2A: Naive RAG (chunk-level, same granularity as CoMeT) ───
        print('\n--- [2A] Naive RAG (chunk-level summary embed) ---')
        chunk_size = 4
        chunks = []
        for i in range(0, n_turns, chunk_size):
            chunk_turns = turns[i:i+chunk_size]
            chunks.append('\n'.join(chunk_turns))
        n_chunks = len(chunks)
        print(f'  Chunks: {n_chunks} (size={chunk_size})')

        chunk_summaries = []
        for chunk in chunks:
            summary = slm.invoke(
                '다음 대화를 핵심 내용 위주로 2-3문장으로 요약해줘. '
                '구체적인 숫자, 고유명사, 전문용어를 반드시 포함해.\n\n' + chunk
            ).content
            chunk_summaries.append(summary)

        chunk_embeddings = raw_openai.embeddings.create(
            model=config.retrieval.embedding_model,
            input=chunk_summaries,
        )
        chunk_vectors = [item.embedding for item in chunk_embeddings.data]

        naive_rag_results = []
        naive_rag_top3 = []
        for i, q in enumerate(QUESTIONS, 1):
            q_emb = raw_openai.embeddings.create(
                model=config.retrieval.embedding_model,
                input=q['q'],
            ).data[0].embedding

            similarities = []
            for j, cv in enumerate(chunk_vectors):
                sim = sum(a*b for a, b in zip(q_emb, cv))
                similarities.append((j, sim))
            similarities.sort(key=lambda x: -x[1])
            top_chunks = similarities[:5]

            top_context = '\n\n'.join(
                f'[Chunk {idx+1}] {chunks[idx]}' for idx, _ in top_chunks
            )
            a = llm.invoke(
                f'아래 관련 대화 조각들을 참고해서 질문에 답해줘.\n\n'
                f'## 관련 대화\n{top_context}\n\n## 질문\n{q["q"]}'
            ).content

            hit = check_answer(a, q['expected'])
            naive_rag_results.append(hit)

            top1_text = chunks[top_chunks[0][0]]
            top1_hit = any(kw.lower() in top1_text.lower() for kw in q['expected'])
            top3_texts = ' '.join(chunks[idx] for idx, _ in top_chunks[:3])
            top3_hit = any(kw.lower() in top3_texts.lower() for kw in q['expected'])
            naive_rag_top3.append(top3_hit)

            print(f'  Q{i:02d} {"✅" if hit else "❌"} top1={"✅" if top1_hit else "❌"} top3={"✅" if top3_hit else "❌"} {q["q"][:45]}')

    # ── 2B: CoMeT RAG (agent-based tool calling) ───────────────
    print(f'\n--- [2B] CoMeT RAG (agent tool calling) ---')
    print(f'  Vectors: {memo._vector_index.count}')

    rag_agent = create_react_agent(llm, tools)
    rag_sys_prompt = (
        'You are a memory retrieval agent. '
        'You have access to a structured memory system with these tools:\n'
        '- retrieve_memory: Semantic search returning summaries and triggers (lightweight).\n'
        '  "summary_query" = keywords describing WHAT you want to find.\n'
        '  "trigger_query" = a sentence describing WHY/WHEN you need this info.\n'
        '  Both fields are REQUIRED.\n'
        '- read_memory_node: Read full raw content of a specific node. Use ONLY when summaries lack details.\n'
        '- get_memory_index: List all nodes with summaries and triggers.\n'
        '- search_memory: Search by topic tag.\n\n'
        'Strategy: Use retrieve_memory first. Try to answer from the returned summaries. '
        'Only call read_memory_node if the summaries do not contain the specific facts/numbers needed. '
        'Answer in Korean, preserving original English technical terms as-is.'
    )

    comet_rag_results = []
    for i, q in enumerate(QUESTIONS, 1):
        response = rag_agent.invoke({
            'messages': [
                {'role': 'system', 'content': rag_sys_prompt},
                {'role': 'user', 'content': q['q']},
            ]
        })
        a = response['messages'][-1].content
        hit = check_answer(a, q['expected'])
        comet_rag_results.append(hit)

        retrieves = sum(
            1 for m in response['messages']
            if hasattr(m, 'tool_calls') and m.tool_calls
            for tc in m.tool_calls if tc['name'] == 'retrieve_memory'
        )
        reads = sum(
            1 for m in response['messages']
            if hasattr(m, 'tool_calls') and m.tool_calls
            for tc in m.tool_calls if tc['name'] == 'read_memory_node'
        )
        print(f'  Q{i:02d} {"✅" if hit else "❌"} retrieves={retrieves} reads={reads} {q["q"][:45]}')

    # ── Phase 2 Summary ───────────────────────────────────────────
    print('\n' + '=' * 70)
    print('PHASE 2 RESULTS: CoMeT RAG (agent tool calling)')
    print('=' * 70)
    print(f'{"CoMeT RAG Accuracy":<35} {f"{sum(comet_rag_results)}/{n_questions}":>12}')
    if not phase2_only:
        print(f'{"Naive RAG (chunk embed + LLM)":<35} {f"{sum(naive_rag_results)}/{n_questions}":>12}')

    if not phase2_only:
        # ═══════════════════════════════════════════════════════════════
        # Overall Summary
        # ═══════════════════════════════════════════════════════════════
        print('\n' + '=' * 70)
        print('OVERALL SUMMARY')
        print('=' * 70)
        print(f'Source: {len(SELECTED_CONVS)} real conversations')
        print(f'Turns: {n_turns} | Questions: {n_questions} (10 single-topic + 10 cross-topic)')
        print(f'CoMeT Nodes: {len(nodes)} | Vectors: {memo._vector_index.count}')
        print(f'Naive RAG Chunks: {n_chunks} (chunk_size={chunk_size})')
        print()
        print('Session Memory:')
        print(f'  Full Context:  {sum(full_results)}/{n_questions} ({full_chars:,} chars = 100%)')
        print(f'  Naive Summary: {sum(naive_results)}/{n_questions} ({naive_chars:,} chars = {naive_chars/full_chars*100:.1f}%)')
        print(f'  CoMeT:         {sum(comet_results)}/{n_questions} ({comet_chars:,} chars = {comet_chars/full_chars*100:.1f}%)')
        print()
        print('RAG Retrieval:')
        print(f'  Naive RAG:     {sum(naive_rag_results)}/{n_questions}')
        print(f'  CoMeT RAG:     {sum(comet_rag_results)}/{n_questions}')
        print('=' * 70)


if __name__ == '__main__':
    main()
