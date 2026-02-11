"""Measure latency and token usage for all 4 modes.

Uses cached .bench_cache data for CoMeT — no ingestion needed.

Modes compared:
  1. Full Context Injection (baseline)
  2. Naive Summary
  3. CoMeT Session Memory (context window)
  4. CoMeT RAG (agent tool calling)
"""
import os
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from loguru import logger

from comet import CoMeT, scope
from benchmark_real import QUESTIONS, SELECTED_CONVS, extract_turns, check_answer


def measure_single_call(llm, prompt, q, expected):
    t0 = time.time()
    response = llm.invoke(prompt)
    latency = time.time()-t0
    hit = check_answer(response.content, expected)
    usage = response.usage_metadata or {}
    return {
        'hit': hit,
        'latency': latency,
        'prompt_tokens': usage.get('input_tokens', 0),
        'completion_tokens': usage.get('output_tokens', 0),
    }


def print_mode_results(name, results, n_questions):
    hits = sum(r['hit'] for r in results)
    lats = [r['latency'] for r in results]
    pt = sum(r['prompt_tokens'] for r in results)
    ct = sum(r['completion_tokens'] for r in results)
    total = pt+ct
    cost = pt*2/1_000_000 + ct*8/1_000_000

    print(f'\n  Accuracy: {hits}/{n_questions}')
    print(f'  Avg latency: {sum(lats)/n_questions:.2f}s (min={min(lats):.1f}s max={max(lats):.1f}s)')
    print(f'  Total tokens: prompt={pt:,} completion={ct:,} total={total:,}')
    print(f'  Avg tokens/q: prompt={pt//n_questions:,} completion={ct//n_questions:,} total={total//n_questions:,}')
    print(f'  Est. cost (gpt-4.1): ${cost:.4f}')
    return {'hits': hits, 'avg_lat': sum(lats)/n_questions, 'prompt': pt, 'completion': ct, 'total': total, 'cost': cost}


@scope
def main(config):
    cache_dir = os.path.join(os.path.dirname(__file__), '.bench_cache')
    cache_index = os.path.join(cache_dir, 'store', 'index.json')
    if not os.path.exists(cache_index):
        logger.error('No cached data found. Run benchmark_real.py first.')
        return

    config.storage.base_path = f'{cache_dir}/store'
    config.storage.raw_path = f'{cache_dir}/store/raw'
    config.retrieval.vector_db_path = f'{cache_dir}/vectors'

    turns = extract_turns('comet/conversations.json', SELECTED_CONVS)
    full_context = '\n'.join(f'[Turn {i+1}] {t}' for i, t in enumerate(turns))
    full_chars = len(full_context)

    llm = ChatOpenAI(model='gpt-4.1')
    slm = ChatOpenAI(model=config.slm_model)
    memo = CoMeT(config)
    nodes = memo.list_memories()
    tools = memo.get_tools()
    comet_context = memo.get_context_window(max_nodes=50)
    comet_chars = len(comet_context)
    n_questions = len(QUESTIONS)

    print(f'Source: {len(SELECTED_CONVS)} conversations, {len(turns)} turns, {full_chars:,} chars')
    print(f'CoMeT: {len(nodes)} nodes | {memo._vector_index.count} vectors | context {comet_chars:,} chars ({comet_chars/full_chars*100:.1f}%)')
    print()

    summaries = {}

    # ═══════════════════════════════════════════════════════════════
    # 1. Full Context Injection
    # ═══════════════════════════════════════════════════════════════
    print('=' * 80)
    print('[1] FULL CONTEXT INJECTION (baseline)')
    print('=' * 80)

    results_full = []
    for i, q in enumerate(QUESTIONS, 1):
        r = measure_single_call(
            llm,
            f'아래 대화 기록을 보고 질문에 답해줘. 기록에 없으면 "정보 없음"이라고 해.\n\n'
            f'## 대화 기록\n{full_context}\n\n## 질문\n{q["q"]}',
            q['q'], q['expected'],
        )
        results_full.append(r)
        print(f'  Q{i:02d} {"✅" if r["hit"] else "❌"} {r["latency"]:.1f}s  p={r["prompt_tokens"]:,} c={r["completion_tokens"]:,}  {q["q"][:40]}')

    summaries['Full Context'] = print_mode_results('Full Context', results_full, n_questions)

    # ═══════════════════════════════════════════════════════════════
    # 2. Naive Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('[2] NAIVE SUMMARY')
    print('=' * 80)

    print('  Generating summary...')
    t0 = time.time()
    summary_response = llm.invoke(
        '다음 대화 기록을 핵심 내용 위주로 요약해줘. '
        '구체적인 숫자, 전문 용어, 수식, 고유명사, 약어를 반드시 원문 그대로 포함해서 요약해.\n\n'
        f'{full_context}'
    )
    summary_time = time.time()-t0
    naive_summary = summary_response.content
    summary_usage = summary_response.usage_metadata or {}
    summary_prompt = summary_usage.get('input_tokens', 0)
    summary_comp = summary_usage.get('output_tokens', 0)
    print(f'  Summary: {len(naive_summary):,} chars in {summary_time:.1f}s (p={summary_prompt:,} c={summary_comp:,})')

    results_naive = []
    for i, q in enumerate(QUESTIONS, 1):
        r = measure_single_call(
            llm,
            f'아래 요약본만 보고 질문에 답해줘. 요약에 없는 내용은 "정보 없음"이라고 해.\n\n'
            f'## 요약\n{naive_summary}\n\n## 질문\n{q["q"]}',
            q['q'], q['expected'],
        )
        results_naive.append(r)
        print(f'  Q{i:02d} {"✅" if r["hit"] else "❌"} {r["latency"]:.1f}s  p={r["prompt_tokens"]:,} c={r["completion_tokens"]:,}  {q["q"][:40]}')

    # Add summary generation cost to first question's tokens
    results_naive[0]['prompt_tokens'] += summary_prompt
    results_naive[0]['completion_tokens'] += summary_comp
    summaries['Naive Summary'] = print_mode_results('Naive Summary', results_naive, n_questions)

    # ═══════════════════════════════════════════════════════════════
    # 3. CoMeT Session Memory
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('[3] CoMeT SESSION MEMORY (context window)')
    print('=' * 80)

    results_session = []
    for i, q in enumerate(QUESTIONS, 1):
        r = measure_single_call(
            llm,
            f'아래 요약된 메모리 정보를 보고 질문에 답해줘. 정보가 없으면 "정보 없음"이라고 해.\n\n'
            f'## 메모리\n{comet_context}\n\n## 질문\n{q["q"]}',
            q['q'], q['expected'],
        )
        results_session.append(r)
        print(f'  Q{i:02d} {"✅" if r["hit"] else "❌"} {r["latency"]:.1f}s  p={r["prompt_tokens"]:,} c={r["completion_tokens"]:,}  {q["q"][:40]}')

    summaries['CoMeT Session'] = print_mode_results('CoMeT Session', results_session, n_questions)

    # ═══════════════════════════════════════════════════════════════
    # 4. CoMeT RAG (agent tool calling)
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('[4] CoMeT RAG (agent tool calling)')
    print('=' * 80)

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

    results_rag = []
    for i, q in enumerate(QUESTIONS, 1):
        t0 = time.time()
        response = rag_agent.invoke({
            'messages': [
                {'role': 'system', 'content': rag_sys_prompt},
                {'role': 'user', 'content': q['q']},
            ]
        })
        latency = time.time()-t0

        a = response['messages'][-1].content
        hit = check_answer(a, q['expected'])

        pt = sum(
            (m.usage_metadata or {}).get('input_tokens', 0)
            for m in response['messages']
            if hasattr(m, 'usage_metadata') and m.usage_metadata
        )
        ct = sum(
            (m.usage_metadata or {}).get('output_tokens', 0)
            for m in response['messages']
            if hasattr(m, 'usage_metadata') and m.usage_metadata
        )
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

        results_rag.append({
            'hit': hit, 'latency': latency,
            'prompt_tokens': pt, 'completion_tokens': ct,
            'retrieves': retrieves, 'reads': reads,
        })
        print(
            f'  Q{i:02d} {"✅" if hit else "❌"} {latency:.1f}s  '
            f'p={pt:,} c={ct:,}  '
            f'ret={retrieves} read={reads}  '
            f'{q["q"][:35]}'
        )

    total_ret = sum(r['retrieves'] for r in results_rag)
    total_read = sum(r['reads'] for r in results_rag)
    summaries['CoMeT RAG'] = print_mode_results('CoMeT RAG', results_rag, n_questions)
    print(f'  Tool calls: retrieve={total_ret} read={total_read} (avg {(total_ret+total_read)/n_questions:.1f}/q)')

    # ═══════════════════════════════════════════════════════════════
    # Comparison Table
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('COMPARISON SUMMARY')
    print('=' * 80)
    header = f'{"":25} {"Accuracy":>10} {"Avg Lat":>10} {"Prompt":>12} {"Compl":>12} {"Total Tok":>12} {"Tok/Q":>10} {"Cost":>10}'
    print(header)
    print('-' * len(header))
    for name, s in summaries.items():
        print(
            f'{name:25} '
            f'{f"{s["hits"]}/{n_questions}":>10} '
            f'{f"{s["avg_lat"]:.2f}s":>10} '
            f'{s["prompt"]:>12,} '
            f'{s["completion"]:>12,} '
            f'{s["total"]:>12,} '
            f'{s["total"]//n_questions:>10,} '
            f'{f"${s["cost"]:.4f}":>10}'
        )
    print('=' * len(header))


if __name__ == '__main__':
    main()
