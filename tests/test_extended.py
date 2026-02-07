"""Extended Link Navigation Test: 52 turns from 5 conversations.

Tests whether CoMeT agent follows inter-node links to find
cross-topic answers that span multiple memory nodes.

Conversations:
  [32] B200 활용 아이디어 (world model, MCG, meta-context)
  [07] KTO BCO DPO 비교 (hybrid GRPO+DPO, geometric reward)
  [08] SFT vs DPO 차이 (persona chat, unpaired preference)
  [40] MoLoRA로 8B 한계 해결 (mixture of LoRAs, per-persona)
  [54] 멀티모달 모델 메커니즘 (CAD→DSL, vision encoder, DPO data)
"""
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from comet import CoMeT, scope


TEST_DATA = Path(__file__).parent/'test_data_extended.json'

# 10 questions designed to test link navigation:
# - Some require info from a single deep node (baseline)
# - Some require connecting info across conversations (cross-topic)
# - Some require following links from one node to a related one
QUESTIONS = [
    # --- Single-topic (within one conversation) ---
    {
        'q': 'video 모델과 world 모델의 수학적 관계를 수식으로 설명해봐.',
        'expected': ['f(s, t)', 'f(s, a, t)', 'superset', '상위'],
        'type': 'single',
    },
    {
        'q': 'MCT의 short/mid/long 구분은 AR에서 몇 프레임 기준이야?',
        'expected': ['1~10', '10~40', '40'],
        'type': 'single',
    },
    {
        'q': 'MoLoRA에서 공용 LoRA가 필요한 이유가 뭐야?',
        'expected': ['공통', '공용', '변하지 않는'],
        'type': 'single',
    },
    # --- Cross-topic (require linking across conversations) ---
    {
        'q': 'DPO와 GRPO를 하이브리드로 쓰자고 제안한 이유가 뭐야?',
        'expected': ['기하', '언어', 'geometric', 'language', '리워드'],
        'type': 'cross',
    },
    {
        'q': 'CAD→DSL 멀티모달 모델에서 DPO 데이터의 negative 케이스는 어떻게 구성했어?',
        'expected': ['syntax', 'visual confusion', '비전 인코더'],
        'type': 'cross',
    },
    {
        'q': '8B 모델의 크기 한계를 MoLoRA로 해결할 때 SFT와 DPO 단계는 어떻게 나눠?',
        'expected': ['LoRA', 'SFT', 'DPO'],
        'type': 'cross',
    },
    {
        'q': '페르소나 채팅에서 SFT 후 DPO vs KTO 중 뭐가 좋다고 했어?',
        'expected': ['DPO', 'KTO', 'unpaired', '맥락'],
        'type': 'cross',
    },
    {
        'q': 'MCG 개선에서 learned metric의 문제점이 뭐라고 했어?',
        'expected': ['sampling', '샘플링'],
        'type': 'single',
    },
    {
        'q': 'teacher forcing에 attention masking을 쓰자고 한 이유가 뭐야?',
        'expected': ['teacher forcing', 'attention', 'mask'],
        'type': 'single',
    },
    {
        'q': 'SFT 데이터를 만들 때 상용 LLM과 파인튜닝 모델의 차이가 뭐라고 했어?',
        'expected': ['말투', '맥락', '내용'],
        'type': 'cross',
    },
]


def check(answer: str, expected: list[str]) -> bool:
    a = answer.lower()
    return any(kw.lower() in a for kw in expected)


@scope
def main(config):
    config.storage.base_path = './memory_store_extended'
    config.storage.raw_path = './memory_store_extended/raw'

    with open(TEST_DATA, 'r', encoding='utf-8') as f:
        all_messages = json.load(f)

    llm = ChatOpenAI(model=config.main_model)
    n = len(all_messages)

    # ─── Phase 1: Naive ──────────────────────────────────────
    print('=' * 60)
    print(f'[Phase 1] Naive Summarizer — {n} turns')
    print('=' * 60)

    conv_text = '\n'.join(f'[Turn {i+1}] {m}' for i, m in enumerate(all_messages))
    summary = llm.invoke(
        '다음 대화 기록을 핵심 내용 위주로 요약해줘. '
        '구체적인 숫자, 전문 용어, 수식, 고유명사, 약어를 반드시 원문 그대로 포함해서 요약해.\n\n'
        f'{conv_text}'
    ).content

    print(f'[Summary] ({len(summary)} chars)')
    print(summary[:500] + '...\n')

    naive_results = []
    for i, t in enumerate(QUESTIONS, 1):
        a = llm.invoke(
            f'아래 요약본만 보고 질문에 답해줘. 요약에 없는 내용은 "정보 없음"이라고 해.\n\n'
            f'## 요약\n{summary}\n\n## 질문\n{t["q"]}'
        ).content
        found = check(a, t['expected'])
        naive_results.append(found)
        print(f'  Q{i:02d}[{t["type"]:6s}] {"✅" if found else "❌"} {t["q"][:45]}')

    # ─── Phase 2: CoMeT ────────────────────────────────────
    print('\n' + '=' * 60)
    print(f'[Phase 2] CoMeT Agent — {n} turns')
    print('=' * 60)

    memo = CoMeT(config)
    existing = memo.list_memories()
    if not existing:
        print('[Building memory...]')
        for content in all_messages:
            memo.add(content)
        memo.force_compact()
        existing = memo.list_memories()
    print(f'[Nodes: {len(existing)}]')

    # Show node structure
    for node_info in existing:
        node = memo._store.get_node(node_info['node_id'])
        if node:
            n_links = len(node.links)
            print(f'  {node.node_id}: {node.topic_tags} -> {n_links} links')

    tools = memo.get_tools()
    agent = create_react_agent(llm, tools)
    sys_prompt = (
        'You are a memory retrieval agent. '
        'Use get_memory_index first, then read_memory_node for relevant nodes. '
        'IMPORTANT: When you read a node, check "Linked nodes" in the output. '
        'If the current node does not fully answer the question, '
        'follow the links and read those connected nodes too. '
        'Do NOT answer from summaries alone — always read the raw data. '
        'Answer in Korean, preserving original English technical terms as-is.'
    )

    comet_results = []
    link_followed = []
    for i, t in enumerate(QUESTIONS, 1):
        response = agent.invoke({
            'messages': [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': t['q']},
            ]
        })
        a = response['messages'][-1].content
        found = check(a, t['expected'])
        comet_results.append(found)

        tc_list = [
            tc['name']
            for m in response['messages']
            if hasattr(m, 'tool_calls') and m.tool_calls
            for tc in m.tool_calls
        ]
        read_count = tc_list.count('read_memory_node')
        followed = read_count > 1
        link_followed.append(followed)

        indicator = '🔗' if followed else '  '
        print(f'  Q{i:02d}[{t["type"]:6s}] {"✅" if found else "❌"} {indicator} reads={read_count} {t["q"][:40]}')

    # ─── Results ─────────────────────────────────────────────
    print('\n' + '=' * 60)
    n_n = sum(naive_results)
    n_c = sum(comet_results)
    n_l = sum(link_followed)
    print(f'Naive: {n_n}/10 | CoMeT: {n_c}/10 | Links followed: {n_l}/10')
    print('-' * 60)
    for i, t in enumerate(QUESTIONS):
        n = '✅' if naive_results[i] else '❌'
        c = '✅' if comet_results[i] else '❌'
        l = '🔗' if link_followed[i] else '  '
        print(f'  Q{i+1:02d} {n} vs {c} {l}  [{t["type"]:6s}] {t["q"][:42]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
