"""CoMeT Agent Retrieval Test: Can an agent find answers using CoMeT tools?

Tests whether the CoMeT summary+trigger system provides enough information
for an agent to locate and retrieve specific facts from raw memory.
"""
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from comet import CoMeT, scope


TEST_DATA = Path(__file__).parent/'test_data.json'
MAX_TURNS = 15

QUESTIONS = [
    {
        'q': '월드모델에 필요한 GPU 파라미터 규모가 어느 정도라고 했어?',
        'expected': ['2B', '8B'],
    },
    {
        'q': 'video 모델과 world 모델 사이의 수학적 관계가 뭐라고 했어?',
        'expected': ['superset'],
    },
    {
        'q': 'N-of-1 신호가 약한 이유를 뭐라고 설명했어?',
        'expected': ['multi-map'],
    },
]


@scope
def main(config):
    config.storage.base_path = './memory_store_test'
    config.storage.raw_path = './memory_store_test/raw'

    memo = CoMeT(config)

    existing = memo.list_memories()
    if not existing:
        print('[Setup] No existing memories. Building from test data...')
        with open(TEST_DATA, 'r', encoding='utf-8') as f:
            messages = json.load(f)[:MAX_TURNS]

        for content in messages:
            memo.add(content)
        memo.force_compact()
        print(f'[Setup] Created {len(memo.list_memories())} nodes')
    else:
        print(f'[Setup] Found {len(existing)} existing nodes')

    tools = memo.get_tools()
    agent_llm = ChatOpenAI(model=config.main_model)
    agent = create_react_agent(agent_llm, tools)

    system_prompt = (
        'You are a memory retrieval agent. '
        'You have access to a memory system. Use get_memory_index to see available memory nodes, '
        'then use read_memory_node to read the raw conversation data from relevant nodes. '
        'Answer the user\'s question based ONLY on the information found in memory. '
        'Always retrieve the raw data before answering — summaries alone are not enough. '
        'Answer in Korean.'
    )

    print('=' * 60)
    print('[Agent Retrieval Test]')
    print(f'Agent LLM: {config.main_model}')
    print(f'Memory nodes: {len(memo.list_memories())}')
    print(f'Tools: {[t.name for t in tools]}')
    print('=' * 60)

    results = []
    for i, test in enumerate(QUESTIONS, 1):
        question = test['q']
        expected = test['expected']

        print(f'\n[Q{i}] {question}')
        print(f'  Expected keywords: {expected}')

        response = agent.invoke({
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': question},
            ]
        })

        answer = response['messages'][-1].content
        print(f'  Answer: {answer[:200]}')

        found = all(kw.lower() in answer.lower() for kw in expected)
        results.append(found)
        print(f'  Result: {"✅ PASS" if found else "❌ FAIL"}')

        tool_calls = [
            m for m in response['messages']
            if hasattr(m, 'tool_calls') and m.tool_calls
        ]
        if tool_calls:
            for tc_msg in tool_calls:
                for tc in tc_msg.tool_calls:
                    print(f'    Tool: {tc["name"]}({tc["args"]})')

    print('\n' + '=' * 60)
    passed = sum(results)
    total = len(results)
    print(f'[Final] {passed}/{total} passed')
    print('=' * 60)


if __name__ == '__main__':
    main()
