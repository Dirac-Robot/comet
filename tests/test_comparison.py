"""Extended Comparison: 31-turn conversation — Naive Summarizer vs CoMeT.

Tests with the full B200 conversation (31 user turns) including
deep-detail questions from the later turns (MCT, meta-context, gating, EMA).
"""
import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from comet import CoMeT, scope


TEST_DATA = Path(__file__).parent/'test_data.json'

QUESTIONS = [
    {
        'q': '월드모델에 필요한 파라미터 규모가 어느 정도라고 했어?',
        'expected': ['2B', '8B', '2b', '8b', '20억', '80억'],
    },
    {
        'q': 'video 모델과 world 모델의 수학적 관계가 뭐야?',
        'expected': ['superset', '상위 집합', '상위집합'],
    },
    {
        'q': 'meta-context token(MCT)의 short/mid/long 구분은 어떤 기준이야?',
        'expected': ['AR', 'auto-regressive', '프레임', '1~10', '10~40', '40'],
    },
    {
        'q': 'MCT 경계에서 차이가 안 나게 하려고 어떤 방법을 제안했어?',
        'expected': ['EMA'],
    },
    {
        'q': 'gating을 두겠다고 한 이유가 뭐야?',
        'expected': ['feature', '피처', '가져갈', '냅둘'],
    },
]


def check(answer: str, expected: list[str]) -> bool:
    a = answer.lower()
    return any(kw.lower() in a for kw in expected)


@scope
def main(config):
    config.storage.base_path = './memory_store_full'
    config.storage.raw_path = './memory_store_full/raw'

    with open(TEST_DATA, 'r', encoding='utf-8') as f:
        all_messages = json.load(f)

    llm = ChatOpenAI(model=config.main_model)
    n_turns = len(all_messages)

    # ─── Phase 1: Naive Summarizer ───────────────────────────
    print('=' * 60)
    print(f'[Phase 1] Naive Summarizer — {n_turns} turns')
    print('=' * 60)

    conversation_text = '\n'.join(
        f'[Turn {i+1}] {msg}' for i, msg in enumerate(all_messages)
    )

    summary = llm.invoke(
        '다음 대화 기록을 핵심 내용 위주로 요약해줘. '
        '구체적인 숫자, 전문 용어, 수식, 고유명사, 약어를 반드시 원문 그대로 포함해서 요약해.\n\n'
        f'{conversation_text}'
    ).content

    print(f'\n[Summary] ({len(summary)} chars)')
    print(summary[:600])
    if len(summary) > 600:
        print('...')

    print('\n' + '-' * 60)
    naive_results = []
    for i, t in enumerate(QUESTIONS, 1):
        answer = llm.invoke(
            f'아래 요약본만 보고 질문에 답해줘. 요약에 없는 내용은 "정보 없음"이라고 해.\n\n'
            f'## 요약\n{summary}\n\n## 질문\n{t["q"]}'
        ).content
        found = check(answer, t['expected'])
        naive_results.append(found)
        print(f'[Q{i}] {t["q"]}')
        print(f'  A: {answer[:150]}')
        print(f'  {"✅" if found else "❌"}')

    # ─── Phase 2: CoMeT Agent ──────────────────────────────
    print('\n' + '=' * 60)
    print(f'[Phase 2] CoMeT Agent — {n_turns} turns')
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

    tools = memo.get_tools()
    agent = create_react_agent(llm, tools)
    sys_prompt = (
        'You are a memory retrieval agent. '
        'Use get_memory_index first, then read_memory_node for ALL potentially relevant nodes. '
        'Do NOT answer from summaries alone — always read the raw data. '
        'Answer in Korean, preserving original English technical terms as-is.'
    )

    comet_results = []
    for i, t in enumerate(QUESTIONS, 1):
        response = agent.invoke({
            'messages': [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': t['q']},
            ]
        })
        answer = response['messages'][-1].content
        found = check(answer, t['expected'])
        comet_results.append(found)

        tc_names = [
            tc['name']
            for m in response['messages']
            if hasattr(m, 'tool_calls') and m.tool_calls
            for tc in m.tool_calls
        ]
        print(f'[Q{i}] {t["q"]}')
        print(f'  A: {answer[:150]}')
        print(f'  Tools: {tc_names}')
        print(f'  {"✅" if found else "❌"}')

    # ─── Comparison Table ────────────────────────────────────
    print('\n' + '=' * 60)
    n_pass = sum(naive_results)
    c_pass = sum(comet_results)
    print(f'[Result] Naive {n_pass}/{len(QUESTIONS)} vs CoMeT {c_pass}/{len(QUESTIONS)}')
    print('-' * 60)
    for i, t in enumerate(QUESTIONS):
        n = '✅' if naive_results[i] else '❌'
        c = '✅' if comet_results[i] else '❌'
        print(f'  Q{i+1}: {n} vs {c}  {t["q"][:45]}')
    print('=' * 60)


if __name__ == '__main__':
    main()
