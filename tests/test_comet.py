"""CoMeT Integration Test: Real chat data from B200 conversation."""
import json
from pathlib import Path

from comet import CoMeT, scope


TEST_DATA = Path(__file__).parent/'test_data.json'
MAX_TURNS = 15


@scope
def main(config):
    config.storage.base_path = './memory_store_test'
    config.storage.raw_path = './memory_store_test/raw'

    memo = CoMeT(config)

    with open(TEST_DATA, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    messages = messages[:MAX_TURNS]

    print('=' * 60)
    print(f'[CoMeT Test] {len(messages)} turns from B200 conversation')
    print(f'SLM: {config.slm_model} | Main: {config.main_model}')
    print(f'Compact threshold: load >= {config.compacting.load_threshold}, max_buffer={config.compacting.max_l1_buffer}')
    print('=' * 60)

    compacted_nodes = []
    for i, content in enumerate(messages, 1):
        display = content[:80].replace('\n', ' ')
        print(f'\n[Turn {i:02d}] {display}...')

        node = memo.add(content)

        load = memo.last_load
        if load:
            print(f'  CogLoad: flow={load.logic_flow}, level={load.load_level}')
            if load.reasoning:
                print(f'  Reason: {load.reasoning[:80]}')

        if node:
            compacted_nodes.append(node)
            print(f'  >>> COMPACTED -> {node.node_id}')
            print(f'      Summary: {node.summary}')
            print(f'      Trigger: {node.trigger[:80]}')
            print(f'      Tags: {node.topic_tags}')
            print(f'      Links: {node.links}')

    # Force compact remaining
    print('\n' + '=' * 60)
    print('[Force compacting remaining buffer...]')
    final = memo.force_compact()
    if final:
        compacted_nodes.append(final)
        print(f'  Final node: {final.node_id}')
        print(f'  Summary: {final.summary}')
        print(f'  Trigger: {final.trigger[:80]}')

    # Results
    print('\n' + '=' * 60)
    print(f'[Results] {len(compacted_nodes)} nodes created from {len(messages)} turns')
    print('-' * 60)

    for node in compacted_nodes:
        print(f'\n  [{node.node_id}]')
        print(f'  Summary: {node.summary}')
        print(f'  Trigger: {node.trigger}')
        print(f'  Tags: {node.topic_tags}')
        print(f'  Links: {node.links}')

    # Navigation test
    if compacted_nodes:
        test_node = compacted_nodes[0]
        print('\n' + '=' * 60)
        print(f'[Navigation Test] node={test_node.node_id}')
        print('-' * 60)

        print('\n[depth=0]:')
        print(memo.read_memory(test_node.node_id, depth=0))

        print('\n[depth=1]:')
        print(memo.read_memory(test_node.node_id, depth=1))

        print('\n[depth=2]:')
        raw = memo.read_memory(test_node.node_id, depth=2)
        if raw and len(raw) > 300:
            print(raw[:300] + '...')
        else:
            print(raw)

    # Context window
    print('\n' + '=' * 60)
    print('[Context Window]')
    print(memo.get_context_window())

    # Tool integration test
    print('\n' + '=' * 60)
    print('[Tool Integration]')
    tools = memo.get_tools()
    print(f'Available tools: {[t.name for t in tools]}')


if __name__ == '__main__':
    main()
