"""Demo: CoMeT Dynamic Resolution Memory System."""
from comet import CoMeT, scope


@scope
def main(config):
    memo = CoMeT(config)
    
    print("=" * 60)
    print("[CoMeT Demo - Dynamic Resolution Memory]")
    print(f"SLM Model: {config.slm_model}")
    print(f"Main Model: {config.main_model}")
    print(f"Compacting threshold: load >= {config.compacting.load_threshold}")
    print("=" * 60)
    print()
    
    # 다양한 주제의 대화 시뮬레이션
    conversations = [
        # Topic 1: 여행
        "다음주에 제주도 휴가 가려고 하는데 추천해줘",
        "성산일출봉이랑 우도 가볼까 생각 중이야",
        "숙소는 서귀포 쪽이 좋을까?",
        
        # Topic 2: 업무 (주제 전환!)
        "이번 분기 매출 목표 정리해줘",
        "경쟁사 A사 분석 자료도 필요해",
        "다음 회의 때 발표할 PPT 준비해야 해",
        
        # Topic 3: 일상 (또 다른 주제 전환!)
        "오늘 저녁 뭐 먹지?",
        "근처에 새로 생긴 이탈리안 어때?",
    ]
    
    print("[Adding conversations...]")
    print("-" * 60)
    
    for i, content in enumerate(conversations, 1):
        print(f"\n#{i}: {content}")
        node = memo.add(content)
        
        if node:
            print(f"  → 🗂️ COMPACTED! Node: {node.node_id}")
            print(f"     Summary: {node.summary[:60]}...")
        else:
            load = memo.last_load
            if load:
                print(f"  → L1 Buffer (flow: {load.logic_flow}, load: {load.load_level})")
    
    # Force compact remaining buffer
    print("\n" + "=" * 60)
    print("[Force compacting remaining buffer...]")
    final_node = memo.force_compact()
    if final_node:
        print(f"  → Final node: {final_node.node_id}")
        print(f"     Summary: {final_node.summary}")
    
    # Navigation demo
    print("\n" + "=" * 60)
    print("[Navigation Demo - read_memory with depth]")
    print("-" * 60)
    
    all_nodes = memo.list_memories()
    if all_nodes:
        first_node = all_nodes[0]['node_id']
        
        print(f"\n[depth=0] Summary only:")
        print(memo.read_memory(first_node, depth=0))
        
        print(f"\n[depth=1] With metadata:")
        print(memo.read_memory(first_node, depth=1))
        
        print(f"\n[depth=2] Full raw data:")
        print(memo.read_memory(first_node, depth=2))
    
    # Context window demo
    print("\n" + "=" * 60)
    print("[Context Window for Agent]")
    print("-" * 60)
    print(memo.get_context_window())


if __name__ == "__main__":
    main()
