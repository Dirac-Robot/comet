"""Demo: Conversation Memory with ERASE filtering."""
from erase import ConversationMemory, scope


@scope
def main(config):
    memory = ConversationMemory(config)
    
    print("=" * 60)
    print("[ERASE Conversation Memory Demo]")
    print(f"Retention threshold: {config.threshold.retention}")
    print(f"Erasure threshold: {config.threshold.erasure}")
    print("=" * 60)
    print()
    
    # 여러 주제의 대화 시뮬레이션
    conversations = [
        ("user", "다음주에 제주도 휴가 가려고 하는데 추천해줘"),
        ("assistant", "제주도 동쪽 코스 추천드려요. 성산일출봉, 섭지코지, 우도가 좋아요."),
        ("user", "숙소는 어디가 좋을까?"),
        ("assistant", "서귀포 쪽 펜션이나 호텔 추천드려요. 중문 관광단지도 좋습니다."),
        ("user", "이번 분기 매출 목표 정리해줘"),
        ("assistant", "이번 분기 매출 목표는 50억원입니다. 주요 KPI는 신규 고객 1000명 유치입니다."),
        ("user", "경쟁사 분석 자료도 필요해"),
        ("assistant", "경쟁사 A사는 최근 신제품을 출시했고, B사는 가격 인하 전략을 쓰고 있어요."),
        ("user", "오늘 저녁 뭐 먹지?"),
        ("assistant", "근처에 새로 생긴 이탈리안 레스토랑 어때요? 파스타가 맛있대요."),
    ]
    
    # 대화 추가
    print("[Adding conversations to memory...]")
    for role, content in conversations:
        memory.add(role, content)
    print(f"  Added {len(conversations)} messages\n")
    
    # 테스트: 모든 청크 점수 보기
    query = "매출 목표가 뭐였지?"
    
    print("=" * 60)
    print(f"[Query] '{query}'")
    print("=" * 60)
    print()
    
    # score_all: 모든 청크 (필터링 전)
    all_chunks = memory.score_all(query)
    print(f"[ALL chunks with scores] ({len(all_chunks)} total):")
    print("-" * 60)
    
    for c in all_chunks:
        # 통과 여부 판단
        passed = c.retention_score >= config.threshold.retention and c.erasure_score < config.threshold.erasure
        status = "✅ KEEP" if passed else "❌ EXCLUDE"
        reason = ""
        if not passed:
            if c.retention_score < config.threshold.retention:
                reason = "(low retention)"
            elif c.erasure_score >= config.threshold.erasure:
                reason = "(high erasure - off-topic)"
        print(f"  {status} R={c.retention_score:.2f} E={c.erasure_score:.2f} {reason}")
        print(f"         {c.content[:55]}...")
        print()
    
    # retrieve: 필터링된 결과만
    print("=" * 60)
    print("[FILTERED result (via retrieve)]:")
    print("-" * 60)
    filtered = memory.retrieve(query)
    print(f"  Kept {len(filtered)} / {len(all_chunks)} chunks")


if __name__ == "__main__":
    main()
