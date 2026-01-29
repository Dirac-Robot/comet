"""Demo: ERASE + Web Search RAG"""
from erase import ERASE, scope
from erase.web_search import search_web, format_search_results


@scope
def main(config):
    erase = ERASE(config)

    # 실제 쿼리
    query = "Python asyncio 사용법"
    
    print("=" * 60)
    print(f"[ERASE Web RAG Demo]")
    print(f"Query: {query}")
    print("=" * 60)
    print()

    # 1. 웹 검색
    print("[Step 1] Searching the web...")
    results = search_web(query, max_results=8)
    
    # 검색 결과를 텍스트로 변환
    raw_context = ""
    for i, r in enumerate(results, 1):
        raw_context += f"\n[{i}] {r.get('title', '')}\n{r.get('body', '')}\n"
    
    print(f"Found {len(results)} results")
    print()

    # 2. ERASE 없이 (기존 RAG 방식)
    print("[Step 2] Without ERASE (traditional RAG):")
    print("-" * 40)
    print(raw_context[:500] + "..." if len(raw_context) > 500 else raw_context)
    print()

    # 3. ERASE 적용 (query-aware filtering)
    print("[Step 3] With ERASE (query-aware filtering):")
    print("-" * 40)
    print(f"Retention threshold: {config.threshold.retention}")
    print(f"Erasure threshold: {config.threshold.erasure}")
    print()
    
    memories = erase(raw_context, query=query)
    
    print(f"Kept {len(memories)} / {len(results)} chunks:")
    for mem in memories:
        indicator = "⚠️" if mem.erasure_score >= 0.3 else "✅"
        print(f"  {indicator} R={mem.retention_score:.2f} E={mem.erasure_score:.2f}")
        print(f"     {mem.content[:60]}...")
        print()

    # 4. 비교: 제거된 것들
    print("[Step 4] What was EXCLUDED:")
    print("-" * 40)
    
    # 실제로 제거된 청크를 보려면 threshold 이전 상태가 필요
    # 간단히 낮은 threshold로 다시 실행해서 비교
    config_lenient = config.copy()
    config_lenient.threshold.retention = 0.0
    config_lenient.threshold.erasure = 1.0
    erase_lenient = ERASE(config_lenient)
    all_chunks = erase_lenient(raw_context, query=query)
    
    excluded = [c for c in all_chunks if c not in memories]
    if excluded:
        for mem in excluded:
            print(f"  ❌ R={mem.retention_score:.2f} E={mem.erasure_score:.2f}")
            print(f"     {mem.content[:60]}...")
            print()
    else:
        print("  (Nothing excluded)")


if __name__ == "__main__":
    main()
