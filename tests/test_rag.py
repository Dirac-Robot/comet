"""Ultra Test: End-to-end RAG pipeline with multi-topic ingestion + semantic retrieval.

Tests:
1. Ingestion: Add 5 distinct topic clusters → sensor triggers compacting → VectorIndex upserts
2. recall_mode: Verify LLM classifies personal vs active vs both
3. RAG Retrieval: Semantic queries → dual-path (summary + trigger) search → RRF fusion
4. Cross-topic: Queries that span multiple nodes
5. Trigger perspective: LLM-centric trigger phrasing verification
"""
import shutil
import tempfile

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

from comet import CoMeT, scope


CONVERSATIONS = {
    'server_incident': [
        '2월 1일 새벽 3시에 서버 장애가 발생했어. 원인은 DB 커넥션 풀 고갈이었어.',
        '복구까지 2시간 반 걸렸고, 매출 손실이 약 3200만원이었어.',
        '재발 방지 대책으로 커넥션 풀 모니터링 알람이랑 자동 스케일링을 도입하기로 했어.',
        '장애 보고서 작성해서 CTO한테 공유했어. 다음주 월요일에 포스트모템 미팅 예정이야.',
    ],
    'user_preferences': [
        '나는 코드 리뷰할 때 항상 성능 이슈를 먼저 본다.',
        '파이썬 코딩할 때 타입 힌트를 반드시 쓰는 걸 선호해.',
        '회의는 30분 이내로 끝내는 걸 좋아하고, 길어지면 집중력이 떨어져.',
    ],
    'api_auth': [
        'API 인증 방식은 JWT를 쓰기로 결정했어. 액세스 토큰 만료는 15분, 리프레시 토큰은 7일.',
        'OAuth 2.0 Authorization Code Flow를 적용하고, PKCE도 함께 적용한다.',
        '토큰 저장은 httpOnly 쿠키로 하고, CSRF 방어를 위해 SameSite=Strict 설정.',
        '외부 서비스 연동용 API 키는 별도로 발급하고, rate limiting은 분당 100회로 세팅.',
    ],
    'ml_training': [
        'SFT 학습 시 learning rate는 2e-5, batch size는 32로 설정했어.',
        'LoRA rank는 16, alpha는 32를 사용. 타겟 모듈은 q_proj, v_proj만.',
        'DPO 학습에서 beta=0.1로 하니까 너무 conservative해서 0.05로 낮췄더니 성능이 올랐어.',
        '학습 데이터 전처리 시 512 토큰 이상 잘리는 샘플이 15%나 돼서 max_length를 1024로 올렸어.',
    ],
    'deployment': [
        '배포 파이프라인은 GitHub Actions → Docker Build → ECR Push → ECS Deploy 순서야.',
        'staging 환경에서 canary 배포 5% 트래픽으로 30분 모니터링 후 전체 배포.',
        'rollback은 이전 task definition 버전으로 자동 전환되도록 설정해놨어.',
        '배포 시 헬스체크 interval은 10초, threshold는 3회 실패 시 unhealthy 판정.',
    ],
}

QUERIES = [
    {
        'query': '서버 장애 복구 시간이 얼마나 걸렸지?',
        'expected_topic': 'server_incident',
        'expected_keywords': ['2시간', '반'],
    },
    {
        'query': 'JWT 토큰 만료 시간 설정이 어떻게 되어있어?',
        'expected_topic': 'api_auth',
        'expected_keywords': ['15분', '7일'],
    },
    {
        'query': 'DPO 학습 beta 값을 얼마로 바꿨더니 성능이 올랐어?',
        'expected_topic': 'ml_training',
        'expected_keywords': ['0.05'],
    },
    {
        'query': '배포할 때 canary 트래픽 비율이 얼마야?',
        'expected_topic': 'deployment',
        'expected_keywords': ['5%'],
    },
    {
        'query': '내가 코드 리뷰할 때 제일 먼저 확인하는 게 뭐야?',
        'expected_topic': 'user_preferences',
        'expected_keywords': ['성능'],
    },
    {
        'query': '장애 원인이 뭐였고 방지 대책은?',
        'expected_topic': 'server_incident',
        'expected_keywords': ['커넥션', '풀', '모니터링'],
    },
    {
        'query': 'LoRA 설정값 알려줘',
        'expected_topic': 'ml_training',
        'expected_keywords': ['16', '32', 'q_proj'],
    },
    {
        'query': 'CSRF 방어 설정이 어떻게 되어있어?',
        'expected_topic': 'api_auth',
        'expected_keywords': ['SameSite', 'Strict'],
    },
]


def check(answer_summary: str, keywords: list[str]) -> bool:
    a = answer_summary.lower()
    return any(kw.lower() in a for kw in keywords)


@scope
def main(config):
    tmpdir = tempfile.mkdtemp(prefix='comet_rag_test_')
    config.storage.base_path = f'{tmpdir}/store'
    config.storage.raw_path = f'{tmpdir}/store/raw'
    config.retrieval.vector_db_path = f'{tmpdir}/vectors'

    logger.info(f'Test directory: {tmpdir}')

    memo = CoMeT(config)

    # ─── Phase 1: Ingestion ─────────────────────────────────
    print('=' * 70)
    print('[Phase 1] Ingestion — 5 topic clusters')
    print('=' * 70)

    all_nodes = []
    for topic_name, messages in CONVERSATIONS.items():
        print(f'\n--- Topic: {topic_name} ({len(messages)} turns) ---')
        for msg in messages:
            node = memo.add(msg)
            if node:
                all_nodes.append(node)
                print(f'  >> COMPACTED: {node.node_id}')
                print(f'     Summary: {node.summary}')
                print(f'     Trigger: {node.trigger[:80]}')
                print(f'     recall_mode: {node.recall_mode}')
                print(f'     Tags: {node.topic_tags}')

    final = memo.force_compact()
    if final:
        all_nodes.append(final)
        print(f'\n  >> FINAL COMPACT: {final.node_id}')
        print(f'     Summary: {final.summary}')
        print(f'     Trigger: {final.trigger[:80]}')
        print(f'     recall_mode: {final.recall_mode}')

    print(f'\n[Ingestion Result] {len(all_nodes)} nodes from {sum(len(v) for v in CONVERSATIONS.values())} turns')
    print(f'[VectorIndex] {memo._vector_index.count} vectors indexed')

    # ─── Phase 2: Node Inspection ───────────────────────────
    print('\n' + '=' * 70)
    print('[Phase 2] Node Inspection — recall_mode + trigger perspective')
    print('=' * 70)

    for node in all_nodes:
        print(f'\n  [{node.node_id}]')
        print(f'    Summary: {node.summary}')
        print(f'    Trigger: {node.trigger}')
        print(f'    recall_mode: {node.recall_mode}')
        print(f'    Tags: {node.topic_tags}')
        print(f'    Links: {node.links}')

    passive_count = sum(1 for n in all_nodes if n.recall_mode in ('passive', 'both'))
    active_count = sum(1 for n in all_nodes if n.recall_mode == 'active')
    print(f'\n  passive/both: {passive_count} | active: {active_count}')

    trigger_perspective_ok = all(
        '내가' in n.trigger or '내게' in n.trigger or '필요' in n.trigger
        for n in all_nodes
    )
    print(f'  Trigger LLM perspective: {"✅ All LLM-centric" if trigger_perspective_ok else "⚠️ Some may be user-centric"}')

    # ─── Phase 3: RAG Retrieval ─────────────────────────────
    print('\n' + '=' * 70)
    print('[Phase 3] RAG Retrieval — 8 queries')
    print('=' * 70)

    results = []
    for i, q in enumerate(QUERIES, 1):
        retrieved = memo.retrieve(q['query'], top_k=3)
        if retrieved:
            top = retrieved[0]
            raw = memo._store.get_raw(top.node.content_key) or ''
            hit = check(raw, q['expected_keywords'])
            results.append(hit)
            print(f'\n  Q{i:02d} {"✅" if hit else "❌"} {q["query"]}')
            print(f'      Top node: {top.node.node_id} (score={top.relevance_score:.4f})')
            print(f'      Summary: {top.node.summary}')
            print(f'      Raw match: {hit}')
            if len(retrieved) > 1:
                print(f'      #2: {retrieved[1].node.node_id} (score={retrieved[1].relevance_score:.4f})')
        else:
            results.append(False)
            print(f'\n  Q{i:02d} ❌ {q["query"]} — no results')

    # ─── Phase 4: Context Window ────────────────────────────
    print('\n' + '=' * 70)
    print('[Phase 4] Context Window')
    print('=' * 70)
    print(memo.get_context_window(max_nodes=10))

    # ─── Phase 5: Tools ─────────────────────────────────────
    print('\n' + '=' * 70)
    print('[Phase 5] Tool Registry')
    print('=' * 70)
    tools = memo.get_tools()
    for t in tools:
        print(f'  - {t.name}: {t.description[:70]}')

    # ─── Summary ────────────────────────────────────────────
    print('\n' + '=' * 70)
    correct = sum(results)
    total = len(results)
    print(f'[RESULT] RAG Retrieval: {correct}/{total} ({correct/total*100:.0f}%)')
    print(f'         Nodes: {len(all_nodes)} | Vectors: {memo._vector_index.count}')
    print(f'         Passive/Both: {passive_count} | Active: {active_count}')
    print(f'         Trigger perspective: {"✅" if trigger_perspective_ok else "⚠️"}')
    print('=' * 70)

    shutil.rmtree(tmpdir)
    print(f'\nCleaned up: {tmpdir}')


if __name__ == '__main__':
    main()
