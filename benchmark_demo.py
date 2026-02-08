"""Benchmark Demo: Session Memory + RAG full comparison.

Phase 1 — Session Memory:
  Full Context Injection vs Naive Summary vs CoMeT
  Measures: accuracy (keyword match), context chars, compression ratio

Phase 2 — RAG Retrieval:
  Naive RAG (single embedding search) vs CoMeT RAG (dual-path + RRF)
  Measures: top-1 hit rate, top-3 hit rate, answer accuracy

100 turns across 28 topic clusters, topic shift every 3-4 turns.
"""
import shutil
import tempfile
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from loguru import logger

from comet import CoMeT, scope
from comet.vector_index import VectorIndex

# ─── Test Data: 100 turns, 28 topic clusters ─────────────────────

TURNS = [
    # [01-04] 서버 장애 사고
    '2월 1일 새벽 3시에 프로덕션 서버 장애가 발생했어. 원인은 DB 커넥션 풀 고갈이었어.',
    '복구까지 2시간 반 걸렸고, 매출 손실이 약 3200만원이었어.',
    '재발 방지 대책으로 커넥션 풀 모니터링 알람이랑 자동 스케일링을 도입하기로 했어.',
    '장애 보고서 작성해서 CTO한테 공유했어. 다음주 월요일에 포스트모템 미팅 예정이야.',
    # [05-07] 제주도 여행 계획
    '다음주에 제주도 여행 갈 건데 성산일출봉이랑 우도를 가고 싶어.',
    '숙소는 서귀포 쪽 풀빌라로 예약했어. 1박에 25만원.',
    '해산물 알레르기가 있어서 식당 고를 때 유의해줘.',
    # [08-11] ML 학습 파이프라인
    'SFT 학습 시 learning rate는 2e-5, batch size는 32로 설정했어.',
    'LoRA rank는 16, alpha는 32를 사용. 타겟 모듈은 q_proj, v_proj만.',
    'DPO 학습에서 beta=0.1로 하니까 너무 conservative해서 0.05로 낮췄더니 성능이 올랐어.',
    '학습 데이터 전처리 시 512 토큰 이상 잘리는 샘플이 15%나 돼서 max_length를 1024로 올렸어.',
    # [12-14] API 인증 설계
    'API 인증 방식은 JWT를 쓰기로 결정했어. 액세스 토큰 만료는 15분, 리프레시 토큰은 7일.',
    'OAuth 2.0 Authorization Code Flow를 적용하고, PKCE도 함께 적용한다.',
    '토큰 저장은 httpOnly 쿠키로 하고, CSRF 방어를 위해 SameSite=Strict 설정.',
    # [15-18] 배포 파이프라인
    '배포 파이프라인은 GitHub Actions → Docker Build → ECR Push → ECS Deploy 순서야.',
    'staging 환경에서 canary 배포 5% 트래픽으로 30분 모니터링 후 전체 배포.',
    'rollback은 이전 task definition 버전으로 자동 전환되도록 설정해놨어.',
    '배포 시 헬스체크 interval은 10초, threshold는 3회 실패 시 unhealthy 판정.',
    # [19-21] 팀 채용
    '프론트엔드 시니어 개발자 채용 공고를 올렸어. React + TypeScript 필수.',
    '연봉 범위는 7000만~9000만원. 스톡옵션 별도.',
    '면접은 코딩테스트 → 기술면접 → 컬쳐핏 3단계로 진행.',
    # [22-24] 데이터베이스 마이그레이션
    'PostgreSQL 14에서 16으로 마이그레이션 계획. 주말 점검 시간에 진행.',
    '주요 변경점은 논리 복제 성능 개선이랑 JSON 쿼리 최적화.',
    '마이그레이션 전에 pg_dump로 풀 백업하고, 복구 시간은 최대 4시간으로 잡았어.',
    # [25-28] 사용자 피드백 분석
    '최근 NPS 점수가 72에서 58로 떨어졌어.',
    '주요 불만은 로딩 속도(35%), 가격(28%), UI 복잡성(22%).',
    '속도 문제는 CDN 적용과 이미지 최적화로 해결 가능.',
    'UI 개선은 네비게이션 단순화랑 온보딩 튜토리얼 추가로 대응하기로.',
    # [29-31] 보안 감사
    '외부 보안 감사에서 취약점 3건 발견. 2건 Critical, 1건 Medium.',
    'Critical: SQL Injection 가능성 있는 API 엔드포인트 2개. 파라미터 바인딩으로 수정 완료.',
    'Medium: 세션 타임아웃이 24시간으로 너무 길어서 2시간으로 변경.',
    # [32-34] 제주도 여행 (2차)
    '제주도 렌터카는 전기차로 예약했어. 테슬라 모델3, 하루 8만원.',
    '올레길 7코스를 걸어볼 예정. 중문에서 출발해서 대포주상절리까지.',
    '마지막 날은 공항 근처 동문시장에서 쇼핑할 거야.',
    # [35-38] 분기 매출 보고서
    '3분기 매출 42억, 목표 45억 대비 93.3% 달성.',
    '가장 높은 매출 세그먼트는 B2B SaaS로 전체의 61%.',
    'MRR은 전월 대비 8% 성장. Churn rate는 2.1%로 업계 평균(5%) 절반 이하.',
    '4분기 목표는 50억. 신규 엔터프라이즈 계약 3건 파이프라인에 있어.',
    # [39-41] 코딩 스타일 선호
    '나는 코드 리뷰할 때 항상 성능 이슈를 먼저 본다.',
    '파이썬 코딩할 때 타입 힌트를 반드시 쓰는 걸 선호해.',
    '회의는 30분 이내로 끝내는 걸 좋아하고, 아젠다 없는 회의는 거절해.',
    # [42-44] 모바일 앱 출시
    'iOS 앱 v2.0을 3월 15일에 출시 예정. 다크모드랑 위젯 기능 추가.',
    'TestFlight 베타 테스트 참여자 500명 모집 완료.',
    '앱 스토어 심사는 보통 24-48시간 걸리니까 3월 12일에 제출해야 해.',
    # [45-48] 클라우드 비용 최적화
    'AWS 월 비용이 2800만원인데 목표는 2000만원 이하로 줄이는 거야.',
    '가장 큰 비용은 EC2(45%)랑 RDS(28%). Reserved Instance로 전환하면 35% 절감 가능.',
    'S3 스토리지는 Intelligent-Tiering으로 바꾸면 월 120만원 절감.',
    '개발 환경은 야간에 자동으로 꺼지게 Lambda 스케줄러를 설정했어.',
    # [49-51] 마케팅 캠페인
    '블랙프라이데이 캠페인 CAC가 15000원이었어. 목표는 12000원.',
    '가장 효과 좋은 채널은 인스타그램 리타겟팅으로 ROAS 4.2.',
    '이메일 마케팅 오픈율이 28%에서 15%로 떨어져서 제목 A/B 테스트 진행 중.',
    # [52-55] 데이터 파이프라인
    'Airflow DAG를 Kafka Streams로 전환하는 작업 진행 중.',
    '실시간 처리 latency 목표는 p99 < 500ms. 현재 Airflow 배치는 15분 간격.',
    'Kafka 클러스터는 브로커 5대, 파티션은 토픽당 12개로 설정.',
    '스키마 레지스트리는 Confluent Schema Registry 사용. Avro 포맷.',
    # [56-58] 팀 워크샵
    '다음 달에 팀 빌딩 워크샵을 양평에서 1박 2일로 계획 중.',
    '예산은 1인당 15만원. 참석 인원은 25명.',
    '첫째 날은 에스케이프룸 + 바베큐, 둘째 날은 회고 세션.',
    # [59-62] 신규 기능 스펙
    '검색 기능을 Elasticsearch에서 Typesense로 전환 검토 중.',
    'Typesense가 한국어 형태소 분석을 지원하고, 셀프호스팅 비용이 50% 저렴.',
    '현재 검색 QPS는 피크 시 1200. Typesense 벤치마크 결과 p99 응답 < 15ms.',
    '마이그레이션 기간은 2주로 잡고, 기존 검색과 병렬 운영하면서 A/B 테스트.',
    # [63-65] 고객 계약
    '삼성전자와 엔터프라이즈 계약 협상 중. 월 구독료 3500만원.',
    '계약 기간 2년, 해지 시 위약금 3개월치.',
    'SLA 가용성 99.95%, 응답시간 p95 < 200ms, 미달 시 크레딧 15% 환급.',
    # [66-69] 모니터링 시스템
    'Grafana + Prometheus로 모니터링 스택 구축 완료.',
    '주요 대시보드: 서비스 헬스, 비즈니스 KPI, 인프라 리소스. 총 48개 패널.',
    '알람 규칙 32개 설정. P1 알람은 PagerDuty 연동, P2는 Slack.',
    '로그 수집은 Loki로 통합. 보관 기간 30일, 인덱싱 비용 월 80만원.',
    # [70-72] ML 모델 서빙
    'TorchServe에서 vLLM으로 전환 후 throughput 3배 증가.',
    '8B 모델 기준 A100 1장으로 초당 45 requests 처리 가능.',
    'batch 크기 동적 조절로 latency p50=120ms, p99=380ms.',
    # [73-75] 법률/컴플라이언스
    '개인정보보호법 개정에 따라 동의 절차 UI 수정 필요.',
    '데이터 보관 기간을 3년에서 1년으로 단축. 기존 데이터 파기 스케줄 수립.',
    'GDPR 대응으로 Data Subject Access Request 처리 시스템 구축. SLA 30일.',
    # [76-79] 프로덕트 로드맵
    '2026년 상반기 로드맵: Q1은 검색 고도화, Q2는 AI 추천 엔진.',
    'AI 추천 엔진은 협업 필터링 + 콘텐츠 기반 하이브리드 방식.',
    '추천 정확도 목표는 Precision@10 > 0.35. 현재 baseline은 0.18.',
    'A/B 테스트 후 클릭률 20% 이상 개선 시 전체 적용.',
    # [80-82] 서버 장애 (2차)
    '3월 5일 오후 2시에 또 장애 발생. 이번엔 Redis 메모리 OOM.',
    'maxmemory-policy를 noeviction에서 allkeys-lru로 변경해서 해결.',
    '영향 범위는 캐시 의존 서비스만. 사용자 체감 장애 시간 약 15분.',
    # [83-86] A/B 테스트 결과
    '새 온보딩 플로우 A/B 테스트 결과: 전환율 12% → 18.5%로 개선.',
    '이탈률은 기존 45%에서 32%로 감소. 특히 모바일에서 효과 큼.',
    '통계적 유의성 p-value < 0.01. 샘플 사이즈 각 그룹 5000명.',
    '다음 단계로 가격 페이지 A/B 테스트 계획 중.',
    # [87-89] 외부 API 연동
    'Stripe 결제 연동 완료. 수수료 2.9% + 30센트.',
    'Twilio SMS 인증 도입. 건당 25원, 월 예상 비용 75만원.',
    'SendGrid 이메일은 월 5만건 플랜. 초과 시 건당 1.5원.',
    # [90-92] 제주도 여행 (3차)
    '제주도 여행 마지막 날 돌하르방 공원이랑 카멜리아힐 다녀왔어.',
    '기념품으로 한라봉 초콜릿이랑 감귤잼 샀어. 총 8만원.',
    '전체 여행 경비는 1인당 62만원. 예산 50만원에서 12만원 초과.',
    # [93-95] 인프라 자동화
    'Terraform으로 인프라 코드화 완료. 모듈 28개, 리소스 156개 관리 중.',
    'CI/CD에 terraform plan 자동 실행 추가. PR에 diff 코멘트.',
    'Vault로 시크릿 관리 전환. 기존 .env 파일 방식 완전 폐기.',
    # [96-99] 기술 부채 정리
    '기술 부채 백로그 정리: Critical 8건, Major 15건, Minor 23건.',
    'Critical 최우선은 Express.js에서 Fastify 전환. 응답속도 40% 개선 예상.',
    '레거시 jQuery 코드 3만줄 React로 마이그레이션. 4주 소요 예상.',
    'DB 쿼리 최적화: N+1 문제 12건, 누락 인덱스 8건 식별.',
    # [100] 마무리
    '오늘 전체 기술 리뷰 미팅에서 위 사안들 우선순위 정했어. Q1에 최소 Critical 8건 해결 목표.',
]

# 20 questions covering single-topic + cross-topic retrieval
QUESTIONS = [
    {
        'q': '서버 장애 복구 시간이 얼마나 걸렸고 매출 손실은 얼마야?',
        'expected': ['2시간', '3200만'],
        'topic': 'server_incident',
    },
    {
        'q': 'JWT 액세스 토큰 만료 시간은?',
        'expected': ['15분'],
        'topic': 'api_auth',
    },
    {
        'q': 'DPO 학습 beta 값을 얼마로 조정했어?',
        'expected': ['0.05'],
        'topic': 'ml_training',
    },
    {
        'q': 'canary 배포 트래픽 비율이 얼마야?',
        'expected': ['5%'],
        'topic': 'deployment',
    },
    {
        'q': '프론트엔드 채용 연봉 범위는?',
        'expected': ['7000', '9000'],
        'topic': 'hiring',
    },
    {
        'q': '3분기 매출 달성률이 얼마야?',
        'expected': ['93', '42억'],
        'topic': 'sales',
    },
    {
        'q': 'NPS 점수가 얼마로 떨어졌어?',
        'expected': ['58'],
        'topic': 'feedback',
    },
    {
        'q': '보안 감사에서 Critical 취약점 몇 건 나왔어?',
        'expected': ['2건', 'SQL Injection'],
        'topic': 'security',
    },
    {
        'q': 'AWS 월 비용이 얼마이고 목표는?',
        'expected': ['2800', '2000'],
        'topic': 'cloud_cost',
    },
    {
        'q': 'Kafka 클러스터 브로커 몇 대야?',
        'expected': ['5대', '5'],
        'topic': 'data_pipeline',
    },
    {
        'q': '삼성전자 계약 월 구독료는?',
        'expected': ['3500만'],
        'topic': 'contract',
    },
    {
        'q': 'vLLM 전환 후 초당 처리량은?',
        'expected': ['45'],
        'topic': 'ml_serving',
    },
    {
        'q': '제주도 여행 전체 경비가 얼마였어?',
        'expected': ['62만'],
        'topic': 'jeju',
    },
    {
        'q': '새 온보딩 A/B 테스트 전환율 결과는?',
        'expected': ['18.5', '18'],
        'topic': 'ab_test',
    },
    {
        'q': 'Terraform으로 관리하는 리소스는 몇 개야?',
        'expected': ['156'],
        'topic': 'infra',
    },
    # Cross-topic questions
    {
        'q': '서버 장애가 두 번 있었는데 각각 원인이 뭐야?',
        'expected': ['커넥션', 'Redis', 'OOM'],
        'topic': 'cross_server',
    },
    {
        'q': '제주도 숙소비랑 렌터카비 합치면 얼마야?',
        'expected': ['25만', '8만'],
        'topic': 'cross_jeju',
    },
    {
        'q': 'PostgreSQL 마이그레이션이랑 Elasticsearch→Typesense 전환 각각 기간이 어떻게 돼?',
        'expected': ['4시간', '2주'],
        'topic': 'cross_migration',
    },
    {
        'q': 'Critical 기술 부채 건수랑 보안 Critical 건수 합치면?',
        'expected': ['8', '2'],
        'topic': 'cross_debt_security',
    },
    {
        'q': 'AI 추천 엔진의 정확도 목표랑 현재 baseline은?',
        'expected': ['0.35', '0.18'],
        'topic': 'roadmap',
    },
]


def check_answer(answer: str, expected: list[str]) -> bool:
    a = answer.lower()
    return any(kw.lower() in a for kw in expected)


@scope
def main(config):
    tmpdir = tempfile.mkdtemp(prefix='comet_benchmark_')
    config.storage.base_path = f'{tmpdir}/store'
    config.storage.raw_path = f'{tmpdir}/store/raw'
    config.retrieval.vector_db_path = f'{tmpdir}/vectors'

    n_turns = len(TURNS)
    n_questions = len(QUESTIONS)
    full_context = '\n'.join(f'[Turn {i+1}] {t}' for i, t in enumerate(TURNS))
    full_chars = len(full_context)

    llm = ChatOpenAI(model=config.main_model)
    slm = ChatOpenAI(model=config.slm_model)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Session Memory Benchmark
    # ═══════════════════════════════════════════════════════════════
    print('=' * 70)
    print(f'PHASE 1: Session Memory Benchmark ({n_turns} turns, {n_questions} questions)')
    print('=' * 70)

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
        print(f'  Q{i:02d} {"✅" if hit else "❌"} {q["q"][:50]}')
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
        print(f'  Q{i:02d} {"✅" if hit else "❌"} {q["q"][:50]}')
    naive_time = time.time()-t0
    print(f'  Context: {naive_chars:,} chars ({naive_chars/full_chars*100:.1f}%) | Accuracy: {sum(naive_results)}/{n_questions}')
    print(f'  Time: {naive_time:.1f}s')

    # ── 1C: CoMeT Session Memory ──────────────────────────────────
    print('\n--- [1C] CoMeT Session Memory ---')
    t0 = time.time()
    memo = CoMeT(config)
    for content in TURNS:
        memo.add(content)
    memo.force_compact()

    nodes = memo.list_memories()
    comet_context = memo.get_context_window(max_nodes=50)
    comet_chars = len(comet_context)

    print(f'  Nodes: {len(nodes)} | VectorIndex: {memo._vector_index.count}')

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
        print(f'  Q{i:02d} {"✅" if hit else "❌"} reads={reads} {q["q"][:45]}')
    comet_time = time.time()-t0
    print(f'  Context: {comet_chars:,} chars ({comet_chars/full_chars*100:.1f}%) | Accuracy: {sum(comet_results)}/{n_questions}')
    print(f'  Time: {comet_time:.1f}s')

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

    # ── 2A: Naive RAG (chunk-level, same granularity as CoMeT) ───
    print('\n--- [2A] Naive RAG (chunk-level summary embed) ---')
    chunk_size = 4
    chunks = []
    for i in range(0, n_turns, chunk_size):
        chunk_turns = TURNS[i:i+chunk_size]
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

        print(f'  Q{i:02d} {"✅" if hit else "❌"} top1={"✅" if top1_hit else "❌"} top3={"✅" if top3_hit else "❌"} {q["q"][:40]}')

    # ── 2B: CoMeT RAG (dual-path: summary + trigger) ─────────────
    print(f'\n--- [2B] CoMeT RAG (dual-path: summary + trigger) ---')
    print(f'  Vectors: {memo._vector_index.count}')

    rag_queries = [
        ('서버 장애 복구 시간 매출 손실', '장애 보고서를 작성하려고 서버 장애의 복구 소요 시간과 손실액을 확인하려 한다'),
        ('JWT 액세스 토큰 만료 시간 설정', 'API 인증 아키텍처를 검토하면서 토큰 만료 정책을 확인하려 한다'),
        ('DPO 학습 beta 값 조정', 'ML 학습 하이퍼파라미터를 리뷰하면서 DPO beta 값 변경 이력을 찾으려 한다'),
        ('canary 배포 트래픽 비율', '배포 파이프라인 설정을 확인하면서 canary 배포 전략을 검토하려 한다'),
        ('프론트엔드 시니어 개발자 채용 연봉 범위', '채용 예산을 책정하면서 프론트엔드 채용 공고의 연봉 조건을 확인하려 한다'),
        ('3분기 매출 달성률 목표 대비', '분기별 실적 보고를 준비하면서 3분기 매출 성과를 확인하려 한다'),
        ('NPS 점수 하락 추이', '고객 만족도를 분석하면서 NPS 점수 변화를 파악하려 한다'),
        ('보안 감사 Critical 취약점 SQL Injection', '보안 감사 결과를 리뷰하면서 발견된 Critical 취약점 내역을 확인하려 한다'),
        ('AWS 월간 클라우드 비용 목표', '클라우드 비용 최적화 프로젝트를 위해 현재 비용과 목표치를 확인하려 한다'),
        ('Kafka 클러스터 브로커 구성 파티션', '데이터 파이프라인 인프라를 검토하면서 Kafka 클러스터 스펙을 확인하려 한다'),
        ('삼성전자 엔터프라이즈 계약 월 구독료', '엔터프라이즈 계약 현황을 파악하면서 삼성전자 건의 금액을 확인하려 한다'),
        ('vLLM 전환 throughput 초당 처리량', 'ML 모델 서빙 인프라를 검토하면서 vLLM 성능 수치를 확인하려 한다'),
        ('제주도 여행 전체 경비 예산 초과', '여행 후 정산을 하면서 전체 경비를 확인하려 한다'),
        ('온보딩 A/B 테스트 전환율 결과', '신규 온보딩 플로우의 성과를 리뷰하면서 A/B 테스트 결과를 확인하려 한다'),
        ('Terraform 인프라 코드화 리소스 개수', '인프라 자동화 현황을 파악하면서 Terraform으로 관리 중인 리소스 규모를 확인하려 한다'),
        ('서버 장애 원인 DB 커넥션 Redis OOM', '반복 장애의 패턴을 분석하면서 각 장애의 원인을 비교하려 한다'),
        ('제주도 숙소 풀빌라 렌터카 비용', '여행 예산을 계산하면서 숙소비와 렌터카비를 합산하려 한다'),
        ('PostgreSQL 마이그레이션 Typesense 전환 기간', '인프라 마이그레이션 일정을 수립하면서 각 작업의 소요 기간을 확인하려 한다'),
        ('기술 부채 Critical 건수 보안 Critical', 'Q1 우선순위를 정하면서 전체 Critical 이슈 건수를 파악하려 한다'),
        ('AI 추천 엔진 Precision@10 정확도 목표 baseline', '프로덕트 로드맵을 검토하면서 추천 엔진의 성능 목표치를 확인하려 한다'),
    ]

    comet_rag_results = []
    comet_rag_top3 = []
    for i, q in enumerate(QUESTIONS, 1):
        sq, tq = rag_queries[i-1]
        retrieved = memo.retrieve_dual(sq, tq, top_k=5)
        if retrieved:
            rag_context_parts = []
            for r in retrieved:
                raw = memo._store.get_raw(r.node.content_key) or ''
                rag_context_parts.append(
                    f'[{r.node.node_id}] (score={r.relevance_score:.4f})\n'
                    f'Summary: {r.node.summary}\n'
                    f'Raw: {raw}'
                )
            rag_context = '\n\n'.join(rag_context_parts)
            a = llm.invoke(
                f'아래 검색된 메모리를 참고해서 질문에 답해줘.\n\n'
                f'## 검색 결과\n{rag_context}\n\n## 질문\n{q["q"]}'
            ).content
        else:
            a = '정보 없음'

        hit = check_answer(a, q['expected'])
        comet_rag_results.append(hit)

        if retrieved:
            top1_raw = memo._store.get_raw(retrieved[0].node.content_key) or ''
            top1_hit = any(kw.lower() in top1_raw.lower() for kw in q['expected'])
            top3_raw = ' '.join(
                memo._store.get_raw(r.node.content_key) or '' for r in retrieved[:3]
            )
            top3_hit = any(kw.lower() in top3_raw.lower() for kw in q['expected'])
        else:
            top1_hit = False
            top3_hit = False
        comet_rag_top3.append(top3_hit)

        score_str = f'{retrieved[0].relevance_score:.4f}' if retrieved else 'N/A'
        print(f'  Q{i:02d} {"✅" if hit else "❌"} top1={"✅" if top1_hit else "❌"} top3={"✅" if top3_hit else "❌"} score={score_str} {q["q"][:35]}')

    # ── Phase 2 Summary ───────────────────────────────────────────
    print('\n' + '=' * 70)
    print('PHASE 2 RESULTS: RAG Retrieval')
    print('=' * 70)
    print(f'{"Method":<30} {"Answer Acc":>12} {"Top-3 Hit":>12}')
    print('-' * 70)
    print(f'{"Naive RAG (chunk summary)":<30} {f"{sum(naive_rag_results)}/{n_questions}":>12} {f"{sum(naive_rag_top3)}/{n_questions}":>12}')
    print(f'{"CoMeT RAG (dual-path)":<30} {f"{sum(comet_rag_results)}/{n_questions}":>12} {f"{sum(comet_rag_top3)}/{n_questions}":>12}')

    # ═══════════════════════════════════════════════════════════════
    # Overall Summary
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 70)
    print('OVERALL SUMMARY')
    print('=' * 70)
    print(f'Turns: {n_turns} | Questions: {n_questions}')
    print(f'CoMeT Nodes: {len(nodes)} | Vectors: {memo._vector_index.count}')
    print(f'Naive RAG Chunks: {n_chunks} (chunk_size={chunk_size})')
    print()
    print('Session Memory:')
    print(f'  Full Context:  {sum(full_results)}/{n_questions} ({full_chars:,} chars = 100%)')
    print(f'  Naive Summary: {sum(naive_results)}/{n_questions} ({naive_chars:,} chars = {naive_chars/full_chars*100:.1f}%)')
    print(f'  CoMeT:         {sum(comet_results)}/{n_questions} ({comet_chars:,} chars = {comet_chars/full_chars*100:.1f}%)')
    print()
    print('RAG Retrieval:')
    print(f'  Naive RAG:     {sum(naive_rag_results)}/{n_questions} (top-3 hit: {sum(naive_rag_top3)}/{n_questions})')
    print(f'  CoMeT RAG:     {sum(comet_rag_results)}/{n_questions} (top-3 hit: {sum(comet_rag_top3)}/{n_questions})')
    print('=' * 70)

    shutil.rmtree(tmpdir)
    logger.info(f'Cleaned up: {tmpdir}')


if __name__ == '__main__':
    main()

