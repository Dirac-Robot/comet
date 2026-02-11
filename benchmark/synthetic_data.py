"""
벤치마크용 합성 데이터: 10개 주제, 각 3-4턴, 어려운 기술적 내용

각 주제는 구체적인 숫자, 이름, 설정값 등 "정답"이 될 수 있는 정보 포함.
주제 전환은 급격하게 일어남.
"""

SYNTHETIC_CONVERSATIONS = [
    # Topic 1: Kubernetes 클러스터 설정
    {
        "topic": "kubernetes_cluster",
        "turns": [
            ("user", "우리 프로덕션 k8s 클러스터 노드 수가 몇 개였지?"),
            ("assistant", "현재 프로덕션 클러스터는 worker 노드 12개, control plane 3개로 총 15개 노드입니다. 각 worker는 c5.2xlarge 인스턴스예요."),
            ("user", "메모리 limit은 어떻게 설정했어?"),
            ("assistant", "기본 Pod 메모리 limit은 2Gi, 최대 8Gi입니다. ML 워크로드용 namespace는 예외로 32Gi까지 허용했어요."),
        ],
        "questions": [
            ("프로덕션 k8s worker 노드 수는?", "12개"),
            ("ML 워크로드 최대 메모리 limit은?", "32Gi"),
        ]
    },
    
    # Topic 2: 데이터베이스 마이그레이션
    {
        "topic": "db_migration",
        "turns": [
            ("user", "PostgreSQL에서 TiDB로 마이그레이션 일정 어떻게 됐어?"),
            ("assistant", "3월 15일에 read replica 전환, 4월 1일에 primary 전환 예정입니다. 롤백 윈도우는 72시간이에요."),
            ("user", "데이터 양은?"),
            ("assistant", "현재 785GB, 마이그레이션 대상 테이블은 142개입니다. users 테이블이 가장 커서 단독 230GB예요."),
        ],
        "questions": [
            ("TiDB primary 전환 예정일은?", "4월 1일"),
            ("users 테이블 크기는?", "230GB"),
        ]
    },
    
    # Topic 3: ML 모델 하이퍼파라미터
    {
        "topic": "ml_hyperparameters",
        "turns": [
            ("user", "지난주 실험한 Transformer 모델 learning rate 얼마였어?"),
            ("assistant", "base lr 3e-4, warmup 2000 steps, cosine decay로 최종 1e-6까지 떨어뜨렸어요."),
            ("user", "배치 사이즈랑 gradient accumulation은?"),
            ("assistant", "per-GPU batch 32, gradient accumulation 8 steps라서 effective batch는 256이에요. A100 8장 사용."),
            ("user", "결과는?"),
            ("assistant", "validation loss 0.847, BLEU 42.3으로 baseline 대비 +2.1 개선됐어요."),
        ],
        "questions": [
            ("Transformer 모델의 effective batch size는?", "256"),
            ("validation loss는?", "0.847"),
            ("warmup steps는?", "2000"),
        ]
    },
    
    # Topic 4: 보안 인시던트
    {
        "topic": "security_incident",
        "turns": [
            ("user", "저번 달 보안 인시던트 원인이 뭐였어?"),
            ("assistant", "S3 버킷 misconfiguration이었어요. public access가 실수로 열려있었고, 12시간 동안 노출됐습니다."),
            ("user", "영향 범위는?"),
            ("assistant", "로그 데이터 47GB가 노출됐는데, PII는 없었고 서버 메트릭만 있었어요. 고객 데이터는 안전합니다."),
        ],
        "questions": [
            ("S3 버킷 노출 시간은?", "12시간"),
            ("노출된 로그 데이터 크기는?", "47GB"),
        ]
    },
    
    # Topic 5: API Rate Limiting
    {
        "topic": "api_rate_limiting",
        "turns": [
            ("user", "외부 API rate limit 설정 어떻게 했어?"),
            ("assistant", "Free tier는 100 req/min, Pro는 1000 req/min, Enterprise는 10000 req/min입니다."),
            ("user", "burst limit은?"),
            ("assistant", "각각 150%, 200%, 300%까지 5초간 burst 허용해요. Enterprise는 30000 req/min burst 가능."),
        ],
        "questions": [
            ("Pro tier의 rate limit은?", "1000 req/min"),
            ("Enterprise burst limit은?", "30000 req/min"),
        ]
    },
    
    # Topic 6: CI/CD 파이프라인
    {
        "topic": "cicd_pipeline",
        "turns": [
            ("user", "CI 파이프라인 평균 시간 얼마야?"),
            ("assistant", "unit test 3분, integration test 8분, build 4분으로 총 15분 정도예요. E2E는 별도로 25분."),
            ("user", "캐시 hit rate는?"),
            ("assistant", "npm cache 92%, docker layer cache 87%입니다. 캐시 미스 시 빌드가 35분까지 늘어나요."),
        ],
        "questions": [
            ("integration test 시간은?", "8분"),
            ("캐시 미스 시 빌드 시간은?", "35분"),
            ("docker layer cache hit rate는?", "87%"),
        ]
    },
    
    # Topic 7: 결제 시스템
    {
        "topic": "payment_system",
        "turns": [
            ("user", "결제 수수료율 어떻게 되지?"),
            ("assistant", "국내 카드 2.5%, 해외 카드 3.2%, 계좌이체 1.0%입니다. 월 거래액 1억 넘으면 0.3% 할인."),
            ("user", "정산 주기는?"),
            ("assistant", "T+2 정산이 기본이고, 프리미엄 가맹점은 T+1 가능해요. 수수료 0.1% 추가됩니다."),
        ],
        "questions": [
            ("해외 카드 수수료율은?", "3.2%"),
            ("T+1 정산 추가 수수료는?", "0.1%"),
        ]
    },
    
    # Topic 8: 로깅 인프라
    {
        "topic": "logging_infra",
        "turns": [
            ("user", "일일 로그 볼륨 얼마야?"),
            ("assistant", "평균 2.3TB/day, 피크 시 4.1TB까지 올라가요. 압축 후 저장은 약 380GB입니다."),
            ("user", "보관 기간은?"),
            ("assistant", "hot storage 7일, warm 30일, cold 365일입니다. compliance 로그는 7년 보관."),
        ],
        "questions": [
            ("피크 시 일일 로그 볼륨은?", "4.1TB"),
            ("compliance 로그 보관 기간은?", "7년"),
        ]
    },
    
    # Topic 9: 마이크로서비스 아키텍처
    {
        "topic": "microservices",
        "turns": [
            ("user", "현재 서비스 개수 몇 개야?"),
            ("assistant", "core 서비스 23개, supporting 서비스 47개로 총 70개입니다. 이 중 deprecated 5개는 올해 제거 예정."),
            ("user", "서비스 간 통신은?"),
            ("assistant", "sync는 gRPC가 85%, REST가 15%예요. async는 Kafka로 일일 메시지 1.2억 건 처리합니다."),
        ],
        "questions": [
            ("core 서비스 개수는?", "23개"),
            ("일일 Kafka 메시지 처리량은?", "1.2억 건"),
        ]
    },
    
    # Topic 10: 모니터링 알림 설정
    {
        "topic": "monitoring_alerts",
        "turns": [
            ("user", "Slack 알림 임계값 어떻게 설정했어?"),
            ("assistant", "CPU 80% 5분 지속, Memory 85% 3분 지속, Disk 90%면 즉시 알림입니다."),
            ("user", "PagerDuty는?"),
            ("assistant", "5xx error rate 1% 초과 시 P2, 5% 초과 시 P1으로 에스컬레이션됩니다. P1은 5분 내 응답 필수."),
        ],
        "questions": [
            ("Memory 알림 임계값은?", "85% 3분 지속"),
            ("P1 에스컬레이션 조건은?", "5xx error rate 5% 초과"),
            ("P1 응답 시간 SLA는?", "5분"),
        ]
    },
]

# 전체 대화를 flat하게 만들기
def get_flat_conversation() -> list[tuple[str, str]]:
    """모든 주제의 대화를 순서대로 합침."""
    result = []
    for topic_data in SYNTHETIC_CONVERSATIONS:
        result.extend(topic_data["turns"])
    return result

# 모든 질문과 정답
def get_all_questions() -> list[tuple[str, str, str]]:
    """(question, answer, topic) 튜플 리스트."""
    result = []
    for topic_data in SYNTHETIC_CONVERSATIONS:
        for q, a in topic_data["questions"]:
            result.append((q, a, topic_data["topic"]))
    return result

if __name__ == "__main__":
    print(f"Total topics: {len(SYNTHETIC_CONVERSATIONS)}")
    print(f"Total turns: {len(get_flat_conversation())}")
    print(f"Total questions: {len(get_all_questions())}")
    
    print("\nSample questions:")
    for q, a, t in get_all_questions()[:5]:
        print(f"  [{t}] Q: {q} → A: {a}")
