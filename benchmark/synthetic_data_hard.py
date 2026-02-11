"""
HARD 벤치마크용 합성 데이터
- 100+ 턴으로 토큰 수 대폭 증가
- Needle-in-Haystack: 중요 정보를 노이즈로 둘러싸기
- 유사 숫자 혼동: 12, 15, 12.5, 120 등 비슷한 값 반복
- 극한의 주제 전환: 20개 주제
"""

HARD_CONVERSATIONS = [
    # Topic 1: 서버 인프라 A (노이즈)
    {
        "topic": "server_infra_a",
        "turns": [
            ("user", "서버 A 클러스터 현황 알려줘"),
            ("assistant", "서버 A 클러스터는 현재 15대의 물리 서버로 구성되어 있고, 각각 128GB RAM을 탑재했습니다."),
            ("user", "CPU 코어는?"),
            ("assistant", "서버당 32코어이고, 총 480코어 용량입니다. 현재 평균 사용률은 67%예요."),
        ],
        "questions": []  # 노이즈용
    },
    
    # Topic 2: 프로젝트 알파 (NEEDLE - 중요 정보)
    {
        "topic": "project_alpha",
        "turns": [
            ("user", "프로젝트 알파 예산 확정됐어?"),
            ("assistant", "네, 최종 예산은 12억 3천만원으로 확정됐습니다. 1차 집행은 4억 2천만원이에요."),
            ("user", "일정은?"),
            ("assistant", "킥오프는 3월 17일, 1차 마일스톤은 6월 23일, 최종 납품은 12월 8일입니다."),
        ],
        "questions": [
            ("프로젝트 알파 최종 예산은?", "12억 3천만원"),
            ("프로젝트 알파 1차 마일스톤은?", "6월 23일"),
        ]
    },
    
    # Topic 3: 서버 인프라 B (노이즈 + 혼동 숫자)
    {
        "topic": "server_infra_b",
        "turns": [
            ("user", "서버 B 클러스터는?"),
            ("assistant", "서버 B는 12대로 구성되어 있고, 각각 256GB RAM입니다. A보다 메모리 집약적이에요."),
            ("user", "GPU는?"),
            ("assistant", "서버당 A100 4장씩, 총 48장입니다. 추론용으로 사용 중이에요."),
            ("user", "전력 소비는?"),
            ("assistant", "클러스터 전체 평균 127kW이고, 피크 시 153kW까지 올라갑니다."),
        ],
        "questions": []
    },
    
    # Topic 4: 고객사 계약 (NEEDLE)
    {
        "topic": "client_contract",
        "turns": [
            ("user", "삼성전자 계약 조건 어떻게 됐어?"),
            ("assistant", "월 구독료 8,500만원, 계약 기간 36개월입니다. 총 계약액은 30억 6천만원이에요."),
            ("user", "SLA는?"),
            ("assistant", "가용성 99.95%, 응답시간 200ms 이하 보장입니다. 위반 시 월 요금의 15% 배상."),
        ],
        "questions": [
            ("삼성전자 월 구독료는?", "8,500만원"),
            ("삼성전자 SLA 가용성은?", "99.95%"),
        ]
    },
    
    # Topic 5: 팀 현황 (노이즈)
    {
        "topic": "team_status",
        "turns": [
            ("user", "개발팀 인원 현황"),
            ("assistant", "백엔드 12명, 프론트엔드 8명, DevOps 4명으로 총 24명입니다."),
            ("user", "신규 채용은?"),
            ("assistant", "이번 분기 목표는 백엔드 3명, ML 엔지니어 2명입니다. 현재 면접 진행 중인 후보가 7명이에요."),
        ],
        "questions": []
    },
    
    # Topic 6: 프로젝트 베타 (NEEDLE - 알파와 혼동 유도)
    {
        "topic": "project_beta",
        "turns": [
            ("user", "프로젝트 베타 예산은?"),
            ("assistant", "베타는 8억 7천만원 규모입니다. 알파보다 작지만 ROI 기대치는 높아요."),
            ("user", "일정은 어떻게 돼?"),
            ("assistant", "킥오프 4월 5일, 1차 마일스톤 7월 15일, 최종 납품 11월 30일입니다."),
        ],
        "questions": [
            ("프로젝트 베타 예산은?", "8억 7천만원"),
            ("프로젝트 베타 1차 마일스톤은?", "7월 15일"),
        ]
    },
    
    # Topic 7: 네트워크 설정 (노이즈)
    {
        "topic": "network_config",
        "turns": [
            ("user", "내부 네트워크 대역폭 얼마야?"),
            ("assistant", "코어 스위치 간 100Gbps, 서버 연결은 25Gbps입니다. 총 용량은 1.2Tbps예요."),
            ("user", "외부 연결은?"),
            ("assistant", "인터넷 회선 10Gbps 2개로 이중화했고, 평균 사용률 45%입니다."),
            ("user", "CDN은?"),
            ("assistant", "Cloudflare Enterprise 사용 중이고, 월 트래픽 평균 850TB입니다."),
        ],
        "questions": []
    },
    
    # Topic 8: 보안 인시던트 기록 (NEEDLE)
    {
        "topic": "security_log",
        "turns": [
            ("user", "작년 보안 인시던트 통계 알려줘"),
            ("assistant", "총 147건 발생했고, Critical 3건, High 23건, Medium 121건이었습니다."),
            ("user", "Critical 3건이 뭐였어?"),
            ("assistant", "1월 15일 DB 접근 키 노출(2시간 47분 노출), 5월 3일 DDoS 공격(서비스 23분 다운), 9월 22일 내부자 데이터 유출 시도(미수)입니다."),
        ],
        "questions": [
            ("작년 Critical 보안 인시던트 수는?", "3건"),
            ("1월 15일 DB 키 노출 시간은?", "2시간 47분"),
        ]
    },
    
    # Topic 9: 비용 최적화 (노이즈 + 혼동 숫자)
    {
        "topic": "cost_optimization",
        "turns": [
            ("user", "클라우드 비용 현황"),
            ("assistant", "월 평균 12억원이고, 작년 대비 23% 증가했어요. 주요 원인은 GPU 인스턴스 확대입니다."),
            ("user", "절감 목표는?"),
            ("assistant", "올해 목표는 15% 절감, 즉 월 1억 8천만원 줄이는 거예요. 현재까지 7% 달성했습니다."),
        ],
        "questions": []
    },
    
    # Topic 10: 데이터베이스 성능 (NEEDLE - 유사 숫자)
    {
        "topic": "db_performance",
        "turns": [
            ("user", "메인 DB 성능 지표 알려줘"),
            ("assistant", "초당 쿼리 수 평균 12,500건, 피크 시 18,700건입니다. 평균 응답 시간 8.3ms예요."),
            ("user", "슬로우 쿼리는?"),
            ("assistant", "100ms 이상 쿼리가 일 평균 847건 발생합니다. 가장 느린 쿼리는 리포트용으로 2.3초 걸려요."),
        ],
        "questions": [
            ("메인 DB 초당 쿼리 수 평균은?", "12,500건"),
            ("가장 느린 쿼리 응답 시간은?", "2.3초"),
        ]
    },
    
    # Topic 11: 마케팅 캠페인 A (노이즈)
    {
        "topic": "marketing_a",
        "turns": [
            ("user", "봄 캠페인 성과 어때?"),
            ("assistant", "총 광고비 3억 2천만원 집행했고, 신규 가입 12,400명 유치했어요. CAC가 25,800원입니다."),
            ("user", "ROI는?"),
            ("assistant", "예상 LTV 기준 ROAS 2.7배예요. 목표인 2.5배를 초과 달성했습니다."),
        ],
        "questions": []
    },
    
    # Topic 12: API 사용량 (NEEDLE)
    {
        "topic": "api_usage",
        "turns": [
            ("user", "이번 달 API 사용량 현황"),
            ("assistant", "총 호출 수 8억 7천만 건이고, 일 평균 2,900만 건입니다. 피크 시간대는 오후 2-4시예요."),
            ("user", "에러율은?"),
            ("assistant", "전체 에러율 0.0023%이고, 5xx 에러는 0.00078%입니다. SLA 기준 0.01% 이하라 안전합니다."),
        ],
        "questions": [
            ("이번 달 API 총 호출 수는?", "8억 7천만 건"),
            ("API 5xx 에러율은?", "0.00078%"),
        ]
    },
    
    # Topic 13: 서버 인프라 C (노이즈 + 혼동)
    {
        "topic": "server_infra_c",
        "turns": [
            ("user", "서버 C 클러스터는?"),
            ("assistant", "테스트/스테이징용으로 8대 운영 중이고, 각각 64GB RAM입니다. 프로덕션의 1/2 스펙이에요."),
            ("user", "비용은?"),
            ("assistant", "월 1,200만원 정도 나가고, 야간에는 자동 스케일다운해서 40% 절감 중입니다."),
        ],
        "questions": []
    },
    
    # Topic 14: 고객사 계약 2 (NEEDLE - 첫 번째와 혼동)
    {
        "topic": "client_contract_2",
        "turns": [
            ("user", "LG화학 계약 어떻게 됐어?"),
            ("assistant", "월 구독료 6,200만원, 계약 기간 24개월입니다. 총 계약액은 14억 8천8백만원이에요."),
            ("user", "특이사항은?"),
            ("assistant", "온프레미스 설치 옵션 포함이고, 설치비 별도 2억원입니다. SLA는 99.9% 가용성이에요."),
        ],
        "questions": [
            ("LG화학 월 구독료는?", "6,200만원"),
            ("LG화학 SLA 가용성은?", "99.9%"),
        ]
    },
    
    # Topic 15: 기술 부채 (노이즈)
    {
        "topic": "tech_debt",
        "turns": [
            ("user", "기술 부채 현황"),
            ("assistant", "현재 측정된 기술 부채는 총 1,247 story point입니다. 이 중 긴급이 312 SP예요."),
            ("user", "해결 계획은?"),
            ("assistant", "분기당 20% 해결 목표이고, 이번 분기는 250 SP 처리할 예정입니다."),
            ("user", "인력 할당은?"),
            ("assistant", "전체 개발 리소스의 15%를 기술 부채 해결에 배정했어요."),
        ],
        "questions": []
    },
    
    # Topic 16: ML 모델 성능 (NEEDLE - 비슷한 숫자들)
    {
        "topic": "ml_model_perf",
        "turns": [
            ("user", "추천 모델 성능 어때?"),
            ("assistant", "CTR 예측 정확도 89.3%, 실제 CTR 개선 12.7% 달성했습니다."),
            ("user", "A/B 테스트 결과는?"),
            ("assistant", "대조군 대비 전환율 15.2% 상승이고, p-value 0.0023으로 통계적 유의미해요."),
            ("user", "다음 목표는?"),
            ("assistant", "정확도 92% 이상, 추론 지연시간 50ms 이하 달성이 목표입니다."),
        ],
        "questions": [
            ("추천 모델 CTR 예측 정확도는?", "89.3%"),
            ("A/B 테스트 전환율 상승폭은?", "15.2%"),
        ]
    },
    
    # Topic 17: 백업 정책 (노이즈)
    {
        "topic": "backup_policy",
        "turns": [
            ("user", "백업 정책 현황"),
            ("assistant", "풀 백업은 매주 일요일, 증분 백업은 매일 새벽 3시에 진행합니다."),
            ("user", "보관 기간은?"),
            ("assistant", "일일 백업 30일, 주간 백업 52주, 월간 백업 7년 보관입니다."),
            ("user", "복구 테스트는?"),
            ("assistant", "매월 1회 DR 테스트 진행하고, 목표 RTO 4시간, RPO 1시간입니다."),
        ],
        "questions": []
    },
    
    # Topic 18: 프로젝트 감마 (NEEDLE - 알파/베타와 혼동)
    {
        "topic": "project_gamma",
        "turns": [
            ("user", "프로젝트 감마 현황"),
            ("assistant", "감마는 R&D 프로젝트로 예산 4억 5천만원입니다. 실험적 성격이라 예산 유동적이에요."),
            ("user", "인력은?"),
            ("assistant", "풀타임 3명, 파트타임 2명 투입 중입니다. 6월까지 POC 완료 목표예요."),
            ("user", "예상 성과는?"),
            ("assistant", "성공 시 연간 운영비 17% 절감 효과 예상됩니다."),
        ],
        "questions": [
            ("프로젝트 감마 예산은?", "4억 5천만원"),
            ("프로젝트 감마 POC 완료 목표 시기는?", "6월"),
        ]
    },
    
    # Topic 19: 지표 대시보드 (노이즈 + 숫자 폭탄)
    {
        "topic": "metrics_dashboard",
        "turns": [
            ("user", "핵심 지표 대시보드 수치들"),
            ("assistant", "DAU 127,000, MAU 1,850,000, 일 평균 세션 수 347,000, 평균 세션 시간 8분 23초입니다."),
            ("user", "전환 지표는?"),
            ("assistant", "회원가입 전환율 4.7%, 결제 전환율 2.3%, 재구매율 34.8%입니다."),
            ("user", "이탈률은?"),
            ("assistant", "첫 달 이탈률 42%, 3개월 이탈률 58%, 12개월 잔존율 23.7%예요."),
        ],
        "questions": []
    },
    
    # Topic 20: 인시던트 특정 (NEEDLE - 매우 구체적)
    {
        "topic": "specific_incident",
        "turns": [
            ("user", "지난주 목요일 장애 원인 뭐였어?"),
            ("assistant", "2월 1일 목요일 14:23에 발생한 장애는 Redis 클러스터 메모리 부족이었어요."),
            ("user", "영향 범위는?"),
            ("assistant", "전체 사용자의 37%가 영향받았고, 복구까지 47분 걸렸습니다. 매출 손실 약 1,800만원 추정."),
            ("user", "재발 방지책은?"),
            ("assistant", "메모리 알림 임계값을 80%에서 70%로 낮추고, 자동 스케일아웃 정책 추가했어요."),
        ],
        "questions": [
            ("2월 1일 장애 발생 시각은?", "14:23"),
            ("2월 1일 장애 복구 시간은?", "47분"),
            ("2월 1일 장애로 인한 매출 손실은?", "1,800만원"),
        ]
    },
]

def get_flat_conversation() -> list[tuple[str, str]]:
    """모든 주제의 대화를 순서대로 합침."""
    result = []
    for topic_data in HARD_CONVERSATIONS:
        result.extend(topic_data["turns"])
    return result

def get_all_questions() -> list[tuple[str, str, str]]:
    """(question, answer, topic) 튜플 리스트."""
    result = []
    for topic_data in HARD_CONVERSATIONS:
        for q, a in topic_data["questions"]:
            result.append((q, a, topic_data["topic"]))
    return result

if __name__ == "__main__":
    flat = get_flat_conversation()
    questions = get_all_questions()
    
    print(f"Total topics: {len(HARD_CONVERSATIONS)}")
    print(f"Total turns: {len(flat)}")
    print(f"Total questions: {len(questions)}")
    
    # 토큰 수 대략 추정
    total_chars = sum(len(t[1]) for t in flat)
    print(f"Approx tokens: {total_chars // 4}")
    
    print("\nQuestions (NEEDLE positions):")
    for q, a, t in questions:
        print(f"  [{t}] Q: {q} → A: {a}")
