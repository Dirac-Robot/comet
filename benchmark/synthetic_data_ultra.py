"""
ULTRA-HARD 벤치마크: 150+ 턴, 20+ 주제
- 각 주제가 6-8턴에 걸쳐 상세히 논의됨
- 더 많은 주제, 더 긴 컨텍스트
"""

ULTRA_HARD_CONVERSATIONS = {
    "project_alpha": [
        ("user", "프로젝트 알파 예산 회의 시작할게요. 현재 책정된 예산이 얼마죠?"),
        ("assistant", "프로젝트 알파의 초기 책정 예산은 10억원이었습니다."),
        ("user", "근데 추가 요구사항이 들어왔잖아요. 최종 예산은요?"),
        ("assistant", "추가 요구사항 반영 후 최종 예산은 12억 3천만원으로 확정되었습니다."),
        ("user", "1차 마일스톤은 언제로 잡혀있어요?"),
        ("assistant", "1차 마일스톤은 6월 23일입니다. 이때까지 MVP 완성 목표입니다."),
        ("user", "2차 마일스톤은요?"),
        ("assistant", "2차 마일스톤은 9월 15일, 베타 릴리즈 목표입니다."),
    ],
    "project_beta": [
        ("user", "프로젝트 베타 진행상황 공유해주세요."),
        ("assistant", "베타 프로젝트는 현재 설계 단계입니다."),
        ("user", "예산은 얼마로 책정됐나요?"),
        ("assistant", "베타 프로젝트 예산은 8억 7천만원입니다."),
        ("user", "ROI 기대치는 어떻게 되나요?"),
        ("assistant", "18개월 내 250% ROI를 기대하고 있습니다."),
        ("user", "팀 규모는요?"),
        ("assistant", "백엔드 5명, 프론트엔드 3명, QA 2명 총 10명입니다."),
    ],
    "samsung_contract": [
        ("user", "삼성전자 계약 조건 검토해봅시다."),
        ("assistant", "삼성전자와의 계약서를 준비했습니다."),
        ("user", "월 구독료가 얼마로 협의됐죠?"),
        ("assistant", "월 구독료는 8,500만원으로 협의되었습니다."),
        ("user", "계약 기간은요?"),
        ("assistant", "최소 계약 기간은 36개월입니다."),
        ("user", "위약금 조건은요?"),
        ("assistant", "중도해지 시 남은 기간 월 구독료의 30%가 위약금입니다."),
    ],
    "samsung_sla": [
        ("user", "삼성 계약의 SLA 조건도 확인해야 해요."),
        ("assistant", "SLA 조건 문서를 가져왔습니다."),
        ("user", "가용성 보장이 몇 퍼센트죠?"),
        ("assistant", "가용성 99.95%를 보장합니다."),
        ("user", "응답시간 기준은요?"),
        ("assistant", "평균 응답시간 200ms 이하를 보장합니다."),
        ("user", "SLA 미달성 시 패널티는요?"),
        ("assistant", "월 구독료의 최대 20%까지 크레딧으로 환급됩니다."),
    ],
    "lg_contract": [
        ("user", "LG화학 계약 조건도 정리해주세요."),
        ("assistant", "LG화학 계약서 초안입니다."),
        ("user", "월 구독료는 얼마예요?"),
        ("assistant", "LG화학 월 구독료는 6,200만원입니다."),
        ("user", "SLA 가용성은요?"),
        ("assistant", "99.9% 가용성을 보장합니다."),
        ("user", "계약 기간은?"),
        ("assistant", "24개월 계약입니다."),
    ],
    "sk_contract": [
        ("user", "SK하이닉스 계약도 논의해야죠."),
        ("assistant", "SK하이닉스 계약 조건입니다."),
        ("user", "월 구독료는요?"),
        ("assistant", "월 7,800만원입니다."),
        ("user", "특이 조건이 있나요?"),
        ("assistant", "연간 선결제 시 10% 할인 조건이 있습니다."),
    ],
    "security_incident": [
        ("user", "작년 보안 인시던트 리포트 공유해주세요."),
        ("assistant", "2025년 보안 인시던트 리포트입니다."),
        ("user", "Critical 등급 인시던트가 몇 건이었죠?"),
        ("assistant", "Critical 등급 인시던트는 총 3건이었습니다."),
        ("user", "가장 심각했던 건 뭐였어요?"),
        ("assistant", "1월 15일 DB 암호화 키 노출 사건이 가장 심각했으며, 2시간 47분간 지속되었습니다."),
        ("user", "해결은 어떻게 됐나요?"),
        ("assistant", "키 롤오버 후 영향받은 데이터 전체 재암호화했습니다."),
    ],
    "db_performance": [
        ("user", "DB 성능 지표 확인해볼게요."),
        ("assistant", "메인 DB 성능 리포트입니다."),
        ("user", "초당 쿼리 수 평균은요?"),
        ("assistant", "평균 초당 12,500건의 쿼리를 처리합니다."),
        ("user", "가장 느린 쿼리 응답시간은요?"),
        ("assistant", "최악의 경우 2.3초까지 걸리는 쿼리가 있습니다."),
        ("user", "어떤 쿼리예요?"),
        ("assistant", "대시보드용 집계 쿼리입니다. 최적화 예정입니다."),
    ],
    "api_metrics": [
        ("user", "API 사용량 지표 확인합시다."),
        ("assistant", "이번 달 API 메트릭스입니다."),
        ("user", "총 호출 수가 얼마나 되죠?"),
        ("assistant", "이번 달 API 총 호출 수는 8억 7천만 건입니다."),
        ("user", "5xx 에러율은요?"),
        ("assistant", "5xx 에러율은 0.00078%입니다."),
        ("user", "가장 많이 호출되는 엔드포인트는요?"),
        ("assistant", "/api/v2/users 엔드포인트가 전체의 34%를 차지합니다."),
    ],
    "ml_model": [
        ("user", "추천 모델 성능 리뷰할게요."),
        ("assistant", "추천 모델 v3.2 성능 리포트입니다."),
        ("user", "CTR 예측 정확도가 얼마죠?"),
        ("assistant", "CTR 예측 정확도는 89.3%입니다."),
        ("user", "A/B 테스트에서 전환율 상승은요?"),
        ("assistant", "A/B 테스트 결과 전환율이 15.2% 상승했습니다."),
        ("user", "모델 추론 속도는요?"),
        ("assistant", "평균 추론 속도 23ms입니다."),
    ],
    "project_gamma": [
        ("user", "프로젝트 감마 논의할게요."),
        ("assistant", "감마 프로젝트 개요서입니다."),
        ("user", "예산이 얼마로 책정됐나요?"),
        ("assistant", "프로젝트 감마 예산은 4억 5천만원입니다."),
        ("user", "POC 완료 목표 시기는요?"),
        ("assistant", "6월까지 POC 완료가 목표입니다."),
        ("user", "리스크 요소는요?"),
        ("assistant", "주요 리스크는 외부 API 의존성입니다."),
    ],
    "feb_incident": [
        ("user", "2월 1일 장애 사후 분석 해봅시다."),
        ("assistant", "2월 1일 장애 분석 보고서입니다."),
        ("user", "장애가 몇 시에 발생했죠?"),
        ("assistant", "장애는 14:23에 발생했습니다."),
        ("user", "복구까지 얼마나 걸렸어요?"),
        ("assistant", "복구까지 47분이 소요되었습니다."),
        ("user", "매출 손실은 얼마였죠?"),
        ("assistant", "장애로 인한 매출 손실은 약 1,800만원으로 추정됩니다."),
        ("user", "원인이 뭐였어요?"),
        ("assistant", "DB 커넥션 풀 고갈이 원인이었습니다."),
    ],
    "server_cluster_a": [
        ("user", "서버 클러스터 A 현황 공유해주세요."),
        ("assistant", "클러스터 A 상태 리포트입니다."),
        ("user", "물리 서버가 몇 대죠?"),
        ("assistant", "물리 서버 24대로 구성되어 있습니다."),
        ("user", "총 CPU 코어 수는요?"),
        ("assistant", "총 1,152개 코어를 보유하고 있습니다."),
        ("user", "메모리 용량은요?"),
        ("assistant", "총 메모리는 6,144GB입니다."),
    ],
    "server_cluster_b": [
        ("user", "클러스터 B 현황도 확인해볼게요."),
        ("assistant", "클러스터 B 리포트입니다."),
        ("user", "서버 대수는요?"),
        ("assistant", "물리 서버 16대입니다."),
        ("user", "GPU 구성은요?"),
        ("assistant", "A100 80GB GPU 64장 구성입니다."),
        ("user", "주 용도는요?"),
        ("assistant", "ML 학습 및 추론 전용 클러스터입니다."),
    ],
    "network_bandwidth": [
        ("user", "네트워크 대역폭 현황 확인해볼게요."),
        ("assistant", "네트워크 인프라 현황입니다."),
        ("user", "외부 인터넷 대역폭이 얼마죠?"),
        ("assistant", "외부 인터넷 연결은 40Gbps입니다."),
        ("user", "평균 사용률은요?"),
        ("assistant", "평균 사용률은 67%입니다."),
        ("user", "피크 시간대는 언제예요?"),
        ("assistant", "오후 2-4시에 피크가 발생합니다."),
    ],
    "cdn_usage": [
        ("user", "CDN 사용 현황 리뷰합시다."),
        ("assistant", "Cloudflare Enterprise 리포트입니다."),
        ("user", "월간 트래픽이 얼마나 되죠?"),
        ("assistant", "월간 데이터 전송량은 850TB입니다."),
        ("user", "캐시 히트율은요?"),
        ("assistant", "전체 캐시 히트율은 94.7%입니다."),
        ("user", "비용은 얼마예요?"),
        ("assistant", "월 CDN 비용은 3,200만원입니다."),
    ],
    "hiring_plan": [
        ("user", "올해 채용 계획 공유해주세요."),
        ("assistant", "2026년 채용 계획입니다."),
        ("user", "개발자 몇 명 채용 예정이에요?"),
        ("assistant", "개발자 15명 채용 예정입니다."),
        ("user", "연봉 예산은요?"),
        ("assistant", "평균 연봉 8,500만원 기준으로 책정했습니다."),
        ("user", "채용 일정은요?"),
        ("assistant", "상반기 10명, 하반기 5명 목표입니다."),
    ],
    "office_expansion": [
        ("user", "사무실 확장 계획은 어떻게 되나요?"),
        ("assistant", "사무실 확장 검토 보고서입니다."),
        ("user", "필요 면적이 얼마예요?"),
        ("assistant", "추가로 300평이 필요합니다."),
        ("user", "예상 임대료는요?"),
        ("assistant", "평당 월 12만원, 총 월 3,600만원 예상입니다."),
        ("user", "이전 시기는요?"),
        ("assistant", "8월 말 이전 목표입니다."),
    ],
    "marketing_budget": [
        ("user", "마케팅 예산 검토해봅시다."),
        ("assistant", "2026년 마케팅 예산안입니다."),
        ("user", "총 예산이 얼마예요?"),
        ("assistant", "연간 마케팅 예산은 15억원입니다."),
        ("user", "분기별 배분은요?"),
        ("assistant", "Q1: 4억, Q2: 3억, Q3: 3억, Q4: 5억 배분입니다."),
        ("user", "주요 채널은요?"),
        ("assistant", "디지털 광고 60%, 컨퍼런스 25%, 콘텐츠 15% 배분입니다."),
    ],
    "compliance_audit": [
        ("user", "컴플라이언스 감사 결과 공유해주세요."),
        ("assistant", "2025년 감사 결과 보고서입니다."),
        ("user", "주요 지적사항은요?"),
        ("assistant", "접근권한 관리 미흡이 주요 지적사항이었습니다."),
        ("user", "개선 기한은요?"),
        ("assistant", "3월 말까지 개선 완료해야 합니다."),
        ("user", "담당자는 누구예요?"),
        ("assistant", "보안팀 김철수 팀장이 담당합니다."),
    ],
}


def get_flat_conversation_ultra() -> list[tuple[str, str]]:
    """모든 대화를 순서대로 평탄화."""
    result = []
    for topic, turns in ULTRA_HARD_CONVERSATIONS.items():
        result.extend(turns)
    return result


def get_all_questions_ultra() -> list[tuple[str, str, str]]:
    """(질문, 정답, 토픽) 튜플 리스트."""
    return [
        # 프로젝트 알파
        ("프로젝트 알파 최종 예산은?", "12억 3천만원", "project_alpha"),
        ("프로젝트 알파 2차 마일스톤은?", "9월 15일", "project_alpha"),
        # 프로젝트 베타
        ("프로젝트 베타 예산은?", "8억 7천만원", "project_beta"),
        ("프로젝트 베타 팀 규모는?", "10명", "project_beta"),
        # 삼성 계약
        ("삼성전자 월 구독료는?", "8,500만원", "samsung_contract"),
        ("삼성전자 위약금 조건은?", "30%", "samsung_contract"),
        # 삼성 SLA
        ("삼성전자 SLA 가용성은?", "99.95%", "samsung_sla"),
        ("삼성 SLA 미달 패널티는?", "20%", "samsung_sla"),
        # LG 계약
        ("LG화학 월 구독료는?", "6,200만원", "lg_contract"),
        ("LG화학 계약 기간은?", "24개월", "lg_contract"),
        # SK 계약
        ("SK하이닉스 월 구독료는?", "7,800만원", "sk_contract"),
        ("SK 선결제 할인율은?", "10%", "sk_contract"),
        # 보안 인시던트
        ("작년 Critical 보안 인시던트 수는?", "3건", "security_incident"),
        ("1월 15일 DB 키 노출 시간은?", "2시간 47분", "security_incident"),
        # DB 성능
        ("메인 DB 초당 쿼리 수 평균은?", "12,500건", "db_performance"),
        ("가장 느린 쿼리 응답 시간은?", "2.3초", "db_performance"),
        # API 메트릭
        ("이번 달 API 총 호출 수는?", "8억 7천만 건", "api_metrics"),
        ("가장 많이 호출되는 엔드포인트는?", "/api/v2/users", "api_metrics"),
        # ML 모델
        ("추천 모델 CTR 예측 정확도는?", "89.3%", "ml_model"),
        ("추천 모델 평균 추론 속도는?", "23ms", "ml_model"),
        # 프로젝트 감마
        ("프로젝트 감마 예산은?", "4억 5천만원", "project_gamma"),
        ("프로젝트 감마 주요 리스크는?", "외부 API 의존성", "project_gamma"),
        # 2월 장애
        ("2월 1일 장애 발생 시각은?", "14:23", "feb_incident"),
        ("2월 1일 장애 원인은?", "DB 커넥션 풀 고갈", "feb_incident"),
        # 서버 클러스터 A
        ("클러스터 A 물리 서버 수는?", "24대", "server_cluster_a"),
        ("클러스터 A 총 메모리는?", "6,144GB", "server_cluster_a"),
        # 서버 클러스터 B
        ("클러스터 B GPU 구성은?", "A100 64장", "server_cluster_b"),
        ("클러스터 B 주 용도는?", "ML 학습 및 추론", "server_cluster_b"),
        # 네트워크
        ("외부 인터넷 대역폭은?", "40Gbps", "network_bandwidth"),
        ("네트워크 피크 시간대는?", "오후 2-4시", "network_bandwidth"),
        # CDN
        ("CDN 월간 데이터 전송량은?", "850TB", "cdn_usage"),
        ("CDN 월 비용은?", "3,200만원", "cdn_usage"),
        # 채용
        ("올해 개발자 채용 예정 인원은?", "15명", "hiring_plan"),
        ("채용 평균 연봉 기준은?", "8,500만원", "hiring_plan"),
        # 사무실
        ("사무실 확장 필요 면적은?", "300평", "office_expansion"),
        ("사무실 월 임대료 예상은?", "3,600만원", "office_expansion"),
        # 마케팅
        ("연간 마케팅 예산은?", "15억원", "marketing_budget"),
        ("Q4 마케팅 예산은?", "5억", "marketing_budget"),
        # 컴플라이언스
        ("컴플라이언스 주요 지적사항은?", "접근권한 관리 미흡", "compliance_audit"),
        ("컴플라이언스 개선 담당자는?", "김철수", "compliance_audit"),
    ]
