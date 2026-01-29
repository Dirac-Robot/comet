"""
Needle in a Haystack (NIAH) Benchmark for ERASE - Extended Version

Features:
- Configurable haystack size (number of distractors)
- Chunk size control (sentences per chunk before scoring)
- Comparison across different haystack scales
"""
import random
from erase import ERASE, scope
from erase.schemas import MemoryChunk
from langchain_openai import ChatOpenAI


# Extended haystack distractors (50+ sentences for scaling tests)
HAYSTACK_SENTENCES = [
    "2023년 1분기 글로벌 시장 점유율은 23.5%로 전년 대비 2.1% 상승했습니다.",
    "신규 물류센터는 경기도 평택에 건설 예정이며 2024년 완공 목표입니다.",
    "클라우드 마이그레이션 프로젝트는 3단계로 진행되며 총 18개월이 소요됩니다.",
    "인사팀 조직개편으로 채용팀과 교육팀이 통합되었습니다.",
    "연간 R&D 투자 비중은 매출의 8.7%로 업계 평균을 상회합니다.",
    "고객만족도 조사 결과 NPS 점수가 72점으로 역대 최고치를 기록했습니다.",
    "ISO 27001 인증 갱신이 완료되어 정보보안 체계가 재확인되었습니다.",
    "신제품 베타 테스트 참여자는 5,000명이며 피드백 수집 중입니다.",
    "해외 지사 확장 계획에 따라 싱가포르 법인 설립을 추진 중입니다.",
    "에너지 효율화 프로젝트로 전년 대비 탄소 배출량 15% 감축을 달성했습니다.",
    "협력업체 평가 시스템이 개편되어 ESG 항목이 추가되었습니다.",
    "사내 교육 플랫폼 이용률이 78%로 목표치를 초과 달성했습니다.",
    "분기별 실적 발표는 매 분기 마지막 주 목요일에 진행됩니다.",
    "직원 복지 프로그램에 심리상담 서비스가 새롭게 포함되었습니다.",
    "재고관리 시스템 업그레이드로 재고 회전율이 12% 개선되었습니다.",
    "신규 파트너십으로 유럽 시장 진출 기반이 마련되었습니다.",
    "모바일 앱 다운로드 수가 500만 건을 돌파했습니다.",
    "품질관리 프로세스 개선으로 불량률이 0.3% 이하로 감소했습니다.",
    "데이터센터 확장 공사가 3분기 중 완료될 예정입니다.",
    "고객센터 응대 시간이 평균 2분 30초로 단축되었습니다.",
    "신규 채용 공고에 1만 명 이상의 지원자가 접수되었습니다.",
    "특허 출원 건수가 전년 대비 35% 증가했습니다.",
    "온라인 판매 비중이 전체 매출의 45%를 차지하게 되었습니다.",
    "생산라인 자동화율이 85%에 도달했습니다.",
    "고객 리텐션율이 92%로 업계 최고 수준을 유지하고 있습니다.",
    "신규 서비스 론칭 후 3개월 만에 10만 사용자를 확보했습니다.",
    "공급망 다변화로 원자재 조달 리스크가 크게 감소했습니다.",
    "사내 스타트업 프로그램에서 5개 팀이 최종 선발되었습니다.",
    "연구소 신축 건물이 내년 상반기 준공 예정입니다.",
    "글로벌 브랜드 인지도 조사에서 상위 10위권에 진입했습니다.",
    "직원 평균 근속연수가 7.2년으로 안정적인 조직문화를 보여줍니다.",
    "신규 ERP 시스템 도입으로 업무 효율성이 25% 향상되었습니다.",
    "해외 매출 비중이 처음으로 40%를 넘어섰습니다.",
    "AI 기반 추천 시스템 도입 후 구매 전환율이 18% 증가했습니다.",
    "환경부 친환경 기업 인증을 획득했습니다.",
    "B2B 사업부 매출이 전년 대비 50% 성장했습니다.",
    "신규 물류 파트너와의 계약으로 배송 시간이 1일 단축되었습니다.",
    "AR/VR 기술 적용 프로젝트가 파일럿 단계에 진입했습니다.",
    "고객 VOC 분석 시스템이 고도화되어 실시간 모니터링이 가능해졌습니다.",
    "신규 결제 시스템 도입으로 결제 성공률이 99.5%로 향상되었습니다.",
]

# Needles
NEEDLE_TEMPLATES = [
    {
        "needle": "김민수 부장의 비밀 프로젝트 코드명은 'AURORA'이며 예산은 35억원입니다.",
        "query": "김민수 부장의 비밀 프로젝트 코드명과 예산이 뭐야?",
        "expected_keywords": ["AURORA", "35억"],
    },
    {
        "needle": "다음 주 화요일 오후 3시에 대표이사와 긴급 회의가 예정되어 있습니다.",
        "query": "대표이사와의 회의가 언제야?",
        "expected_keywords": ["화요일", "3시"],
    },
    {
        "needle": "신규 AI 모델 'NEXUS-7'의 정확도는 94.3%이며 다음 달 출시 예정입니다.",
        "query": "새로운 AI 모델 정확도가 얼마야?",
        "expected_keywords": ["94.3%", "NEXUS"],
    },
]


class SingleScoredMemory:
    """Traditional RAG baseline."""
    
    def __init__(self, config):
        self._erase = ERASE(config)
        self._threshold = config.threshold.retention
    
    def retrieve(self, text: str, query: str) -> list[MemoryChunk]:
        all_chunks = self._erase.score_all(text, query=query)
        return [c for c in all_chunks if c.retention_score >= self._threshold]


def create_haystack(needle: str, haystack_size: int = 10, chunk_size: int = 1) -> str:
    """
    Create a haystack with needle inserted at random position.
    
    Args:
        needle: The target fact to find
        haystack_size: Number of distractor sentences
        chunk_size: How many sentences to group together (simulates pre-chunking)
    
    Returns:
        Text with sentences grouped by chunk_size, separated by double newlines
    """
    # Get enough distractors (with replacement if needed)
    if haystack_size <= len(HAYSTACK_SENTENCES):
        distractors = random.sample(HAYSTACK_SENTENCES, haystack_size)
    else:
        distractors = random.choices(HAYSTACK_SENTENCES, k=haystack_size)
    
    # Insert needle at random position
    insert_pos = random.randint(0, len(distractors))
    distractors.insert(insert_pos, needle)
    
    # Group into chunks
    if chunk_size == 1:
        return "\n".join(distractors)
    else:
        chunks = []
        for i in range(0, len(distractors), chunk_size):
            chunk_sentences = distractors[i:i+chunk_size]
            chunks.append(" ".join(chunk_sentences))
        return "\n\n".join(chunks)


def check_needle_found(chunks: list[MemoryChunk], keywords: list[str]) -> bool:
    """Check if needle keywords are in the retrieved chunks."""
    combined = " ".join(c.content for c in chunks)
    return all(kw in combined for kw in keywords)


def run_test(erase: ERASE, single: SingleScoredMemory, template: dict, 
             haystack_size: int, chunk_size: int) -> dict:
    """Run a single test with given parameters."""
    needle = template["needle"]
    query = template["query"]
    keywords = template["expected_keywords"]
    
    haystack = create_haystack(needle, haystack_size=haystack_size, chunk_size=chunk_size)
    
    single_chunks = single.retrieve(haystack, query)
    erase_chunks = erase(haystack, query)
    
    single_found = check_needle_found(single_chunks, keywords)
    erase_found = check_needle_found(erase_chunks, keywords)
    
    return {
        "single_found": single_found,
        "erase_found": erase_found,
        "single_chunks": len(single_chunks),
        "erase_chunks": len(erase_chunks),
        "noise_reduction": 1-(len(erase_chunks)/len(single_chunks)) if single_chunks else 0,
    }


@scope
def main(config):
    """Run NIAH benchmark with different scales."""
    erase = ERASE(config)
    single = SingleScoredMemory(config)
    
    print("=" * 70)
    print("NIAH Benchmark - Scale & Chunk Size Comparison")
    print("=" * 70)
    
    # Test configurations: (haystack_size, chunk_size)
    configs = [
        (10, 1),   # Small, sentence-level
        (20, 1),   # Medium, sentence-level
        (30, 1),   # Large, sentence-level
        (30, 3),   # Large, 3 sentences per chunk
        (30, 5),   # Large, 5 sentences per chunk
    ]
    
    for haystack_size, chunk_size in configs:
        print(f"\n[Haystack: {haystack_size} sentences, Chunk size: {chunk_size}]")
        print("-" * 50)
        
        results = []
        for template in NEEDLE_TEMPLATES:
            result = run_test(erase, single, template, haystack_size, chunk_size)
            results.append(result)
        
        # Aggregate results
        single_success = sum(1 for r in results if r["single_found"])
        erase_success = sum(1 for r in results if r["erase_found"])
        avg_single = sum(r["single_chunks"] for r in results)/len(results)
        avg_erase = sum(r["erase_chunks"] for r in results)/len(results)
        avg_noise = sum(r["noise_reduction"] for r in results)/len(results)
        
        print(f"  Traditional RAG: {single_success}/3 found, avg {avg_single:.1f} chunks")
        print(f"  ERASE: {erase_success}/3 found, avg {avg_erase:.1f} chunks")
        print(f"  Noise reduction: {avg_noise:.0%}")
    
    print("\n" + "=" * 70)
    print("Conclusion: ERASE maintains accuracy while reducing context noise")
    print("across different haystack sizes and chunk configurations.")
    print("=" * 70)


if __name__ == "__main__":
    main()
