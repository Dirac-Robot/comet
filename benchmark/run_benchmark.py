"""
CoMeT vs Baseline 벤치마크

비교 대상:
1. Baseline: 전체 대화를 단순 요약한 뒤 질문에 답변
2. CoMeT: 구조화된 메모리 + read_memory 툴로 필요한 정보 탐색
"""
import json
from typing import Optional
from dataclasses import dataclass, field

from ato.adict import ADict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from loguru import logger

from comet import CoMeT, scope
from benchmark.synthetic_data import (
    SYNTHETIC_CONVERSATIONS,
    get_flat_conversation,
    get_all_questions,
)


@dataclass
class BenchmarkResult:
    question: str
    expected: str
    topic: str
    baseline_answer: str = ""
    comet_answer: str = ""
    baseline_correct: bool = False
    comet_correct: bool = False


def judge_answer(expected: str, actual: str, llm: ChatOpenAI) -> bool:
    """LLM으로 정답 여부 판단."""
    # "I don't know" 또는 유사 답변은 무조건 오답
    if any(phrase in actual.lower() for phrase in ["i don't know", "모르", "알 수 없", "not available", "no information"]):
        return False
    
    prompt = f"""Judge if the actual answer correctly contains the expected information.

Expected answer: {expected}
Actual answer: {actual}

Rules:
- CORRECT: The actual answer contains the expected value/information
- INCORRECT: The actual answer is wrong, missing, or says "I don't know"

Respond with only "CORRECT" or "INCORRECT"."""
    
    response = llm.invoke(prompt)
    return "CORRECT" in response.content.upper()


class BaselineAgent:
    """단순 요약 기반 에이전트."""
    
    def __init__(self, config: ADict):
        self._llm = ChatOpenAI(model=config.main_model)
        self._summary: str = ""
    
    def ingest(self, conversations: list[tuple[str, str]]):
        """대화 전체를 요약."""
        conv_text = '\n'.join([f"[{role}] {content}" for role, content in conversations])
        
        prompt = f"""Summarize the following conversation history.
Keep all important details, numbers, dates, and specific values.

Conversation:
{conv_text}

Provide a comprehensive summary:"""
        
        response = self._llm.invoke(prompt)
        self._summary = response.content
        logger.info(f"Baseline summary length: {len(self._summary)} chars")
    
    def answer(self, question: str) -> str:
        """요약 기반으로 질문에 답변."""
        prompt = f"""Based on the following summary, answer the question.
If the information is not available, say "I don't know".

Summary:
{self._summary}

Question: {question}

Answer concisely:"""
        
        response = self._llm.invoke(prompt)
        return response.content


class CoMeTAgent:
    """CoMeT 구조화 메모리 기반 에이전트."""
    
    def __init__(self, config: ADict):
        self._config = config
        self._memo = CoMeT(config)
        self._llm = ChatOpenAI(model=config.main_model)
    
    def ingest(self, conversations: list[tuple[str, str]]):
        """대화를 CoMeT에 추가 (자동 compacting)."""
        for role, content in conversations:
            node = self._memo.add(f"[{role}] {content}")
            if node:
                logger.debug(f"Compacted: {node.node_id}")
        
        # 남은 버퍼 강제 compacting
        self._memo.force_compact()
        logger.info(f"CoMeT nodes: {len(self._memo.list_memories())}")
    
    def answer(self, question: str) -> str:
        """메모리 탐색 후 질문에 답변."""
        # Step 1: 컨텍스트 윈도우 가져오기 (요약 레벨)
        context = self._memo.get_context_window(max_nodes=20)
        
        # Step 2: 관련 노드 찾기
        nodes = self._memo.list_memories()
        
        # Step 3: 질문 기반으로 depth=2 탐색할 노드 선택
        select_prompt = f"""Given these memory summaries, which node IDs are most relevant to answer the question?

{context}

Question: {question}

Return only the node IDs as comma-separated list (e.g., mem_xxx, mem_yyy), or "NONE" if no relevant nodes:"""
        
        select_response = self._llm.invoke(select_prompt)
        selected_ids = [id.strip() for id in select_response.content.split(',') if id.strip().startswith('mem_')]
        
        # Step 4: 선택된 노드의 raw 데이터 가져오기
        detailed_context = []
        for node_id in selected_ids[:3]:  # 최대 3개
            raw = self._memo.read_memory(node_id, depth=2)
            if raw:
                detailed_context.append(raw)
        
        # Step 5: 최종 답변
        final_context = '\n\n'.join(detailed_context) if detailed_context else context
        
        answer_prompt = f"""Based on the following context, answer the question.
If the information is not available, say "I don't know".

Context:
{final_context}

Question: {question}

Answer concisely:"""
        
        response = self._llm.invoke(answer_prompt)
        return response.content


@scope
def main(config):
    print("=" * 60)
    print("[CoMeT vs Baseline Benchmark]")
    print(f"Questions: {len(get_all_questions())}")
    print(f"Topics: {len(SYNTHETIC_CONVERSATIONS)}")
    print("=" * 60)
    
    # 에이전트 초기화
    baseline = BaselineAgent(config)
    comet_agent = CoMeTAgent(config)
    judge_llm = ChatOpenAI(model='gpt-4o-mini')
    
    # 대화 주입
    conversations = get_flat_conversation()
    print(f"\n[Ingesting {len(conversations)} turns...]")
    
    baseline.ingest(conversations)
    comet_agent.ingest(conversations)
    
    # 벤치마크 실행
    results: list[BenchmarkResult] = []
    questions = get_all_questions()
    
    print(f"\n[Testing {len(questions)} questions...]")
    print("-" * 60)
    
    for i, (question, expected, topic) in enumerate(questions, 1):
        result = BenchmarkResult(
            question=question,
            expected=expected,
            topic=topic,
        )
        
        # Baseline 답변
        result.baseline_answer = baseline.answer(question)
        result.baseline_correct = judge_answer(expected, result.baseline_answer, judge_llm)
        
        # CoMeT 답변
        result.comet_answer = comet_agent.answer(question)
        result.comet_correct = judge_answer(expected, result.comet_answer, judge_llm)
        
        status_b = "✅" if result.baseline_correct else "❌"
        status_c = "✅" if result.comet_correct else "❌"
        
        print(f"Q{i}: {question[:40]}...")
        print(f"     Expected: {expected}")
        print(f"     Baseline: {status_b} {result.baseline_answer[:50]}...")
        print(f"     CoMeT:  {status_c} {result.comet_answer[:50]}...")
        print()
        
        results.append(result)
    
    # 최종 결과
    baseline_score = sum(1 for r in results if r.baseline_correct)
    comet_score = sum(1 for r in results if r.comet_correct)
    
    print("=" * 60)
    print("[FINAL RESULTS]")
    print(f"Baseline: {baseline_score}/{len(results)} ({100*baseline_score/len(results):.1f}%)")
    print(f"CoMeT:  {comet_score}/{len(results)} ({100*comet_score/len(results):.1f}%)")
    print("=" * 60)
    
    # 결과 저장
    output = {
        'baseline_score': baseline_score,
        'comet_score': comet_score,
        'total': len(results),
        'results': [
            {
                'question': r.question,
                'expected': r.expected,
                'topic': r.topic,
                'baseline': {'answer': r.baseline_answer, 'correct': r.baseline_correct},
                'comet': {'answer': r.comet_answer, 'correct': r.comet_correct},
            }
            for r in results
        ]
    }
    
    with open('benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\nResults saved to benchmark_results.json")


if __name__ == "__main__":
    main()
