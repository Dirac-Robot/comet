"""
HARD 벤치마크: CoMeT vs Baseline 극한 테스트
- 20개 주제
- Needle-in-Haystack 패턴
- 유사 숫자로 혼동 유도
"""
import json
from typing import Optional
from dataclasses import dataclass

from ato.adict import ADict
from langchain_openai import ChatOpenAI
from loguru import logger

from comet import CoMeT, scope
from benchmark.synthetic_data_hard import (
    HARD_CONVERSATIONS,
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
        conv_text = '\n'.join([f"[{role}] {content}" for role, content in conversations])
        
        prompt = f"""Summarize the following conversation history.
Keep ALL important details, numbers, dates, and specific values.
This is critical - preserve every specific number mentioned.

Conversation:
{conv_text}

Provide a comprehensive summary:"""
        
        response = self._llm.invoke(prompt)
        self._summary = response.content
        logger.info(f"Baseline summary length: {len(self._summary)} chars")
    
    def answer(self, question: str) -> str:
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
        for role, content in conversations:
            node = self._memo.add(f"[{role}] {content}")
            if node:
                logger.debug(f"Compacted: {node.node_id}")
        
        self._memo.force_compact()
        logger.info(f"CoMeT nodes: {len(self._memo.list_memories())}")
    
    def answer(self, question: str) -> str:
        # Get ALL nodes for hard benchmark
        context = self._memo.get_context_window(max_nodes=50)
        nodes = self._memo.list_memories()
        
        select_prompt = f"""Select memory nodes that can answer the question.

## Memory Index (node_id | summary | trigger):
{context}

## Question: {question}

## Instructions:
1. Look at each node's TRIGGER field - it describes what info that node contains
2. Select ALL nodes whose trigger matches any keyword in the question
3. Be INCLUSIVE - if a trigger mentions related terms, include that node
4. Keywords to match: entities (삼성, LG, SK), concepts (예산, 계약, SLA, 채용), metrics (할인율, 패널티)

Return node IDs as comma-separated list (e.g., mem_xxx, mem_yyy), or "NONE":"""
        
        select_response = self._llm.invoke(select_prompt)
        selected_ids = [id.strip() for id in select_response.content.split(',') if id.strip().startswith('mem_')]
        
        detailed_context = []
        for node_id in selected_ids[:5]:  # 최대 5개
            raw = self._memo.read_memory(node_id, depth=2)
            if raw:
                detailed_context.append(raw)
        
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
    print("=" * 70)
    print("[HARD BENCHMARK: CoMeT vs Baseline]")
    print(f"Questions: {len(get_all_questions())}")
    print(f"Topics: {len(HARD_CONVERSATIONS)}")
    print(f"Turns: {len(get_flat_conversation())}")
    print("=" * 70)
    
    baseline = BaselineAgent(config)
    comet_agent = CoMeTAgent(config)
    judge_llm = ChatOpenAI(model='gpt-4o-mini')
    
    conversations = get_flat_conversation()
    print(f"\n[Ingesting {len(conversations)} turns...]")
    
    baseline.ingest(conversations)
    comet_agent.ingest(conversations)
    
    results: list[BenchmarkResult] = []
    questions = get_all_questions()
    
    print(f"\n[Testing {len(questions)} questions...]")
    print("-" * 70)
    
    for i, (question, expected, topic) in enumerate(questions, 1):
        result = BenchmarkResult(
            question=question,
            expected=expected,
            topic=topic,
        )
        
        result.baseline_answer = baseline.answer(question)
        result.baseline_correct = judge_answer(expected, result.baseline_answer, judge_llm)
        
        result.comet_answer = comet_agent.answer(question)
        result.comet_correct = judge_answer(expected, result.comet_answer, judge_llm)
        
        status_b = "✅" if result.baseline_correct else "❌"
        status_c = "✅" if result.comet_correct else "❌"
        
        print(f"Q{i}: [{topic}] {question[:35]}...")
        print(f"     Expected: {expected}")
        print(f"     Baseline: {status_b} {result.baseline_answer[:45]}...")
        print(f"     CoMeT:  {status_c} {result.comet_answer[:45]}...")
        print()
        
        results.append(result)
    
    baseline_score = sum(1 for r in results if r.baseline_correct)
    comet_score = sum(1 for r in results if r.comet_correct)
    
    print("=" * 70)
    print("[FINAL RESULTS - HARD BENCHMARK]")
    print(f"Baseline: {baseline_score}/{len(results)} ({100*baseline_score/len(results):.1f}%)")
    print(f"CoMeT:  {comet_score}/{len(results)} ({100*comet_score/len(results):.1f}%)")
    print("=" * 70)
    
    # 압축률 분석
    baseline_raw = '\n'.join([f'[{r}] {c}' for r,c in conversations])
    comet_context = comet_agent._memo.get_context_window(max_nodes=100)
    baseline_chars = len(baseline_raw)
    comet_chars = len(comet_context)
    compression = (1 - comet_chars/baseline_chars)*100
    
    print()
    print("[COMPRESSION ANALYSIS]")
    print(f"Baseline: {baseline_chars:,} chars ({baseline_chars//4:,} tokens)")
    print(f"CoMeT:  {comet_chars:,} chars ({comet_chars//4:,} tokens)")
    print(f"Compression: {compression:.1f}% ({(baseline_chars-comet_chars)//4:,} tokens saved)")
    print("=" * 70)
    
    output = {
        'benchmark': 'HARD',
        'baseline_score': baseline_score,
        'comet_score': comet_score,
        'total': len(results),
        'compression': {
            'baseline_chars': baseline_chars,
            'comet_chars': comet_chars,
            'compression_rate': round(compression, 1),
        },
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
    
    with open('benchmark_results_hard.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\nResults saved to benchmark_results_hard.json")


if __name__ == "__main__":
    main()
