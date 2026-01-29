"""ERASE Benchmark Runner - Compare Single vs Dual-scored memory."""
import json
from pathlib import Path

from erase import ERASE, scope
from erase.schemas import MemoryChunk
from tests.benchmark.schemas import BenchmarkTask, BenchmarkResult, BenchmarkSummary


class SingleScoredMemory:
    """Traditional RAG: Only uses relevance score."""
    
    def __init__(self, config):
        self._erase = ERASE(config)
        self._threshold = config.threshold.retention
    
    def retrieve(self, text: str, query: str) -> list[MemoryChunk]:
        all_chunks = self._erase.score_all(text, query=query)
        return [c for c in all_chunks if c.retention_score >= self._threshold]


def load_tasks() -> list[BenchmarkTask]:
    """Load benchmark tasks from JSON."""
    data_path = Path(__file__).parent/'data'/'tasks.json'
    with open(data_path) as f:
        data = json.load(f)
    return [BenchmarkTask(**t) for t in data]


def check_keywords(chunks: list[MemoryChunk], keywords: list[str]) -> tuple[int, int]:
    """Check how many keywords are found in chunks."""
    combined_text = ' '.join(c.content for c in chunks)
    found = sum(1 for kw in keywords if kw in combined_text)
    return found, len(keywords)


def run_task(task: BenchmarkTask, erase: ERASE, single: SingleScoredMemory, config) -> BenchmarkResult:
    """Run a single benchmark task."""
    # Get results from both approaches
    single_results = single.retrieve(task.memory_bank, task.query)
    dual_results = erase(task.memory_bank, task.query)
    all_chunks = erase.score_all(task.memory_bank, task.query)
    
    # Count excluded by ERASE
    excluded = 0
    for c in all_chunks:
        in_single = c.retention_score >= config.threshold.retention
        in_dual = c.retention_score >= config.threshold.retention and c.erasure_score < config.threshold.erasure
        if in_single and not in_dual:
            excluded += 1
    
    # Check expected keywords
    keep_found, keep_total = check_keywords(dual_results, task.expected_keep)
    keep_precision = keep_found/keep_total if keep_total > 0 else 1.0
    
    # Check excluded keywords (should NOT be in dual results)
    exclude_found, exclude_total = check_keywords(dual_results, task.expected_exclude)
    exclude_precision = 1.0-(exclude_found/exclude_total) if exclude_total > 0 else 1.0
    
    # Noise reduction
    noise_reduction = 1.0-(len(dual_results)/len(single_results)) if len(single_results) > 0 else 0.0
    
    # Pass if both precisions are above threshold
    passed = keep_precision >= 0.5 and exclude_precision >= 0.5
    
    return BenchmarkResult(
        task_id=task.id,
        task_name=task.name,
        single_scored_chunks=len(single_results),
        dual_scored_chunks=len(dual_results),
        excluded_by_erase=excluded,
        keep_precision=keep_precision,
        exclude_precision=exclude_precision,
        noise_reduction=noise_reduction,
        passed=passed
    )


@scope
def main(config):
    """Run all benchmark tasks."""
    tasks = load_tasks()
    erase = ERASE(config)
    single = SingleScoredMemory(config)
    
    print("=" * 70)
    print("ERASE Benchmark: Single vs Dual-scored Memory")
    print("=" * 70)
    print(f"Retention threshold: {config.threshold.retention}")
    print(f"Erasure threshold: {config.threshold.erasure}")
    print(f"Tasks: {len(tasks)}")
    print()
    
    results = []
    for task in tasks:
        result = run_task(task, erase, single, config)
        results.append(result)
        
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"[{status}] {task.name}")
        print(f"       Single: {result.single_scored_chunks} chunks | Dual: {result.dual_scored_chunks} chunks")
        print(f"       Noise reduction: {result.noise_reduction:.0%}")
        print(f"       Keep precision: {result.keep_precision:.0%} | Exclude precision: {result.exclude_precision:.0%}")
        print()
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    avg_noise = sum(r.noise_reduction for r in results)/len(results)
    avg_keep = sum(r.keep_precision for r in results)/len(results)
    avg_exclude = sum(r.exclude_precision for r in results)/len(results)
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Passed: {passed}/{len(results)}")
    print(f"  Avg noise reduction: {avg_noise:.0%}")
    print(f"  Avg keep precision: {avg_keep:.0%}")
    print(f"  Avg exclude precision: {avg_exclude:.0%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
