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


def check_keywords(chunks: list[MemoryChunk], keywords: list[str]) -> tuple[list[str], list[str]]:
    """Check which keywords are found/missing in chunks."""
    combined_text = ' '.join(c.content for c in chunks)
    found = [kw for kw in keywords if kw in combined_text]
    missing = [kw for kw in keywords if kw not in combined_text]
    return found, missing


def run_task(task: BenchmarkTask, erase: ERASE, single: SingleScoredMemory, config) -> tuple[BenchmarkResult, dict]:
    """Run a single benchmark task. Returns result and details."""
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
    
    # Check expected keywords (should be KEPT)
    keep_found, keep_missing = check_keywords(dual_results, task.expected_keep)
    keep_precision = len(keep_found)/len(task.expected_keep) if task.expected_keep else 1.0
    
    # Check excluded keywords (should NOT be in dual results)
    exclude_leaked, exclude_blocked = check_keywords(dual_results, task.expected_exclude)
    exclude_precision = len(exclude_blocked)/len(task.expected_exclude) if task.expected_exclude else 1.0
    
    # Noise reduction
    noise_reduction = 1.0-(len(dual_results)/len(single_results)) if len(single_results) > 0 else 0.0
    
    # Pass if both precisions are above threshold
    passed = keep_precision >= 0.5 and exclude_precision >= 0.5
    
    result = BenchmarkResult(
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
    
    details = {
        'keep_found': keep_found,
        'keep_missing': keep_missing,  # False negatives - important info filtered out!
        'exclude_leaked': exclude_leaked,  # False positives - noise that got through
        'exclude_blocked': exclude_blocked,
    }
    
    return result, details


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
        result, details = run_task(task, erase, single, config)
        results.append(result)
        
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"[{status}] {task.name}")
        print(f"       Single: {result.single_scored_chunks} | Dual: {result.dual_scored_chunks} | Noise ↓ {result.noise_reduction:.0%}")
        
        # Show what was kept (expected)
        if details['keep_found']:
            print(f"       ✅ Kept: {', '.join(details['keep_found'])}")
        
        # IMPORTANT: Show what was accidentally filtered (false negatives)
        if details['keep_missing']:
            print(f"       ⚠️ MISSING (false negative): {', '.join(details['keep_missing'])}")
        
        # Show what noise got through (false positives)
        if details['exclude_leaked']:
            print(f"       ⚠️ LEAKED (false positive): {', '.join(details['exclude_leaked'])}")
        
        # Show what was correctly blocked
        if details['exclude_blocked']:
            print(f"       ❌ Blocked: {', '.join(details['exclude_blocked'])}")
        
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
    print(f"  Avg keep precision (↑ better): {avg_keep:.0%}")
    print(f"  Avg exclude precision (↑ better): {avg_exclude:.0%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
