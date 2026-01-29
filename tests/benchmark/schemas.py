"""Schemas for ERASE benchmark framework."""
from pydantic import BaseModel, Field


class BenchmarkTask(BaseModel):
    """A single benchmark test case."""
    id: str = Field(description='Unique task identifier')
    name: str = Field(description='Human-readable task name')
    description: str = Field(description='What this test validates')
    memory_bank: str = Field(description='The conversation/document data')
    query: str = Field(description='The query to test')
    expected_keep: list[str] = Field(
        default_factory=list,
        description='Keywords that SHOULD be in retrieved chunks'
    )
    expected_exclude: list[str] = Field(
        default_factory=list,
        description='Keywords that should be EXCLUDED from retrieved chunks'
    )


class BenchmarkResult(BaseModel):
    """Result from running a single benchmark task."""
    task_id: str
    task_name: str
    
    # Chunk counts
    single_scored_chunks: int
    dual_scored_chunks: int
    excluded_by_erase: int
    
    # Metrics
    keep_precision: float = Field(description='% of expected_keep found in results')
    exclude_precision: float = Field(description='% of expected_exclude NOT in results')
    noise_reduction: float = Field(description='% reduction in chunks by ERASE')
    
    # Pass/fail
    passed: bool = Field(description='Overall test passed')


class BenchmarkSummary(BaseModel):
    """Summary of all benchmark results."""
    total_tasks: int
    passed_tasks: int
    avg_noise_reduction: float
    avg_keep_precision: float
    avg_exclude_precision: float
