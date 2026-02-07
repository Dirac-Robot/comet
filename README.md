# ☄️ CoMeT — Cognitive Memory Tree

**Lossless structured memory for AI agents.**

CoMeT compresses long conversations into a navigable tree of memory nodes.  
Unlike naive summarization that loses details, CoMeT preserves raw data behind structured summaries — agents read summaries first, then drill into raw data only when needed.

## Architecture

```
User Input
    │
    ▼
┌─────────┐    SLM (fast)     ┌───────────┐
│  Sensor  │ ───────────────▶ │ L1 Buffer │
└─────────┘   entity/intent   └─────┬─────┘
                                    │ cognitive load trigger
                                    ▼
                              ┌───────────┐
                              │ Compacter │  LLM (slow)
                              └─────┬─────┘
                                    │ summary + trigger + tags
                                    ▼
                              ┌───────────┐
                              │   Store   │  depth 0/1/2
                              └───────────┘
```

### Dual-Speed Layer
- **Fast (Sensor)**: SLM extracts entities/intent per turn, detects topic shifts via cognitive load assessment
- **Slow (Compacter)**: Main LLM structures accumulated L1 buffer into `MemoryNode` with summary, trigger, and topic tags

### Dynamic Resolution (depth 0 → 1 → 2)

| Depth | Content | Use Case |
|-------|---------|----------|
| 0 | Summary + Trigger | Agent's initial context window |
| 1 | + Topic tags + Links | Navigation / node selection |
| 2 | Full raw data + Links | Fact retrieval |

### Topic-Aware Auto-Linking
Nodes share a global topic tag set. The compacter reuses existing tags when possible, enabling automatic bidirectional linking between related nodes across different conversation segments.

## Benchmark (52 turns, 5 conversations, 10 questions)

| Method | Context Cost | Accuracy |
|--------|-------------|----------|
| Full Context Injection | 5,198 chars (100%) | 10/10 |
| **CoMeT** | **1,397 chars (27%)** | **9/10** |
| Naive Summary | 1,179 chars (23%) | 1/10 |

- CoMeT uses **27% of the tokens** while retaining **90% accuracy**
- 6/10 questions required **link traversal** (agent read 2-3 nodes)
- Cross-topic questions: CoMeT 5/5 vs Naive 0/5

## Quick Start

```python
from comet import CoMeT, scope

@scope
def main(config):
    memo = CoMeT(config)

    # Add conversation turns
    memo.add("B200 4대로 월드모델 학습 가능할까?")
    memo.add("2B면 충분하고 커봐야 8B")
    memo.add("DPO 데이터는 negative를 syntax error로 구성했어")

    # Force compact remaining buffer
    memo.force_compact()

    # Navigation
    for node in memo.list_memories():
        print(memo.read_memory(node['node_id'], depth=0))

    # Agent tools (LangChain compatible)
    tools = memo.get_tools()
    # → get_memory_index, read_memory_node, search_memory

main()
```

## Configuration ([ato](https://github.com/Dirac-Robot/ato))

```python
# comet/config.py
@scope.observe(default=True)
def default(config):
    config.slm_model = 'gpt-4o-mini'
    config.main_model = 'gpt-4o'
    config.compacting.load_threshold = 3
    config.compacting.max_l1_buffer = 5

@scope.observe
def local_slm(config):
    config.slm_model = 'ollama/gemma3:4b'

@scope.observe
def aggressive(config):
    config.compacting.load_threshold = 2
    config.compacting.max_l1_buffer = 3
```

```bash
# Use default
python main.py

# Local SLM + aggressive compacting
python main.py local_slm aggressive
```

## Project Structure

```
comet/
├── orchestrator.py    # CoMeT main class
├── sensor.py          # L1 extraction + cognitive load (SLM)
├── compacter.py       # L1→L2 structuring + auto-linking (LLM)
├── storage.py         # JSON key-value store + navigation
├── schemas.py         # MemoryNode, L1Memory, CognitiveLoad
├── config.py          # ato scope configuration
└── templates/         # Prompt templates
```
