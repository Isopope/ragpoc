# Elysia Decision Tree in LangGraph

A complete reimplementation of the **Elysia agentic decision tree system** using **LangGraph**, maintaining architectural fidelity while leveraging LangGraph's graph execution capabilities.

## Overview

Elysia is an agentic RAG (Retrieval-Augmented Generation) framework built around a decision tree that:
1. **Routes** user queries through decision nodes
2. **Executes** retrieval tools (Query, Aggregate, Summarize)
3. **Manages** a persistent environment of retrieved objects
4. **Learns** from errors through feedback loops

This LangGraph implementation provides the same functionality with improved graph visualization and Pythonic async/await patterns.

## Architecture

### 1. State Management (`state.py`)

Maps Elysia's `TreeData` and `Environment` concepts into a unified `ElysiaState`:

```python
ElysiaState = {
    "user_prompt": str,
    "conversation_history": list,
    "current_branch": str,
    "environment": dict,           # {tool_name: {collection: [ToolResult]}}
    "collection_metadata": dict,   # Schema info for each collection
    "errors": list,                # Error tracking for feedback
    ...
}
```

**Key Differences from Elysia:**
- Unified state instead of separate `TreeData`, `Environment`, `Atlas` objects
- Async-first design
- Typed with Pydantic where applicable

### 2. Decision Nodes (`decision_nodes.py`)

Implements tree structure with support for multiple configurations:

```
MultibranchTree:
  base (root)
    ├── summarize
    ├── text_response
    └── search (branch)
        ├── query
        └── aggregate

OneBranchTree:
  base (root: all tools available)
    ├── query
    ├── aggregate
    ├── summarize
    ├── text_response
    └── visualize
```

Each node maps to Elysia's `DecisionNode` with instruction text and available actions.

### 3. Tools (`tools.py`)

Implements retrieval tools as async classes:

| Tool | Purpose | Mirrors |
|------|---------|---------|
| `QueryTool` | Semantic/keyword search | `elysia/tools/retrieval/query.py` |
| `AggregateTool` | Statistical operations | `elysia/tools/retrieval/aggregate.py` |
| `SummarizationTool` | LLM-based summarization | `elysia/tools/postprocessing/summarise_items.py` |

Each tool:
- Takes collection names, parameters
- Returns structured `ToolResult` objects
- Updates the environment
- Tracks errors

### 4. LLM Integration (`llm_integration.py`)

**Decision Making:**
- Uses LangChain ChatOpenAI with structured outputs
- Mirrors Elysia's DSPy `DecisionPrompt` signature
- Supports DSPy-compatible signatures for gradual migration

**Response Generation:**
- Synthesizes retrieved information
- Maintains conversation context
- Handles error recovery

### 5. Main Graph (`graph.py`)

The `ElysiaGraph` class orchestrates the full workflow:

```
START
  ↓
DECISION NODE
  ├─→ execute_tool? → TOOL EXECUTION → PROCESS RESULT → (DECISION | RESPONSE | END)
  ├─→ generate_response? → RESPONSE GENERATION → END
  └─→ end? → END
```

**Flow:**
1. **Decide**: LLM chooses next action based on user prompt, available tools, previous errors
2. **Execute**: Run selected tool (Query, Aggregate, etc.)
3. **Process**: Evaluate results, decide next step
4. **Generate**: Create final response or loop back to Decide

## Mapping: Elysia → LangGraph

| Elysia Component | LangGraph Equivalent |
|------------------|---------------------|
| `Tree` class | `ElysiaGraph` |
| `TreeData` | `ElysiaState` |
| `Environment` | `state["environment"]` |
| `DecisionNode` | `decision_nodes.DecisionNode` |
| `Tool` (base class) | `QueryTool`, `AggregateTool`, etc. |
| DSPy `Signature` | Pydantic `BaseModel` + `DecisionMaker` |
| `ChainOfThought` | `DecisionMaker.decide()` with reasoning |
| Multi-branch tree | `MultibranchTree` |
| Single-branch tree | `OneBranchTree` |

## Usage

### Basic Example

```python
from langgraph_implementation.graph import ElysiaGraph
import asyncio

async def main():
    # Create graph
    graph = ElysiaGraph(mode="multibranch")
    
    # Run query
    result = await graph.run(
        user_prompt="What are the most expensive products?",
        collection_names=["products"],
    )
    
    print(result["response"])
    print(f"Decision history: {result['decision_history']}")

asyncio.run(main())
```

### With Collection Metadata

```python
metadata = {
    "products": {
        "length": 1200,
        "summary": "E-commerce product catalog",
        "fields": [
            {"name": "name", "type": "text"},
            {"name": "price", "type": "float"},
        ]
    }
}

result = await graph.run(
    user_prompt="Show products under $100",
    collection_names=["products"],
    collection_metadata=metadata,
)
```

### Custom Tree Configuration

```python
from langgraph_implementation.decision_nodes import TreeBuilder

builder = TreeBuilder()

# Add custom branch
builder.add_branch(
    branch_id="analysis",
    instruction="Choose an analysis type",
    tools=[
        {"name": "statistical_analysis", ...},
        {"name": "trend_analysis", ...},
    ],
    is_root=False,
    parent_branch_id="base",
)

# Build graph with custom tree
graph = ElysiaGraph(mode="custom")
graph.tree_builder = builder
```

## Key Features

### 1. **Error Recovery**
- Tracks errors per tool execution
- Provides error history to LLM for learning
- Supports retry logic with feedback

### 2. **Multiturning**
- Maintains conversation history
- Reuses retrieved objects across turns
- Context-aware decision making

### 3. **Extensibility**
- Add custom tools by extending `QueryTool` base class
- Implement custom LLM integration
- Define new tree structures

### 4. **State Persistence**
- Full state snapshots available
- Environment updates tracked
- Decision history for debugging

### 5. **Async-First Design**
- All operations are async
- Parallel execution ready
- Compatible with async frameworks

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4-turbo-preview

# Weaviate Configuration
WCD_URL=https://cluster.weaviate.cloud
WCD_API_KEY=your-api-key

# Optional: LangChain callbacks
LANGCHAIN_CALLBACKS_ENABLED=true
```

### Graph Parameters

```python
ElysiaGraph(
    mode="multibranch",              # multibranch, onebranch
    weaviate_client=client,          # Real Weaviate client
    llm_client=chat_openai,          # Custom LLM client
    model_name="gpt-4-turbo-preview" # Model selection
)
```

## Differences from Elysia

### Advantages
- Native async/await support
- Better graph visualization (LangGraph Studio compatible)
- Simpler state management
- Easier testing and debugging

### Considerations
- DSPy signatures → Pydantic models (gradual migration support)
- No built-in Weaviate preprocessing (use separately)
- Different error handling patterns
- Tool definitions more explicit

## Running Examples

```bash
# Run all examples
python -m langgraph_implementation.examples

# Or import and run specific examples
from langgraph_implementation.examples import example_multibranch
import asyncio
asyncio.run(example_multibranch())
```

## Directory Structure

```
langgraph_implementation/
├── __init__.py                    # Package exports
├── state.py                       # State definitions & utilities
├── decision_nodes.py              # Tree structure & nodes
├── tools.py                       # Retrieval tool implementations
├── llm_integration.py             # LLM decision-making
├── graph.py                       # Main ElysiaGraph orchestrator
├── examples.py                    # Usage examples
└── README.md                      # This file
```

## Future Enhancements

- [ ] Direct Weaviate integration (async client)
- [ ] LangGraph Studio visualization support
- [ ] Advanced error recovery strategies
- [ ] DSPy compatibility layer
- [ ] Benchmark suite vs original Elysia
- [ ] Custom tracer/observer for debugging
- [ ] Streaming response support
- [ ] Multi-modal retrieval support

## Integration with Original Elysia

This can coexist with original Elysia by:

1. **Using same Weaviate clusters**: Share preprocessed collections
2. **Gradual migration**: Run LangGraph in parallel, compare results
3. **Hybrid approach**: Use LangGraph's strengths + Elysia's preprocessing

```python
# Use Elysia preprocessing
from elysia import preprocess
preprocess("my_collection")

# Use LangGraph for execution
from langgraph_implementation import ElysiaGraph
graph = ElysiaGraph()
result = await graph.run(
    user_prompt="Query my data",
    collection_names=["my_collection"],
)
```

## Requirements

```
langraph>=0.1.0
langgraph-cli>=0.1.0
langchain>=0.1.0
pydantic>=2.0
```

## License & Attribution

This implementation follows the original Elysia architecture as documented in the source code. Designed to provide a faster, more Pythonic alternative while maintaining compatibility with the Elysia ecosystem.
