# Elysia LangGraph - Technical Architecture

## Overview

This document provides detailed technical information about the LangGraph implementation of Elysia's decision tree system.

## System Architecture

### 1. State Management Layer (`state.py`)

**Purpose**: Unified state representation replacing Elysia's `TreeData` and `Environment` classes.

**Key Data Structures**:

```
ElysiaState (TypedDict)
├── Session Info
│   ├── user_id: str
│   ├── conversation_id: str
│   ├── user_prompt: str
│   └── conversation_history: list
│
├── Tree Navigation
│   ├── current_branch: str
│   ├── decision_history: list[str]
│   ├── tree_depth: int
│   └── max_tree_depth: int
│
├── Environment (Retrieved Data)
│   ├── environment: dict[tool_name][collection_name] = [ToolResult]
│   ├── hidden_environment: dict (internal storage)
│   └── collection_metadata: dict
│
├── Error Tracking
│   ├── errors: list[dict]
│   └── previous_attempts: list[dict]
│
└── Output
    ├── final_response: str
    └── messages: list[dict]
```

**Environment Structure**:
```python
{
    "query": {
        "documents": [ToolResult, ToolResult, ...],
        "products": [ToolResult, ...]
    },
    "aggregate": {
        "products": [ToolResult]
    }
}
```

Each `ToolResult` contains:
- `tool_name`: "query" | "aggregate" | "summarize"
- `collection_names`: list of collections targeted
- `objects`: list[RetrievedObject] with UUID, properties, timestamps
- `metadata`: query parameters, execution stats
- `status`: TaskStatus enum

### 2. Decision Node Layer (`decision_nodes.py`)

**Purpose**: Tree structure management and branching logic.

**Tree Types**:

#### MultibranchTree
```
base (DecisionNode: root)
  ├── summarize (CitedSummarizer)
  ├── text_response (FakeTextResponse)
  ├── visualize (Visualise)
  └── search (DecisionNode: branch)
      ├── query (Query tool)
      └── aggregate (Aggregate tool)
```

#### OneBranchTree
```
base (DecisionNode: root)
├── query
├── aggregate
├── summarize
├── text_response
└── visualize
```

**DecisionNode Structure**:
```python
{
    "node_id": str,
    "instruction": str,  # LLM instruction
    "status": str,       # Current status message
    "options": dict,     # {tool_name: tool_definition}
    "parent_node": str,  # Parent node ID
    "is_root": bool,
    "visited_count": int,
    "error_history": list
}
```

### 3. Tool Execution Layer (`tools.py`)

**Tool Base Interface**:
```python
class Tool:
    async def execute(
        state: ElysiaState,
        **tool_specific_kwargs
    ) -> ElysiaState
```

**Implemented Tools**:

1. **QueryTool**
   - Retrieves specific data from collections
   - Search types: hybrid, vector, keyword, filter_only
   - Supports complex filtering via Pydantic models
   - Returns up to `limit` results

2. **AggregateTool**
   - Statistical operations: MIN, MAX, MEAN, SUM, COUNT
   - Grouping by properties
   - Optional filtering before aggregation
   - Returns aggregated metrics

3. **SummarizationTool**
   - Summarizes retrieved objects
   - Uses LLM to synthesize information
   - Extracts from environment or takes specific objects

**Tool Execution Flow**:
```
Input → Validate → Execute → Format Result → Update Environment → Return Stats
```

### 4. LLM Integration Layer (`llm_integration.py`)

**Purpose**: LLM-based decision making and response generation.

**Decision Making Process**:

1. **Format Context**: Build decision prompt with:
   - User query and conversation history
   - Available actions at current node
   - Unavailable actions and why
   - Tree of future possible actions
   - Previous errors (for learning)
   - Current environment snapshot

2. **LLM Call**: Send to LLM (GPT-4, Claude, etc.)
   - Returns `DecisionOutput`:
     ```
     {
       "action": str,          # Selected tool
       "reasoning": str,       # Why chosen
       "confidence": 0.0-1.0,  # Confidence level
       "parameters": dict      # Tool parameters
     }
     ```

3. **Validation**: Ensure action is in `available_actions`

4. **Error Recovery**: If error history exists, adjust prompt

**Response Generation Process**:

1. Format retrieved information from environment
2. Build response prompt with:
   - User query
   - Retrieved data
   - Conversation context
3. LLM synthesizes comprehensive response
4. Return to user

### 5. Graph Orchestration Layer (`graph.py`)

**Purpose**: Main graph execution and node coordination.

**Graph Structure** (LangGraph):
```
START
  ↓
DECISION NODE ──→ (LLM selects action)
  ├─→ execute_tool? 
  │    ├→ EXECUTE TOOL NODE
  │    │   ├→ Query/Aggregate/Summarize
  │    │   └→ Update environment
  │    ├→ PROCESS RESULT NODE
  │    │   └→ Decide: more data needed? → route
  │    └→ Routes to: DECISION | RESPONSE | END
  │
  ├─→ generate_response?
  │    └→ RESPONSE GENERATION NODE
  │        └→ LLM synthesizes response
  │
  └─→ end? → END
```

**Node Implementations**:

1. **node_decide**: Decision-making node
   - Calls LLM's `decide()`
   - Updates decision history
   - Enforces recursion limit

2. **node_execute_tool**: Tool execution node
   - Routes to specific tool handler
   - Passes parameters from LLM output
   - Catches and logs errors

3. **node_process_result**: Result processing node
   - Evaluates retrieval success
   - Decides if more queries needed
   - Routes to next appropriate node

4. **node_generate_response**: Response generation node
   - Calls LLM's `generate_response()`
   - Uses environment data
   - Formats for delivery

**Routing Logic**:

After DECISION:
- If `action == "end"` or `tree_depth > max` → END
- If `action in tool_list` → EXECUTE_TOOL
- If `action == "generate_response"` → RESPONSE

After EXECUTE_TOOL:
- If has results → PROCESS_RESULT
- Otherwise → END

After PROCESS_RESULT:
- If enough data → RESPONSE
- If reached limit → RESPONSE
- Otherwise → DECISION (loop)

### 6. Configuration Layer (`config.py`)

**Environment Variables**:
```
OPENAI_API_KEY          - LLM API key
OPENAI_MODEL            - Model selection
WCD_URL                 - Weaviate endpoint
WCD_API_KEY             - Weaviate credentials
LANGGRAPH_MODE          - multibranch | onebranch
MAX_TREE_DEPTH          - Recursion limit
USE_WEAVIATE            - Enable real data retrieval
USE_LLM                 - Enable LLM decisions
```

**Config Class**: Centralized configuration with validation and dict export.

## Data Flow Example

### Query: "What are the top 5 most expensive products?"

1. **Initialization**
   ```
   State created: tree_depth=0, environment={}
   ```

2. **First Decision**
   ```
   DECISION NODE →
   Context: user_prompt, available_tools = [query, aggregate]
   LLM selects: "aggregate" (better for top-N stats)
   ```

3. **Tool Execution**
   ```
   EXECUTE_TOOL NODE →
   AggregateTool.execute(
     collection_names=["products"],
     aggregations={"price": ["MAX"]},
     groupby_property="price",
     limit=5
   ) →
   Returns: ToolResult with top 5 prices
   Environment updated with aggregation results
   ```

4. **Result Processing**
   ```
   PROCESS_RESULT NODE →
   Has 5 aggregation results
   tree_depth < max_depth
   Decision: enough data, go to response
   ```

5. **Response Generation**
   ```
   RESPONSE NODE →
   LLM.generate_response(
     user_prompt="What are the top 5 most expensive products?",
     environment={aggregate results}
   ) →
   Returns formatted response
   ```

6. **Return**
   ```
   Result:
   {
     "response": "Based on the data...",
     "decision_history": ["aggregate"],
     "tree_depth": 1,
     ...
   }
   ```

## Comparison: Elysia vs LangGraph Implementation

| Aspect | Elysia | LangGraph |
|--------|--------|-----------|
| State Management | TreeData + Environment classes | Single ElysiaState TypedDict |
| Graph Representation | Manual dict of DecisionNodes | LangGraph StateGraph |
| Async Support | Partial (some async tools) | Full async/await throughout |
| LLM Integration | DSPy Signatures & ChainOfThought | LangChain + Pydantic models |
| Tool Execution | Direct method calls | Async node functions |
| Error Recovery | Manual feedback loops | Built-in retry mechanisms |
| Visualization | Custom rendering | LangGraph Studio compatible |
| Testing | Manual test harness | pytest + async fixtures |

## Performance Characteristics

- **Decision Latency**: ~1-2s (LLM call dominated)
- **Tool Execution**: ~0.1-0.5s (mock), varies with Weaviate
- **Total Query Time**: 2-10s (depends on # decisions × LLM latency)
- **Memory Usage**: ~50MB baseline + environment size

## Extension Points

### Adding Custom Tools
```python
class CustomTool:
    async def execute(self, state: ElysiaState, **kwargs) -> ElysiaState:
        # Implementation
        result = ToolResult(...)
        return add_to_environment(state, "custom", result)

executor.tools["custom"] = CustomTool()
```

### Custom Tree Structure
```python
builder = TreeBuilder()
builder.add_branch("custom", instruction="...", tools=[...])
graph.tree_builder = builder
```

### Custom LLM Provider
```python
class CustomDecisionMaker(DecisionMaker):
    async def decide(self, context, previous_failures):
        # Custom logic
        return decision
        
graph.decision_maker = CustomDecisionMaker(model_name)
```

## Debugging & Monitoring

**Debug Utilities**:
- `DebugHelper.print_state_snapshot()`: Print state at any point
- `DebugHelper.print_execution_summary()`: Summary of full run
- `MetricsCollector`: Track metrics across runs

**Logging**:
- LangChain verbose mode
- LangGraph debug output
- Custom trace logging

## Known Limitations & Future Work

1. **Weaviate Integration**: Currently mocked, needs real client
2. **DSPy Compatibility**: Gradual migration, not 100% drop-in
3. **Streaming**: No streaming response support yet
4. **Distributed Execution**: Single-machine only
5. **Feedback Learning**: Error history collected but not yet used for prompt optimization

## References

- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- Elysia Original: https://github.com/weaviate/elysia
- LangChain: https://python.langchain.com/
