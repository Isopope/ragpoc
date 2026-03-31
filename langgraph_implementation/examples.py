"""
Example usage of the Elysia LangGraph implementation.
Demonstrates how to set up and run the decision tree.
"""

import asyncio
from langgraph_implementation.graph import ElysiaGraph
from langgraph_implementation.state import create_initial_state


async def example_basic_usage():
    """
    Basic example: Run a single query through the Elysia LangGraph.
    """
    print("=" * 60)
    print("ELYSIA LANGGRAPH - BASIC EXAMPLE")
    print("=" * 60)
    
    # Initialize the graph
    graph = ElysiaGraph(
        mode="onebranch",  # or "multibranch"
        model_name="gpt-4-turbo-preview",
    )
    
    # Example collections
    collection_names = ["documents", "products", "tickets"]
    
    # Collection metadata (would come from Weaviate preprocessing)
    collection_metadata = {
        "documents": {
            "length": 5000,
            "summary": "Academic papers and research documents",
            "fields": [
                {"name": "title", "type": "text"},
                {"name": "content", "type": "text"},
                {"name": "author", "type": "text"},
            ],
        },
        "products": {
            "length": 1200,
            "summary": "E-commerce product catalog",
            "fields": [
                {"name": "name", "type": "text"},
                {"name": "price", "type": "float"},
                {"name": "description", "type": "text"},
            ],
        },
    }
    
    # Run the graph
    result = await graph.run(
        user_prompt="What are the most expensive products in the AI category?",
        user_id="user-123",
        conversation_id="conv-456",
        collection_names=collection_names,
        collection_metadata=collection_metadata,
    )
    
    # Display results
    print(f"\nUser Query: {result['user_prompt']}")
    print(f"\nFinal Response:\n{result['response']}")
    print(f"\nDecision History: {result['decision_history']}")
    print(f"Tree Depth: {result['tree_depth']}")
    print(f"Errors: {len(result['errors'])}")
    
    return result


async def example_multibranch():
    """
    Multi-branch example: Uses the branching decision tree structure.
    """
    print("\n" + "=" * 60)
    print("ELYSIA LANGGRAPH - MULTIBRANCH EXAMPLE")
    print("=" * 60)
    
    graph = ElysiaGraph(
        mode="multibranch",
        model_name="gpt-4-turbo-preview",
    )
    
    # Show the tree structure
    print("\nTree Structure:")
    tree_structure = graph.tree_builder.get_branch_structure()
    for branch_id, branch_info in tree_structure.items():
        print(f"\n  Branch: {branch_id}")
        print(f"    Options: {branch_info['options']}")
        print(f"    Is Root: {branch_info['is_root']}")
    
    # Run
    result = await graph.run(
        user_prompt="Give me an average rating of products and count by category",
        collection_names=["products"],
    )
    
    print(f"\nResponse: {result['response']}")
    print(f"Decision Path: {' -> '.join(result['decision_history'])}")
    
    return result


async def example_with_conversation_history():
    """
    Example with conversation history for multi-turn interactions.
    """
    print("\n" + "=" * 60)
    print("ELYSIA LANGGRAPH - CONVERSATION EXAMPLE")
    print("=" * 60)
    
    graph = ElysiaGraph(mode="onebranch")
    
    # First turn
    print("\n--- Turn 1 ---")
    result1 = await graph.run(
        user_prompt="Show me all documents about machine learning",
        user_id="user-789",
        conversation_id="conv-789",
        collection_names=["documents"],
    )
    print(f"Response: {result1['response']}")
    
    # Second turn (would normally include history from previous turn)
    print("\n--- Turn 2 ---")
    result2 = await graph.run(
        user_prompt="Summarize those documents",
        user_id="user-789",
        conversation_id="conv-789",  # Same conversation
        collection_names=["documents"],
    )
    print(f"Response: {result2['response']}")
    
    return result1, result2


async def example_error_handling():
    """
    Example demonstrating error handling and recovery.
    """
    print("\n" + "=" * 60)
    print("ELYSIA LANGGRAPH - ERROR HANDLING EXAMPLE")
    print("=" * 60)
    
    graph = ElysiaGraph(mode="onebranch")
    
    # Try with empty collections (will result in no data)
    result = await graph.run(
        user_prompt="Find data that doesn't exist",
        collection_names=[],  # No collections available
    )
    
    print(f"\nErrors encountered: {len(result['errors'])}")
    for error in result['errors']:
        print(f"  - {error.get('tool_name')}: {error.get('message')}")
    
    print(f"\nResponse: {result['response']}")
    
    return result


async def example_compare_modes():
    """
    Compare single-branch vs multi-branch execution.
    """
    print("\n" + "=" * 60)
    print("ELYSIA LANGGRAPH - MODE COMPARISON")
    print("=" * 60)
    
    query = "What are the top-selling products?"
    collections = ["products"]
    
    # Single branch
    print("\n--- Single Branch Mode ---")
    graph_single = ElysiaGraph(mode="onebranch")
    result_single = await graph_single.run(
        user_prompt=query,
        collection_names=collections,
    )
    print(f"Tree Depth: {result_single['tree_depth']}")
    print(f"Decisions: {result_single['decision_history']}")
    
    # Multi branch
    print("\n--- Multi Branch Mode ---")
    graph_multi = ElysiaGraph(mode="multibranch")
    result_multi = await graph_multi.run(
        user_prompt=query,
        collection_names=collections,
    )
    print(f"Tree Depth: {result_multi['tree_depth']}")
    print(f"Decisions: {result_multi['decision_history']}")
    
    return result_single, result_multi


async def main():
    """Run all examples."""
    
    # Run examples
    try:
        # Basic usage
        await example_basic_usage()
        
        # Multibranch
        await example_multibranch()
        
        # Conversation
        await example_with_conversation_history()
        
        # Error handling
        await example_error_handling()
        
        # Compare modes
        await example_compare_modes()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
