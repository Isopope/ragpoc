import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph_implementation.graph import ElysiaGraph

async def main():
    print("Initializing ElysiaGraph in multibranch mode...")
    graph = ElysiaGraph(mode="multibranch")
    
    print("\nRunning test query...")
    result = await graph.run(
        user_prompt="Can you summarize the documents about machine learning?",
        user_id="test_user",
        conversation_id="test_conv",
        collection_names=["Documents"]
    )
    
    print("\n--- TEST RESULTS ---")
    print(f"Final Response: {result.get('response')}")
    print(f"Tree Depth: {result.get('tree_depth')}")
    print("\nDecision History:")
    for d in result.get('decision_history', []):
        print(f"  - {d}")
        
    print("\nErrors (if any):")
    for e in result.get('errors', []):
        print(f"  - {e}")
        
    print("\nExtracted Metadata (checking follow-up/title nodes):")
    metadata = result.get('metadata', {})
    print(f"  - Mode: {metadata.get('mode')}")

if __name__ == "__main__":
    asyncio.run(main())
