import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"

def test_rag_service():
    """Test the RAG service with sample queries"""
    
    # Check health
    print("=== Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health: {response.json()}")
    
    # Check index status
    print("\n=== Index Status ===")
    response = requests.get(f"{BASE_URL}/index-status")
    print(f"Index Status: {response.json()}")
    
    # Test query
    print("\n=== Test Query ===")
    query_data = {
        "query": "What is the name of the scientist widely acclaimed as the foundational figure of modern physics?",
        "top_k": 1
    }
    
    response = requests.post(
        f"{BASE_URL}/query",
        json=query_data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {len(result['retrieved_chunks'])} chunks")
        print(f"Scores: {result['scores']}")
        
        # Print first chunk as example
        if result['retrieved_chunks']:
            print(f"\nFirst retrieved chunk:\n{result['retrieved_chunks'][0][:200]}...")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_rag_service()