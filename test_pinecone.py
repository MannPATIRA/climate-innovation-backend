from common.pinecone_store import PineconeStore
import os
from dotenv import load_dotenv
from common.pinecone_store import PineconeStore
import time

def test_pinecone_operations():
    """Test Pinecone operations with text chunks"""
    
    # Initialize store
    store = PineconeStore(
        index_name="test-index",
        model="multilingual-e5-large"
    )
    
    # Test data
    test_chunks = [
        "Apple is a popular fruit known for its sweetness.",
        "The tech company Apple makes iPhones and MacBooks.",
        "Machine learning is transforming technology.",
        "Natural language processing helps computers understand text."
    ]
    
    test_metadata = [
        {"category": "fruit", "source": "test"},
        {"category": "technology", "source": "test"},
        {"category": "AI", "source": "test"},
        {"category": "AI", "source": "test"}
    ]

    print("\nTesting Pinecone operations...")

    # Test adding chunks
    print("\n1. Testing add_chunks...")
    success = store.add_chunks(test_chunks, test_metadata, namespace="test_namespace")
    print(f"Add chunks successful: {success}")

    # Wait a moment for indexing
    time.sleep(2)

    # Test querying chunks
    print("\n2. Testing query_chunk...")
    query = "Tell me about Apple technology"
    results = store.query_chunk(query, top_k=2, namespace="test_namespace")
    print("\nQuery results:")
    for result in results:
        print(f"Score: {result['score']}")
        print(f"Metadata: {result['metadata']}")
        print("---")


if __name__ == "__main__":
    try:
        test_pinecone_operations()
        print("\nAll tests completed!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")