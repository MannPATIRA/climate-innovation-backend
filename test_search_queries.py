from common.pinecone_store import PineconeStore
import time

def test_search_query_embeddings():
    """Test Pinecone operations with search queries and similarity matching"""
    
    # Initialize store
    store = PineconeStore(
        index_name="climate-index",
    )
    
    # Test search queries
    search_queries = [
        # Government and Policy Reports
        "UK climate policy report",
        "UK Net Zero strategy report", 
        "Climate adaptation report UK",
        "UK government climate risk assessment",
        
        # Scientific Reports
        "UK climate change scientific assessment",
        "UK carbon emissions report",
        "UK climate projections Met Office",
        
        # Regional Reports
        "London climate risk report",
        "Scotland climate impact assessment", 
        "Wales climate adaptation strategy"
    ]
    
    # Metadata for categorization
    metadata_list = [
        {"category": "government", "type": "policy", "source": "gov.uk"},
        {"category": "government", "type": "strategy", "source": "gov.uk"},
        {"category": "government", "type": "adaptation", "source": "theccc.org.uk"},
        {"category": "government", "type": "assessment", "source": "gov.uk"},
        
        {"category": "scientific", "type": "assessment", "source": "nature.com"},
        {"category": "scientific", "type": "emissions", "source": "report"},
        {"category": "scientific", "type": "projections", "source": "met_office"},
        
        {"category": "regional", "type": "assessment", "source": "london"},
        {"category": "regional", "type": "assessment", "source": "scotland"},
        {"category": "regional", "type": "strategy", "source": "wales"}
    ]

    print("\nTesting search query embeddings...")

    # Add search queries to Pinecone
    print("\n1. Adding search queries to index...")
    metadata_list = [{**metadata_list[i], "content": search_queries[i]} for i in range(len(search_queries))]
    success = store.add_chunks(search_queries, metadata_list, namespace="search_queries")
    print(f"Add queries successful: {success}")

    # Wait for indexing
    time.sleep(2)

    # Test similarity search with different queries
    test_queries = [
        "Nigeria climate risk report",
    ]

    print("\n2. Testing similarity search...")
    for query in test_queries:
        print(f"\nFinding similar queries for: '{query}'")
        results = store.query_chunk(query, top_k=3, namespace="search_queries")
        print("\nMost similar search queries:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"Content: {result.metadata['content']}")
            print(f"Category: {result.metadata['category']}")
            print(f"Type: {result.metadata['type']}")

if __name__ == "__main__":
    try:
        test_search_query_embeddings()
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}") 