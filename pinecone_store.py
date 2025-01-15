import pinecone
from typing import List, Dict, Any

class PineconeStore:
    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initialize Pinecone client with API credentials and index information.
        
        Args:
            api_key (str): Pinecone API key
            environment (str): Pinecone environment (e.g., "us-west1-gcp")
            index_name (str): Name of the Pinecone index to use
        """
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    def add_embeddings(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str]) -> bool:
        """
        Add embeddings to Pinecone index with associated metadata.
        
        Args:
            vectors (List[List[float]]): List of embedding vectors
            metadata (List[Dict]): List of metadata dictionaries for each vector
            ids (List[str]): List of unique IDs for each vector
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            items_to_upsert = list(zip(ids, vectors, metadata))
            self.index.upsert(vectors=items_to_upsert)
            return True
        except Exception as e:
            print(f"Error adding embeddings: {str(e)}")
            return False

    def query_embeddings(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Query the Pinecone index for similar vectors.
        
        Args:
            query_vector (List[float]): The query embedding vector
            top_k (int): Number of results to return
        
        Returns:
            List[Dict]: List of matching results with scores and metadata
        """
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            print(f"Error querying embeddings: {str(e)}")
            return []

    def delete_embeddings(self, ids: List[str]) -> bool:
        """
        Delete embeddings from the Pinecone index by their IDs.
        
        Args:
            ids (List[str]): List of IDs to delete
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.index.delete(ids=ids)
            return True
        except Exception as e:
            print(f"Error deleting embeddings: {str(e)}")
            return False
