from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    """
    Abstract base class for vector stores.
    Defines the interface that all vector store implementations must follow.
    """
    
    @abstractmethod
    def add_embeddings(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str]) -> bool:
        """
        Add embeddings to the vector store with associated metadata.
        
        Args:
            vectors (List[List[float]]): List of embedding vectors
            metadata (List[Dict]): List of metadata dictionaries for each vector
            ids (List[str]): List of unique IDs for each vector
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def query_embeddings(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """
        Query the vector store for similar vectors.
        
        Args:
            query_vector (List[float]): The query embedding vector
            top_k (int): Number of results to return
        
        Returns:
            List[Dict]: List of matching results with scores and metadata
        """
        pass

    @abstractmethod
    def delete_embeddings(self, ids: List[str]) -> bool:
        """
        Delete embeddings from the vector store by their IDs.
        
        Args:
            ids (List[str]): List of IDs to delete
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def add_chunks(self, chunks: List[str], metadata: List[Dict[str, Any]] = None, 
                  namespace: str = "") -> bool:
        """
        Add text chunks to vector store after converting to embeddings.
        
        Args:
            chunks (List[str]): List of text chunks to embed and store
            metadata (List[Dict], optional): List of metadata for each chunk
            namespace (str, optional): Namespace for the vectors
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass

    @abstractmethod
    def query_chunk(self, query_text: str, top_k: int = 5, 
                   namespace: str = "") -> List[Dict]:
        """
        Query the store using a text chunk.
        
        Args:
            query_text (str): The text to search for
            top_k (int): Number of results to return
            namespace (str, optional): Namespace to search in
            
        Returns:
            List[Dict]: List of matching results with scores and metadata
        """
        pass
