from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

class PineconeStore:
    def __init__(self, index_name: str,
                 model: str = "multilingual-e5-large"):
        """
        Initialize Pinecone client with API credentials and index information.
        
        Args:
            index_name (str): Name of the Pinecone index to use
            embedding_dim (int): Dimension of vectors (default 512)
            model (str): Name of the embedding model to use (default multilingual-e5-large)
        """
        load_dotenv()
        self.model = model
        # Get API key from environment variables
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        if not self.pc.has_index(index_name):
          self.pc.create_index(
            name=index_name,
            dimension=1024,
            metric='cosine',
            spec=ServerlessSpec(
              cloud="aws",
              region="us-east-1"
            ),
          )
        # need to wait before runnning this on creation since the index may not be created yet
        index_description = self.pc.describe_index(index_name)
        index_host = index_description['host']
        # Initialize index using host
        self.index = self.pc.Index(host=index_host)

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
      
    def add_chunks(self, chunks: List[str], metadata: List[Dict[str, Any]] = None, 
                  namespace: str = "") -> bool:
        """
        Add text chunks to Pinecone index after converting to embeddings.
        
        Args:
            chunks (List[str]): List of text chunks to embed and store
            metadata (List[Dict], optional): List of metadata for each chunk
            namespace (str, optional): Namespace for the vectors
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate embeddings using Pinecone's inference API
            embeddings = self.pc.inference.embed(
                model=self.model,
                inputs=chunks,
                parameters={"input_type": "passage", "truncate": "END"}
            )

            # Prepare records for upsert
            records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                record = {
                    "id": f"chunk_{i}",
                    "values": embedding['values'],
                    "metadata": metadata[i] if metadata else {"text": chunk}
                }
                records.append(record)

            # Upsert to index
            self.index.upsert(vectors=records, namespace=namespace)
            return True
        except Exception as e:
            print(f"Error adding chunks: {str(e)}")
            return False

    def query_chunk(self, query_text: str, top_k: int = 5, 
                   namespace: str = "") -> List[Dict]:
        """
        Query the index using a text chunk.
        
        Args:
            query_text (str): The text to search for
            top_k (int): Number of results to return
            namespace (str, optional): Namespace to search in
            
        Returns:
            List[Dict]: List of matching results with scores and metadata
        """
        try:
            # Generate embedding for query
            query_embedding = self.pc.inference.embed(
                model=self.model,
                inputs=[query_text],
                parameters={"input_type": "query"}
            )

            # Query the index
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding[0]['values'],
                top_k=top_k,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            print(f"Error querying chunks: {str(e)}")
            return []

    @staticmethod
    def delete_index(index_name: str) -> bool:
        """
        Delete a Pinecone index by name.
        
        Args:
            index_name (str): Name of the index to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            load_dotenv()
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found in environment variables")
                
            pc = Pinecone(api_key=api_key)
            pc.delete_index(index_name)
            print(f"Successfully deleted index: {index_name}")
            return True
        except Exception as e:
            print(f"Error deleting index {index_name}: {str(e)}")
            return False
