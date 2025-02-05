from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
# If you have a base class VectorStore, you can still inherit from it:
from common.vector_store import VectorStore

import os
import hashlib
from dotenv import load_dotenv
from typing import List, Dict, Any

# Import the BM25Encoder from pinecone_text
from pinecone_text.sparse import BM25Encoder

class HybridPineconeStore(VectorStore):
    def __init__(self, index_name: str, model: str = "multilingual-e5-large"):
        """
        Initialize Pinecone client with API credentials and index information.
        Also prepare a BM25Encoder for sparse vectors (used in hybrid search).
        
        Args:
            index_name (str): Name of the Pinecone index to use
            model (str): Name of the embedding model to use (default multilingual-e5-large)
        """
        load_dotenv()
        self.model = model
        
        # Get API key from environment variables
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        
        # Create (if necessary) and connect to the index
        if not self.pc.has_index(index_name):
            self.pc.create_index(
                name=index_name,
                dimension=1024,  # e5-large is 1024-dimensional
                metric='cosine',
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        index_description = self.pc.describe_index(index_name)
        index_host = index_description['host']
        self.index = self.pc.Index(host=index_host)

        # Initialize BM25Encoder for generating sparse embeddings
        # (used when we want to do hybrid search)
        self.bm25_encoder = BM25Encoder()

    def add_embeddings(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: List[str],
        use_hybrid: bool = False,
        raw_texts: List[str] = None
    ) -> bool:
        """
        Add embeddings to Pinecone index with associated metadata.
        Optionally, if use_hybrid=True, also add sparse vectors generated 
        from the raw text using BM25.
        
        Args:
            vectors (List[List[float]]): List of dense embedding vectors
            metadata (List[Dict]): List of metadata dictionaries for each vector
            ids (List[str]): List of unique IDs for each vector
            use_hybrid (bool): Whether to add sparse vectors too (default: False)
            raw_texts (List[str]): Original raw texts corresponding to vectors, 
                                   needed only if use_hybrid=True so we can encode them.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare list of upsert items
            items_to_upsert = []
            
            for i, emb in enumerate(vectors):
                # Build the base record (dense vector + metadata)
                record = {
                    "id": ids[i],
                    "values": emb,
                    "metadata": metadata[i]
                }
                
                # If we want hybrid, we also add a sparse vector from BM25
                if use_hybrid and raw_texts is not None:
                    # Encode the text as a document (BM25Encoder) -> sparse dict
                    sparse_vec = self.bm25_encoder.encode_documents([raw_texts[i]])[0]
                    record["sparse_values"] = sparse_vec
                
                items_to_upsert.append(record)
            
            # Upsert to Pinecone
            self.index.upsert(vectors=items_to_upsert)
            return True
        except Exception as e:
            print(f"Error adding embeddings: {str(e)}")
            return False

    def query_embeddings(
        self,
        query_vector: List[float],
        top_k: int = 5,
        use_hybrid: bool = False,
        query_text: str = None
    ) -> List[Dict]:
        """
        Query the Pinecone index for similar vectors using a dense vector.
        If use_hybrid=True, also generate a sparse BM25 vector from `query_text`.
        
        Args:
            query_vector (List[float]): The query embedding vector (dense)
            top_k (int): Number of results to return
            use_hybrid (bool): If True, do a hybrid query (dense + sparse)
            query_text (str): The actual text of the query for BM25 encoding 
                              if use_hybrid=True.
        
        Returns:
            List[Dict]: List of matching results with scores and metadata
        """
        try:
            if use_hybrid and query_text:
                # Encode the query as a sparse vector
                sparse_query = self.bm25_encoder.encode_queries([query_text])[0]
                
                results = self.index.query(
                    vector=query_vector,
                    sparse_vector=sparse_query,
                    top_k=top_k,
                    include_metadata=True
                )
            else:
                # Dense-only query
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

    def add_chunks(
        self,
        chunks: List[str],
        metadata: List[Dict[str, Any]] = None, 
        namespace: str = "",
        use_hybrid: bool = False
    ) -> bool:
        """
        Add text chunks to Pinecone index after converting to embeddings 
        (and optionally generating BM25 sparse vectors).
        
        Args:
            chunks (List[str]): List of text chunks to embed and store
            metadata (List[Dict], optional): List of metadata for each chunk
            namespace (str, optional): Namespace for the vectors
            use_hybrid (bool): If True, store both dense + sparse embeddings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            batch_size = 96
            for i in range(0, len(chunks), batch_size):
                # Get batch
                batch_chunks = chunks[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size] if metadata else None
                
                # 1) Generate dense embeddings for the chunk batch
                embeddings = self.pc.inference.embed(
                    model=self.model,
                    inputs=batch_chunks,
                    parameters={"input_type": "passage"}
                )
                
                # 2) (Optional) Generate sparse embeddings if hybrid
                sparse_vectors = []
                if use_hybrid:
                    # encode_documents returns a list of dicts w/ "indices" and "values"
                    sparse_vectors = self.bm25_encoder.encode_documents(batch_chunks)
                
                # 3) Prepare upsert data
                records = []
                for j, (chunk, emb) in enumerate(zip(batch_chunks, embeddings)):
                    chunk_id = hashlib.sha256(chunk.encode()).hexdigest()
                    
                    record = {
                        "id": chunk_id,
                        "values": emb['values'],
                        "metadata": batch_metadata[j] if batch_metadata else {"text": chunk}
                    }
                    
                    if use_hybrid and sparse_vectors:
                        record["sparse_values"] = sparse_vectors[j]

                    records.append(record)
                
                # 4) Upsert
                self.index.upsert(vectors=records, namespace=namespace)
            
            return True
        except Exception as e:
            print(f"Error adding chunks: {str(e)}")
            return False

    def query_chunk(
        self,
        query_text: str,
        top_k: int = 5,
        namespace: str = "",
        use_hybrid: bool = False
    ) -> List[Dict]:
        """
        Query the index using a text chunk. We generate a dense embedding for `query_text`.
        If use_hybrid=True, we also generate a BM25 sparse vector from `query_text`.
        
        Args:
            query_text (str): The text to search for
            top_k (int): Number of results to return
            namespace (str, optional): Namespace to search in
            use_hybrid (bool): If True, do a hybrid query (dense + sparse)
            
        Returns:
            List[Dict]: List of matching results with scores and metadata
        """
        try:
            # 1) Generate a dense embedding for the query
            query_embedding = self.pc.inference.embed(
                model=self.model,
                inputs=[query_text],
                parameters={"input_type": "query"}
            )
            dense_vec = query_embedding[0]["values"]
            
            # 2) If hybrid, generate sparse vector from BM25
            if use_hybrid:
                sparse_query = self.bm25_encoder.encode_queries([query_text])[0]
                results = self.index.query(
                    vector=dense_vec,
                    sparse_vector=sparse_query,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace
                )
            else:
                results = self.index.query(
                    vector=dense_vec,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace
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
