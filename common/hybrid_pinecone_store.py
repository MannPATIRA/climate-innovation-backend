from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from common.vector_store import VectorStore
import os
import hashlib
from dotenv import load_dotenv
from typing import List, Dict, Any

class HybridPineconeStore(VectorStore):
    def __init__(self,
                 dense_index_name: str = "test-index",
                 hybrid_index_name: str = "hybrid-index",
                 model: str = "multilingual-e5-large"):
        """
        Single class to manage BOTH a dense-only index and a hybrid index.
        This version automatically handles a naive sparse embedding generation
        inside the class (for demonstration).
        """
        load_dotenv()
        self.model = model
        # Get API key from environment variables
        api_key = os.getenv('PINECONE_API_KEY')
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Pinecone Client
        self.pc = Pinecone(api_key=api_key)
        
        # ------------------------------
        # 1) Set up the Dense Index
        # ------------------------------
        self.dense_index_name = dense_index_name
        if not self.pc.has_index(self.dense_index_name):
            self.pc.create_index(
                name=self.dense_index_name,
                dimension=1024,  # Adjust to match your dense model dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )
        dense_desc = self.pc.describe_index(self.dense_index_name)
        dense_host = dense_desc['host']
        self.dense_index = self.pc.Index(host=dense_host)
        
        # ------------------------------
        # 2) Set up the Hybrid Index
        # ------------------------------
        self.hybrid_index_name = hybrid_index_name
        if not self.pc.has_index(self.hybrid_index_name):
            self.pc.create_index(
                name=self.hybrid_index_name,
                dimension=1024,  # Adjust to match your dense model dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
            )
        hybrid_desc = self.pc.describe_index(self.hybrid_index_name)
        hybrid_host = hybrid_desc['host']
        self.hybrid_index = self.pc.Index(host=hybrid_host)

        # -----------------------------------------------------
        # A naive, local vocabulary mapping token -> token_id
        # In a real system, you'd persist this or use a library
        # -----------------------------------------------------
        self.token2id = {}
        self.next_token_id = 1  # Start token IDs at 1 (arbitrary choice)
    
    def _compute_sparse_representation(self, text: str) -> Dict[str, List]:
        """
        Naive example of generating a sparse representation for a single piece
        of text. We'll tokenize by splitting on whitespace, then assign each token
        an ID, and (for demonstration) we'll weight each token by 1.0.

        In a real system, you'd do something more sophisticated (BM25, TF-IDF, etc.).
        """
        # 1) Simple tokenize
        tokens = text.lower().split()
        
        # 2) Build lists for 'indices' and 'values'
        indices = []
        values = []
        
        # We can do a frequency-based approach or a simple '1.0' for each token
        # For demonstration, let's do: freq of token in this text
        token_counts = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
        
        for token, freq in token_counts.items():
            # If it's a new token, assign a new ID
            if token not in self.token2id:
                self.token2id[token] = self.next_token_id
                self.next_token_id += 1
            
            token_id = self.token2id[token]
            indices.append(token_id)
            # Naive weighting: let's just store the frequency
            # or you could do something like freq / max_freq
            values.append(float(freq))
        
        return {
            "indices": indices,
            "values": values
        }

    # ----------------------------------------------------------------
    # Single method to add text chunks to EITHER the dense index or
    # the hybrid index. The method internally handles:
    #   - generating dense embeddings
    #   - generating naive sparse embeddings
    # ----------------------------------------------------------------
    def add_texts(
        self,
        texts: List[str],
        metadata_list: List[Dict[str, Any]] = None,
        use_hybrid: bool = False,
        namespace: str = ""
    ) -> bool:
        """
        The caller only provides raw text (and optional metadata). This method:
          1) Generates a dense embedding for each text (via Pinecone Inference).
          2) Generates a naive sparse embedding (via _compute_sparse_representation)
             if use_hybrid=True.
          3) Upserts to the chosen index (dense or hybrid).
        
        Args:
            texts (List[str]): The text documents or chunks
            metadata_list (List[Dict[str, Any]]): Any metadata
            use_hybrid (bool): If True, upsert to hybrid index with sparse + dense
            namespace (str): Optional Pinecone namespace
        """
        try:
            index = self.hybrid_index if use_hybrid else self.dense_index
            
            # 1) Generate dense embeddings
            embeddings = self.pc.inference.embed(
                model=self.model,
                inputs=texts,
                parameters={"input_type": "passage"}
            )
            
            # 2) Prepare upsert items
            upsert_data = []
            for i, text in enumerate(texts):
                # build ID from the text
                doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                
                # build metadata
                if metadata_list and i < len(metadata_list):
                    doc_metadata = metadata_list[i]
                else:
                    doc_metadata = {"text": text}
                
                # The dense vector
                dense_vec = embeddings[i]["values"]
                
                # For hybrid, also generate a naive sparse vector
                if use_hybrid:
                    sparse_vec = self._compute_sparse_representation(text)
                    record = {
                        "id": doc_id,
                        "values": dense_vec,
                        "sparse_values": sparse_vec,
                        "metadata": doc_metadata
                    }
                else:
                    record = {
                        "id": doc_id,
                        "values": dense_vec,
                        "metadata": doc_metadata
                    }
                
                upsert_data.append(record)
            
            # 3) Upsert
            index.upsert(vectors=upsert_data, namespace=namespace)
            return True
        
        except Exception as e:
            print(f"Error adding texts: {e}")
            return False

    # ----------------------------------------------------------------
    # Query method that automatically:
    #   - Creates a dense query embedding
    #   - Creates a naive sparse embedding if it's a hybrid query
    # ----------------------------------------------------------------
    def query_text(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = False,
        namespace: str = ""
    ) -> List[Dict]:
        """
        The caller only provides raw text as the query. We:
          1) Generate a dense query embedding.
          2) Generate a naive sparse query embedding (if use_hybrid=True).
          3) Query Pinecone with both dense + sparse or just dense.

        Args:
            query (str): Raw text query
            top_k (int): # of results
            use_hybrid (bool): If True, query the hybrid index with sparse + dense
            namespace (str): Optionally search within a namespace
        """
        try:
            index = self.hybrid_index if use_hybrid else self.dense_index
            
            # 1) Dense embedding for the query
            query_embedding = self.pc.inference.embed(
                model=self.model,
                inputs=[query],
                parameters={"input_type": "query"}
            )
            dense_vec = query_embedding[0]["values"]
            
            # 2) If hybrid, compute sparse
            if use_hybrid:
                sparse_vec = self._compute_sparse_representation(query)
                results = index.query(
                    vector=dense_vec,
                    sparse_vector=sparse_vec,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace
                )
            else:
                results = index.query(
                    vector=dense_vec,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace
                )
            
            return results.matches
        
        except Exception as e:
            print(f"Error querying text: {e}")
            return []

    @staticmethod
    def delete_index(index_name: str) -> bool:
        """
        Delete a Pinecone index by name.
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
