import asyncio
import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from common.pinecone_store import PineconeStore
from common.hybrid_pinecone_store import HybridPineconeStore

class MockQueryProcessor:
    def __init__(self):
        pass

    async def process_stream(self, query: str, chat_history: list, completion_callback):
        """Generator function that yields streaming responses"""
        # Example predefined response
        mock_response = (
            "### Climate Challenges\n\n"
            "Climate change is one of the most significant global challenges we face today. "
            "It is linked to various climate and weather extremes, leading to several problems "
            "affecting millions worldwide. Some key climate challenges include:\n\n"
            "- **Catastrophic Heat Waves**: These intense heat events lead to severe health impacts, "
            "increased mortality, and stresses on energy and water resources.\n"
            "- **Wildfires**: Often exacerbated by higher temperatures and prolonged dry conditions, "
            "wildfires destroy vast landscapes, endanger fauna and flora, and threaten human lives and properties.\n"
            "- **Floods**: Increased rainfall and rising sea levels result in severe flooding, "
            "displacing communities, damaging infrastructure, and affecting agriculture.\n\n"
            "These climate-related events underscore the need for innovative approaches and strengthened "
            "global collaboration to mitigate the impacts of climate change.\n\n"
            "{'topics': [{'topic': 'Heat Wave Impacts', 'source': 'Climate Risk Report'}, "
            "{'topic': 'Wildfire Threats', 'source': 'Environmental Assessment'}, "
            "{'topic': 'Flood Disasters', 'source': 'Global Climate Study'}]}"
        )
        
        full_response = ""
        
        # Simulate streaming by yielding chunks with delays
        for chunk in mock_response:
            
            full_response += chunk
            # print(chunk)
            await asyncio.sleep(0.01)  # Add delay to simulate real streaming
            yield chunk
        
        # After the stream is complete, call the completion callback
        await completion_callback(full_response)

class QueryProcessor:
    def __init__(self):
        # Initialize the chain by calling the helper method
        self.chain = self._create_chain()
        self.pinecone_store =  PineconeStore(
            index_name="test-index",
            model="multilingual-e5-large"
        )
    
    def _get_relevant_chunks(self, query: str) -> str:
        """Helper function to get and format relevant chunks from Pinecone"""
        # Query for similar chunks
        matches = self.pinecone_store.query_chunk(
            query_text=query,
            top_k=3,  # Adjust number of chunks as needed
            namespace="reports"
        )
        
        # Format the chunks with their metadata
        formatted_context = []
        for i, match in enumerate(matches, 1):
            if match.metadata and 'content' in match.metadata:
                chunk_text = match.metadata['content']
                chunk_title = match.metadata["report_title"]
                formatted_context.append(f"Chunk {i}:\n{chunk_text}\n\nSource:\n{chunk_title}\n\n")
        
        # Join all formatted chunks
        return "\n".join(formatted_context)

    def _create_chain(self):
        """Helper method to create and configure the LangChain processing chain"""
        model = ChatOpenAI(model="gpt-4o")

        prompt = ChatPromptTemplate.from_template(
            "Answer the following query in markdown format using the provided context\n"
            "Query: {query}\n"
            "Context \n\n: {context}\n"
            "Your response must directly start with the markdown"
            "After responding, extract 3-5 main points that answer the query from this response along with their sources.\n"
            "When you extract these point, you must ensure that they come from the chunks, don't use your own knowledge"
            "Format the output as a JSON list of objects, where each object has 'topic' and 'source' keys.\n"
            "Format: {{'topics': [{{'topic': 'topic 1', 'source': 'source 1'}}, {{'topic': 'topic 2', 'source': 'source 2'}}]}}"
        )
        parser = StrOutputParser()

        return (
            {
                "context": lambda x: self._get_relevant_chunks(x["query"]),
                "query": lambda x: x["query"]
            }
            | prompt 
            | model 
            | parser
        )

    async def process_stream(self, query: str, chat_history: list, completion_callback):
        """Generator function that yields streaming responses"""
        full_response = ""
        
        async for chunk in self.chain.astream({"query": query}):
            full_response += chunk
            # print(chunk)
            yield chunk
        
        # After the stream is complete, call the completion callback
        await completion_callback(full_response)

class HybridQueryProcessor:
    def __init__(self):
        # Initialize the chain by calling the helper method
        self.chain = self._create_chain()
        
        # Initialize the Pinecone store WITH two indexes (dense + hybrid).
        # We'll keep 'test-index' for dense usage, 'hybrid-index' for the hybrid usage.
        self.pinecone_store = HybridPineconeStore(
            dense_index_name="test-index",
            hybrid_index_name="hybrid-index",
            model="multilingual-e5-large"
        )
    
    def _get_relevant_chunks(self, query: str) -> str:
        """Helper function to get and format relevant chunks from Pinecone using hybrid search"""
        # Query for similar chunks from the 'hybrid' index
        matches = self.pinecone_store.query_chunk(
            query_text=query,
            top_k=3,
            namespace="reports",
            use_hybrid=True  
        )
        
        # Format the chunks with their metadata
        formatted_context = []
        for i, match in enumerate(matches, 1):
            if match.metadata and 'content' in match.metadata:
                chunk_text = match.metadata['content']
                chunk_title = match.metadata.get("report_title", "Unknown Source")
                formatted_context.append(f"Chunk {i}:\n{chunk_text}\n\nSource:\n{chunk_title}\n\n")
        
        # Join all formatted chunks
        return "\n".join(formatted_context)

    def _create_chain(self):
        """Helper method to create and configure the LangChain processing chain"""
        model = ChatOpenAI(model="gpt-4")

        prompt = ChatPromptTemplate.from_template(
            "Answer the following query in markdown format using the provided context\n"
            "Query: {query}\n"
            "Context:\n\n {context}\n"
            "Your response must directly start with the markdown. "
            "After responding, extract 3-5 main points that answer the query from this response along with their sources.\n"
            "When you extract these point, you must ensure that they come from the chunks, don't use your own knowledge.\n"
            "Format the output as a JSON list of objects, where each object has 'topic' and 'source' keys.\n"
            "Format: {{'topics': [{{'topic': 'topic 1', 'source': 'source 1'}}, {{'topic': 'topic 2', 'source': 'source 2'}}]}}"
        )
        parser = StrOutputParser()
        return (
            {
                "context": lambda x: self._get_relevant_chunks(x["query"]),
                "query": lambda x: x["query"]
            }
            | prompt
            | model
            | parser
        )

    async def process_stream(self, query: str, chat_history: list, completion_callback):
        """Generator function that yields streaming responses"""
        full_response = ""
        
        async for chunk in self.chain.astream({"query": query}):
            full_response += chunk
            yield chunk
        
        # After the stream is complete, call the completion callback
        await completion_callback(full_response)

class AdvancedQueryProcessor:
    def __init__(self):
        # The default chain (technique 1) is built at initialization.
        self.chain = self._create_default_chain()
        self.pinecone_store = PineconeStore(
            index_name="test-index",
            model="multilingual-e5-large"
        )

    def _get_relevant_chunks(self, query: str) -> str:
        """
        Retrieve and format relevant chunks from Pinecone using dense retrieval.
        """
        matches = self.pinecone_store.query_chunk(
            query_text=query,
            top_k=3,  # Adjust as needed
            namespace="reports"
        )
        formatted_context = []
        for i, match in enumerate(matches, 1):
            if match.metadata and "content" in match.metadata:
                chunk_text = match.metadata["content"]
                chunk_title = match.metadata.get("report_title", "Unknown Source")
                formatted_context.append(
                    f"Chunk {i}:\n{chunk_text}\n\nSource:\n{chunk_title}\n\n"
                )
        return "\n".join(formatted_context)

    def _get_relevant_chunks_list(self, query: str) -> list:
        """
        Retrieve a list of chunks (for Fusion-in-Decoder style processing).
        """
        matches = self.pinecone_store.query_chunk(
            query_text=query,
            top_k=3,
            namespace="reports"
        )
        chunks = []
        for i, match in enumerate(matches, 1):
            if match.metadata and "content" in match.metadata:
                chunk_text = match.metadata["content"]
                chunk_title = match.metadata.get("report_title", "Unknown Source")
                chunks.append(f"Chunk {i}:\n{chunk_text}\n\nSource:\n{chunk_title}")
        return chunks

    def _get_sparse_chunks(self, query: str) -> str:
        """
        Dummy implementation of sparse retrieval (e.g., BM25).
        In a real system, replace this with a call to a BM25 engine or similar.
        """
        return f"Sparse retrieval results for query: {query}"

    # === Chain Builders for Different Techniques ===

    def _create_default_chain(self):
        """
        Default chain using the provided context from dense retrieval.
        (Technique 1)
        """
        model = ChatOpenAI(model="gpt-4o")
        prompt = ChatPromptTemplate.from_template(
            "Answer the following query in markdown format using the provided context.\n"
            "Query: {query}\n"
            "Context:\n\n{context}\n"
            "Ensure your answer starts with markdown formatting. "
            "After answering, extract 3-5 main points along with their sources in JSON format.\n"
            "Format: {{'topics': [{{'topic': 'topic 1', 'source': 'source 1'}}, "
            "{{'topic': 'topic 2', 'source': 'source 2'}}]}}"
        )
        parser = StrOutputParser()
        chain = (
            {
                "context": lambda x: self._get_relevant_chunks(x["query"]),
                "query": lambda x: x["query"]
            }
            | prompt
            | model
            | parser
        )
        return chain

    def _build_chain_hybrid_retrieval(self, query: str):
        """
        Technique 2: Hybrid Retrieval
        Combines dense (via Pinecone) and sparse (simulated) retrieval.
        """
        dense_context = self._get_relevant_chunks(query)
        sparse_context = self._get_sparse_chunks(query)
        combined_context = (
            f"{dense_context}\n\n---\n\nSparse Retrieval Context:\n{sparse_context}"
        )
        model = ChatOpenAI(model="gpt-4o")
        prompt = ChatPromptTemplate.from_template(
            "Answer the following query in markdown format using the combined context provided below.\n"
            "Query: {query}\n"
            "Combined Context:\n\n{context}\n"
            "Ensure your answer starts with markdown formatting and ends by extracting 3-5 main points along with their sources in JSON format."
        )
        parser = StrOutputParser()
        chain = (
            {
                "context": lambda x: combined_context,
                "query": lambda x: x["query"]
            }
            | prompt
            | model
            | parser
        )
        return chain

    def _build_chain_fusion_in_decoder(self, query: str):
        """
        Technique 3: Fusion-in-Decoder (FiD)
        Provides each retrieved chunk separately.
        """
        chunks = self._get_relevant_chunks_list(query)
        fid_context = "\n\n---\n\n".join(chunks)
        model = ChatOpenAI(model="gpt-4o")
        prompt = ChatPromptTemplate.from_template(
            "Answer the following query in markdown format using each provided evidence chunk separately.\n"
            "Query: {query}\n"
            "Evidence Chunks:\n\n{context}\n"
            "Ensure that your answer starts with markdown formatting and concludes with 3-5 main points along with their sources in JSON format."
        )
        parser = StrOutputParser()
        chain = (
            {
                "context": lambda x: fid_context,
                "query": lambda x: x["query"]
            }
            | prompt
            | model
            | parser
        )
        return chain

    def _build_chain_dynamic_prompt_engineering(self, query: str):
        """
        Technique 4: Dynamic & Context-Aware Prompt Engineering.
        Adjusts prompt instructions based on query length/complexity.
        """
        complexity_note = (
            "This appears to be a complex query. Please provide a detailed, structured answer."
            if len(query) > 50
            else "Provide a concise and clear answer."
        )
        context = self._get_relevant_chunks(query)
        model = ChatOpenAI(model="gpt-4o")
        prompt = ChatPromptTemplate.from_template(
            f"Note: {complexity_note}\n"
            "Answer the following query in markdown format using the provided context.\n"
            "Query: {{query}}\n"
            "Context:\n\n{{context}}\n"
            "After your answer, extract 3-5 main points along with their sources in JSON format."
        )
        parser = StrOutputParser()
        chain = (
            {
                "context": lambda x: context,
                "query": lambda x: x["query"]
            }
            | prompt
            | model
            | parser
        )
        return chain

    def _build_chain_iterative_retrieval(self, query: str):
        """
        Technique 5: Iterative and Multi-Hop Retrieval.
        Performs an initial pass and then refines the query for additional context.
        """
        # First pass (simulate an initial answer; here we use the default context)
        initial_context = self._get_relevant_chunks(query)
        # Simulate refining the query (for example, by appending a clarifying instruction)
        refined_query = query + " Please provide additional details on any overlooked aspects."
        additional_context = self._get_relevant_chunks(refined_query)
        combined_context = (
            f"{initial_context}\n\nAdditional Context:\n{additional_context}"
        )
        model = ChatOpenAI(model="gpt-4o")
        prompt = ChatPromptTemplate.from_template(
            "Based on the updated context, answer the following query in markdown format.\n"
            "Query: {query}\n"
            "Combined Context:\n\n{context}\n"
            "Ensure your answer starts with markdown formatting and concludes by extracting 3-5 main points with their sources in JSON format."
        )
        parser = StrOutputParser()
        chain = (
            {
                "context": lambda x: combined_context,
                "query": lambda x: x["query"]
            }
            | prompt
            | model
            | parser
        )
        return chain

    def _build_chain_rerank_verification(self, query: str):
        """
        Technique 6: Re-Ranking and Verification.
        Simulates re-ranking of retrieved chunks before generation.
        """
        chunks = self._get_relevant_chunks_list(query)
        # Simulate re-ranking (here we simply sort alphabetically as a placeholder).
        sorted_chunks = sorted(chunks)
        reranked_context = "\n\n---\n\n".join(sorted_chunks)
        model = ChatOpenAI(model="gpt-4o")
        prompt = ChatPromptTemplate.from_template(
            "Answer the following query in markdown format using the re-ranked evidence below.\n"
            "Query: {query}\n"
            "Re-ranked Evidence:\n\n{context}\n"
            "After answering, verify the evidence by extracting 3-5 main points along with their sources in JSON format."
        )
        parser = StrOutputParser()
        chain = (
            {
                "context": lambda x: reranked_context,
                "query": lambda x: x["query"]
            }
            | prompt
            | model
            | parser
        )
        return chain

    def _build_chain_dynamic_index_update(self, query: str):
        """
        Technique 7: Dynamic Index Updates and Continual Learning.
        (Assumes that your PineconeStore can update its index in real time.)
        """
        # If your PineconeStore has an update method, call it here.
        if hasattr(self.pinecone_store, "update_index"):
            self.pinecone_store.update_index()
        context = self._get_relevant_chunks(query)
        model = ChatOpenAI(model="gpt-4o")
        prompt = ChatPromptTemplate.from_template(
            "After updating the index, answer the following query in markdown format using the latest context.\n"
            "Query: {query}\n"
            "Context:\n\n{context}\n"
            "Provide your answer starting with markdown formatting and extract 3-5 key points with their sources in JSON format."
        )
        parser = StrOutputParser()
        chain = (
            {
                "context": lambda x: context,
                "query": lambda x: x["query"]
            }
            | prompt
            | model
            | parser
        )
        return chain

    # === Single process_stream function with technique selection ===

    async def process_stream(self, query: str, chat_history: list, completion_callback, technique = 1):
        """
        Generator function that yields streaming responses from the chosen chain.
        
        :param query: The user query.
        :param chat_history: A list of previous conversation turns (unused in this demo).
        :param technique: An integer representing the desired advanced RAG technique.
                          Mapping:
                            1 - Default (dense retrieval)
                            2 - Hybrid Retrieval (dense + sparse)
                            3 - Fusion-in-Decoder (FiD)
                            4 - Dynamic Prompt Engineering
                            5 - Iterative Retrieval
                            6 - Re-Ranking & Verification
                            7 - Dynamic Index Update
        :param completion_callback: A callback to invoke with the full response after streaming.
        """
        full_response = ""
        # Select the appropriate chain based on the technique number.
        if technique == 1:
            chain = self.chain  # default chain
            chain_input = {"query": query}
        elif technique == 2:
            chain = self._build_chain_hybrid_retrieval(query)
            chain_input = {"query": query}
        elif technique == 3:
            chain = self._build_chain_fusion_in_decoder(query)
            chain_input = {"query": query}
        elif technique == 4:
            chain = self._build_chain_dynamic_prompt_engineering(query)
            chain_input = {"query": query}
        elif technique == 5:
            chain = self._build_chain_iterative_retrieval(query)
            chain_input = {"query": query}
        elif technique == 6:
            chain = self._build_chain_rerank_verification(query)
            chain_input = {"query": query}
        elif technique == 7:
            chain = self._build_chain_dynamic_index_update(query)
            chain_input = {"query": query}
        else:
            # Fallback to default chain if an unknown technique is passed.
            chain = self.chain
            chain_input = {"query": query}

        # Use asynchronous streaming to yield chunks.
        async for chunk in chain.astream(chain_input):
            full_response += chunk
            yield chunk

        # After the stream is complete, call the completion callback.
        await completion_callback(full_response)