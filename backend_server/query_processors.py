import asyncio
import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from common.pinecone_store import PineconeStore

class MockQueryProcessor:
    def __init__(self):
        pass

    async def process_stream(self, query: str, chat_history: list, completion_callback):
        """Generator function that yields streaming responses"""
        # Example predefined response
        mock_response = (
            "Based on the provided context, "
            "here is a summary of the key points:\n\n"
            "1. The document discusses various aspects of software architecture\n"
            "2. It emphasizes the importance of scalability\n"
            "3. Security considerations are highlighted\n\n"
            "The analysis suggests that...\n"
            "{'topics': ['Software Architecture', 'System Scalability', 'Security Principles']}"
        )
        
        full_response = ""
        
        # Simulate streaming by yielding chunks with delays
        for chunk in mock_response:
            
            full_response += chunk
            print(chunk)
            await asyncio.sleep(0.02)  # Add delay to simulate real streaming
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        
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
                formatted_context.append(f"Chunk {i}:\n{chunk_text}\n")
        
        # Join all formatted chunks
        return "\n".join(formatted_context)

    def _create_chain(self):
        """Helper method to create and configure the LangChain processing chain"""
        model = ChatOpenAI(model="gpt-4o")

        prompt = ChatPromptTemplate.from_template(
            "Answer the following query professionally using the provided context\n"
            "Query: {query}\n"
            "Context \n\n: {context}\n"
            "After responding, extract 3-5 main topics from this response.\n"
            "Format these topics as a JSON list of strings under a 'topics' key.\n"
            "Format: {{'topics': ['topic 1', 'topic 2', 'topic 3']}}"
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
            print(chunk)
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        
        # After the stream is complete, call the completion callback
        await completion_callback(full_response)