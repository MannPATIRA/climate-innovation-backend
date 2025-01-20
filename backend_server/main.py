import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
import json
from fastapi.responses import StreamingResponse
from common.pinecone_store import PineconeStore
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
HF_MODEL_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

# Access HF_TOKEN from environment
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("No HF_TOKEN found in environment variables!")

# Pydantic model for request validation
class Query(BaseModel):
    query: str

class MockQueryProcessor:
    def __init__(self):
        pass

    async def process_stream(self, query: str, completion_callback):
        """Generator function that yields streaming responses"""
        # Example predefined response
        mock_response = [
            "Based on the provided context, ",
            "here is a summary of the key points:\n\n",
            "1. The document discusses various aspects of software architecture\n",
            "2. It emphasizes the importance of scalability\n",
            "3. Security considerations are highlighted\n\n",
            "The analysis suggests that...\n",
            "{'topics': ['Software Architecture', 'System Scalability', 'Security Principles']}"
        ]
        
        full_response = ""
        
        # Simulate streaming by yielding chunks with delays
        for chunk in mock_response:
            full_response += chunk
            print(chunk)
            await asyncio.sleep(0.05)  # Add delay to simulate real streaming
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        
        # After the stream is complete, call the completion callback
        await completion_callback(query, full_response)

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
            "Context: {context}\n"
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

    async def process_stream(self, query: str, completion_callback):
        """Generator function that yields streaming responses"""
        full_response = ""
        
        async for chunk in self.chain.astream({"query": query}):
            full_response += chunk
            print(chunk)
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        
        # After the stream is complete, call the completion callback
        await completion_callback(query, full_response)

# Initialize the query processor
query_processor = MockQueryProcessor()

@app.get("/")
async def home():
    return {"message": "Hello from the Hugging Face LLaMA backend from Aaryan Purohit!"}

@app.post("/api/reports/query")
async def stream_query(query: Query):
    async def streaming_completion_callback(query: str, full_response: str):
        """Callback function called when streaming is complete"""
        # Here you would typically save to your database
        print(f"Completed processing query:\n {query}")
        print(f"Full response:\n {full_response}")
        # Add your database saving logic here

    try:
        # Create the streaming response with the completion callback
        return StreamingResponse(
            query_processor.process_stream(
                query.query, 
                completion_callback=streaming_completion_callback
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
