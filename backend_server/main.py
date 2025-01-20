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
from .query_processors import MockQueryProcessor, QueryProcessor

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


# Initialize the query processor
query_processor = QueryProcessor()

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
