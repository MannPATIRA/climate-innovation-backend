import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from fastapi.responses import StreamingResponse
from common.pinecone_store import PineconeStore
from .query_processors import MockQueryProcessor, QueryProcessor
from common.supabase_client import init_supabase
from supabase import Client
from backend_server.chat_repository import ChatNotFoundError, InvalidSourceTypeError, ChatRepository
from gatherers import SemanticScholarInformationGatherer

supabase: Client = init_supabase()
# Load environment variables
load_dotenv(override=True)

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
    chat_id: str


# Initialize the query processor
# query_processor = QueryProcessor()
query_processor = MockQueryProcessor()

@app.get("/")
async def home():
    return {"message": "Hello from the Hugging Face LLaMA backend from Aaryan Purohit!"}


chat_repository = ChatRepository(supabase_client=supabase)
@app.get("/api/chat/{chat_id}")
async def get_chat(chat_id: int):
    try:
        return chat_repository.get_chat(chat_id)
    except ChatNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/{source_type}/chat")
async def create_chat(source_type: str):
    try:
        return chat_repository.create_chat(source_type)
    except InvalidSourceTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reports/query")
async def stream_query(query: Query):
    try:
        # Get chat and verify it exists
        chat = chat_repository.get_chat(query.chat_id)
        current_count = chat.get('message_count', 0)
        
        # Get chat history
        chat_history = chat_repository.get_chat_history(query.chat_id)
        
        # Store user message
        new_message_order = current_count + 1
        chat_repository.add_message(
            query.chat_id,
            query.query,
            new_message_order,
            True
        )
        
        # Update message count
        chat_repository.update_message_count(query.chat_id, new_message_order)

        async def streaming_completion_callback(full_response: str):
            """Callback function called when streaming is complete"""
            # Store assistant response
            response_order = new_message_order + 1
            chat_repository.add_message(
                query.chat_id,
                full_response,
                response_order,
                False
            )
                
            # Update message count again
            chat_repository.update_message_count(query.chat_id, response_order)
            
            print(f"Completed processing query:\n {query.query}")
            print(f"Full response:\n {full_response}")

        return StreamingResponse(
            query_processor.process_stream(
                query.query,
                chat_history=chat_history,
                completion_callback=streaming_completion_callback
            ),
            media_type="text/event-stream"
        )
    except ChatNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/construct_graph/{doi}")
async def construct_graph(doi: str):
    authors = SemanticScholarInformationGatherer.get_authors_from_doi(doi)

    for author in authors:
        SemanticScholarInformationGatherer.get_author_info(author["authorId"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
