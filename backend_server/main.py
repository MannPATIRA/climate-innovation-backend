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

supabase: Client = init_supabase()
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
    chat_id: str


# Initialize the query processor
query_processor = MockQueryProcessor()

@app.get("/")
async def home():
    return {"message": "Hello from the Hugging Face LLaMA backend from Aaryan Purohit!"}

@app.get("/api/chat/{chat_id}")
async def get_chat(chat_id: int):
    try:
        # Query the chat record by ID
        result = supabase.table("chats")\
            .select("*")\
            .eq("id", chat_id)\
            .single()\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail=f"Chat with ID {chat_id} not found")
            
        return result.data
        
    except Exception as e:
        if "404" in str(e):  # Handle Supabase's not found error
            raise HTTPException(status_code=404, detail=f"Chat with ID {chat_id} not found")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/{source_type}/chat")
async def create_chat(source_type: str):
    try:
        # Validate source_type and map to chat type
        if source_type not in ["reports", "papers"]:
            raise HTTPException(status_code=400, detail="Invalid source type. Must be 'reports' or 'papers'")
        
        # Insert new chat record
        data = supabase.table("chats").insert({
            "type": source_type.rstrip('s'),  # Convert 'reports' to 'report', 'papers' to 'paper'
        }).execute()
        
        return data.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reports/query")
async def stream_query(query: Query):
    # First get the chat and verify it exists
    chat_result = supabase.table("chats")\
        .select("*")\
        .eq("id", query.chat_id)\
        .execute()
        
    if not chat_result.data:
        raise HTTPException(status_code=404, detail=f"Chat with ID {query.chat_id} not found")
    
    chat = chat_result.data[0]
    current_count = chat.get('message_count', 0)
    
    # Get chat history
    history_result = supabase.table("chat_messages")\
        .select("*")\
        .eq("chat_id", query.chat_id)\
        .order("order")\
        .execute()
    
    chat_history = history_result.data
    # Store user message
    new_message_order = current_count + 1
    supabase.table("chat_messages").insert({
        "content": query.query,
        "order": new_message_order,
        "user_message": True,
        "chat_id": query.chat_id
    }).execute()
    
    # Update message count
    supabase.table("chats")\
        .update({"message_count": new_message_order})\
        .eq("id", query.chat_id)\
        .execute()
    

    async def streaming_completion_callback(full_response: str):
        """Callback function called when streaming is complete"""
        # Store assistant response
        response_order = new_message_order + 1
        supabase.table("chat_messages").insert({
            "content": full_response,
            "order": response_order,
            "user_message": False,
            "chat_id": query.chat_id
        }).execute()
            
        # Update message count again
        supabase.table("chats")\
            .update({"message_count": response_order})\
            .eq("id", query.chat_id)\
            .execute()
        # Here you would typically save to your database
        print(f"Completed processing query:\n {query.query}")
        print(f"Full response:\n {full_response}")
        # Add your database saving logic here

    try:
        # Create the streaming response with the completion callback
        return StreamingResponse(
            query_processor.process_stream(
                query.query,
                chat_history=chat_history,
                completion_callback=streaming_completion_callback
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
