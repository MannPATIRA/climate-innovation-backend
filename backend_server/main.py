import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from common.pinecone_store import PineconeStore
from ranking_model.paper import Paper
from .query_processors import MockQueryProcessor, QueryProcessor
from common.supabase_client import init_supabase
from supabase import Client
from backend_server.chat_repository import ChatNotFoundError, InvalidSourceTypeError, ChatRepository
from .gatherers import OpenAlexInformationGatherer, authors_from_doi
from ranking_model.author import Author
from ranking_model.grant import Grant
from enum import Enum
from ranking_model.ranker_manager import RankerManager
from ranking_model.ranker import RegressionRanker, OnlineRankSVMRanker

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

# Add new Pydantic models for the request and response
class PaperQuery(BaseModel):
    query: str
    top_k: Optional[int] = 1000

class GrantData(BaseModel):
    title: str
    category: str
    value: float
    funder: str
    organisation: str
class AuthorData(BaseModel):
    name: str
    citations: int
    dob: str
    organisation_history: str
    orcid: str
    hindex: int
    grants: Optional[List[GrantData]] = []
    grant_org_name: Optional[str] = None
    website: Optional[str] = None
    openAlexid: str
    works_count: int
class PaperData(BaseModel):
    paper_id: str
    openalex_id: str
    title: str
    relevancy: float
    doi: str
    abstract: str
    publication_date: str
    authors: List[AuthorData]

# For paper feedback (accept/reject)
class PaperFeedback(BaseModel):
    paper: PaperData
    accepted: bool

# For author feedback (accept/reject)
class AuthorFeedback(BaseModel):
    paper: PaperData
    author_name: str
    accepted: bool

class AuthorState(str, Enum):
    UNCONTACTED = "uncontacted"
    INTERESTED = "interested"
    UNINTERESTED = "uninterested"
    BLOCKED = "blocked"

class AuthorCreate(BaseModel):
    name: str
    institution: str
    note: Optional[str] = None
    openalex_id: str

class AuthorUpdate(BaseModel):
    note: Optional[str] = None
    state: Optional[AuthorState] = None



def build_author_from_dict(data: dict) -> Author:
    grants = []
    for grant in data.get("grants", []):
        grants.append(Grant(
            title=grant["title"],
            category=grant["category"],
            value=grant["value"],
            funder=grant["funder"],
            organisation=grant["organisation"]
        ))
    return Author(
         name=data["name"],
         citations=data["citations"],
         dob=data["dob"],
         organisation_history=data["organisation_history"],
         orcid=data["orcid"],
         hindex=data["hindex"],
         grants=grants,
         grant_org_name=data.get("grant_org_name"),
         website=data.get("website"),
         openAlexid=data["openAlexid"],
         works_count=data["works_count"]
    )

    
def build_paper_from_dict(data: dict) -> Paper:
    authors = [build_author_from_dict(a) for a in data.get("authors", [])]
    return Paper(
        paper_id=data["paper_id"],
        openalex_id=data["openalex_id"],
        title=data["title"],
        relevancy=data["relevancy"],
        doi=data["doi"],
        abstract=data["abstract"],
        publication_date=data["publication_date"],
        authors=authors
    )


# Initialize the query processor
# query_processor = QueryProcessor()
query_processor = QueryProcessor()

ranker_classes = {
    'regression': RegressionRanker,
    'svm': OnlineRankSVMRanker,
}
ranker = RankerManager(supabase, "main_ranker_manager", ranker_classes, 0.01)
ranker.load_model()

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

@app.post("/api/papers/search")
async def search_papers(query: PaperQuery):
    try:
        # Initialize PineconeStore
        pinecone_store = PineconeStore(index_name="climate-index")
        
        # Query the papers namespace
        results = pinecone_store.query_chunk(
            query_text=query.query,
            top_k=query.top_k,
            namespace="papers"
        )

        # Format the results
        paper_results = []
        for match in results:
            metadata = match.metadata
            authors = authors_from_doi(metadata.get("doi"))
            details = OpenAlexInformationGatherer.get_details_from_paper_id(metadata.get("openalex_id"))
            paper_results.append(Paper(
                paper_id=metadata.get("paper_id"),
                openalex_id=metadata.get("openalex_id"),
                title=details["title"],
                relevancy=match.score,
                doi=metadata.get("doi"),
                abstract=details["abstract"],
                publication_date=details["publication_date"],
                authors=authors
            ))

        return ranker.ranker(paper_results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/papers/feedback")
async def paper_feedback(feedback: PaperFeedback):
    """
    Endpoint for paper acceptance or rejection.
    The frontend sends the paper (as JSON) along with a flag (accepted: true/false).
    The ranker then updates its model weights using the paper's relevancy.
    """
    try:
        paper_obj = build_paper_from_dict(feedback.paper.model_dump())
        if feedback.accepted:
            ranker.accept_paper(paper_obj)
            message = "Paper accepted. Model weights updated."
        else:
            ranker.delete_paper(paper_obj)
            message = "Paper rejected. Model weights updated."
        # message = "Paper ranking is disabled"
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/authors/feedback")
async def author_feedback(feedback: AuthorFeedback):
    """
    Endpoint for author acceptance or rejection.
    The frontend sends the paper (with the list of authors), the specific author name,
    and whether that author is accepted (true) or rejected (false).
    The ranker then updates its author model weights accordingly.
    """
    try:
        paper_obj = build_paper_from_dict(feedback.paper.model_dump())
        target_author = None
        for author in paper_obj.authors:
            if author.name == feedback.author_name:
                target_author = author
                break
        if target_author is None:
            raise HTTPException(status_code=404, detail="Author not found in paper.")
        if feedback.accepted:
            ranker.accept_author(paper_obj, target_author)
            message = "Author accepted. Model weights updated."
        else:
            ranker.delete_author(paper_obj, target_author)
            message = "Author rejected. Model weights updated."
        # message = "Author ranking is disabled"
        return {"message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/crm/authors")
async def create_author(author: AuthorCreate):
    try:
        # Check if author already exists
        existing = supabase.table("author_crm").select("*").eq("openalex_id", author.openalex_id).execute()
        
        if existing.data:
            raise HTTPException(status_code=400, detail="Author already exists in CRM")
        
        # Create new author
        result = supabase.table("author_crm").insert({
            "name": author.name,
            "institution": author.institution,
            "note": author.note,
            "openalex_id": author.openalex_id,
            "state": AuthorState.UNCONTACTED
        }).execute()
        
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/crm/authors/{author_id}/note")
async def update_author_note(author_id: int, note_update: AuthorUpdate):
    try:
        result = supabase.table("author_crm").update({"note": note_update.note}).eq("id", author_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Author not found")
            
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/crm/authors/{author_id}/state")
async def update_author_state(author_id: int, state_update: AuthorUpdate):
    try:
        result = supabase.table("author_crm").update({"state": state_update.state}).eq("id", author_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Author not found")
            
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crm/authors")
async def get_authors():
    try:
        result = supabase.table("author_crm").select("*").execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/crm/authors/{author_id}")
async def get_author(author_id: int):
    try:
        result = supabase.table("author_crm").select("*").eq("id", author_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Author not found")
            
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/crm/authors/{author_id}")
async def delete_author(author_id: int):
    try:
        result = supabase.table("author_crm").delete().eq("id", author_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Author not found")
            
        return {"message": "Author deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats")
async def get_all_chats():
    try:
        result = chat_repository.get_all_chats()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str):
    try:
        result = chat_repository.get_chat_history(chat_id)
        if not result:
            raise HTTPException(status_code=404, detail=f"No messages found for chat {chat_id}")
        return result
    except ChatNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
