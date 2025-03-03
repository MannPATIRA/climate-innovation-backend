import os
from re import L
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from common.pinecone_store import PineconeStore
from ranking_model.OnlineSVMRanker import OnlineSVMRanker
from ranking_model.paper import Paper
from .query_processors import MockQueryProcessor, QueryProcessor
from common.supabase_client import init_supabase
from supabase import Client
from backend_server.chat_repository import ChatNotFoundError, InvalidSourceTypeError, ChatRepository
from .gatherers import get_all_author_info
from ranking_model.ranker import RegressionRanker
from ranking_model.author import Author
from ranking_model.grant import Grant
import climate_graph.graph
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from ranking_model.ranker_manager import RankerManager
from ranking_model.RegressionRanker import RegressionRanker
import json

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

class GraphQuery(BaseModel):
    authorid: str
    paperid: str

class AuthQuery(BaseModel):
    authorid: str

class GraphNextQuery(BaseModel):
    authorid: str

class GraphPrecomputationQuery(BaseModel):
    authorids: list[str]
    paperid: str

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

author_queue = []
global_paperid = ""



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
query_processor = MockQueryProcessor()

ranker_classes = {
    'regression': RegressionRanker,
    'svm': OnlineSVMRanker,
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
        
        # Check if this is the first message (current_count == 0)
        is_first_message = current_count == 0
        
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
            
            # Generate and update chat name if this is the first message
            if is_first_message:
                chat_name = await query_processor.generate_chat_name(query.query)
                chat_repository.update_chat_name(query.chat_id, chat_name)
            
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
        seen_paper_ids = set()  # To avoid duplicates
        
        async def generate_events():
            # First event: papers with basic author info
            for match in results:
                metadata = match.metadata
                paper_id = metadata.get("paper_id")
                print(f"Paper ID: {paper_id}")
                if paper_id in seen_paper_ids:
                    continue
                seen_paper_ids.add(paper_id)
                
                # Get paper details from database instead of OpenAlex
                paper_records = supabase.table('papers')\
                    .select('*')\
                    .eq('id', int(float(paper_id)))\
                    .execute()
                
                if not paper_records.data:
                    continue  # Skip if paper not found in database
                
                details = paper_records.data[0]
                
                # Get authors from paper_authors table
                author_records = supabase.table('paper_authors')\
                    .select('authors(*)')\
                    .eq('paper_id', int(float(paper_id)))\
                    .execute()
                print(f"Number of author records: {len(author_records.data)}")

                authors = [Author(
                    name=author['authors']['display_name'],
                    citations=author['authors'].get('citations', 0),
                    hindex=author['authors'].get('h_index', 0),
                    organisation_history=author['authors'].get('institutions', []),
                    orcid=author['authors'].get('orcid'),
                    works_count=author['authors'].get('works_count', 0),
                    openAlexid=author['authors'].get('openalex_id'),
                    dob=None,  # Will be updated in second event
                    grants=[],  # Will be updated in second event
                    grant_org_name=None,  # Will be updated in second event
                    website=None  # Will be updated in second event
                ) for author in author_records.data]

                paper = Paper(
                    paper_id=str(paper_id),
                    openalex_id=metadata.get("openalex_id"),
                    title=details["title"],
                    relevancy=match.score,
                    doi=metadata.get("doi"),
                    abstract=details["abstract"],
                    publication_date=details.get("publication_date"),
                    citations=details.get("cited_by_count", 0),
                    authors=authors
                )
                paper_results.append(paper)

            yield f"data: {json.dumps({'type': 'initial', 'papers': [p.model_dump() for p in paper_results]})}\n\n"

            # Second event: additional author details
            author_updates = {}
            for paper in paper_results:
                for author in paper.authors:
                    if author.openAlexid not in author_updates:
                        # Get additional author info
                        author_info = get_all_author_info(author.openAlexid)
                        author_updates[author.openAlexid] = {
                            'dob': author_info.dob,
                            'grants': [g.model_dump() for g in author_info.grants],
                            'grant_org_name': author_info.grant_org_name,
                            'website': author_info.website
                        }

            yield f"data: {json.dumps({'type': 'author_details', 'updates': author_updates})}\n\n"

        return StreamingResponse(generate_events(), media_type="text/event-stream")
        
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

# Create a thread pool for blocking calls.
executor = ThreadPoolExecutor(max_workers=5)
precomputation_store = {}
computed_store = {}

def compute_author_connections(authorid: str, paperid: str) -> dict:
    """
    This function wraps the slow computation.
    It calls the blocking function to compute connections for a given author.
    """
    # Here, climate_graph.graph.get_relevant_authors is assumed to be a blocking call.
    result = climate_graph.graph.get_relevant_authors(author_id=authorid, paper_id=paperid)
    return result

# Background worker to precompute queued author connections.
async def background_worker():
    while True:
        # Iterate over a snapshot of keys in the precomputation store.
        for authorid in list(precomputation_store.keys()):
            record = precomputation_store.get(authorid)
            if record and not record["computed"]:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, compute_author_connections, authorid, record["paperid"]
                )
                record["result"] = result
                record["computed"] = True
            if record and record["computed"]:
                # Move the computed result into the computed_store.
                computed_store[authorid] = record["result"]
                del precomputation_store[authorid]
        await asyncio.sleep(2)

@app.post('/api/graph/get_auth_info')
async def get_auth_info(data: AuthQuery):
    try:
        authorid = data.authorid

        author_info = get_all_author_info(authorid)
        return author_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/graph/get_initial_connections')
async def get_initial_connections(data: GraphQuery):
    try:
        authorid = data.authorid
        paperid = data.paperid
        global global_paperid
        global_paperid = paperid

        # Synchronously compute immediate connections.
        immediate_connections = climate_graph.graph.get_relevant_authors(author_id=authorid, paper_id=paperid)

        # Queue each connected author for background precomputation,
        # if not already computed.
        for author in immediate_connections:
            aid = author.get('authorId')
            if aid and aid not in computed_store and aid not in precomputation_store:
                precomputation_store[aid] = {
                    "paperid": paperid,
                    "computed": False,
                    "result": None
                }
        print("PRECOMP: ", precomputation_store)
        print("COMP: ", computed_store)
        return {'connections': immediate_connections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/graph/get_next_connections')
async def get_next_connections(gnq: GraphNextQuery):
    try:
        print("ENTERING GNC")
        authorid = gnq.authorid
        if authorid in computed_store:
            print("Authpr In Computed Store")

            for connection in computed_store[authorid]:
                precomputation_store[connection['authorId']] = {
                    "paperid": global_paperid,
                    "computed": False,
                    "result": None
                }

            print("PRECOMP: ", precomputation_store)
            print("COMP: ", computed_store)
            return {"connections": computed_store[authorid]}
        elif authorid in precomputation_store:
            print("Authpr In PRECOMP Store")

            # Prioritize this author: compute immediately.
            result = await asyncio.get_event_loop().run_in_executor(
                executor, compute_author_connections, authorid, global_paperid
            )

            print("Authpr has been computed")

            computed_store[authorid] = result
            del precomputation_store[authorid]

            print("deleted from precomp and returning")

            precomputation_store[authorid] = {
                "paperid": global_paperid,
                "computed": False,
                "result": None
            }

            print("PRECOMP: ", precomputation_store)
            print("COMP: ", computed_store)
            return {"connections": result}
        else:
            print("Author not queued")


            # Not queued; compute on demand and store.
            result = await asyncio.get_event_loop().run_in_executor(
                executor, compute_author_connections, authorid, global_paperid
            )
            computed_store[authorid] = result
            print("Has been computed")

            precomputation_store[authorid] = {
                "paperid": global_paperid,
                "computed": False,
                "result": None
            }
            print("PRECOMP: ", precomputation_store)
            print("COMP: ", computed_store)
            return {"connections": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # A simple connection manager for handling websocket connections.
# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: List[WebSocket] = []
# # A simple connection manager for handling websocket connections.
# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: List[WebSocket] = []

#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         self.active_connections.append(websocket)

#     def disconnect(self, websocket: WebSocket):
#         if websocket in self.active_connections:
#             self.active_connections.remove(websocket)

#     async def broadcast(self, message: dict):
#         for connection in self.active_connections:
#             try:
#                 await connection.send_json(message)
#             except Exception:
#                 # In production, add logging or error handling here.
#                 pass

# manager = ConnectionManager()

# @app.websocket("/ws/precomputations")
# async def websocket_precomputations(websocket: WebSocket):
#     """
#     WebSocket endpoint that sends computed records to the front end as soon as they are ready.
#     """
#     await manager.connect(websocket)
#     try:
#         while True:
#             # Keep the connection open. You can also include a ping/pong or
#             # simply await for any message (if two-way communication is needed).
#             await asyncio.sleep(1)
#     except WebSocketDisconnect:
#         manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_worker())
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