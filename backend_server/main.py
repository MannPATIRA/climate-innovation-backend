import os
from re import L
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from common.pinecone_store import PineconeStore
from ranking_model.SevenQRanker import SevenQRanker
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
from common.neo4j_client import Neo4jClient
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

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
    user_email: str

class AuthorUpdate(BaseModel):
    note: Optional[str] = None
    state: Optional[AuthorState] = None

class ChatCreate(BaseModel):
    user_email: str

author_queue = []
global_paperid = ""

class NetworkQuery(BaseModel):
    author_id: str
    limit: Optional[int] = 50
# Add new Pydantic model for natural language graph queries
class NaturalLanguageGraphQuery(BaseModel):
    author_id: str
    query: str
    limit: Optional[int] = 50

class CipherQuery(BaseModel):
    """Query to be executed on the Neo4j database"""
    query: str = Field(
        description="The cipher query based on the natural language query",
    )

class SaveQueryRequest(BaseModel):
    cipher_query: str
    name: str
    user: str

class RenameQueryRequest(BaseModel):
    name: str

class RawCipherQuery(BaseModel):
    """Model for executing raw cipher queries"""
    query: str
    params: Optional[Dict] = {}
    limit: Optional[int] = 50


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
    'SevenQ': (SevenQRanker, 0.8),
    'regression': (RegressionRanker, 0.1),
    'svm': (OnlineSVMRanker, 0.1),
}
ranker = RankerManager(supabase, "main_ranker_manager", ranker_classes, 0.01)
ranker.load_model()

@app.get("/")
async def home():
    return {"message": "Hello from the Hugging Face LLaMA backend from Aaryan Purohit!"}

from google.oauth2 import id_token
from google.auth.transport import requests

@app.get("/api/login/gauth")
async def validate_gauth(token: str):
    try:
        # Verify the Google token first
        idinfo = id_token.verify_oauth2_token(
            token.token,
            requests.Request(),
            os.getenv("GOOGLE_CLIENT_ID")
        )
    except ValueError:
        # Invalid token
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


    user_email = idinfo['email']
    res = supabase.table('users')\
        .select('*')\
        .eq('user_email', user_email)\
        .execute()
    if res.data:
        return {"message": "User found", "user_email": user_email}
    else:
        res = supabase.table('login_attempt').insert({"user_email": user_email}).execute()
        if res.data:
            raise HTTPException(status_code=404, detail="User not found")
        else:
            raise HTTPException(status_code=500, detail="Failed to document user")


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
async def create_chat(source_type: str, chatCreate: ChatCreate):
    try:
        return chat_repository.create_chat(source_type, chatCreate.user_email)
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
        existing = supabase.table("author_crm").select("*")\
            .eq("openalex_id", author.openalex_id)\
            .eq("user_email", author.user_email).execute()
        
        if existing.data:
            raise HTTPException(status_code=400, detail="Author already exists in CRM")
        
        # Create new author
        result = supabase.table("author_crm").insert({
            "name": author.name,
            "institution": author.institution,
            "note": author.note,
            "openalex_id": author.openalex_id,
            "state": AuthorState.UNCONTACTED,
            "user_email": author.user_email
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
async def get_authors(user_email: str):
    try:
        result = supabase.table("author_crm").select("*").eq("user_email", user_email).execute()
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
async def get_all_chats(user_email: str):
    try:
        result = chat_repository.get_all_chats(user_email)
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

def serialize_neo4j_graph(graph):
    """Convert Neo4j graph object into JSON-serializable format"""
    return {
        "nodes": [
            {
                "id": node.id,
                "labels": list(node.labels),
                "properties": dict(node)
            }
            for node in graph.nodes
        ],
        "relationships": [
            {
                "id": rel.id,
                "type": rel.type,
                "start_node": rel.start_node.id,
                "end_node": rel.end_node.id,
                "properties": dict(rel)
            }
            for rel in graph.relationships
        ]
    }

@app.post("/api/graph/coauthor_network")
async def get_coauthor_network(query: NetworkQuery):
    """Get the coauthor network for a given author"""
    try:
        neo4j_client = Neo4jClient(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        graph = neo4j_client.get_coauthor_network(
            author_id=query.author_id,
            limit=query.limit
        )
        
        serializable_graph = serialize_neo4j_graph(graph)
        neo4j_client.close()
        return {"graph": serializable_graph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/graph/topic_network") 
async def get_topic_network(query: NetworkQuery):
    """Get the topic-based network for a given author"""
    try:
        neo4j_client = Neo4jClient(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        graph = neo4j_client.get_topic_network(
            author_id=query.author_id,
            limit=query.limit
        )
        
        serializable_graph = serialize_neo4j_graph(graph)
        neo4j_client.close()
        return {"graph": serializable_graph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/graph/author_topics")
async def get_author_topics(query: NetworkQuery):
    """Get all topics researched by an author"""
    try:
        neo4j_client = Neo4jClient(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        graph = neo4j_client.get_author_topics(
            author_id=query.author_id,
            limit=query.limit
        )
        
        serializable_graph = serialize_neo4j_graph(graph)
        neo4j_client.close()
        return {"graph": serializable_graph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/graph/natural_language_query")
async def natural_language_graph_query(query_data: NaturalLanguageGraphQuery):
    """
    Generate and execute a Cypher query from natural language, centered around a specific author.
    Returns the serialized graph result.
    """
    try:
        # Initialize ChatOpenAI
        llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        ).with_structured_output(CipherQuery)
        
        # Create system and human messages for the prompt with examples
        system_message = SystemMessage(content="""
            You are a Neo4j Cypher query generator. Your task is to convert natural language queries 
            into valid Cypher queries that explore academic collaboration networks.
            
            The database schema includes:
            - (Author)-[AUTHORED]->(Work)
            - (Work)-[MENTIONS]->(Topic)
            - (Author)-[RESEARCHES]->(Topic)
            - (Author)-[AFFILIATED_WITH]->(Institution)
            
            Here are example query patterns to follow:

            1. For coauthor networks:
            Query: "Show me this author's coauthors and their shared papers"
            ```
            MATCH (a:Author {id: $author_id})-[auth1:AUTHORED]->(w:Work)<-[auth2:AUTHORED]-(co:Author)
            WHERE a <> co
            RETURN DISTINCT a, auth1, w, auth2, co
            ```

            2. For topic-based networks:
            Query: "Show me authors who research similar topics"
            ```
            MATCH (a:Author {id: $author_id})-[r1:RESEARCHES]->(t:Topic)<-[r2:RESEARCHES]-(other:Author)
            WHERE a <> other
            RETURN DISTINCT a, r1, t, r2, other
            ```

            3. For author's research topics:
            Query: "Show me all topics this author researches"
            ```
            MATCH (a:Author {id: $author_id})-[r:RESEARCHES]->(t:Topic)
            RETURN DISTINCT a, r, t
            ORDER BY r.paperCount DESC
            ```

            Always follow these rules:
            1. Start queries with MATCH (a:Author {id: $author_id})
            2. Return graph structures (nodes and relationships)
            3. Use DISTINCT in RETURN statements
            4. Do not include LIMIT clauses
            5. Ensure all node labels and relationship types match the schema
            6. Use meaningful variable names
            7. Include relevant WHERE clauses for filtering
            8. Use parameters with $ prefix (especially for author_id)
            9. Maintain variable scope by including necessary variables in all WITH clauses.
        """)
        
        human_message = HumanMessage(content=f"""
            Generate a Cypher query for the following request, starting with author ID '{query_data.author_id}':
            {query_data.query}
        """)
        
        # Generate Cypher query using LLM
        response = await llm.ainvoke([system_message, human_message])
        cypher_query = response.query.strip().replace("```cypher", "").replace("```", "")
        print("CYPHER QUERY: ", cypher_query)
        print("____________end of cipher query____________")
        # Initialize Neo4j client
        neo4j_client = Neo4jClient(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        # Execute the generated query
        graph = neo4j_client.execute_custom_query(
            query=cypher_query,
            params={"author_id": query_data.author_id},
            limit=query_data.limit
        )
        
        # Serialize the result
        serializable_graph = serialize_neo4j_graph(graph)
        neo4j_client.close()
        
        return {
            "graph": serializable_graph,
            "generated_query": cypher_query
        }
    except Exception as e:
        print("ERROR: ", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/nodes/{node_type}/{node_id}")
async def get_node_properties(
    node_type: str, 
    node_id: str
):
    """
    Get properties of a node by its type and OpenAlex ID.
    
    Args:
        node_type: One of 'work', 'author', 'topic', or 'institution'
        node_id: OpenAlex ID (URL or just the ID portion)
    """
    try:
        # Map route parameters to Neo4j node types
        type_mapping = {
            'work': 'Work',
            'author': 'Author', 
            'topic': 'Topic',
            'institution': 'Institution'
        }
        
        if node_type.lower() not in type_mapping:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid node type. Must be one of: {', '.join(type_mapping.keys())}"
            )
            
        # Convert ID to full OpenAlex URL if needed
        if not node_id.startswith('https://'):
            base_url = 'https://openalex.org'
            # If ID starts with letter identifier (W, A, etc), keep it
            if node_id[0].upper() in ['W', 'A', 'T', 'I']:
                node_id = f"{base_url}/{node_id}"
            else:
                # Otherwise assume it's a work ID and add W prefix
                node_id = f"{base_url}/W{node_id}"
                
        neo4j_client = Neo4jClient(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        result = neo4j_client.get_node_by_id(
            node_type=type_mapping[node_type.lower()],
            node_id=node_id
        )
        
        neo4j_client.close()
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Node not found with ID: {node_id}"
            )
            
        return result
        
    except Exception as e:
        print("ERROR: ", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/graph/queries")
async def save_query(query: SaveQueryRequest):
    """Save a cipher query to the database"""
    try:
        result = supabase.table("saved_cipher_queries").insert({
            "cipher_query": query.cipher_query,
            "name": query.name,
            "user": query.user
        }).execute()
        
        return result.data[0]
    except Exception as e:
        print("ERROR: ", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/graph/queries/{query_id}/rename")
async def rename_query(query_id: int, rename_request: RenameQueryRequest):
    """Rename a saved cipher query"""
    try:
        result = supabase.table("saved_cipher_queries").update({
            "name": rename_request.name
        }).eq("id", query_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Query not found")
            
        return result.data[0]
    except Exception as e:
        print("ERROR: ", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/queries")
async def get_saved_queries(user: str):
    """Get all saved queries for a user"""
    try:
        result = supabase.table("saved_cipher_queries").select("*").eq("user", user).execute()
        return result.data
    except Exception as e:
        print("ERROR: ", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/graph/execute_cipher")
async def execute_cipher_query(query_data: RawCipherQuery):
    """
    Execute a raw Cypher query and return the graph result.
    Uses the same serialization format as the natural language query endpoint.
    """
    try:
        # Initialize Neo4j client
        neo4j_client = Neo4jClient(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        
        # Execute the query
        graph = neo4j_client.execute_custom_query(
            query=query_data.query,
            params=query_data.params,
            limit=query_data.limit
        )
        
        # Serialize the result
        serializable_graph = serialize_neo4j_graph(graph)
        neo4j_client.close()
        
        return {
            "graph": serializable_graph,
            "executed_query": query_data.query
        }
    except Exception as e:
        print("ERROR: ", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/graph/queries/{query_id}")
async def delete_query(query_id: int):
    """Delete a saved cipher query"""
    try:
        # Check if the query exists first
        check_result = supabase.table("saved_cipher_queries").select("id").eq("id", query_id).execute()
        
        if not check_result.data:
            raise HTTPException(status_code=404, detail="Query not found")
        
        # Delete the query
        result = supabase.table("saved_cipher_queries").delete().eq("id", query_id).execute()
        
        return {"message": f"Query with ID {query_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
