import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from google.oauth2 import id_token
from google.auth.transport import requests

import climate_graph.graph
from backend_server.chat_repository import ChatNotFoundError, InvalidSourceTypeError, ChatRepository
from common.neo4j_client import Neo4jClient
from common.pinecone_store import PineconeStore
from common.supabase_client import init_supabase_async
from ranking_model.OnlineSVMRanker import OnlineSVMRanker
from ranking_model.RegressionRanker import RegressionRanker
from ranking_model.SevenQRanker import SevenQRanker
from ranking_model.author import Author
from ranking_model.grant import Grant
from ranking_model.paper import Paper
from ranking_model.ranker_manager import RankerManager
from .author_building import get_all_author_info
from .models import (
    Query,
    PaperQuery,
    GraphQuery,
    AuthQuery,
    GraphNextQuery,
    PaperFeedback,
    AuthorFeedback,
    AuthorState,
    AuthorCreate,
    AuthorUpdate,
    ChatCreate,
    NetworkQuery,
    NaturalLanguageGraphQuery,
    CipherQuery,
    SaveQueryRequest,
    RenameQueryRequest,
    RawCipherQuery
)
from .query_processors import QueryProcessor
from common.async_neo4j_client import AsyncNeo4jClient

# Replace the sync client initialization with async
supabase = None  # Will be initialized in startup event
neo4j_client = None  # Will be initialized in startup event
# Load environment variables
load_dotenv(override=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code (previously in @app.on_event("startup"))
    global supabase, chat_repository, ranker, neo4j_client
    
    # Initialize Supabase
    supabase = await init_supabase_async()
    print("Supabase initialized: ", supabase)
    chat_repository = ChatRepository(supabase_client=supabase)
    
    # Initialize ranker with async supabase client
    ranker = RankerManager(supabase, "main_ranker_manager", ranker_classes, 0.01)
    await ranker.load_model()
    
    # Initialize Neo4j client
    neo4j_client = AsyncNeo4jClient(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
        ssh_host=os.getenv("REMOTE_SERVER_HOST"),
        ssh_user=os.getenv("REMOTE_SERVER_USER"),
        ssh_password=os.getenv("REMOTE_SERVER_PASSWORD")
    )
    await neo4j_client.initialize()
    print("Neo4j client initialized")
    
    # Start background worker
    asyncio.create_task(background_worker())
    print("Background worker started")
    
    yield  # This is where FastAPI serves requests
    
    # Shutdown code (previously in @app.on_event("shutdown"))
    if neo4j_client:
        await neo4j_client.close()
        print("Neo4j client closed")

# Create the FastAPI app with the lifespan
app = FastAPI(lifespan=lifespan)

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


def build_author_from_dict(data: dict) -> Author:
    """
    Build an Author object from a dictionary of author data.

    Parameters
    ----------
    data : dict
        A dictionary containing author information. Expected keys include:
        - "name" (str): The name of the author.
        - "citations" (int): The number of citations the author has.
        - "dob" (str): The date of birth of the author.
        - "organisation_history" (list): A list of organisations the author has been affiliated with.
        - "orcid" (str): The ORCID identifier of the author.
        - "hindex" (int): The h-index of the author.
        - "grants" (list): A list of grants received by the author. Each grant is a dictionary with keys:
            - "title" (str): The title of the grant.
            - "category" (str): The category of the grant.
            - "value" (float): The value of the grant.
            - "funder" (str): The funder of the grant.
            - "organisation" (str): The organisation providing the grant.
        - "grant_org_name" (str, optional): The name of the grant organisation.
        - "website" (str, optional): The website of the author.
        - "openAlexid" (str): The OpenAlex ID of the author.
        - "works_count" (int): The number of works published by the author.

    Returns
    -------
    Author
        An Author object populated with the provided data.
    """
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
    """
    Build a Paper object from a dictionary of paper data.

    Parameters
    ----------
    data : dict
        A dictionary containing paper information. Expected keys include:
        - "paper_id" (str): The ID of the paper.
        - "openalex_id" (str): The OpenAlex ID of the paper.
        - "title" (str): The title of the paper.
        - "relevancy" (float): The relevancy score of the paper.
        - "doi" (str): The DOI of the paper.
        - "abstract" (str): The abstract of the paper.
        - "publication_date" (str): The publication date of the paper.
        - "authors" (list): A list of authors of the paper. Each author is a dictionary with keys:
            - "name" (str): The name of the author.
            - "citations" (int): The number of citations the author has.
            - "dob" (str): The date of birth of the author.
            - "organisation_history" (list): A list of organisations the author has been affiliated with.
            - "orcid" (str): The ORCID identifier of the author.
            - "hindex" (int): The h-index of the author.
            - "grants" (list): A list of grants received by the author. Each grant is a dictionary with keys:
                - "title" (str): The title of the grant.
                - "category" (str): The category of the grant.
                - "value" (float): The value of the grant.
                - "funder" (str): The funder of the grant.
                - "organisation" (str): The organisation providing the grant.
            - "grant_org_name" (str, optional): The name of the grant organisation.
            - "website" (str, optional): The website of the author.
            - "openAlexid" (str): The OpenAlex ID of the author.
            - "works_count" (int): The number of works published by the author.

    Returns
    -------
    Paper
        A Paper object populated with the provided data.
    """
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
query_processor = QueryProcessor()
# query_processor = MockQueryProcessor()

ranker_classes = {
    'SevenQ': (SevenQRanker, 0.8),
    'regression': (RegressionRanker, 0.1),
    'svm': (OnlineSVMRanker, 0.1),
}

# Don't initialize ranker here, move to startup event
ranker = None

@app.get("/")
async def home():
    return {"message": "Hello from the Hugging Face LLaMA backend from Aaryan Purohit!"}


@app.get("/api/login/gauth")
async def validate_gauth(token: str):
    """
    Validate a Google OAuth token and check if the user exists in the database.

    Parameters
    ----------
    token : str
        The Google OAuth token to be validated.

    Returns
    -------
    dict
        A dictionary containing a message and the user's email if the user is found.
        If the user is not found, raises an HTTPException with status code 404.
        If the token is invalid, raises an HTTPException with status code 401.
        If there is an internal server error, raises an HTTPException with status code 500.
    """

    try:

        # Verify the Google token first
        idinfo = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            os.getenv("GOOGLE_CLIENT_ID")
        )

    except ValueError:

        # Invalid token
        raise HTTPException(status_code=401, detail="Invalid token")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    user_email = idinfo['email']

    res = await supabase.table('users') \
        .select('*') \
        .eq('user_email', user_email) \
        .execute()

    if res.data:

        return {"message": "User found", "user_email": user_email}

    else:

        res = await supabase.table('login_attempt').insert({"user_email": user_email}).execute()

        if res.data:

            raise HTTPException(status_code=404, detail="User not found")

        else:

            raise HTTPException(status_code=500, detail="Failed to document user")


@app.get("/api/chat/{chat_id}")
async def get_chat(chat_id: int):
    """
    Retrieve a chat by its ID.

    Parameters
    ----------
    chat_id : int
        The ID of the chat to be retrieved.

    Returns
    -------
    dict
        A dictionary containing the chat details if found.
        If the chat is not found, raises an HTTPException with status code 404.
        If there is an internal server error, raises an HTTPException with status code 500.
    """
    try:
        return await chat_repository.get_chat(chat_id)
    except ChatNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/{source_type}/chat")
async def create_chat(source_type: str, chatCreate: ChatCreate):
    """
    Create a new chat.

    Parameters
    ----------
    source_type : str
        The source type for the chat.
    chatCreate : ChatCreate
        The chat creation details.

    Returns
    -------
    dict
        A dictionary containing the created chat details.
        If the source type is invalid, raises an HTTPException with status code 400.
        If there is an internal server error, raises an HTTPException with status code 500.
    """
    try:
        return await chat_repository.create_chat(source_type, chatCreate.user_email)
    except InvalidSourceTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reports/query")
async def stream_query(query: Query):
    """
    Stream query results.

    Parameters
    ----------
    query : Query
        The query details.

    Returns
    -------
    StreamingResponse
        A streaming response with the query results.
        If the chat is not found, raises an HTTPException with status code 404.
        If there is an internal server error, raises an HTTPException with status code 500.
    """
    try:
        # Get chat and verify it exists
        chat = await chat_repository.get_chat(query.chat_id)
        current_count = chat.get('message_count', 0)

        # Check if this is the first message (current_count == 0)
        is_first_message = current_count == 0

        # Get chat history
        chat_history = await chat_repository.get_chat_history(query.chat_id)

        # Store user message
        new_message_order = current_count + 1
        await chat_repository.add_message(
            query.chat_id,
            query.query,
            new_message_order,
            True
        )

        # Update message count
        await chat_repository.update_message_count(query.chat_id, new_message_order)

        async def streaming_completion_callback(full_response: str):
            """Callback function called when streaming is complete"""
            # Store assistant response
            response_order = new_message_order + 1
            await chat_repository.add_message(
                query.chat_id,
                full_response,
                response_order,
                False
            )

            # Update message count again
            await chat_repository.update_message_count(query.chat_id, response_order)

            # Generate and update chat name if this is the first message
            if is_first_message:
                chat_name = await query_processor.generate_chat_name(query.query)
                await chat_repository.update_chat_name(query.chat_id, chat_name)

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
    """
    Search for papers based on the query.

    Parameters
    ----------
    query : PaperQuery
        The query details.

    Returns
    -------
    StreamingResponse
        A streaming response with the search results.
        If there is an internal server error, raises an HTTPException with status code 500.
    """
    try:
        # Initialize PineconeStore
        pinecone_store = PineconeStore(index_name="climate-index")

        # Query the papers namespace
        results = pinecone_store.query_chunk(
            query_text=query.query,
            top_k=query.top_k,
            namespace="papers"
        )

        async def generate_events():
            # Format the results
            paper_results = []
            seen_paper_ids = set()  # To avoid duplicates

            # First event: papers with basic author info
            for match in results:
                metadata = match.metadata
                paper_id = metadata.get("paper_id")
                print(f"Paper ID: {paper_id}")
                if paper_id in seen_paper_ids:
                    continue
                seen_paper_ids.add(paper_id)

                # Get paper details from database instead of OpenAlex
                paper_records = await supabase.table('papers') \
                    .select('*') \
                    .eq('id', int(float(paper_id))) \
                    .execute()

                if not paper_records.data:
                    continue  # Skip if paper not found in database

                details = paper_records.data[0]

                # Get authors from paper_authors table
                author_records = await supabase.table('paper_authors') \
                    .select('authors(*)') \
                    .eq('paper_id', int(float(paper_id))) \
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

            ranked_papers = paper_results
            yield f"data: {json.dumps({'type': 'initial', 'papers': [p.model_dump() for p in ranked_papers]})}\n\n"
            print("Should have yielded first all papers ")

            # Second event: additional author details
            author_updates = {}
            for paper in ranked_papers:
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
    """
    Background worker to precompute queued author connections.

    This function continuously iterates over the precomputation store, checks if the computation
    for each author is done, and if not, runs the computation in an executor. The result is then
    stored in the computed store and removed from the precomputation store.

    Returns
    -------
    None
    """
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
    """
    Retrieve author information based on the provided author ID.

    Parameters
    ----------
    data : AuthQuery
        The query details containing the author ID.

    Returns
    -------
    dict
        A dictionary containing the author information.
        If there is an internal server error, raises an HTTPException with status code 500.
    """
    try:
        authorid = data.authorid

        author_info = get_all_author_info(authorid)
        return author_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/graph/get_initial_connections')
async def get_initial_connections(data: GraphQuery):
    """
    Retrieve initial connections for a given author and paper.

    Parameters
    ----------
    data : GraphQuery
        The query details containing the author ID and paper ID.

    Returns
    -------
    dict
        A dictionary containing the initial connections.
        If there is an internal server error, raises an HTTPException with status code 500.
    """
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
    """
    Retrieve the next set of connections for a given author.

    Parameters
    ----------
    gnq : GraphNextQuery
        The query details containing the author ID.

    Returns
    -------
    dict
        A dictionary containing the next set of connections.
        If there is an internal server error, raises an HTTPException with status code 500.
    """
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


@app.post("/api/crm/authors")
async def create_author(author: AuthorCreate):
    try:
        # Check if author already exists
        existing = await supabase.table("author_crm").select("*") \
            .eq("openalex_id", author.openalex_id) \
            .eq("user_email", author.user_email).execute()

        if existing.data:
            raise HTTPException(status_code=400, detail="Author already exists in CRM")

        # Create new author
        result = await supabase.table("author_crm").insert({
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
        result = await supabase.table("author_crm").update({"note": note_update.note}).eq("id", author_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Author not found")

        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/api/crm/authors/{author_id}/state")
async def update_author_state(author_id: int, state_update: AuthorUpdate):
    try:
        result = await supabase.table("author_crm").update({"state": state_update.state}).eq("id", author_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Author not found")

        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/crm/authors")
async def get_authors(user_email: str):
    try:
        result = await supabase.table("author_crm").select("*").eq("user_email", user_email).execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/crm/authors/{author_id}")
async def get_author(author_id: int):
    try:
        result = await supabase.table("author_crm").select("*").eq("id", author_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Author not found")

        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/crm/authors/{author_id}")
async def delete_author(author_id: int):
    try:
        result = await supabase.table("author_crm").delete().eq("id", author_id).execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="Author not found")

        return {"message": "Author deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chats")
async def get_all_chats(user_email: str):
    try:
        result = await chat_repository.get_all_chats(user_email)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str):
    try:
        result = await chat_repository.get_chat_history(chat_id)
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
        graph = await neo4j_client.get_coauthor_network(
            author_id=query.author_id,
            limit=query.limit
        )

        serializable_graph = serialize_neo4j_graph(graph)
        return {"graph": serializable_graph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/graph/topic_network")
async def get_topic_network(query: NetworkQuery):
    """Get the topic-based network for a given author"""
    try:
        graph = await neo4j_client.get_topic_network(
            author_id=query.author_id,
            limit=query.limit
        )

        serializable_graph = serialize_neo4j_graph(graph)
        return {"graph": serializable_graph}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/graph/author_topics")
async def get_author_topics(query: NetworkQuery):
    """Get all topics researched by an author"""
    try:
        graph = await neo4j_client.get_author_topics(
            author_id=query.author_id,
            limit=query.limit
        )

        serializable_graph = serialize_neo4j_graph(graph)
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

        # Execute the generated query
        graph = await neo4j_client.execute_custom_query(
            query=cypher_query,
            params={"author_id": query_data.author_id},
            limit=query_data.limit
        )

        # Serialize the result
        serializable_graph = serialize_neo4j_graph(graph)

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

        result = await neo4j_client.get_node_by_id(
            node_type=type_mapping[node_type.lower()],
            node_id=node_id
        )

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
        result = await supabase.table("saved_cipher_queries").insert({
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
        result = await supabase.table("saved_cipher_queries").update({
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
        result = await supabase.table("saved_cipher_queries").select("*").eq("user", user).execute()
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
        # Execute the query
        graph = await neo4j_client.execute_custom_query(
            query=query_data.query,
            params=query_data.params,
            limit=query_data.limit
        )

        # Serialize the result
        serializable_graph = serialize_neo4j_graph(graph)

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
        check_result = await supabase.table("saved_cipher_queries").select("id").eq("id", query_id).execute()

        if not check_result.data:
            raise HTTPException(status_code=404, detail="Query not found")

        # Delete the query
        result = await supabase.table("saved_cipher_queries").delete().eq("id", query_id).execute()

        return {"message": f"Query with ID {query_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
