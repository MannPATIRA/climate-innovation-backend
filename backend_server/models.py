from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

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