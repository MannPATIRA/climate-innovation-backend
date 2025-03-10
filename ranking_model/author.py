from pydantic import BaseModel
from typing import List, Optional, Union, Any, Dict
from .grant import Grant

class Author(BaseModel):
    name: Optional[str] = None
    citations: Optional[int] = None
    dob: Optional[str] = None
    organisation_history: Optional[Union[str, List[str], List[Any]]] = None  # Allow both string and list formats
    website: Optional[str] = None
    orcid: Optional[str] = None
    hindex: Optional[int] = None
    grants: List[Grant] = []
    grant_org_name: Optional[str] = None
    openAlexid: Optional[str] = None
    works_count: Optional[Union[int, str]] = None  # Allow both int and string formats
    score: Optional[float] = None  # This will be computed by the Ranker
    profile: Optional[Dict] = None
    employment: Optional[Dict] = None

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return f"Author(name={self.name}, score={self.score:.3f})" if self.score is not None else f"Author(name={self.name})"

