from pydantic import BaseModel
from typing import List, Optional, Union
from .author import Author

class Paper(BaseModel):
    paper_id: Optional[str] = None
    openalex_id: Optional[str] = None
    title: Optional[str] = None
    relevancy: Optional[float] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    authors: List[Author] = []
    citations: Optional[int] = None
    score: Optional[float] = None
    init_authors: Optional[List[Author]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.init_authors is None and self.authors is not None:
            self.init_authors = self.authors.copy()

    def __repr__(self):
        return f"Paper(id={self.paper_id}, score={self.score:.3f if self.score else None}, authors={self.authors})"



