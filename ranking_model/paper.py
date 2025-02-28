class Paper:
    def __init__(self, paper_id=None, openalex_id=None, title=None, relevancy=None, authors=None, doi=None, abstract=None, publication_date=None, citations=None):
        """
        Represents a paper.
        """
        self.publication_date = publication_date
        self.abstract = abstract
        self.doi = doi
        self.openalex_id = openalex_id
        self.paper_id = paper_id
        self.title = title
        self.relevancy = relevancy
        self.authors = authors  # List of Author instances
        self.score = None  # Overall paper score (computed later)
        self.citations=citations
        self.init_authors = authors.copy()  # List of Author instances
    
    def __repr__(self):
        return f"Paper(id={self.paper_id}, score={self.score:.3f}, authors={self.authors})"



