class Paper:
    def __init__(self, paper_id, openalex_id, title, relevancy, authors, doi, abstract, publication_date):
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
    
    def __repr__(self):
        return f"Paper(id={self.paper_id}, score={self.score:.3f}, authors={self.authors})"



