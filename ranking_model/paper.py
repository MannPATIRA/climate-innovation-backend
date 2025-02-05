class Paper:
    def __init__(self, paper_id, name, title, institution, institution_size, funding, citations, relevancy, authors):
        """
        Represents a paper.
        :param paper_id: A unique identifier.
        :param name: Name or short code for the paper.
        :param title: Full title.
        :param institution: Institution that produced the paper.
        :param institution_size: Number of people at the institution.
        :param funding: Amount of funding (e.g., in thousands).
        :param citations: Number of citations for the paper.
        :param relevancy: A precomputed relevancy score for the search topic.
        :param authors: List of Author objects.
        """
        self.paper_id = paper_id
        self.name = name
        self.title = title
        self.institution = institution
        self.institution_size = institution_size
        self.funding = funding
        self.citations = citations
        self.relevancy = relevancy
        self.authors = authors  # List of Author instances
        self.score = None  # Overall paper score (computed later)
    
    def __repr__(self):
        return f"Paper(id={self.paper_id}, score={self.score:.3f}, authors={self.authors})"


