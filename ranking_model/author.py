class Author:
    def __init__(self, name=None, citations=None, dob=None, organisation_history=None, orcid=None, hindex=None, grants=None, grant_org_name=None, website=None, openAlexid=None, works_count=None):
        """
        Represents an author.
        :param name: Author's name.
        :param citations: Number of citations.
        :param dob: Date of birth.
        :param hindex: h-index.
        """
        self.works_count = works_count
        self.openAlexid = openAlexid
        self.organisation_history = organisation_history
        self.website = website
        self.name = name
        self.citations = citations
        self.dob = dob
        self.hindex = hindex
        self.score = None  # This will be computed by the Ranker
        self.grants = grants if grants else []
        self.orcid = orcid
        self.grant_org_name = grant_org_name

    def __repr__(self):
        return f"Author(name={self.name}, score={self.score:.3f})" if self.score is not None else f"Author(name={self.name})"

