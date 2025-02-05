import numpy as np

class Author:
    def __init__(self, name, citations, dob, orcid, hindex, grants, org_name, website):
        """
        Represents an author.
        :param name: Author's name.
        :param citations: Number of citations.
        :param dob: Date of birth.
        :param hindex: h-index.
        """
        self.website = website
        self.name = name
        self.citations = citations
        self.dob = dob
        self.hindex = hindex
        self.score = None  # This will be computed by the Ranker
        self.grants = grants if grants else []
        self.orcid = orcid
        self.org_name = org_name

    def get_feature_vector(self):
        """Return a numpy array of features used for ranking (order: citations, hindex)."""
        return np.array([self.citations, self.hindex], dtype=float)

    def __repr__(self):
        return f"Author(name={self.name}, score={self.score:.3f})" if self.score is not None else f"Author(name={self.name})"

