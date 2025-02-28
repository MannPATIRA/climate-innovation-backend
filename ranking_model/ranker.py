from abc import ABC, abstractmethod
import numpy as np
from typing import List
from supabase import Client
import functools

from .author import Author
from .paper import Paper

class Ranker(ABC):
    """
    Abstract base class for Ranker implementations.
    """
    def __init__(self, supabase_client: Client, model_name: str, learning_rate: float = 0.01):
        self.supabase = supabase_client
        self.model_name = model_name
        self.learning_rate = learning_rate

    def get_extended_feature_vector(self, author: Author) -> np.ndarray:
        """
        Build an extended feature vector for the author including:
            - citations
            - hindex
            - total grant value (sum of values of all grants)
            - number of grants
            - works count
        """
        total_grant_value = sum(grant.value for grant in author.grants) if author.grants else 0.0
        num_grants = len(author.grants)
        works_count = author.works_count
        return np.array([author.citations, author.hindex, total_grant_value, num_grants, works_count], dtype=float)

    @staticmethod
    def save_model_before_rank(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Since this is now a static method, we need to get the instance from args[0]
            instance = args[0]
            instance.save_model()
            return func(*args, **kwargs)
        return wrapper

    @abstractmethod
    @save_model_before_rank
    def rank_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        Ranks a list of papers based on their authors' metrics and paper relevancy.
        Each implementation should define its own ranking algorithm.
        
        (Auxiliary function to save the latest model before ranking)

        Returns a list of papers ranked by their overall score (highest first).
        """
        pass

    @abstractmethod
    def rank_authors(self, authors: List[Author]) -> List[Author]:
        """ 
        Ranks a list of authors based on their metrics.
        Each implementation should define its own ranking algorithm.

        Returns a list of authors ranked by their overall score (highest first).
        """
        pass

    @abstractmethod
    def update_author_model(self, author: Author, label: int):
        """
        Updates the author ranking model based on feedback.
        Implementations should define how author metrics affect the ranking model.
        
        Args:
            author: Author object containing metrics (citations, h-index, etc.)
            label: Binary feedback (1 for positive, 0 for negative)
        """
        pass

    @abstractmethod
    def update_paper_model(self, paper: Paper, label: int):
        """
        Updates the paper ranking model based on feedback.
        Implementations should define how paper metrics affect the ranking model.
        
        Args:
            paper: Paper object containing metrics (relevancy, etc.)
            label: Binary feedback (1 for positive, 0 for negative)
        """
        pass

    @abstractmethod
    def delete_author(self, paper: Paper, author: Author):
        """
        Process a deletion of an author:
          - Update the author model with a rejection (label = 0).
        """
        pass

    @abstractmethod
    def accept_author(self, paper: Paper, author: Author):
        """
        Process an acceptance of an author (update with label = 1).
        """
        pass

    @abstractmethod
    def delete_paper(self, paper: Paper):
        """
        Process a deletion of a paper by updating the paper model with a rejection (label = 0).
        """
        pass

    @abstractmethod
    def accept_paper(self, paper: Paper):
        """
        Process an acceptance of a paper (update with label = 1).
        """
        pass

    @abstractmethod
    def save_model(self):
        """
        Saves the ranker's model state to Supabase.
        Ensure to overwrite (upsert) the latest model in that name:
        ^ as we will already store the feedback so no need to store duplicate model
        """
        pass

    @abstractmethod
    def load_model(self) -> bool:
        """Loads the ranker's model state from Supabase."""
        pass


