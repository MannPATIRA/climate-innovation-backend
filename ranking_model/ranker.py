from abc import ABC, abstractmethod
import numpy as np
from typing import List
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from .author import Author
from .paper import Paper
from .grant import Grant


class Ranker(ABC):
    """
    Abstract base class for Ranker implementations.
    """
    @abstractmethod
    def __init__(self, learning_rate: float = 0.01):
        pass

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

    @abstractmethod
    def rank(self, papers: List[Paper]) -> List[Paper]:
        """
        Ranks a list of papers based on their authors' metrics and paper relevancy.
        Each implementation should define its own ranking algorithm.

        Returns a list of papers ranked by their overall score (highest first).
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
          - Remove the author from the paper.
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

class RegressionRanker(Ranker):
    def __init__(self, learning_rate: float = 0.01):
        """
        The Ranker maintains separate model weights for ranking authors and papers.
          - For authors, we now use an extended feature vector:
                [citations, hindex, total_grant_value, num_grants, works_count]
          - For papers, we use the paper relevancy score.
        """
        self.learning_rate = learning_rate
        self.author_weights = {
            'citations': 0.5,
            'hindex': 0.5,
            'total_grant_value': 0.1,
            'num_grants': 0.1,
            'works_count': 0.1
        }
        self.paper_weights = {'relevancy': 1.0}

    @staticmethod
    def sigmoid(x: float) -> float:
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def rank(self, papers: List[Paper]) -> List[Paper]:
        """
        Rank each paper by computing:
          - Each author's score using the extended feature vector.
          - Authors are sorted within the paper (highest score first).
          - The paper's overall score is computed as the sum of its authors' scores plus
            a contribution from the paper relevancy.
        Returns a list of papers ranked by their overall score (highest first).
        """
        for paper in papers:
            # Compute score for each author using the extended feature vector.
            for author in paper.authors:
                features = self.get_extended_feature_vector(author)
                weights = np.array([
                    self.author_weights['citations'],
                    self.author_weights['hindex'],
                    self.author_weights['total_grant_value'],
                    self.author_weights['num_grants'],
                    self.author_weights['works_count']
                ])
                raw_score = np.dot(features, weights)
                author.score = self.sigmoid(raw_score)
            # Sort authors within the paper (highest score first).
            paper.authors.sort(key=lambda a: a.score, reverse=True)
            # Compute paper relevancy contribution.
            paper_contrib = self.sigmoid(self.paper_weights['relevancy'] * paper.relevancy)
            # Overall paper score: relevancy contribution plus the sum of its authors' scores.
            paper.score = paper_contrib + sum(author.score for author in paper.authors)
        # Return papers sorted by overall score.
        return sorted(papers, key=lambda p: p.score, reverse=True)

    def update_author_model(self, author: Author, label: int):
        """
        Update the author model weights using an online logistic regression update.
        Label is 1 for acceptance and 0 for rejection.
        Uses the extended feature vector.
        """
        features = self.get_extended_feature_vector(author)
        weights = np.array([
            self.author_weights['citations'],
            self.author_weights['hindex'],
            self.author_weights['total_grant_value'],
            self.author_weights['num_grants'],
            self.author_weights['works_count']
        ])
        raw_score = np.dot(features, weights)
        prediction = self.sigmoid(raw_score)
        error = label - prediction
        updated_weights = weights + self.learning_rate * error * features

        # Update the weights dictionary.
        self.author_weights['citations'] = updated_weights[0]
        self.author_weights['hindex'] = updated_weights[1]
        self.author_weights['total_grant_value'] = updated_weights[2]
        self.author_weights['num_grants'] = updated_weights[3]
        self.author_weights['works_count'] = updated_weights[4]

        print("Updated author weights:", self.author_weights)

    def update_paper_model(self, paper: Paper, label: int):
        """
        Update the paper relevancy model weight using an online update rule.
        """
        feature = paper.relevancy  # single feature
        weight = self.paper_weights['relevancy']
        raw_score = weight * feature
        prediction = self.sigmoid(raw_score)
        error = label - prediction
        updated_weight = weight + self.learning_rate * error * feature
        self.paper_weights['relevancy'] = updated_weight
        print("Updated paper weights:", self.paper_weights)

    def delete_author(self, paper: Paper, author: Author):
        """
        Process a deletion of an author:
          - Update the author model with a rejection (label = 0).
          - Remove the author from the paper.
        """
        print(f"Deleting author '{author.name}' from paper '{paper.title}'.")
        self.update_author_model(author, label=0)
        paper.authors = [a for a in paper.authors if a.name != author.name]

    def accept_author(self, paper: Paper, author: Author):
        """
        Process an acceptance of an author (update with label = 1).
        """
        print(f"Accepting author '{author.name}' for paper '{paper.title}'.")
        self.update_author_model(author, label=1)

    def delete_paper(self, paper: Paper):
        """
        Process a deletion of a paper by updating the paper model with a rejection (label = 0).
        """
        print(f"Deleting paper '{paper.title}'.")
        self.update_paper_model(paper, label=0)

    def accept_paper(self, paper: Paper):
        """
        Process an acceptance of a paper (update with label = 1).
        """
        print(f"Accepting paper '{paper.title}'.")
        self.update_paper_model(paper, label=1)
