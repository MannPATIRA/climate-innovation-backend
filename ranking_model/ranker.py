from abc import ABC, abstractmethod
import numpy as np
from typing import List
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from supabase import Client
import pickle
import functools
import gzip
import logging

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

    def save_model_before_rank(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self.save_model()
            return func(self, *args, **kwargs)
        return wrapper

    @abstractmethod
    @save_model_before_rank
    def rank(self, papers: List[Paper]) -> List[Paper]:
        """
        Ranks a list of papers based on their authors' metrics and paper relevancy.
        Each implementation should define its own ranking algorithm.
        
        (Auxilliary function to save the latest model before ranking)

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

class RegressionRanker(Ranker):
    def __init__(self, supabase_client: Client, model_name: str, learning_rate: float = 0.01):
        """
        The Ranker maintains separate model weights for ranking authors and papers.
          - For authors, we now use an extended feature vector:
                [citations, hindex, total_grant_value, num_grants, works_count]
          - For papers, we use the paper relevancy score.
        """
        super().__init__(supabase_client, model_name, learning_rate)
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
            paper_contrib = self.sigmoid(self.paper_weights['relevancy'] * paper.relevancy) # TODO: publication date too?
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

    def save_model(self):
        """
        Saves the RegressionRanker's model state to Supabase.
        """
        try:
            model_state = {
                'author_weights': self.author_weights,
                'paper_weights': self.paper_weights
            }
            serialized_model = pickle.dumps(model_state)
            compressed_model = gzip.compress(serialized_model)
            self.supabase.table('ranker_models').upsert({'model_name': self.model_name, 'model_data': compressed_model}).execute()
        except Exception as e:
            logging.exception(f"Error saving model '{self.model_name}': {e}")

    def load_model(self):
        """
        Loads the RegressionRanker's model state from Supabase.
        """
        response = self.supabase.table('ranker_models').select('model_data').eq('model_name', self.model_name).execute()
        if response.data and response.data[0]:
            serialized_model = response.data[0]['model_data']
            model_state = pickle.loads(gzip.decompress(serialized_model))
            self.author_weights = model_state['author_weights']
            self.paper_weights = model_state['paper_weights']
            return True
        else:
            print(f"Model '{self.model_name}' not found in Supabase.")
            return False

class OnlineRankSVMRanker(Ranker):
    """
    Ranker that uses Support Vector Machines with online learning capabilities.
    
        This ranker maintains two separate SGDClassifier models with hinge loss (SVM):
        - Author Model: Ranks authors based on their feature vectors:
            [citations, hindex, total_grant_value, num_grants, works_count]
        - Paper Model: Ranks papers based on their relevancy score
    
    Key features:
        - Incremental Learning: Uses partial_fit for online updates without full retraining
        - Feature Scaling: Maintains separate StandardScalers for author and paper features
        - Scoring: Combines author and paper scores for final paper ranking
        - Memory Efficient: Suitable for streaming data and continuous updates
    
    The ranking process:
        1. Author scores are computed using the author SVM model
        2. Authors are sorted within each paper by their scores
        3. Paper scores are computed using the paper SVM model
        4. Final paper ranking combines both author and paper scores
    
    Note:
        - Models are initialized on first update
        - Feature scaling parameters are preserved between updates
        - Binary feedback (0/1) is used for model updates
    """

    def __init__(self, supabase_client: Client, model_name: str, learning_rate: float = 0.01):
        """
        Initializes the RankSVMRanker with online SVM models for authors and papers.
        """
        super().__init__(supabase_client, model_name, learning_rate)

        # Initialize online classifiers for authors and papers
        self.author_model = SGDClassifier(
            loss='hinge',
            learning_rate='constant',
            eta0=self.learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=42
        )

        self.paper_model = SGDClassifier(
            loss='hinge',
            learning_rate='constant',
            eta0=self.learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=42
        )

        self.author_scaler = StandardScaler()
        self.paper_scaler = StandardScaler()

        # Flags to check if models are initialized
        self.is_author_model_initialized = False
        self.is_paper_model_initialized = False

    def rank(self, papers: List[Paper]) -> List[Paper]:
        """
        Ranks a list of papers using the trained SGDClassifier models.
        Each author's score is computed, authors are sorted within papers,
        and papers are sorted based on overall score.

        Returns a list of papers ranked by their overall score (highest first).
        """
        for paper in papers:
            # Collect feature vectors for authors
            author_features = [self.get_extended_feature_vector(author) for author in getattr(paper, 'authors', [])]
            author_features = np.array(author_features)

            # Scale features if model is initialized
            if self.is_author_model_initialized and len(author_features) > 0:
                author_features_scaled = self.author_scaler.transform(author_features)
                # Compute scores using decision function
                scores = self.author_model.decision_function(author_features_scaled)
                for author, score in zip(paper.authors, scores):
                    author.score = score
            else:
                # Default score if model not initialized
                for author in getattr(paper, 'authors', []):
                    author.score = 0.0

            # Sort authors by score
            paper.authors.sort(key=lambda a: getattr(a, 'score', 0.0), reverse=True)

            # Compute paper score
            paper_feature = np.array([getattr(paper, 'relevancy', 0.0)]).reshape(1, -1)
            if self.is_paper_model_initialized:
                paper_feature_scaled = self.paper_scaler.transform(paper_feature)
                paper_score = self.paper_model.decision_function(paper_feature_scaled)[0]
            else:
                paper_score = 0.0

            # Aggregate paper score with author scores
            paper.score = paper_score + sum(getattr(author, 'score', 0.0) for author in paper.authors)

        # Sort papers by overall score
        return sorted(papers, key=lambda p: getattr(p, 'score', 0.0), reverse=True)

    def update_author_model(self, author, label: int):
        features = self.get_extended_feature_vector(author).reshape(1, -1)
        labels = np.array([label])

        # Scale features
        if self.is_author_model_initialized:
            features_scaled = self.author_scaler.transform(features)
        else:
            features_scaled = self.author_scaler.fit_transform(features)
            self.is_author_model_initialized = True

        # Partial fit of the model
        self.author_model.partial_fit(features_scaled, labels, classes=np.array([0, 1]))

    def update_paper_model(self, paper, label: int):
        features = np.array([getattr(paper, 'relevancy', 0.0)]).reshape(1, -1)
        labels = np.array([label])

        # Scale features
        if self.is_paper_model_initialized:
            features_scaled = self.paper_scaler.transform(features)
        else:
            features_scaled = self.paper_scaler.fit_transform(features)
            self.is_paper_model_initialized = True

        # Partial fit of the model
        self.paper_model.partial_fit(features_scaled, labels, classes=np.array([0, 1]))

    def delete_author(self, paper, author):
        print(f"Deleting author '{getattr(author, 'name', 'Unknown')}' from paper '{getattr(paper, 'title', 'Unknown')}'.")
        self.update_author_model(author, label=0)
        paper.authors = [a for a in paper.authors if getattr(a, 'name', None) != getattr(author, 'name', None)]

    def accept_author(self, paper, author):
        print(f"Accepting author '{getattr(author, 'name', 'Unknown')}' for paper '{getattr(paper, 'title', 'Unknown')}'.")
        self.update_author_model(author, label=1)

    def delete_paper(self, paper):
        print(f"Deleting paper '{getattr(paper, 'title', 'Unknown')}'.")
        self.update_paper_model(paper, label=0)

    def accept_paper(self, paper):
        print(f"Accepting paper '{getattr(paper, 'title', 'Unknown')}'.")
        self.update_paper_model(paper, label=1)

    def save_model(self):
        """
        Saves the OnlineRankSVMRanker's model state to Supabase.
        """
        try:
            model_state = {
                'author_model': self.author_model,
                'paper_model': self.paper_model,
                'author_scaler': self.author_scaler,
                'paper_scaler': self.paper_scaler,
                'is_author_model_initialized': self.is_author_model_initialized,
                'is_paper_model_initialized': self.is_paper_model_initialized
            }
            serialized_model = pickle.dumps(model_state)
            compressed_model = gzip.compress(serialized_model)
            self.supabase.table('ranker_models').upsert({'model_name': self.model_name, 'model_data': compressed_model}).execute()
        except Exception as e:
            logging.exception(f"Error saving model '{self.model_name}': {e}")

    def load_model(self):
        """
        Loads the OnlineRankSVMRanker's model state from Supabase.
        """
        response = self.supabase.table('ranker_models').select('model_data').eq('model_name', self.model_name).execute()
        if response.data and response.data[0]:
            serialized_model = response.data[0]['model_data']
            model_state = pickle.loads(gzip.decompress(serialized_model))
            self.author_model = model_state['author_model']
            self.paper_model = model_state['paper_model']
            self.author_scaler = model_state['author_scaler']
            self.paper_scaler = model_state['paper_scaler']
            self.is_author_model_initialized = model_state['is_author_model_initialized']
            self.is_paper_model_initialized = model_state['is_paper_model_initialized']
            return True
        else:
            print(f"Model '{self.model_name}' not found in Supabase.")
            return False
