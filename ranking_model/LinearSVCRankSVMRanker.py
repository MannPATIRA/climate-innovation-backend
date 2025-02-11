from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from typing import List
import numpy as np

from .author import Author
from .paper import Paper
from .ranker import Ranker


class LinearSVCRankSVMRanker(Ranker):
    """
    A ranking implementation using the RankSVM algorithm with LinearSVC.

    Key components:
        - LinearSVC: Uses Support Vector Classification with linear kernel
        - Feature Scaling: Maintains StandardScalers for feature normalization
        - Pairwise Learning: Transforms ranking into binary classification of pairs
        - Batch Learning: Requires full retraining when updating models
        
    Note:
        - Not suitable for online learning (requires full retraining)
        - Maintains training data for model updates
        - Computationally intensive for large datasets due to pairwise comparisons
        - Memory usage scales with number of training instances
    """
    def __init__(self, learning_rate: float = 0.01):
        """
        Initializes the RankSVMRanker with models for authors and papers.
        """
        self.learning_rate = learning_rate

        # Initialize models and scalers for authors and papers
        self.author_model = LinearSVC()
        self.author_scaler = StandardScaler()
        self.author_features = []
        self.author_labels = []

        self.paper_model = LinearSVC()
        self.paper_scaler = StandardScaler()
        self.paper_features = []
        self.paper_labels = []

    def rank(self, papers: List[Paper]) -> List[Paper]:
        """
        Ranks a list of papers using the trained RankSVM models.
        Each author's score is computed, authors are sorted within papers,
        and papers are sorted based on overall score.

        Returns a list of papers ranked by their overall score (highest first).
        """
        for paper in papers:
            # Rank authors within the paper
            author_features = []
            for author in paper.authors:
                features = self.get_extended_feature_vector(author)
                author_features.append(features)
            author_features = np.array(author_features)

            # Scale features if model is trained
            if hasattr(self.author_model, 'coef_') and len(author_features) > 0:
                author_features_scaled = self.author_scaler.transform(author_features)
                # Compute scores using the decision function
                scores = self.author_model.decision_function(author_features_scaled)
                for author, score in zip(paper.authors, scores):
                    author.score = score
            else:
                # Default score if model not trained
                for author in paper.authors:
                    author.score = 0.0

            # Sort authors within the paper (highest score first).
            paper.authors.sort(key=lambda a: a.score, reverse=True)

            # Compute paper score
            paper_feature = np.array([paper.relevancy]).reshape(1, -1)
            if hasattr(self.paper_model, 'coef_'):
                paper_feature_scaled = self.paper_scaler.transform(paper_feature)
                paper_score = self.paper_model.decision_function(paper_feature_scaled)[0]
            else:
                paper_score = 0.0

            # Overall paper score: relevancy contribution plus the sum of its authors' scores.
            paper.score = paper_score + sum(author.score for author in paper.authors)

        # Sort papers by overall score
        return sorted(papers, key=lambda p: p.score, reverse=True)

    def update_author_model(self, author: Author, label: int):
        """
        Updates the author ranking model based on feedback.
        Implements pairwise comparisons for RankSVM.
        """
        features = self.get_extended_feature_vector(author)
        self.author_features.append(features)
        self.author_labels.append(label)

        # Train model if sufficient data
        if len(self.author_labels) >= 2:
            self._train_author_model()

    def _train_author_model(self):
        """
        Trains the author RankSVM model using pairwise differences.
        """
        X = np.array(self.author_features)
        y = np.array(self.author_labels)

        # Scale features
        X_scaled = self.author_scaler.fit_transform(X)

        # Generate pairwise differences and labels
        X_pairs, y_pairs = self._generate_pairs(X_scaled, y)

        # Fit the RankSVM model
        if len(X_pairs) > 0:
            self.author_model.fit(X_pairs, y_pairs)

    def update_paper_model(self, paper: Paper, label: int):
        """
        Updates the paper ranking model based on feedback.
        Implements pairwise comparisons for RankSVM.
        """
        features = np.array([paper.relevancy])
        self.paper_features.append(features)
        self.paper_labels.append(label)

        # Train model if sufficient data
        if len(self.paper_labels) >= 2:
            self._train_paper_model()

    def _train_paper_model(self):
        """
        Trains the paper RankSVM model using pairwise differences.
        """
        X = np.array(self.paper_features)
        y = np.array(self.paper_labels)

        # Scale features
        X_scaled = self.paper_scaler.fit_transform(X)

        # Generate pairwise differences and labels
        X_pairs, y_pairs = self._generate_pairs(X_scaled, y)

        # Fit the RankSVM model
        if len(X_pairs) > 0:
            self.paper_model.fit(X_pairs, y_pairs)

    def _generate_pairs(self, X, y):
        """
        Generates pairwise difference vectors and labels for RankSVM.

        Args:
            X: Feature matrix.
            y: Labels vector.

        Returns:
            X_pairs: Pairwise difference of features.
            y_pairs: Labels for pairwise differences.
        """
        X_pairs = []
        y_pairs = []
        n_samples = X.shape[0]
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if y[i] == y[j]:
                    continue  # Skip pairs with the same label
                # Difference between feature vectors
                diff = X[i] - X[j]
                label = np.sign(y[i] - y[j])  # +1 if y[i] > y[j], -1 otherwise
                X_pairs.append(diff * label)
                y_pairs.append(1)
        X_pairs = np.array(X_pairs)
        y_pairs = np.array(y_pairs)
        return X_pairs, y_pairs

    def delete_author(self, paper: Paper, author: Author):
        print(f"Deleting author '{author.name}' from paper '{paper.title}'.")
        self.update_author_model(author, label=0)
        paper.authors = [a for a in paper.authors if a.name != author.name]

    def accept_author(self, paper: Paper, author: Author):
        print(f"Accepting author '{author.name}' for paper '{paper.title}'.")
        self.update_author_model(author, label=1)

    def delete_paper(self, paper: Paper):
        print(f"Deleting paper '{paper.title}'.")
        self.update_paper_model(paper, label=0)

    def accept_paper(self, paper: Paper):
        print(f"Accepting paper '{paper.title}'.")
        self.update_paper_model(paper, label=1)
