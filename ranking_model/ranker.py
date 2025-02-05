from author import Author
from paper import Paper
import numpy as np

class Ranker:
    def __init__(self, learning_rate=0.01):
        """
        The Ranker maintains separate model weights for authors and papers.
          - For authors, we use features: [citations, hindex].
          - For papers, we use the paper's relevancy score.
        """
        self.learning_rate = learning_rate
        self.author_weights = {'citations': 0.5, 'hindex': 0.5}
        self.paper_weights = {'relevancy': 1.0}

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # --------------------------
    # Ranking Functions
    # --------------------------

    def rank(self, papers):
        """
        Given a list of Paper objects, compute a score for each author and each paper.
          - For each paper:
              * Compute each author's score using a logistic model with the current author weights.
              * Rank the authors in descending order (highest score first).
              * Compute a paper relevancy contribution as sigmoid(paper_weight * paper.relevancy).
              * Set the paper's overall score = (paper relevancy contribution) + (sum of its authors' scores).
          - Finally, return the list of papers ranked in descending order.
        """
        for paper in papers:
            # Compute and assign score for each author in the paper.
            for author in paper.authors:
                features = author.get_feature_vector()
                weights = np.array([self.author_weights['citations'], self.author_weights['hindex']])
                raw_score = np.dot(features, weights)
                author.score = self.sigmoid(raw_score)
            # Rank authors within the paper (highest score first).
            paper.authors.sort(key=lambda a: a.score, reverse=True)

            # Compute the paper's relevancy contribution.
            paper_weight = self.paper_weights['relevancy']
            relevancy_contribution = self.sigmoid(paper_weight * paper.relevancy)
            # The overall paper score is defined here as:
            paper.score = relevancy_contribution + sum(author.score for author in paper.authors)
        # Return papers ranked by their score (highest first)
        ranked_papers = sorted(papers, key=lambda p: p.score, reverse=True)
        return ranked_papers

    # --------------------------
    # Model Update Functions
    # --------------------------

    def update_author_model(self, author, label):
        """
        Update the author model weights using an online logistic regression rule.
        :param author: The Author object.
        :param label: 1 for accept, 0 for rejection.
        """
        features = author.get_feature_vector()
        weights = np.array([self.author_weights['citations'], self.author_weights['hindex']])
        raw_score = np.dot(features, weights)
        prediction = self.sigmoid(raw_score)
        error = label - prediction
        updated_weights = weights + self.learning_rate * error * features
        self.author_weights['citations'] = updated_weights[0]
        self.author_weights['hindex'] = updated_weights[1]
        print(f"Updated author weights: {self.author_weights}")

    def update_paper_model(self, paper, label):
        """
        Update the paper model weight (for relevancy) using an online logistic regression rule.
        :param paper: The Paper object.
        :param label: 1 for accept, 0 for rejection.
        """
        feature = paper.relevancy  # single feature
        weight = self.paper_weights['relevancy']
        raw_score = weight * feature
        prediction = self.sigmoid(raw_score)
        error = label - prediction
        updated_weight = weight + self.learning_rate * error * feature
        self.paper_weights['relevancy'] = updated_weight
        print(f"Updated paper weights: {self.paper_weights}")

    # --------------------------
    # User Feedback Functions
    # --------------------------

    def delete_author(self, paper, author):
        """
        Called when the user rejects an author.
          - Update the author model with a rejection signal (label = 0).
          - Remove the author from the paper's author list.
        """
        print(f"Deleting author '{author.name}' from paper '{paper.name}'.")
        self.update_author_model(author, label=0)
        # Remove the author from the paper.
        paper.authors = [a for a in paper.authors if a != author]

    def accept_author(self, paper, author):
        """
        Called when the user accepts an author.
          - Update the author model with an acceptance signal (label = 1).
        """
        print(f"Accepting author '{author.name}' for paper '{paper.name}'.")
        self.update_author_model(author, label=1)

    def delete_paper(self, paper):
        """
        Called when the user rejects a paper.
          - Update the paper model with a rejection signal (label = 0).
        """
        print(f"Deleting paper '{paper.name}'.")
        self.update_paper_model(paper, label=0)
        # The caller can then remove the paper from the ranked list.

    def accept_paper(self, paper):
        """
        Called when the user accepts a paper.
          - Update the paper model with an acceptance signal (label = 1).
        """
        print(f"Accepting paper '{paper.name}'.")
        self.update_paper_model(paper, label=1)
