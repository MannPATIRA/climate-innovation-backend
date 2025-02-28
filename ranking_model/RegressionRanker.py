from ranking_model.author import Author
from ranking_model.paper import Paper
from ranking_model.ranker import Ranker


import numpy as np


import json
import logging
from typing import List


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

    def rank_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        Rank each paper by computing:
          - Each author's score using the extended feature vector.
          - Authors are sorted within the paper (highest score first).
          - The paper's overall score is computed as the sum of its authors' scores plus
            a contribution from the paper relevancy.
        Returns a list of papers ranked by their overall score (highest first).
        """
        for paper in papers:
            paper.authors = self.rank_authors(paper.authors)
            # Compute paper relevancy contribution.
            paper_contrib = self.sigmoid(self.paper_weights['relevancy'] * paper.relevancy) # TODO: publication date too?
            # Overall paper score: relevancy contribution plus the sum of its authors' scores.
            paper.score = paper_contrib + sum(author.score for author in paper.authors)
        # Return papers sorted by overall score.
        return sorted(papers, key=lambda p: p.score, reverse=True)

    def rank_authors(self, authors: List[Author]) -> List[Author]:
        """
        Ranks a list of authors based on their metrics using a weighted sum and sigmoid function.
        Returns a list of authors ranked by their overall score (highest first).
        """
        for author in authors:
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
        return sorted(authors, key=lambda a: a.score, reverse=True)

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

        logging.info(f"Updated author weights: {self.author_weights}")

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
        logging.info(f"Updated paper weights: {self.paper_weights}")

    def delete_author(self, paper: Paper, author: Author):
        """
        Process a deletion of an author:
          - Update the author model with a rejection (label = 0).
        """
        logging.info(f"Deleting author '{author.name}' from paper '{paper.title}'.")
        self.update_author_model(author, label=0)

    def accept_author(self, paper: Paper, author: Author):
        """
        Process an acceptance of an author (update with label = 1).
        """
        logging.info(f"Accepting author '{author.name}' for paper '{paper.title}'.")
        self.update_author_model(author, label=1)

    def delete_paper(self, paper: Paper):
        """
        Process a deletion of a paper by updating the paper model with a rejection (label = 0).
        """
        logging.info(f"Deleting paper '{paper.title}'.")
        self.update_paper_model(paper, label=0)

    def accept_paper(self, paper: Paper):
        """
        Process an acceptance of a paper (update with label = 1).
        """
        logging.info(f"Accepting paper '{paper.title}'.")
        self.update_paper_model(paper, label=1)

    def save_model(self):
        """
        Saves the RegressionRanker's model state to Supabase as JSON.
        """
        try:
            model_state = {
                'author_weights': self.author_weights,
                'paper_weights': self.paper_weights
            }
            # Serialize the model state to a JSON string
            model_data_json = json.dumps(model_state)
            # the ranker manager uses update/insert, so we will too
            response = self.supabase.table('ranker_models').update({'model_data': model_data_json}).eq('model_name', self.model_name).execute()

            # If no rows were updated, it means the model doesn't exist, so insert it
            if not response.data:
                response = self.supabase.table('ranker_models').insert({'model_name': self.model_name, 'model_data': model_data_json}).execute()
                if response.data and response.data[0]:
                    logging.info(f"Model '{self.model_name}' saved successfully.")
                else:
                    logging.error(f"Error saving model '{self.model_name}': {response.status_code} - {response.text}")
            else:
                logging.info(f"Model '{self.model_name}' updated successfully.")
        except Exception as e:
            logging.exception(f"Error saving model '{self.model_name}': {e}")
            logging.error(f"Error saving model '{self.model_name}': {e}")

    def load_model(self):
        """
        Loads the RegressionRanker's model state from Supabase.
        """
        response = self.supabase.table('ranker_models').select('model_data').eq('model_name', self.model_name).execute()
        if response.data and response.data[0]:
            model_data_json = response.data[0]['model_data'] # Load the JSON string
            model_state = json.loads(model_data_json) # Deserialize the JSON string -> Python dictionary
            self.author_weights = model_state['author_weights']
            self.paper_weights = model_state['paper_weights']
            return True
        else:
            logging.info(f"Model '{self.model_name}' not found in Supabase.")
            return False