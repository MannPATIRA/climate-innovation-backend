from ranking_model.author import Author
from ranking_model.paper import Paper
from ranking_model.ranker import Ranker


import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


import base64
import gzip
import json
import logging
import pickle
from typing import List
from supabase import Client


class OnlineSVMRanker(Ranker):
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

    def rank_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        Ranks a list of papers using the trained SGDClassifier models.
        Each author's score is computed, authors are sorted within papers,
        and papers are sorted based on overall score.

        Returns a list of papers ranked by their overall score (highest first).
        """
        for paper in papers:
            paper.authors = self.rank_authors(paper.authors)
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

    def rank_authors_from_paper(self, paper: Paper) -> List[Author]:
        return self.rank_authors(paper.authors)

    def rank_authors(self, authors: List[Author]) -> List[Author]:
        """
        Ranks a list of authors using the trained SGDClassifier model.
        Each author's score is computed based on the decision function of the model.

        Returns a list of authors ranked by their overall score (highest first).
        """
        # Collect feature vectors for authors
        author_features = [self.get_extended_feature_vector(author) for author in authors]
        author_features = np.array(author_features)

        # Scale features if model is initialized
        if self.is_author_model_initialized and len(author_features) > 0:
            author_features_scaled = self.author_scaler.transform(author_features)
            # Compute scores using decision function
            scores = self.author_model.decision_function(author_features_scaled)
            for author, score in zip(authors, scores):
                author.score = score
        else:
            # Default score if model not initialized
            for author in authors:
                author.score = 0.0

        # Sort authors by score
        return sorted(authors, key=lambda a: getattr(a, 'score', 0.0), reverse=True)

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
        logging.info(f"Deleting author '{getattr(author, 'name', 'Unknown')}' from paper '{getattr(paper, 'title', 'Unknown')}'.")
        self.update_author_model(author, label=0)

    def accept_author(self, paper, author):
        logging.info(f"Accepting author '{getattr(author, 'name', 'Unknown')}' for paper '{getattr(paper, 'title', 'Unknown')}'.")
        self.update_author_model(author, label=1)

    def delete_paper(self, paper):
        logging.info(f"Deleting paper '{getattr(paper, 'title', 'Unknown')}'.")
        self.update_paper_model(paper, label=0)

    def accept_paper(self, paper):
        logging.info(f"Accepting paper '{getattr(paper, 'title', 'Unknown')}'.")
        self.update_paper_model(paper, label=1)

    def save_model(self):
        """
        Saves the OnlineSVMRanker's model state to Supabase.
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
            # Serialize the model state using pickle
            serialized_model = pickle.dumps(model_state)
            # Compress the serialized data using gzip
            compressed_model = gzip.compress(serialized_model)
            # Encode the compressed data to a Base64 string
            compressed_model_base64 = base64.b64encode(compressed_model).decode('utf-8')
            # Store the Base64 string in JSON
            model_data_json = json.dumps({'model_data': compressed_model_base64})
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
        Loads the OnlineSVMRanker's model state from Supabase.
        """
        response = self.supabase.table('ranker_models').select('model_data').eq('model_name', self.model_name).execute()
        if response.data and response.data[0]:
            # Load the JSON string from the database
            model_data_json = response.data[0]['model_data']
            # Extract the Base64 string from the JSON
            compressed_model_base64 = json.loads(model_data_json)['model_data']
            # Decode the Base64 string
            compressed_model = base64.b64decode(compressed_model_base64)
            # Decompress the data
            serialized_model = gzip.decompress(compressed_model)
            # Deserialize the model state using pickle
            model_state = pickle.loads(serialized_model)

            self.author_model = model_state['author_model']
            self.paper_model = model_state['paper_model']
            self.author_scaler = model_state['author_scaler']
            self.paper_scaler = model_state['paper_scaler']
            self.is_author_model_initialized = model_state['is_author_model_initialized']
            self.is_paper_model_initialized = model_state['is_paper_model_initialized']
            return True
        else:
            logging.info(f"Model '{self.model_name}' not found in Supabase.")
            return False