from typing import Dict, List, Type, Tuple
from collections import defaultdict
import json
import pickle
import logging

from backend_server.gatherers import OpenAlexInformationGatherer, authors_from_doi, build_author_object
from .RegressionRanker import RegressionRanker
from .OnlineSVMRanker import OnlineSVMRanker
from .ranker import Ranker
from .author import Author
from .paper import Paper
from supabase import Client

class RankerManager(Ranker):
    """
    A meta-ranker that manages multiple ranker instances.
    Allows for training of multiple implementations, and using the one with the best performance.
    """
    def __init__(self, supabase_client: Client, model_name: str, ranker_classes: Dict[str, Tuple[Type[Ranker], float]], learning_rate: float = 0.01):
        """
        Initialize multiple rankers.
        
        Args:
            ranker_classes: Dictionary mapping ranker names to their classes
            learning_rate: Learning rate for all rankers
        """
        super().__init__(supabase_client, model_name, learning_rate)
        self.rankers = {
            name: ranker_class(supabase_client=supabase_client, model_name=f"{model_name}_{name}", learning_rate=learning_rate)
            for name, (ranker_class, wei) in ranker_classes.items()
        }
        for name, ranker in self.rankers.items():
            logging.info(f"Initialized ranker: {name} of type {ranker.__class__.__name__} with model name {ranker.model_name}")
        self.paper_weights = {name: wei for name, (ranker_class, wei) in ranker_classes.items()}
        self.author_weights = {name: wei for name, (ranker_class, wei) in ranker_classes.items()}
        
        self.authors = None
        self.accepted_authors = []
        self.rejected_authors = []
        self.papers = None
        self.accepted_papers = []
        self.rejected_papers = []
        
        self.all_rankings = None

    def _ensemble_rank(self, items: List, ranker_method: str) -> List:
        """
        Applies a weighted ensemble ranking from all managed rankers.
        Args:
            items: List of items to rank (Papers or Authors)
            ranker_method: The ranker method to call ("rank_papers" or "rank_authors")
        Returns:
            List of items ranked by their overall score (highest first).
        """
        self.all_rankings = {
            name: getattr(ranker, ranker_method)(items.copy())
            for name, ranker in self.rankers.items()
        }
        item_scores = defaultdict(float)
        for item in items:
            for ranker_name, ranked_items in self.all_rankings.items():
                # Find item's position in this ranker's list
                position = next(i for i, ranked_item in enumerate(ranked_items) if ranked_item == item)
                # Convert position to score (higher score for better position)
                score = 1.0 / (position + 1)
                # Add weighted contribution from this ranker
                if ranker_method == "rank_papers":
                    item_scores[item] += score * self.paper_weights[ranker_name]
                else:
                    item_scores[item] += score * self.author_weights[ranker_name]
            item.score = item_scores[item]  # Store the ensemble score
            
        # Return items sorted by ensemble score
        return sorted(items, key=lambda item: item.score, reverse=True)

    def rank_papers(self, papers: List[Paper]) -> List[Paper]:
        """
        Implements the rank method from Ranker interface.
        Returns a weighted ensemble ranking from all managed rankers.
        Additionally:
        - updates ranker weighting based on feedback
        - stores the papers / feedback recieved from the last paper set to supabase
        """
        # update weights and store the old information
        if self.all_rankings is not None:
            logging.info("Updating paper model weights and storing feedback")
            self._update_paper_rankers_weights()
            self._store_ranking_papers_feedback()
            logging.info("Updating author model weights and storing feedback")
            self._update_author_rankers_weights()
            self._store_ranking_authors_feedback()
        self.papers = papers
        return self._ensemble_rank(papers, "rank_papers")
    
    def rank_authors(self, authors: List[Author]) -> List[Author]:
        """
        Implements the rank_authors method from Ranker interface.
        Returns a weighted ensemble ranking from all managed rankers.
        """
        if self.all_rankings is not None:
            logging.info("Updating author model weights and storing feedback")
            self._update_author_rankers_weights()
            self._store_ranking_authors_feedback()
        self.authors = authors
        return self._ensemble_rank(authors, "rank_authors")

    def _pairwise_position_loss(self, ranked_items, positive_items, negative_items):
        """
        Computes fraction of positive-negative item pairs where negative is ranked above positive.
        Loss = 0: Perfect ranking (all positive items above negative ones)
        Loss = 1: Worst ranking (all negative items above positive ones)
        """
        pos_positions = [next(i for i, p in enumerate(ranked_items) if p == item) for item in positive_items]
        neg_positions = [next(i for i, p in enumerate(ranked_items) if p == item) for item in negative_items]
        return sum(1 for pos in pos_positions for neg in neg_positions if pos > neg) / (len(pos_positions) * len(neg_positions))
    
    def _relative_position_loss(self, ranked_items, accepted_items, rejected_items):
        """
        Loss based on normalized positions (0-1 range):
        - For accepted: avg position / (n-1) (lower better - want accepted items at top)
        - For rejected: 1 - avg position / (n-1) (lower better - want rejected items at bottom) 
        - Returns average of both losses when both types exist
        """
        n = len(ranked_items)
        if n <= 1: return 0.0
        
        loss_components = []

        if accepted_items:
            acc_positions = [next(i for i, item in enumerate(ranked_items) if item == acc) 
                            for acc in accepted_items]
            # First calculate average position: sum(positions)/len(positions)
            # Then normalize by (n-1) to get value between 0-1
            # n-1 because in zero-based indexing, last position is n-1
            avg_pos = sum(acc_positions)/len(acc_positions)
            loss_components.append(avg_pos / (n - 1))
        
        if rejected_items:
            rej_positions = [next(i for i, item in enumerate(ranked_items) if item == rej) 
                            for rej in rejected_items]
            avg_pos = sum(rej_positions)/len(rej_positions)  
            loss_components.append(1.0 - avg_pos / (n - 1))
        
        return sum(loss_components) / len(loss_components) if loss_components else 0.0

    def _store_ranking_papers_feedback(self):
        """
        Stores ranking feedback (paper_ids, positive_paper_ids, negative_paper_ids) in Supabase.
        Also reset the lists of accepted and rejected.
        """
        try:
            # Extract paper_ids from the lists of papers
            paper_ids = [p.paper_id for p in self.papers]
            pos_paper_ids = [p.paper_id for p in self.accepted_papers]
            neg_paper_ids = [p.paper_id for p in self.rejected_papers]

            # Store paper IDs in ranking_papers_feedback table
            data_to_store = {
                'paper_ids': paper_ids,
                'positive_paper_ids': pos_paper_ids,
                'negative_paper_ids': neg_paper_ids,
                # Store relevancy scores: Keys must be strings in JSON not int8
                'relevancies': [{str(p.paper_id): p.relevancy for p in self.papers}]
            }
            self.supabase.table('ranking_papers_feedback').insert(data_to_store).execute()
            logging.info(f"Ranking feedback stored successfully in Supabase.")
            
            # Reset lists
            self.accepted_papers = []
            self.rejected_papers = []

        except Exception as e:
            logging.error(f"Error storing ranking feedback in Supabase: {e}")

    def _store_ranking_authors_feedback(self):
        """
        Stores ranking feedback (author_ids, positive_author_ids, negative_author_ids) in Supabase.
        Also reset the lists of accepted and rejected authors.
        """
        try:
            # Extract author_ids from the lists of authors
            author_ids = [a.openAlexid for a in self.authors]
            pos_author_ids = [a.openAlexid for a in self.accepted_authors]
            neg_author_ids = [a.openAlexid for a in self.rejected_authors]

            # Store author IDs in ranking_authors_feedback table
            data_to_store = {
                'author_ids': author_ids,
                'positive_author_ids': pos_author_ids,
                'negative_author_ids': neg_author_ids,
                # Store scores: Keys must be strings in JSON not int8
                'scores': [{str(a.openAlexid): a.score for a in self.authors}]
            }
            self.supabase.table('ranking_authors_feedback').insert(data_to_store).execute()
            logging.info(f"Ranking feedback stored successfully in Supabase.")
            # Reset lists
            self.accepted_authors = []
            self.rejected_authors = []

        except Exception as e:
            logging.error(f"Error storing ranking feedback in Supabase: {e}")

    def _update_rankers_weights(self, weights: Dict[str, float], all_items: List, accepted_items: List, rejected_items: List, learning_rate: float = 0.01):
        """
        Evaluate performance of all rankers and update the given weights dictionary accordingly.
        Updates ranker weights based on feedback using gradient descent.
        Lower weights for rankers that contribute to higher loss.
        """
        if not (accepted_items or rejected_items):
            return weights
        
        # Update weights
        for ranker_name in weights:
            # Compute loss contribution from this ranker
            ranker_loss = self._relative_position_loss(all_items, accepted_items, rejected_items)
            
            # Update weight (decrease if ranker contributed to high loss)
            weights[ranker_name] -= learning_rate * ranker_loss
            
            # Ensure weights stay positive
            weights[ranker_name] = max(0.0, weights[ranker_name])
        
        # Normalize weights to sum to 1
        total_weights = sum(weights.values())
        if total_weights > 0:
            weights = {k: v/total_weights for k, v in weights.items()}
        return weights

    def _update_paper_rankers_weights(self, learning_rate=0.01):
        self.paper_weights = self._update_rankers_weights(self.paper_weights, self.papers, self.accepted_papers, self.rejected_papers, learning_rate)

    def _update_author_rankers_weights(self, learning_rate=0.01):
        if self.accepted_authors:
            first_author = self.accepted_authors[0]
        elif self.rejected_authors:
            first_author = self.rejected_authors[0]
        if first_author not in self.authors:
            raise RuntimeError("Author not in papers")
        self.author_weights = self._update_rankers_weights(self.author_weights, self.authors, self.accepted_authors, self.rejected_authors, learning_rate)

    def update_author_model(self, author: Author, label: int):
        """
        Implements the update_author_model method from Ranker interface.
        Updates all managed rankers.
        """
        for ranker in self.rankers.values():
            ranker.update_author_model(author, label)

    def update_paper_model(self, paper: Paper, label: int):
        """
        Updates all managed rankers.
        """
        for ranker in self.rankers.values():
            ranker.update_paper_model(paper, label)

    def delete_author(self, paper: Paper, author: Author):
        """
        Propagates deletion to all managed rankers and stores the rejected author.
        Removes the deleted author from the paper's author list.
        """
        self.authors = paper.init_authors
        for ranker in self.rankers.values():
            ranker.delete_author(paper, author)
        self.rejected_authors.append(author)
        paper.authors = [a for a in paper.authors if a.name != author.name]

    def accept_author(self, paper: Paper, author: Author):
        """
        Propagates acceptance to all managed rankers and stores the accepted author.
        """
        self.authors = paper.init_authors
        for ranker in self.rankers.values():
            ranker.accept_author(paper, author)
        self.accepted_authors.append(author)

    def delete_paper(self, paper: Paper):
        """
        Propagates deletion to all managed rankers and stores the rejected paper.
        """
        for ranker in self.rankers.values():
            ranker.delete_paper(paper)
        self.rejected_papers.append(paper)

    def accept_paper(self, paper: Paper):
        """
        Propagates acceptance to all managed rankers and stores the accepted paper.
        """
        for ranker in self.rankers.values():
            ranker.accept_paper(paper)
        self.accepted_papers.append(paper)

    def save_model(self):
        """Saves the RankerManager's model state to Supabase."""
        try:
            model_state = {
                'paper_weights': self.paper_weights,
                'author_weights': self.author_weights
            }
            model_data_json = json.dumps(model_state)
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
            logging.error(f"Error saving model '{self.model_name}': {e}")

    def _train_new_papers_ranker(self, ranker: Ranker, hist_entry: Dict):
        """
        Trains the given ranker with historical data from a single ranking_papers_feedback entry.
        """
        # Fetch paper IDs from the ranking_papers_feedback table
        paper_ids = hist_entry['paper_ids']
        pos_paper_ids = hist_entry['positive_paper_ids']
        neg_paper_ids = hist_entry['negative_paper_ids']
        relevancies = hist_entry['relevancies'][0]  # Keys are strings of INT8 values

        # Fetch paper data from the paper table using the IDs
        papers = []
        for paper_id in paper_ids:
            paper_response = self.supabase.table('papers').select('*').eq('id', paper_id).execute()
            if paper_response.data:
                paper_data = paper_response.data[0]
                
                # Get authors using DOI
                authors = authors_from_doi(paper_data['doi'])
                
                # Get paper details from OpenAlex
                details = OpenAlexInformationGatherer.get_details_from_paper_id(paper_data['openalex_id'])
                
                # Reconstruct paper object with minimal stored data + retrieved data
                paper = Paper(
                    paper_id=int(paper_data['id']),
                    openalex_id=paper_data['openalex_id'],
                    title=details['title'],
                    relevancy=relevancies[str(paper_data['id'])],  # Use string key to access relevancy
                    authors=authors,
                    doi=paper_data['doi'],
                    abstract=details['abstract'],
                    publication_date=details['publication_date']
                )
                papers.append(paper)
            else:
                logging.warning(f"Paper with ID {paper_id} not found.")
                continue

        # Reconstruct positive and negative papers
        pos_papers = [p for p in papers if p.id in pos_paper_ids]
        neg_papers = [p for p in papers if p.id in neg_paper_ids]

        # Train the ranker with this historical data
        ranker.rank_papers(papers)  # This initializes internal state
        for paper in pos_papers:
            ranker.accept_paper(paper)
            for author in paper.authors:
                ranker.accept_author(paper, author)

        for paper in neg_papers:
            ranker.delete_paper(paper)
            for author in paper.authors:
                ranker.delete_author(paper, author)
        logging.info(f"Ranker {ranker.model_name} trained with historical data.")
        ranker.save_model()

    def _train_new_authors_ranker(self, ranker: Ranker, hist_entry: Dict):
        """
        Trains the given ranker with historical author data from a single ranking_authors_feedback entry.
        """
        # Fetch author IDs from the ranking_authors_feedback table
        author_ids = hist_entry['author_ids']
        pos_author_ids = hist_entry['positive_author_ids']
        neg_author_ids = hist_entry['negative_author_ids']
        scores = hist_entry['scores'][0]  # Keys are strings of openAlexIDs

        # Fetch author data and construct Author objects
        authors = []
        for author_id in author_ids:
            author = build_author_object(author_id)
            if author:
                author.score = scores.get(str(author_id)) # setting the score from the historical data
                authors.append(author)
            else:
                logging.warning(f"Author with ID {author_id} not found.")
                continue

        # Reconstruct positive and negative authors
        pos_authors = [a for a in authors if a.openAlexid in pos_author_ids]
        neg_authors = [a for a in authors if a.openAlexid in neg_author_ids]

        # Train the ranker with this historical data
        ranker.rank_authors(authors)  # This initializes internal state
        for author in pos_authors:
            ranker.accept_author(None, author) # no paper associated with the author

        for author in neg_authors:
            ranker.delete_author(None, author) # no paper associated with the author
        logging.info(f"Ranker {ranker.model_name} trained with historical author data.")
        ranker.save_model()

    def load_model(self):
        """Loads the RankerManager's model state from Supabase."""
        response = self.supabase.table('ranker_models').select('model_data').eq('model_name', self.model_name).execute()
        if response.data and response.data[0]:
            model_data_json = response.data[0]['model_data']
            try:
                model_state = json.loads(model_data_json)
                self.paper_weights = model_state['paper_weights']
                self.author_weights = model_state['author_weights']
                logging.info(f"Model weights loaded successfully for '{self.model_name}'.")
                # Load individual ranker models
                all_rankers_loaded = True
                for name, ranker in self.rankers.items():
                    if not ranker.load_model():
                        all_rankers_loaded = False
                        logging.info(f"Failed to load ranker: {name}. Training with historical data...")

                        # Get historical paper ranking data
                        papers_hist_response = self.supabase.table('ranking_papers_feedback').select('*').execute()
                        if papers_hist_response.data:
                            logging.info("Historical paper ranking data found")
                            for entry in papers_hist_response.data:
                                self._train_new_papers_ranker(ranker, entry)
                            logging.info(f"Completed training {name} with historical paper ranking data")
                        else:
                            logging.info("No historical paper ranking data found.")

                        # Get historical author ranking data
                        authors_hist_response = self.supabase.table('ranking_authors_feedback').select('*').execute()
                        if authors_hist_response.data:
                            logging.info("Historical author ranking data found")
                            for entry in authors_hist_response.data:
                                self._train_new_authors_ranker(ranker, entry)
                            logging.info(f"Completed training {name} with historical author ranking data")
                        else:
                            logging.info("No historical author ranking data found.")

                if all_rankers_loaded:
                    logging.info(f"All ranker models loaded successfully for '{self.model_name}'.")
                else:
                    logging.warning(f"Some ranker models failed to load for '{self.model_name}' and were retrained.")
                return all_rankers_loaded
            except (pickle.UnpicklingError, KeyError, json.JSONDecodeError) as e:
                logging.error(f"Error loading model '{self.model_name}': {e}")
                return False
        else:
            logging.info(f"Model '{self.model_name}' not found in Supabase.")
            return False

if __name__ == "__main__":   
    from common.supabase_client import init_supabase
    from dotenv import load_dotenv
    from .testing_ranker import papers
    ranker_classes = {
        'regression': RegressionRanker,
        'svm': OnlineSVMRanker,
    }

    supabase: Client = init_supabase()
    # Load environment variables
    load_dotenv(override=True)
    ranker = RankerManager(supabase, "TEST_RANKER_MODEL1", ranker_classes)
    ranker.load_model()
    print("FIRST RANKING")
    ranked_papers = ranker.rank_papers(papers)
    ranker.delete_paper(papers[0])
    ranker.accept_paper(papers[1])
    ranker.accept_author(papers[0], papers[0].authors[0])
    ranker.delete_author(papers[0], papers[0].authors[-1])
    print("SECOND RANKING")
    ranked_papers = ranker.rank_papers(papers)
    ranker.save_model()
    ranker.load_model()
    print("\n\n")
    print(papers)
    print("\n\nRANKED:")
    print(ranked_papers)