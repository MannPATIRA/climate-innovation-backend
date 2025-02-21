from typing import Dict, List, Type
from collections import defaultdict
import pickle
import gzip
import logging

from backend_server.gatherers import OpenAlexInformationGatherer, authors_from_doi
from .ranker import Ranker, RegressionRanker, OnlineRankSVMRanker
from .author import Author
from .paper import Paper
from supabase import Client

class RankerManager(Ranker):
    """
    A meta-ranker that manages multiple ranker instances.
    Allows for training of multiple implementations, and using the one with the best performance.
    """
    def __init__(self, supabase_client: Client, model_name: str, ranker_classes: Dict[str, Type[Ranker]], learning_rate: float = 0.01):
        """
        Initialize multiple rankers.
        
        Args:
            ranker_classes: Dictionary mapping ranker names to their classes
            learning_rate: Learning rate for all rankers
        """
        super().__init__(supabase_client, model_name, learning_rate)
        self.rankers = {
            name: ranker_class(supabase_client=supabase_client, model_name=f"{model_name}_{name}", learning_rate=learning_rate)
            for name, ranker_class in ranker_classes.items()
        }
        self.weights = {name: 1.0/len(ranker_classes) for name in ranker_classes} # assign equal weights to them
        
        self.papers = None
        self.accepted_authors = []
        self.rejected_authors = []
        self.accepted_papers = []
        self.rejected_papers = []
        
        self.all_rankings = None

    def rank(self, papers: List[Paper]) -> List[Paper]:
        """
        Implements the rank method from Ranker interface.
        Returns a weighted ensemble ranking from all managed rankers.
        Additionally:
        - updates ranker weighting based on feedback
        - stores the papers / feedback recieved from the last paper set to supabase
        """
        # update weights and store the old information
        if self.all_rankings is not None:
            self._update_rankers_weights()
            self._store_ranking_feedback()
        self.papers = papers
        self.all_rankings = {
            name: ranker.rank(papers.copy())
            for name, ranker in self.rankers.items()
        }
        paper_scores = defaultdict(float)
        for paper in papers:
            for ranker_name, ranked_papers in self.all_rankings.items():
                # Find paper's position in this ranker's list
                position = next(i for i, p in enumerate(ranked_papers) if p == paper)
                # Convert position to score (higher score for better position)
                score = 1.0 / (position + 1)
                # Add weighted contribution from this ranker
                paper_scores[paper] += score * self.weights[ranker_name]
            paper.score = paper_scores[paper]  # Store the ensemble score
            
        # Return papers sorted by ensemble score
        return sorted(papers, key=lambda p: p.score, reverse=True)
    
    def _relative_position_loss(self, ranked_papers, positive_papers, negative_papers):
        """
        Computes fraction of positive-negative paper pairs where negative is ranked above positive.
        Loss = 0: Perfect ranking (all positive papers above negative ones)
        Loss = 1: Worst ranking (all negative papers above positive ones)
        """
        pos_positions = [next(i for i, p in enumerate(ranked_papers) if p == paper) for paper in positive_papers]
        neg_positions = [next(i for i, p in enumerate(ranked_papers) if p == paper) for paper in negative_papers]
        return sum(1 for pos in pos_positions for neg in neg_positions if pos > neg) / (len(pos_positions) * len(neg_positions))
    
    def _store_ranking_feedback(self):
        """
        Stores ranking feedback (paper_ids, positive_paper_ids, negative_paper_ids) in Supabase.
        Also reset the lists of accepted and rejected.
        """
        try:
            # Extract paper_ids from the lists of papers
            paper_ids = [p.paper_id for p in self.papers]
            pos_paper_ids = [p.paper_id for p in self.accepted_papers]
            neg_paper_ids = [p.paper_id for p in self.rejected_papers]

            # Store paper IDs in ranking_feedback table
            data_to_store = {
                'paper_ids': paper_ids,
                'positive_paper_ids': pos_paper_ids,
                'negative_paper_ids': neg_paper_ids,
                # Store relevancy scores: Keys must be strings in JSON not int8
                'relevancies': {str(p.paper_id): p.relevancy for p in self.papers}
            }
            self.supabase.table('ranking_feedback').insert(data_to_store).execute()
            
            # Reset lists
            self.accepted_authors = []
            self.rejected_authors = []
            self.accepted_papers = []
            self.rejected_papers = []

        except Exception as e:
            print(f"Error storing ranking feedback in Supabase: {e}")

    def _update_rankers_weights(self, learning_rate=0.01):
        """
        Evaluate performance of all rankers and update their weights accordingly.
        Updates ranker weights based on feedback using gradient descent.
        Lower weights for rankers that contribute to higher loss.
        """
        if (len(self.accepted_papers) + len(self.rejected_papers)) == 0:
            return
        for ranker_name in self.weights:
            # Compute loss contribution from this ranker
            ranker_papers = self.all_rankings[ranker_name]
            ranker_loss = self._relative_position_loss(ranker_papers, self.accepted_papers, self.rejected_papers)
            
            # Update weight (decrease if ranker contributed to high loss)
            self.weights[ranker_name] -= learning_rate * ranker_loss
            
            # Ensure weights stay positive
            self.weights[ranker_name] = max(0.0, self.weights[ranker_name])
        
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}

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
        """
        for ranker in self.rankers.values():
            ranker.delete_author(paper, author)
        self.rejected_authors.append(author)

    def accept_author(self, paper: Paper, author: Author):
        """
        Propagates acceptance to all managed rankers and stores the accepted author.
        """
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
            ranker_states = {
                name: ranker.save_model()  # Delegate saving to individual rankers
                for name, ranker in self.rankers.items()
            }
            model_state = {
                'weights': self.weights,
                'ranker_states': ranker_states
            }
            serialized_model = pickle.dumps(model_state)
            compressed_model = gzip.compress(serialized_model) #Compress the data
            response = self.supabase.table('ranker_models').upsert({'model_name': self.model_name, 'model_data': compressed_model}).execute()
            if response.error:
                logging.error(f"Error saving model '{self.model_name}': {response.error}")
            else:
                logging.info(f"Model '{self.model_name}' saved successfully.")
        except Exception as e:
            logging.exception(f"Error saving model '{self.model_name}': {e}")

    def _train_new_ranker(self, ranker: Ranker, hist_entry: Dict):
        """
        Trains the given ranker with historical data from a single ranking_feedback entry.
        """
        # Fetch paper IDs from the ranking_feedback table
        paper_ids = hist_entry['paper_ids']
        pos_paper_ids = hist_entry['positive_paper_ids']
        neg_paper_ids = hist_entry['negative_paper_ids']
        relevancies = hist_entry['relevancies']  # Keys are strings of INT8 values

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
                print(f"Paper with ID {paper_id} not found.")
                continue

        # Reconstruct positive and negative papers
        pos_papers = [p for p in papers if p.id in pos_paper_ids]
        neg_papers = [p for p in papers if p.id in neg_paper_ids]

        # Train the ranker with this historical data
        ranker.rank(papers)  # This initializes internal state if needed
        for paper in pos_papers:
            ranker.accept_paper(paper)
            for author in paper.authors:
                ranker.accept_author(paper, author)

        for paper in neg_papers:
            ranker.delete_paper(paper)
            for author in paper.authors:
                ranker.delete_author(paper, author)

    def load_model(self):
        """Loads the RankerManager's model state from Supabase."""
        response = self.supabase.table('ranker_models').select('model_data').eq('model_name', self.model_name).execute()
        if response.data and response.data[0]:
            serialized_model = response.data[0]['model_data']
            try:
                model_state = pickle.loads(serialized_model)
                self.weights = model_state['weights']
                # Load individual ranker models
                all_rankers_loaded = True
                for name, ranker in self.rankers.items():
                    if not ranker.load_model():
                        all_rankers_loaded = False
                        print(f"Failed to load ranker: {name}. Training with historical data...")
                        # Get historical ranking data
                        hist_response = self.supabase.table('ranking_feedback').select('*').execute()
                        if hist_response.data:
                            for entry in hist_response.data:
                                self._train_new_ranker(ranker, entry)

                            print(f"Completed training {name} with historical data")
                        else:
                            print("No historical data found for training")
                return all_rankers_loaded
            except (pickle.UnpicklingError, KeyError) as e:
                print(f"Error loading model '{self.model_name}': {e}")
                return False
        else:
            print(f"Model '{self.model_name}' not found in Supabase.")
            return False

if __name__ == "__main__":   
    from common.supabase_client import init_supabase
    from dotenv import load_dotenv
    from .test_ranker import papers
    ranker_classes = {
        'regression': RegressionRanker,
        'svm': OnlineRankSVMRanker,
    }

    supabase: Client = init_supabase()
    # Load environment variables
    load_dotenv(override=True)
    ranker = RankerManager(supabase, "TEST_RANKER_MODEL", ranker_classes)
    ranker.load_model()
    ranked_papers = ranker.rank(papers)
    ranker.save_model()
    print(papers)
    print(ranked_papers)