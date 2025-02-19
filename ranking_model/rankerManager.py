from typing import Dict, List, Type
from collections import defaultdict
import pickle

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
        """
        # update weights and store the old information
        if self.all_rankings is not None:
            self._update_rankers_weights()
            self._store_ranking_data()
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

    def _store_ranking_data(self):
        """Stores ranking data (papers, positive, negative) as a single entry in Supabase."""
        try:
            paper_data = [{'title': p.title, 'abstract': p.abstract, 'authors': p.authors} for p in self.papers]
            pos_data = [{'title': p.title, 'abstract': p.abstract, 'authors': p.authors} for p in self.accepted_papers]
            neg_data = [{'title': p.title, 'abstract': p.abstract, 'authors': p.authors} for p in self.rejected_papers]

            data_to_store = {
                'model_name': self.model_name,
                'papers': paper_data,
                'positive_papers': pos_data,
                'negative_papers': neg_data,
            }
            self.supabase_client.table('ranking_data').insert(data_to_store).execute()
        except Exception as e:
            print(f"Error storing ranking data in Supabase: {e}")

    def _update_rankers_weights(self, learning_rate=0.01):
        """
        Evaluate performance of all rankers and update their weights accordingly.
        Updates ranker weights based on feedback using gradient descent.
        Lower weights for rankers that contribute to higher loss.
        """
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
        ranker_states = {
            name: ranker.save_model()  # Delegate saving to individual rankers
            for name, ranker in self.rankers.items()
        }
        model_state = {
            'weights': self.weights,
            'ranker_states': ranker_states
        }
        serialized_model = pickle.dumps(model_state)
        self.supabase.table('ranker_models').insert({'model_name': self.model_name, 'model_data': serialized_model}).execute()

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
                        hist_response = self.supabase.table('ranking_data').select('*').execute()
                        if hist_response.data:
                            for entry in hist_response.data:
                                # Reconstruct papers and labels from stored data
                                papers = [Paper(**p) for p in entry['papers']]
                                pos_papers = [Paper(**p) for p in entry['positive_papers']]
                                neg_papers = [Paper(**p) for p in entry['negative_papers']]
                                
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
    ranker_classes = {
        'regression': RegressionRanker,
        'svm': OnlineRankSVMRanker,
    }

    # ranker = RankerManager(supabase_client, model_name, ranker_classes)