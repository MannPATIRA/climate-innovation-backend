from typing import Dict, List, Type
from collections import defaultdict

from .ranker import Ranker, RegressionRanker, OnlineRankSVMRanker
from .author import Author
from .paper import Paper

class RankerManager(Ranker):
    """
    A meta-ranker that manages multiple ranker instances.
    Allows for training of multiple implementations, and using the one with the best performance.
    """
    def __init__(self, ranker_classes: Dict[str, Type[Ranker]], learning_rate: float = 0.01):
        """
        Initialize multiple rankers.
        
        Args:
            ranker_classes: Dictionary mapping ranker names to their classes
            learning_rate: Learning rate for all rankers
        """
        super().__init__(learning_rate)
        self.rankers = {
            name: ranker_class(learning_rate=learning_rate)
            for name, ranker_class in ranker_classes.items()
        }
        self.performance_metrics = defaultdict(list)
        self.weights = {name: 1.0/len(ranker_classes) for name in ranker_classes} # assign equal weights to them

    def rank(self, papers: List[Paper]) -> List[Paper]:
        """
        Implements the rank method from Ranker interface.
        Returns a weighted ensemble ranking from all managed rankers.
        """
        all_rankings = {
            name: ranker.rank(papers.copy())
            for name, ranker in self.rankers.items()
        }
        paper_scores = defaultdict(float)
        for paper in papers:
            for ranker_name, ranked_papers in all_rankings.items():
                # Find paper's position in this ranker's list
                position = next(i for i, p in enumerate(ranked_papers) if p == paper)
                # Convert position to score (higher score for better position)
                score = 1.0 / (position + 1)
                # Add weighted contribution from this ranker
                paper_scores[paper] += score * self.weights[ranker_name]
            paper.score = paper_scores[paper]  # Store the ensemble score
            
        # Return papers sorted by ensemble score
        return sorted(papers, key=lambda p: p.score, reverse=True)
    
    def relative_position_loss(ranked_papers, positive_papers, negative_papers):
        """
        Computes fraction of positive-negative paper pairs where negative is ranked above positive.
        Loss = 0: Perfect ranking (all positive papers above negative ones)
        Loss = 1: Worst ranking (all negative papers above positive ones)
        """
        pos_positions = [next(i for i, p in enumerate(ranked_papers) if p == paper) for paper in positive_papers]
        neg_positions = [next(i for i, p in enumerate(ranked_papers) if p == paper) for paper in negative_papers]
        return sum(1 for pos in pos_positions for neg in neg_positions if pos > neg) / (len(pos_positions) * len(neg_positions))

    def update_weights(self, ranked_papers, positive_papers, negative_papers, learning_rate=0.01):
        """
        Updates ranker weights based on feedback using gradient descent.
        Lower weights for rankers that contribute to higher loss.
        """
        for ranker_name in self.weights:
            # Compute loss contribution from this ranker
            ranker_papers = self.rankers[ranker_name].rank(ranked_papers.copy())
            ranker_loss = self.relative_position_loss(ranker_papers, positive_papers, negative_papers)
            
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
        Propagates deletion to all managed rankers.
        """
        for ranker in self.rankers.values():
            ranker.delete_author(paper, author)

    def accept_author(self, paper: Paper, author: Author):
        """
        Propagates acceptance to all managed rankers.
        """
        for ranker in self.rankers.values():
            ranker.accept_author(paper, author)

    def delete_paper(self, paper: Paper):
        """
        Propagates deletion to all managed rankers.
        """
        for ranker in self.rankers.values():
            ranker.delete_paper(paper)

    def accept_paper(self, paper: Paper):
        """
        Propagates acceptance to all managed rankers.
        """
        for ranker in self.rankers.values():
            ranker.accept_paper(paper)

    def evaluate_and_update_weights(self, test_papers: List[Paper], ground_truth: List[Paper]):
        """
        Evaluate performance of all rankers and update their weights accordingly.
        """
        scores = {}
        for name, ranker in self.rankers.items():
            ranked_papers = ranker.rank(test_papers.copy())
            score = self._calculate_ranking_score(ranked_papers, ground_truth)
            scores[name] = score
            self.performance_metrics[name].append(score)

        # Update weights based on recent performance
        total_score = sum(scores.values())
        if total_score > 0:  # Avoid division by zero
            self.weights = {
                name: score/total_score 
                for name, score in scores.items()
            }

    def _calculate_ranking_score(self, ranked_papers: List[Paper], ground_truth: List[Paper]) -> float:
        """
        Calculate ranking score (e.g., NDCG, MAP, etc.).
        """
        # Implementation depends on specific evaluation needs
        return 0.0  # Replace with actual metric calculation

    def get_ranker_weights(self) -> Dict[str, float]:
        """
        Returns the current weights of each ranker.
        """
        return self.weights.copy()

    def get_best_ranker(self) -> str:
        """
        Returns the name of the best performing ranker based on historical performance.
        """
        avg_scores = {
            name: sum(scores) / len(scores)
            for name, scores in self.performance_metrics.items()
            if scores
        }
        return max(avg_scores.items(), key=lambda x: x[1])[0] if avg_scores else None

if __name__ == "__main__":    
    ranker_classes = {
        'regression': RegressionRanker,
        'svm': OnlineRankSVMRanker,
    }

    ranker = RankerManager(ranker_classes)

    # papers = [...]  # List of papers
    # ranked_papers = ranker.rank(papers)

    # # Process feedback
    # ranker.accept_paper(paper)
    # ranker.delete_author(paper, author)

    # # Evaluate and update ensemble weights
    # test_papers = [...]
    # ground_truth = [...]
    # ranker.evaluate_and_update_weights(test_papers, ground_truth)

    # # Check current weights
    # weights = ranker.get_ranker_weights()
    # print(f"Current ranker weights: {weights}")