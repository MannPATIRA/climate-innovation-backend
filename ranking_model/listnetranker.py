import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from author import Author
from paper import Paper
from ranker import Ranker
from typing import List

class ListNetRanker(Ranker, nn.Module):
    def __init__(self, learning_rate: float = 0.01, input_dim: int = 5, hidden_dim: int = 32):
        """
        Initializes the ListNetRanker.
        We create two networks:
          - paper_net: to score paper-level features.
          - author_net: to score extended author features.
        """
        nn.Module.__init__(self)
        self.learning_rate = learning_rate

        # Paper ranking network.
        self.paper_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Author ranking network.
        self.author_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.paper_optimizer = optim.Adam(self.paper_net.parameters(), lr=self.learning_rate)
        self.author_optimizer = optim.Adam(self.author_net.parameters(), lr=self.learning_rate)

    def extract_paper_features(self, paper: Paper) -> torch.Tensor:
        """
        Extract features from a Paper.
        Features used (order):
          - paper.relevancy (precomputed score)
          - publication age (in days)
          - paper.citations
          - average author citations (if available, else 0)
          - average author h-index (if available, else 0)
        """
        now = datetime.datetime.now()
        publication_age = (now - paper.publication_date).days if hasattr(paper.publication_date, 'days') else 0.0

        if paper.authors and len(paper.authors) > 0:
            avg_author_citations = sum(author.citations for author in paper.authors) / len(paper.authors)
            avg_author_hindex = sum(author.hindex for author in paper.authors) / len(paper.authors)
        else:
            avg_author_citations = 0.0
            avg_author_hindex = 0.0

        features = torch.tensor(
            [paper.relevancy, float(publication_age), paper.citations, avg_author_citations, avg_author_hindex],
            dtype=torch.float32
        )
        return features

    def rank(self, papers: List[Paper]) -> List[Paper]:
        """
        Ranks the list of papers using a combination of paper and author model scores.
        For each paper, we compute:
            overall_score = paper_net(paper_features) + average(author_net(author_features))
        """
        self.paper_net.eval()
        self.author_net.eval()
        ranked = []
        for paper in papers:
            # Compute paper features and score.
            paper_feat = self.extract_paper_features(paper).unsqueeze(0)  # shape [1, input_dim]
            with torch.no_grad():
                paper_score = self.paper_net(paper_feat).item()
            # Compute average author score.
            author_scores = []
            for author in paper.authors:
                # Use the extended feature vector from the Ranker base.
                ext_feat = torch.tensor(self.get_extended_feature_vector(author), dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    author_score = self.author_net(ext_feat).item()
                author_scores.append(author_score)
            if author_scores:
                avg_author_score = sum(author_scores) / len(author_scores)
            else:
                avg_author_score = 0.0

            overall_score = paper_score + avg_author_score
            paper.score = overall_score
            ranked.append(paper)
        # Return papers sorted by overall score (highest first).
        ranked.sort(key=lambda p: p.score, reverse=True)
        return ranked

    def update_author_model(self, author: Author, label: int):
        """
        Update the author network so that the predicted score moves toward label (1.0 for positive, 0.0 for negative).
        We use a simple MSE loss on the extended feature vector.
        """
        self.author_net.train()
        self.author_optimizer.zero_grad()
        ext_feat = torch.tensor(self.get_extended_feature_vector(author), dtype=torch.float32).unsqueeze(0)
        pred = self.author_net(ext_feat).squeeze(0)
        target = torch.tensor([float(label)], dtype=torch.float32)
        loss = F.mse_loss(pred, target)
        loss.backward()
        self.author_optimizer.step()
        return loss.item()

    def update_paper_model(self, paper: Paper, label: int):
        """
        Update the paper network so that the predicted score moves toward label (1.0 for positive, 0.0 for negative).
        We use a simple MSE loss on the paper feature vector.
        """
        self.paper_net.train()
        self.paper_optimizer.zero_grad()
        feat = self.extract_paper_features(paper).unsqueeze(0)
        pred = self.paper_net(feat).squeeze(0)
        target = torch.tensor([float(label)], dtype=torch.float32)
        loss = F.mse_loss(pred, target)
        loss.backward()
        self.paper_optimizer.step()
        return loss.item()

    def delete_author(self, paper: Paper, author: Author):
        """
        Process deletion of an author:
          - Update the author model with negative feedback (label = 0)
          - Remove the author from the paper's author list.
        """
        loss = self.update_author_model(author, label=0)
        if author in paper.authors:
            paper.authors.remove(author)
        return loss

    def accept_author(self, paper: Paper, author: Author):
        """
        Process acceptance of an author:
          - Update the author model with positive feedback (label = 1)
        """
        return self.update_author_model(author, label=1)

    def delete_paper(self, paper: Paper):
        """
        Process deletion of a paper:
          - Update the paper model with negative feedback (label = 0)
        """
        return self.update_paper_model(paper, label=0)

    def accept_paper(self, paper: Paper):
        """
        Process acceptance of a paper:
          - Update the paper model with positive feedback (label = 1)
        """
        return self.update_paper_model(paper, label=1)


if __name__ == "__main__":
    # Create dummy authors.
    now = datetime.datetime.now()
    author1 = Author("Alice", citations=100, dob=now - datetime.timedelta(days=15000),
                     organisation_history=[], orcid="0000-0001", hindex=10, grants=[], grant_org_name="OrgA",
                     website="http://alice.example.com", openAlexid="OA1", works_count=20)
    author2 = Author("Bob", citations=50, dob=now - datetime.timedelta(days=12000),
                     organisation_history=[], orcid="0000-0002", hindex=7, grants=[], grant_org_name="OrgB",
                     website="http://bob.example.com", openAlexid="OA2", works_count=15)

    # Create dummy papers.
    paper1 = Paper(paper_id="P1", openalex_id="OA_P1", title="Climate Impact Study",
                   relevancy=0.8, authors=[author1, author2], doi="doi1", abstract="...", 
                   publication_date=now - datetime.timedelta(days=200), citations=25)
    paper2 = Paper(paper_id="P2", openalex_id="OA_P2", title="Renewable Energy Advances",
                   relevancy=0.9, authors=[author2], doi="doi2", abstract="...", 
                   publication_date=now - datetime.timedelta(days=100), citations=40)
    paper3 = Paper(paper_id="P3", openalex_id="OA_P3", title="Urban Climate Solutions",
                   relevancy=0.7, authors=[author1], doi="doi3", abstract="...", 
                   publication_date=now - datetime.timedelta(days=300), citations=10)

    papers = [paper1, paper2, paper3]

    # Initialize the ListNetRanker.
    ranker = ListNetRanker(learning_rate=0.01, input_dim=5, hidden_dim=32)

    # Rank papers before any feedback.
    ranked_papers = ranker.rank(papers)
    print("Ranked Papers (Before Feedback):")
    for p in ranked_papers:
        print(p)

    # Simulate some feedback:
    # Let's say paper1 and paper2 are accepted (positive feedback) and paper3 is rejected.
    print("\nUpdating models with feedback...")
    loss_p1 = ranker.accept_paper(paper1)
    loss_p2 = ranker.accept_paper(paper2)
    loss_p3 = ranker.delete_paper(paper3)

    # Also update authors: assume author1 gets positive feedback and author2 gets positive feedback.
    loss_a1 = ranker.accept_author(paper1, author1)
    loss_a2 = ranker.accept_author(paper1, author2)

    print(f"Paper update losses: P1 {loss_p1:.4f}, P2 {loss_p2:.4f}, P3 {loss_p3:.4f}")
    print(f"Author update losses: A1 {loss_a1:.4f}, A2 {loss_a2:.4f}")

    # Re-rank papers after feedback.
    ranked_papers_updated = ranker.rank(papers)
    print("\nRanked Papers (After Feedback):")
    for p in ranked_papers_updated:
        print(p)
