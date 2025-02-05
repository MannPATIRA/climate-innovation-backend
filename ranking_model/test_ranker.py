from author import Author
from paper import Paper
from ranker import Ranker

authors1 = [
        Author(name="Alice", citations=150, dob="1975-06-15", hindex=20),
        Author(name="Bob", citations=100, dob="1980-03-22", hindex=15),
        Author(name="Carol", citations=200, dob="1972-11-05", hindex=25)
    ]

authors2 = [
    Author(name="Dave", citations=80, dob="1985-01-10", hindex=12),
    Author(name="Eve", citations=300, dob="1970-09-30", hindex=30)
]

authors3 = [
    Author(name="Frank", citations=50, dob="1990-12-01", hindex=8),
    Author(name="Grace", citations=120, dob="1982-07-07", hindex=18),
    Author(name="Heidi", citations=90, dob="1988-05-20", hindex=14)
]

# Create some sample Paper instances.
paper1 = Paper(
    paper_id=1,
    name="P1",
    title="Research on Neural Networks",
    institution="Uni A",
    institution_size=5000,
    funding=800,
    citations=250,
    relevancy=0.8,
    authors=authors1
)
paper2 = Paper(
    paper_id=2,
    name="P2",
    title="Advances in Quantum Computing",
    institution="Uni B",
    institution_size=3000,
    funding=600,
    citations=150,
    relevancy=0.6,
    authors=authors2
)
paper3 = Paper(
    paper_id=3,
    name="P3",
    title="Innovations in Biotechnology",
    institution="Uni C",
    institution_size=2000,
    funding=500,
    citations=100,
    relevancy=0.9,
    authors=authors3
)

# Assume we have a list of papers from a search.
papers = [paper1, paper2, paper3]

# Instantiate the Ranker.
ranker = Ranker(learning_rate=0.01)

# Rank the papers (this will also rank the authors inside each paper).
ranked_papers = ranker.rank(papers)
print("\nRanked Papers and Authors:")
for p in ranked_papers:
    print(f"Paper: {p.name} (Score: {p.score:.3f})")
    for a in p.authors:
        print(f"   Author: {a.name} (Score: {a.score:.3f})")

# Simulate user feedback:
# Suppose the user rejects author "Bob" from paper1.
ranker.delete_author(paper1, authors1[1])  # Bob is authors1[1]

# Suppose the user accepts author "Eve" from paper2.
ranker.accept_author(paper2, authors2[1])  # Eve is authors2[1]

# Suppose the user rejects paper3.
ranker.delete_paper(paper3)

# Re-rank after the updates.
ranked_papers = ranker.rank(papers)
print("\nAfter Updates - Ranked Papers and Authors:")
for p in ranked_papers:
    print(f"Paper: {p.name} (Score: {p.score:.3f})")
    for a in p.authors:
        print(f"   Author: {a.name} (Score: {a.score:.3f})")