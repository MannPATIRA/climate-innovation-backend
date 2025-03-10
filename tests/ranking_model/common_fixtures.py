import pytest
from ranking_model.author import Author
from ranking_model.paper import Paper

@pytest.fixture
def test_authors():
    authors1 = [
        Author(name="Alice", citations=150, dob="1975-06-15", hindex=20, organisation_history=["Uni A"], orcid="0000-0001-1234-5678", grants=[], grant_org_name="Research Foundation", website="https://alice.com", openAlexid="A12345", works_count=45),
        Author(name="Bob", citations=100, dob="1980-03-22", hindex=15, organisation_history=["Uni B"], orcid="0000-0002-2345-6789", grants=[], grant_org_name="Science Foundation", website="https://bob.com", openAlexid="B12345", works_count=30),
        Author(name="Carol", citations=200, dob="1972-11-05", hindex=25, organisation_history=["Uni C"], orcid="0000-0003-3456-7890", grants=[], grant_org_name="Innovation Fund", website="https://carol.com", openAlexid="C12345", works_count=60)
    ]

    authors2 = [
        Author(name="Dave", citations=80, dob="1985-01-10", hindex=12, organisation_history=["Uni D"], orcid="0000-0004-4567-8901", grants=[], grant_org_name="Tech Foundation", website="https://dave.com", openAlexid="D12345", works_count=25),
        Author(name="Eve", citations=300, dob="1970-09-30", hindex=30, organisation_history=["Uni E"], orcid="0000-0005-5678-9012", grants=[], grant_org_name="Global Research Fund", website="https://eve.com", openAlexid="E12345", works_count=80)
    ]

    authors3 = [
        Author(name="Frank", citations=50, dob="1990-12-01", hindex=8, organisation_history=["Uni F"], orcid="0000-0006-6789-0123", grants=[], grant_org_name="Young Researchers Fund", website="https://frank.com", openAlexid="F12345", works_count=15),
        Author(name="Grace", citations=120, dob="1982-07-07", hindex=18, organisation_history=["Uni G"], orcid="0000-0007-7890-1234", grants=[], grant_org_name="Science Trust", website="https://grace.com", openAlexid="G12345", works_count=40),
        Author(name="Heidi", citations=90, dob="1988-05-20", hindex=14, organisation_history=["Uni H"], orcid="0000-0008-8901-2345", grants=[], grant_org_name="Research Council", website="https://heidi.com", openAlexid="H12345", works_count=30)
    ]
    return authors1, authors2, authors3

@pytest.fixture
def test_papers(test_authors):
    authors1, authors2, authors3 = test_authors
    paper1 = Paper(
        paper_id="1",
        openalex_id="W1234567890",
        title="Research on Neural Networks",
        relevancy=0.8,
        authors=authors1,
        doi="10.1234/nnw.2023.1",
        abstract="This paper explores recent advances in neural networks and their applications.",
        publication_date="2023-01-15"
    )
    paper2 = Paper(
        paper_id="2",
        openalex_id="W2345678901",
        title="Advances in Quantum Computing",
        relevancy=0.6,
        authors=authors2,
        doi="10.1234/qc.2023.1",
        abstract="A comprehensive review of recent developments in quantum computing algorithms.",
        publication_date="2023-02-20"
    )
    paper3 = Paper(
        paper_id="3",
        openalex_id="W3456789012",
        title="Innovations in Biotechnology",
        relevancy=0.9,
        authors=authors3,
        doi="10.1234/biotech.2023.1",
        abstract="Novel approaches in biotechnology and their potential impact on medicine.",
        publication_date="2023-03-10"
    )
    return [paper1, paper2, paper3]
