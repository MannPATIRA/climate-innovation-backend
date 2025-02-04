from abc import ABC

import requests
from pyalex import Works, Authors


class InformationGatherer(ABC):
    pass
    # def gather(self, **kwargs) -> Dict[str, Any]:
    #     pass


class SemanticScholarInformationGatherer(InformationGatherer):

    @staticmethod
    def get_authors_from_doi(doi):

        if "arXiv." in doi:

            arxiv = doi.split("arXiv.")[1]

            r = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv}/authors",
                params={"fields": "authorId,name"}
            )

        else:

            doi = doi.split(".org/")[1]

            r = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}/authors",
                params={"fields": "authorId,name"}
            )

        # First check if the request was successful
        r.raise_for_status()
        
        # Then try to access the data
        response_json = r.json()
        if "data" not in response_json:
            raise ValueError("Invalid API response format: 'data' field missing")
            
        return response_json["data"]

    @staticmethod
    def get_author_info(author_id, exclude=None):

        ps = ["name", "affiliations", "paperCount", "citationCount", "hIndex", "papers.paperId", "externalIds"]

        if exclude is not None:
            ps = ",".join([x for x in ps if x not in exclude])

        else:
            ps = ",".join(ps)

        r = requests.get(
            f"https://api.semanticscholar.org/graph/v1/author/{author_id}",
            params={"fields": ps}
        )

        return r.json()

    @staticmethod
    def get_doi_from_paper_id(id: str):

        r = requests.get(
            f"http://api.semanticscholar.org/graph/v1/paper/{id}",
            params={"fields": "externalIds,name"}
        )

        print(r.json())

        return r.json()["data"]["externalIds"]


class OpenAlexInformationGatherer:

    @staticmethod
    def get_authors_from_doi(doi):
        """
        Retrieve authors of a paper using its DOI.
        """
        # Fetch the work (paper) details using the DOI
        work = Works().filter(doi=doi).get()

        if work:
            # Extract author details
            authors = work[0].get('authorships', [])
            author_list = [{'authorId': author['author']['id'], 'name': author['author']['display_name']} for author in
                           authors]
            return author_list
        else:
            return []

    @staticmethod
    def get_author_info(author_id, exclude=None):
        """
        Retrieve detailed information about an author using their OpenAlex ID.
        """
        # Fetch the author details using the author ID
        author = Authors()[author_id]

        if author:
            # Define the fields to include
            fields = {
                "name": author.get('display_name'),
                "affiliations": [affiliation['display_name'] for affiliation in
                                 author.get('last_known_institution', [])],
                "paperCount": author.get('works_count'),
                "citationCount": author.get('cited_by_count'),
                "hIndex": author.get('h_index'),
                "papers": Works().filter(author={"id": f"https://openalex.org/{author_id}"}).get(),
                "externalIds": author.get('ids')
            }

            # Exclude specified fields
            if exclude:
                for field in exclude:
                    fields.pop(field, None)

            return fields
        else:
            return {}

    @staticmethod
    def get_doi_from_paper_id(paper_id):
        """
        Retrieve the DOI of a paper using its OpenAlex ID.
        """
        # Fetch the work (paper) details using the OpenAlex ID
        work = Works().get(paper_id)

        if work:
            # Extract the DOI
            doi = work.get('doi')
            return doi
        else:
            return None

    @staticmethod
    def get_relevant_concepts_from_paper(paper):
        concepts = paper["concepts"]

        relevant_concepts = list(
            filter(lambda x: x[2] > 0.1, map(lambda y: (y["display_name"], y["level"], y["score"]), concepts)))

        return relevant_concepts


# https://api.semanticscholar.org/graph/v1/paper/ARXIV:2303.11366/authors
# s = SemanticScholarAuthorInformationGatherer().get_authors_from_doi("https://doi.org/10.48550/arXiv.2303.11366")
# print(s)
# print(SemanticScholarAuthorInformationGatherer().get_author_info("2212367248"))

# def construct_graph(doi: str):
#     authors = SemanticScholarInformationGatherer.get_authors_from_doi(doi)
#
#     for author in authors:
#         info = SemanticScholarInformationGatherer.get_author_info(author["authorId"])
#
#         papers = info["papers"]
#
#         paper_dois = list(map(lambda x: SemanticScholarInformationGatherer.get_doi_from_paper_id(x["paperId"]), papers))
#
#         print(paper_dois)

def construct_graph(doi: str):
    authors = OpenAlexInformationGatherer.get_authors_from_doi(doi)

    for author in authors:
        info = OpenAlexInformationGatherer.get_author_info(author["authorId"])

        papers = info["papers"]

        topics = {}

        for paper in papers:
            topics[paper["doi"]] = OpenAlexInformationGatherer.get_relevant_concepts_from_paper(paper)

        original_concepts = topics[doi]

if __name__ == "__main__":
    construct_graph("https://doi.org/10.48550/arXiv.2303.11366")
