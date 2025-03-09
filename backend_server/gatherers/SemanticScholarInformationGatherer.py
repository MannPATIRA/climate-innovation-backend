import requests

from backend_server.gatherers.InformationGatherer import InformationGatherer


class SemanticScholarInformationGatherer(InformationGatherer):
    """
    Information gatherer for Semantic Scholar
    """
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    @staticmethod
    def get_authors_from_doi(doi):
        """
        Gets authors of a work from DOI
        Parameters
        ----------
        doi: str

        Returns
        -------
        Authors
        """
        if "arXiv." in doi:

            arxiv = doi.split("arXiv.")[1]

            r = requests.get(
                f"{SemanticScholarInformationGatherer.BASE_URL}/paper/ARXIV:{arxiv}/authors",
                params={"fields": "authorId,name"}
            )

        else:

            doi = doi.split(".org/")[1]

            r = requests.get(
                f"{SemanticScholarInformationGatherer.BASE_URL}/paper/DOI:{doi}/authors",
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
        """
        Returns author info given author id
        Parameters
        ----------
        author_id: str
        exclude: List[str], all fields to exclude, subset of ["name", "affiliations", "paperCount", "citationCount",
                            "hIndex", "papers.paperId", "externalIds"], default of None

        Returns
        -------
        Dict / JSON - Author information
        """

        ps = ["name", "affiliations", "paperCount", "citationCount", "hIndex", "papers.paperId", "externalIds"]

        if exclude is not None:
            ps = ",".join([x for x in ps if x not in exclude])

        else:
            ps = ",".join(ps)

        r = requests.get(
            f"{SemanticScholarInformationGatherer.BASE_URL}/author/{author_id}",
            params={"fields": ps}
        )

        return r.json()

    @staticmethod
    def get_doi_from_paper_id(paper_id: str):
        """
        DOI from paper id
        Parameters
        ----------
        paper_id: str

        Returns
        -------
        str - DOI
        """

        r = requests.get(
            f"{SemanticScholarInformationGatherer.BASE_URL}/paper/{paper_id}",
            params={"fields": "externalIds,name"}
        )

        print(r.json())

        return r.json()["data"]["externalIds"]

    @staticmethod
    def get_h_index_from_author_name(orcid: str = None, name: str = None):
        """
        Returns h index from an author's name
        Parameters
        ----------
        orcid: str, ORCID
        name: str

        Returns
        -------
        int - h index
        """

        # Search by ORCID otherwise by name
        if orcid:
            url = f"{SemanticScholarInformationGatherer.BASE_URL}/author/{orcid}"
        elif name:
            url = f"{SemanticScholarInformationGatherer.BASE_URL}/author/search"
        else:
            raise ValueError("Either ORCID or author name must be provided.")

        params = {"fields": "hIndex"}
        if name:
            params["query"] = name

        # Get h-index
        r = requests.get(url, params=params)
        response_json = r.json()

        # Return h-index otherwise return -1
        if "hIndex" in response_json:
            return response_json["hIndex"]
        elif "data" in response_json and len(response_json["data"]) > 0:
            return response_json["data"][0].get("hIndex", -1)
        else:
            return -1
