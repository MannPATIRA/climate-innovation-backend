import urllib.parse
from abc import ABC

import requests
from pyalex import Works, Authors

from ranking_model.author import Author
from ranking_model.grant import Grant


class InformationGatherer(ABC):
    pass


class ORCIDInformationGatherer:
    BASE_URL = "https://pub.orcid.org/v3.0/"
    HEADERS = {"Accept": "application/json"}

    @staticmethod
    def get_profile(orcid_id: str):
        url = f"{ORCIDInformationGatherer.BASE_URL}{orcid_id}/person"
        response = requests.get(url, headers=ORCIDInformationGatherer.HEADERS)
        return response.json() if response.status_code == 200 else None

    @staticmethod
    def get_works(orcid_id: str):
        url = f"{ORCIDInformationGatherer.BASE_URL}{orcid_id}/works"
        response = requests.get(url, headers=ORCIDInformationGatherer.HEADERS)
        return response.json() if response.status_code == 200 else None

    @staticmethod
    def get_employments(orcid_id: str):
        url = f"{ORCIDInformationGatherer.BASE_URL}{orcid_id}/employments"
        response = requests.get(url, headers=ORCIDInformationGatherer.HEADERS)
        return response.json() if response.status_code == 200 else None

    @staticmethod
    def get_dob(orcid_id: str):
        url = f"{ORCIDInformationGatherer.BASE_URL}{orcid_id}/biography"
        response = requests.get(url, headers=ORCIDInformationGatherer.HEADERS)
        if response.status_code == 200:
            data = response.json()
            return data.get("date-of-birth")
        return None


class OpenAlexInformationGatherer(InformationGatherer):
    @staticmethod
    def get_authors_from_doi(doi):
        work = Works().filter(doi=doi).get()
        if work:
            authors = work[0].get('authorships', [])
            return [{'authorId': author['author']['id'], 'name': author['author']['display_name']} for author in
                    authors]
        return []

    @staticmethod
    def get_author_info(author_id):
        author = Authors()[author_id]
        if author:
            return {
                "name": author.get('display_name'),
                "citations": author.get('cited_by_count', 0),
                "hIndex": author.get('h_index', 0),
                "externalIds": author.get('ids', {})
            }
        return {}

    #     @staticmethod
    #     def get_doi_from_paper_id(paper_id):
    #         """
    #         Retrieve the DOI of a paper using its OpenAlex ID.
    #         """
    #         # Fetch the work (paper) details using the OpenAlex ID
    #         work = Works().get(paper_id)
    #
    #         if work:
    #             # Extract the DOI
    #             doi = work.get('doi')
    #             return doi
    #         else:
    #             return None
    #
    #     @staticmethod
    #     def get_relevant_concepts_from_paper(paper):
    #         concepts = paper["concepts"]
    #
    #         relevant_concepts = list(
    #             filter(lambda x: x[2] > 0.1, map(lambda y: (y["display_name"], y["level"], y["score"]), concepts)))
    #
    #         return relevant_concepts


class GTRInformationGatherer(InformationGatherer):
    BASE_GTR_URL = "https://gtr.ukri.org/api"
    HEADERS = {"Accept": "application/json"}

    @staticmethod
    def get_gtr_orgs_grants(orcid_id, name):
        """Fetches grant information for an author using GTR API."""

        # Parse the name into some URL format as that's what the API requires
        name_url_parsed = urllib.parse.quote(name)

        # API endpoint to query someone by name
        url = f"{GTRInformationGatherer.BASE_GTR_URL}/search/person?term={name_url_parsed}"
        response = requests.get(url, headers=GTRInformationGatherer.HEADERS)

        if response.status_code == 200:
            data = response.json()

            # Iterate through the results
            for person_details in data.get('facetedSearchResultBean', {}).get("results", []):

                # Get the person JSON
                person = person_details.get("person", {})

                # Try to match the person by name or ORCID
                if ((person.get("firstName") in name and person.get("surname") in name) or
                        (orcid_id is not None
                         and person.get("orcidId") is not None
                         and orcid_id in person.get("orcidId")
                        )
                ):

                    # If we found the person, store this ID for later project comparisons
                    person_id = person["id"]

                    # Get details for the organisation that the researcher is currently at
                    org_details = person_details.get("organisation", {}).get("name", "Unknown")

                    # Query for more details about this author
                    grants_url = f"{GTRInformationGatherer.BASE_GTR_URL}/person/{person_id}"
                    grants_response = requests.get(grants_url, headers=GTRInformationGatherer.HEADERS)

                    if grants_response.status_code == 200:
                        resp = grants_response.json()

                        person_overview = resp.get("personOverview", {})

                        # Get all the grants that they have been involved with
                        grants = map(lambda grant: grant.get("projectComposition", {}),
                                     person_overview.get("projectSearchResult", {}).get("results", []))

                        # create a Grant object for each grant
                        grant_objects = map(lambda grant:
                                            Grant(
                                                title=grant.get("project", {}).get("title", "Unknown title"),
                                                category=grant.get("project", {}).get("grantCategory",
                                                                                      "Unknown category"),
                                                value=grant.get("project", {}).get("fund", {}).get("valuePounds", None),
                                                funder=grant.get("project", {}).get("fund", {}).get("funder", {}).get(
                                                    "name", "Unknown funder"),
                                                organisation=grant.get("leadResearchOrganisation", {})
                                                .get("name", "Unknown organisation")
                                            ), grants)

                        return list(grant_objects), org_details

                    return [], org_details

        return [], ""


######################################
##              UNUSED              ##
######################################
class SemanticScholarInformationGatherer(InformationGatherer):
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    @staticmethod
    def get_authors_from_doi(doi):

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

        r = requests.get(
            f"/paper/{paper_id}",
            params={"fields": "externalIds,name"}
        )

        print(r.json())

        return r.json()["data"]["externalIds"]


def authors_from_doi(doi):
    """
    Takes a doi and gives a bunch of Author objects
    Parameters
    ----------
    doi - doi of paper

    Returns
    -------

    """

    # Get all the authors linked to the paper using OpenAlex
    authors = OpenAlexInformationGatherer.get_authors_from_doi(doi)
    author_objects = []

    for author in authors:
        # Get information on author from PyAlex including ORCID
        author_info = OpenAlexInformationGatherer.get_author_info(author["authorId"])
        orcid = author_info["externalIds"].get("orcid") if "externalIds" in author_info else None
        orcid = orcid.split("org/")[1] if orcid else None

        # Use ORCID's API to get more information about the author
        orcid_data = ORCIDInformationGatherer.get_profile(orcid) if orcid else {}
        employment_data = ORCIDInformationGatherer.get_employments(orcid) if orcid else {}
        dob = ORCIDInformationGatherer.get_dob(orcid) if orcid else None
        website_check = orcid_data.get("researcher-urls", {}).get("researcher-url", [])
        website = website_check[0].get("url", None).get("value") if website_check else None

        # Use GTR API to get grant information about author using ORCID and their name
        (grants, org) = GTRInformationGatherer.get_gtr_orgs_grants(orcid, author.get("name"))

        # Construct Author object
        author_obj = Author(
            name=author_info.get("name", "Unknown"),
            citations=author_info.get("citations", 0),
            hindex=author_info.get("hIndex", 0),
            orcid=orcid,
            dob=dob,
            grants=grants,
            org_name=org,
            website=website
        )

        author_obj.profile = orcid_data
        author_obj.employment = employment_data
        author_objects.append(author_obj)

    return author_objects


if __name__ == "__main__":
    authors_from_doi("https://doi.org/10.48550/arXiv.2402.01928")
