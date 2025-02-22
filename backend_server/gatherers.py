import urllib.parse
from abc import ABC

import requests
from pyalex import Works, Authors

from ranking_model.author import Author
from ranking_model.grant import Grant
import unicodedata
from fuzzywuzzy import fuzz
import re
from scholarly import scholarly


def normalize_name(name):
    """Normalize names by removing accents, converting to lowercase, and stripping whitespace."""
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    return name.lower().strip()


def fuzzy_match(name1, name2, threshold=85):
    """Perform fuzzy matching between two names with a given similarity threshold."""

    # partial_ratio works better when there are initials, but we use token_sort_ratio since this function is only called
    # when org_match is False, so we do not want to match on initials just in case, we match on the name
    return fuzz.token_sort_ratio(name1, name2) >= threshold


def is_name_match(name_variations, person_name, org_match):
    # Normalise the name given by GtR
    person_full_name = normalize_name(person_name)

    # Go through all name variations provided by OpenAlex
    for name in name_variations:

        # Normalise this variation
        norm_name = normalize_name(name)

        # Split into parts of names
        sub_var_names = re.split("[ .-]+", norm_name)
        sub_full_name = re.split("[ .-]+", person_full_name)

        # If the organisations match, we are going to check that any initial matches with the corresponding
        # sub-name in the other name, e.g., between John K Smith and John Kennedy Smith, the K and Kennedy will
        # match however John Calvin Smith will not match with John K Smith, we also match all the non-initial sub
        # names
        if org_match:
            if all(n[0] == g[0] for n, g in zip(sub_var_names, sub_full_name) if len(n) == 1 or len(g) == 1) and \
                    all(n[0] == g[0] for n, g in zip(sub_var_names, sub_full_name) if len(n) > 1 and len(g) > 1):
                return True

        # If the organisations do not match, we use fuzzywuzzy to see if the names match given a certain threshold
        else:
            if fuzzy_match(person_full_name, norm_name, 85):
                return True

    return False


class InformationGatherer(ABC):
    pass


class ORCIDInformationGatherer(InformationGatherer):
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
                "all_names": author.get("display_name_alternatives"),
                "citations": author.get('cited_by_count', 0),
                "hIndex": author.get('summary_stats', {}).get("h_index", -1),
                "externalIds": author.get('ids', {}),
                "works_count": author.get("works_count", 0),
                "organisations": list(
                    map(lambda x: x.get("display_name", "Unknown name"), author.get("last_known_institutions", []))),
                "openAlex_id": author.get("id", "")
            }
        return {}

    @staticmethod
    def get_details_from_paper_id(paper_id):
        """
        Retrieve the DOI of a paper using its OpenAlex ID.
        """

        try:
            id = paper_id.split("org/")[1]
        except:
            id = paper_id

        # Fetch the work (paper) details using the OpenAlex ID
        work = Works()[id]

        if work:
            # Extract the DOI

            return {
                "title": work.get("title", "Unknown title"),
                "publication_date": work.get("publication_date", "Unknown publication date"),
                "abstract": work["abstract"]
            }
        else:
            return None

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
    @staticmethod
    def get_relevant_concepts_from_paper(paper):
        concepts = paper["concepts"]

        relevant_concepts = list(
            filter(lambda x: x[2] > 0.1, map(lambda y: (y["display_name"], y["level"], y["score"]), concepts)))

        return relevant_concepts


class GTRInformationGatherer(InformationGatherer):
    BASE_GTR_URL = "https://gtr.ukri.org/api"
    HEADERS = {"Accept": "application/json"}

    @staticmethod
    def get_gtr_orgs_grants(orcid_id, name_variations, organisations):
        """Fetches grant information for an author using GTR API."""

        # Iterate through all possible name variations for this author
        for name in name_variations:

            # Send request to get grants linked to author
            name_url_parsed = urllib.parse.quote(name)
            url = f"{GTRInformationGatherer.BASE_GTR_URL}/search/person?term={name_url_parsed}"
            response = requests.get(url, headers=GTRInformationGatherer.HEADERS)

            if response.status_code == 200:
                data = response.json()

                # Iterate through each person GtR returned
                for person_details in data.get('facetedSearchResultBean', {}).get("results", []):

                    # Get person object and their organisation
                    person = person_details.get("person", {})
                    org_name = person_details.get("organisation", {}).get("name", "Unknown").strip()

                    # Get details about author
                    person_first = person.get("firstName", "").lower()
                    person_last = person.get("surname", "").lower()
                    person_orcid = person.get("orcidId")

                    # Get whether organisation, name and ORCID matches
                    org_match = org_name in organisations
                    name_match = is_name_match(name_variations, f"{person_first} {person_last}",
                                               org_match)
                    orcid_match = orcid_id and person_orcid and orcid_id in person_orcid

                    # If a suitable combination matches
                    if orcid_match or (name_match and org_match):

                        # Get information about the person's grants
                        person_id = person["id"]
                        grants_url = f"{GTRInformationGatherer.BASE_GTR_URL}/person/{person_id}"
                        grants_response = requests.get(grants_url, headers=GTRInformationGatherer.HEADERS)

                        if grants_response.status_code == 200:
                            resp = grants_response.json()
                            person_overview = resp.get("personOverview", {})

                            # Get details about each grant that the author has had
                            grants = map(lambda grant: grant.get("projectComposition", {}),
                                         person_overview.get("projectSearchResult", {}).get("results", []))

                            # Create a Grant object for each grant
                            grant_objects = map(lambda grant:
                                                Grant(
                                                    title=grant.get("project", {}).get("title", "Unknown title"),
                                                    category=grant.get("project", {}).get("grantCategory",
                                                                                          "Unknown category"),
                                                    value=grant.get("project", {}).get("fund", {}).get("valuePounds",
                                                                                                       None),
                                                    funder=grant.get("project", {}).get("fund", {}).get("funder",
                                                                                                        {}).get(
                                                        "name", "Unknown funder"),
                                                    organisation=grant.get("leadResearchOrganisation", {}).get("name",
                                                                                                               "Unknown organisation")
                                                ), grants)

                            return list(grant_objects), org_name

        return [], ""


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
            f"{SemanticScholarInformationGatherer.BASE_URL}/paper/{paper_id}",
            params={"fields": "externalIds,name"}
        )

        print(r.json())

        return r.json()["data"]["externalIds"]

    @staticmethod
    def get_h_index_from_author_name(orcid: str = None, name: str = None):

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


def authors_from_doi(doi):
    """
    Takes a DOI and retrieves a list of Author objects with additional metadata matching.
    """
    # Get authors from OpenAlex
    authors = OpenAlexInformationGatherer.get_authors_from_doi(doi)
    author_objects = []

    for author in authors:

        # Normalize the author's name
        author_name = normalize_name(author["name"])

        # Get author info from OpenAlex
        author_info = OpenAlexInformationGatherer.get_author_info(author["authorId"])
        author_obj = build_author_object(author_info)
        author_objects.append(author_obj)

    return author_objects

def build_author_object(author_info):
    author_name = author_info.get("name")
    if author_name is None:
        raise Exception("Author name is None")
    
    external_ids = author_info.get("externalIds", {})
    orcid = external_ids.get("orcid")
    orcid = orcid.split("org/")[1] if orcid else None

    # Use ORCID API to get more information
    orcid_data = ORCIDInformationGatherer.get_profile(orcid) if orcid else {}
    employment_data = ORCIDInformationGatherer.get_employments(orcid) if orcid else {}
    dob = ORCIDInformationGatherer.get_dob(orcid) if orcid else None
    website_check = orcid_data.get("researcher-urls", {}).get("researcher-url", [])
    website = website_check[0].get("url", None).get("value") if website_check else None

    # Get grants using GTR API with fuzzy matching
    org_list = author_info.get("organisations", [])
    grants, org = GTRInformationGatherer.get_gtr_orgs_grants(orcid, author_info.get("all_names"), org_list)

    more_info = GoogleScholarInformationGatherer.get_author_info(author_info.get("all_names"), org_list)

    # Get h-index using OpenAlex, if not available then use SemanticScholar to get it
    h_index = author_info.get("hIndex")
    if h_index == -1:
        h_index = SemanticScholarInformationGatherer.get_h_index_from_author_name(orcid, author_name)

    # Construct Author object
    author_obj = Author(
        name=author_name,
        citations=author_info.get("citations", 0),
        hindex=h_index,
        organisation_history=org_list,
        orcid=orcid,
        dob=dob,
        grants=grants,
        grant_org_name=org,
        website=website,
        openAlexid=author_info.get("openAlex_id", "Unknown"),
        works_count=author_info.get("works_count", "Unknown")
    )

    author_obj.profile = orcid_data
    author_obj.employment = employment_data
    return author_obj

###############################################
##          TO BE USED LATER                 ##
###############################################
class GoogleScholarInformationGatherer(InformationGatherer):

    @staticmethod
    def get_author_info(name_variations, organisations):
        """
        Fetches author information from Google Scholar.
        Tries to match both name and organisation.

        So erm... these lot have SO MANY inconsistencies and holes in their data it basically never works rn, will
        have to really work on this to get it to be useful
        """
        for name in name_variations:
            search_query = scholarly.search_author(name)

            for author in search_query:
                scholar_name = author.get("name", "")
                scholar_org = author.get("affiliation", "").strip().lower()

                org_match = any(org.lower() in scholar_org for org in organisations)

                if is_name_match(name_variations, scholar_name, org_match):
                    detailed_author = scholarly.fill(author)

                    return {
                        "name": detailed_author.get("name"),
                        "affiliations": detailed_author.get("affiliation", "Unknown"),
                        "hIndex": detailed_author.get("hindex", -1),
                        "i10Index": detailed_author.get("i10index", -1),
                        "citations": detailed_author.get("citedby", 0),
                        "coauthors": [coauthor["name"] for coauthor in detailed_author.get("coauthors", [])],
                        "publications": [
                            {
                                "title": pub["bib"].get("title", "Unknown"),
                                "citations": pub.get("num_citations", 0),
                                "year": pub["bib"].get("pub_year", "Unknown"),
                                "venue": pub["bib"].get("venue", "Unknown"),
                            }
                            for pub in detailed_author.get("publications", [])
                        ],
                    }

        return {}


if __name__ == "__main__":
    # Dan van der Horst, Saskia A F Vermeylen
    # y = authors_from_doi("https://doi.org/10.1016/j.biombioe.2010.11.029")

    # Saskia E Bakker
    y = authors_from_doi("https://doi.org/10.1099/vir.0.053025-0")

    x = OpenAlexInformationGatherer.get_details_from_paper_id("https://openalex.org/W4400454085")
