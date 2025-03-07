import os
import re
import urllib.parse
from abc import ABC

import pyalex
import requests
import unicodedata
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from pyalex import Works, Authors
from scholarly import scholarly

from ranking_model.author import Author
from ranking_model.grant import Grant

# Load .env
load_dotenv(override=True)

# PyAlex API key config
pyalex.config.api_key = os.getenv("OPENALEX_API_KEY")


def normalize_name(name: str):
    """

    Parameters
    ----------
    name: str, name we want to normalise

    Returns
    -------
    str - normalised name
    """

    # Normalise the name in a consistent format
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')

    return name.lower().strip()


def fuzzy_match(name1, name2, threshold=85):
    """
    Returns whether two names have a similarity score above a certain threshold
    Parameters
    ----------
    name1: str, first name
    name2: str, second name
    threshold: int, threshold above which the names are considered to be the same person's

    Returns
    -------
    bool - whether the similarity score between the names is above the given threshold
    """

    # partial_ratio works better when there are initials, but we use token_sort_ratio since this function is only called
    # when org_match is False, so we do not want to match on initials just in case, we match on the name
    return fuzz.token_sort_ratio(name1, name2) >= threshold


def is_name_match(name_variations, person_name, org_match):
    """
    Given a person's name variations from OpenAlex, the name from the other source, and whether the organisation metadata
    matches, this returns true if we are confident that the names refer to the same person
    Parameters
    ----------
    name_variations: List[str], all known variations of the name according to OpenAlex
    person_name: str, name of person from new source
    org_match: bool, whether the organisations matched

    Returns
    -------
    bool - True if we are confident the names match
    """

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
    """
    Parent class for all Information Gatherers
    """
    pass


class ORCIDInformationGatherer(InformationGatherer):
    """
    Information gatherer for ORCID
    """
    BASE_URL = "https://pub.orcid.org/v3.0/"
    HEADERS = {"Accept": "application/json"}

    @staticmethod
    def get_profile(orcid_id: str):
        """
        Returns the profile of an author given their ORCID
        Parameters
        ----------
        orcid_id: str, ORCID id of author

        Returns
        -------
        Dict - all the details we have about the author on ORCID
        """
        url = f"{ORCIDInformationGatherer.BASE_URL}{orcid_id}/person"
        response = requests.get(url, headers=ORCIDInformationGatherer.HEADERS)
        return response.json() if response.status_code == 200 else None

    @staticmethod
    def get_works(orcid_id: str):
        """
        Returns all the works of an author
        Parameters
        ----------
        orcid_id: str, ORCID of author

        Returns
        -------
        Dict - all works for a given author
        """
        url = f"{ORCIDInformationGatherer.BASE_URL}{orcid_id}/works"
        response = requests.get(url, headers=ORCIDInformationGatherer.HEADERS)
        return response.json() if response.status_code == 200 else None

    @staticmethod
    def get_employments(orcid_id: str):
        """
        Gets the employments of a given author
        Parameters
        ----------
        orcid_id: str, ORCID of author

        Returns
        -------
        Dict - list of all employments
        """
        url = f"{ORCIDInformationGatherer.BASE_URL}{orcid_id}/employments"
        response = requests.get(url, headers=ORCIDInformationGatherer.HEADERS)
        return response.json() if response.status_code == 200 else None

    @staticmethod
    def get_dob(orcid_id: str):
        """
        Date of birth of an author
        Parameters
        ----------
        orcid_id: str, ORCID if of author

        Returns
        -------
        str -  Date of birth of author
        """
        url = f"{ORCIDInformationGatherer.BASE_URL}{orcid_id}/biography"
        response = requests.get(url, headers=ORCIDInformationGatherer.HEADERS)
        if response.status_code == 200:
            data = response.json()
            return data.get("date-of-birth")
        return None


class OpenAlexInformationGatherer(InformationGatherer):
    """
    Information Gatherer for OpenAlex
    """

    @staticmethod
    def get_UK_authors_from_doi(doi: str):
        """
        Gets UK authors given a paper's DOI
        Parameters
        ----------
        doi: str, DOI of paper whose UK authors we want

        Returns
        -------
        List[Dict] - list of authors
        """

        # Get the work
        work = Works().filter(doi=doi).get()

        if work:

            # Get all the authors
            authorships = work[0].get('authorships', [])
            uk_authors = []

            # Go through each author and store in uk_authors if they have been linked to UK institutions
            for authorship in authorships:

                # Get linked institution
                institutions = authorship.get('institutions', [])

                # Check if any of the institutions has a country code of "GB"
                if any(inst.get('country_code') == "GB" for inst in institutions):
                    uk_authors.append({
                        'authorId': authorship['author']['id'],
                        'name': authorship['author']['display_name']
                    })

            return uk_authors

        return []

    @staticmethod
    def get_top_authors_from_doi(doi):
        """
        Only return first, last or corresponding authors from a paper given its DOI
        Parameters
        ----------
        doi: str, the DOI of the paper

        Returns
        -------
        List[Dict] - list of authors
        """

        # Get the work
        work = Works().filter(doi=doi).get()

        top_authors = []

        if work:

            # Get all authors
            authors = work[0].get('authorships', [])

            # Add author if they are first, last or corresponding
            for author in authors:

                if author['author_position'] == 'first' or author['author_position'] == 'last' \
                        or author['is_corresponding']:
                    top_authors.append({'authorId': author['author']['id'], 'name': author['author']['display_name']})

        return top_authors

    @staticmethod
    def get_work_from_doi(doi):
        """
        Returns the work given its doi
        Parameters
        ----------
        doi: str, DOI of paper

        Returns
        -------
        Work - the Work object
        """
        return Works().filter(doi=doi).get()

    @staticmethod
    def get_work_from_paper_id(id):
        """
        Returns the work given its paper id
        Parameters
        ----------
        id: str, DOI of paper

        Returns
        -------
        Work - the Work object
        """
        return Works()[id]

    @staticmethod
    def get_author_info(author_id):
        """
        Returns the information about an author given their author_id
        Parameters
        ----------
        author_id: str, DOI of paper

        Returns
        -------
        Dict - author information
        """

        # Get author
        author = Authors()[author_id]

        # Return only relevant information
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
    def get_works_from_author_id(author_id):
        """
        Get all works given an author_id
        Parameters
        ----------
        author_id: str, author's id whose works we want

        Returns
        -------
        [Work] - the list of works
        """
        works = Works().filter(**{"authorships.author.id": author_id}).get(per_page=200)
        return works if works else []

    @staticmethod
    def get_details_from_paper_id(paper_id):
        """
        Given a paper id, we return title, publication_date and abstract
        Parameters
        ----------
        paper_id: str, id of paper whose info we want

        Returns
        -------
        Dict - title, publication_date and abstract
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

    @staticmethod
    def get_relevant_concepts_from_paper(paper):
        """
        Retuns a paper's concepts
        Parameters
        ----------
        paper: Work

        Returns
        -------
        List - relevant concepts
        """
        concepts = paper["concepts"]

        relevant_concepts = list(
            filter(lambda x: x[2] > 0.1, map(lambda y: (y["display_name"], y["level"], y["score"]), concepts)))

        return relevant_concepts


class GTRInformationGatherer(InformationGatherer):
    """
    Information gatherer for Gateway to Research
    """

    BASE_GTR_URL = "https://gtr.ukri.org/api"
    HEADERS = {"Accept": "application/json"}

    @staticmethod
    def get_gtr_orgs_grants(orcid_id, name_variations, organisations):
        """
        Given an author's ORCID id (which may be None), all possible name variations and their linked organisations,
        returns the grant information that can be found on Gateway to Research

        Parameters
        ----------
        orcid_id: str, ORCID id of author
        name_variations: List[str], all known name variations
        organisations: List[str], all linked institutions

        Returns
        -------
        List[Grant] - all the grants we know the author has received
        """

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


def authors_from_doi(doi):
    """
    Given a doi of a paper, returns a list of author objects with all known information about these authors

    Parameters
    ----------
    doi: str, DOI of paper

    Returns
    -------
    List[Author] - author objects for each author
    """

    # Get authors from OpenAlex
    authors = OpenAlexInformationGatherer.get_UK_authors_from_doi(doi)
    author_objects = []

    for author in authors:
        # Build objects for author
        author_obj = build_author_object(author["authorId"])

        author_objects.append(author_obj)

    return author_objects


def build_author_object(author_id):
    """
    Given openalex author id, it builds an Author object by gathering information using various gatherers.
    Parameters
    ----------
    author_id: str, OpenAlex author id

    Returns
    -------
    Author - Author object for a given author
    """

    # Get information on author from OpenAlex
    author_info = OpenAlexInformationGatherer.get_author_info(author_id)
    author_name = author_info.get("name")
    if author_name is None:
        raise Exception("Author name is None")

    # Get ORCID if available
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


def get_all_author_info(authorid):
    """
    All information available on author given OpenAlex author id
    Parameters
    ----------
    authorid: str, OpenAlex author id

    Returns
    -------
    Author - object for the author
    """

    # Get information about author from OpenAlex
    author_info = OpenAlexInformationGatherer.get_author_info(authorid)
    orcid = author_info["externalIds"].get("orcid") if "externalIds" in author_info else None
    orcid = orcid.split("org/")[1] if orcid else None

    # Use ORCID's API to get more information about the author
    orcid_data = ORCIDInformationGatherer.get_profile(orcid) if orcid else {}
    employment_data = ORCIDInformationGatherer.get_employments(orcid) if orcid else {}
    dob = ORCIDInformationGatherer.get_dob(orcid) if orcid else None
    website_check = orcid_data.get("researcher-urls", {}).get("researcher-url", [])
    website = website_check[0].get("url", None).get("value") if website_check else None

    # Use GTR API to get grant information about author using ORCID and their name
    (grants, org) = GTRInformationGatherer.get_gtr_orgs_grants(orcid, author_info.get("name"),
                                                               author_info.get("organisations", []))

    # Construct Author object
    author_obj = Author(
        name=author_info.get("name", "Unknown"),
        citations=author_info.get("citations", 0),
        hindex=author_info.get("hIndex", 0),
        organisation_history=author_info.get("organisations", []),
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
    authors = OpenAlexInformationGatherer.get_authors_from_doi("https://doi.org/10.1016/j.biombioe.2010.11.029")
    print(authors)
    ids = [author["authorId"] for author in authors]
    works = OpenAlexInformationGatherer.get_works_from_author_id(ids[0])
    print(ids)
    # authors_from_doi("https://doi.org/10.48550/arXiv.2402.01928")

    # x = OpenAlexInformationGatherer.get_details_from_paper_id("https://openalex.org/W4400454085")

    # y = OpenAlexInformationGatherer.get_authors_from_doi()
