import os

import pyalex
from dotenv import load_dotenv

from backend_server.gatherers.GTRInformationGatherer import GTRInformationGatherer
from backend_server.gatherers.ORCIDInformationGatherer import ORCIDInformationGatherer
from backend_server.gatherers.OpenAlexInformationGatherer import OpenAlexInformationGatherer
from backend_server.gatherers.SemanticScholarInformationGatherer import SemanticScholarInformationGatherer
from ranking_model.author import Author

# Load .env
load_dotenv(override=True)

# PyAlex API key config
pyalex.config.api_key = os.getenv("OPENALEX_API_KEY")


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


if __name__ == "__main__":
    authors = OpenAlexInformationGatherer.get_authors_from_doi("https://doi.org/10.1016/j.biombioe.2010.11.029")
    print(authors)
    ids = [author["authorId"] for author in authors]
    works = OpenAlexInformationGatherer.get_works_from_author_id(ids[0])
    print(ids)
    # authors_from_doi("https://doi.org/10.48550/arXiv.2402.01928")

    # x = OpenAlexInformationGatherer.get_details_from_paper_id("https://openalex.org/W4400454085")

    # y = OpenAlexInformationGatherer.get_authors_from_doi()
