import urllib.parse

import requests

from backend_server.gatherers.InformationGatherer import InformationGatherer
from backend_server.gatherers.name_matching_lib import is_name_match
from ranking_model.grant import Grant


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
