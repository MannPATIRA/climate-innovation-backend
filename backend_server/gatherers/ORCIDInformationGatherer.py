import requests

from backend_server.gatherers.InformationGatherer import InformationGatherer


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
