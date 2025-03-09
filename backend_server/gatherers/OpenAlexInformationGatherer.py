from pyalex import Works, Authors
import pyalex
import os
pyalex.config.api_key = os.getenv("OPENALEX_API_KEY")

from backend_server.gatherers.InformationGatherer import InformationGatherer


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
