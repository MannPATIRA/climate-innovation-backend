from abc import ABC, abstractmethod
from typing import List

from pyalex import Works


# Abstract interface for paper sources
class AbstractPaperSource(ABC):
    """
    Adapter class to allow usage of any API as a source of abstracts and papers
    """

    @abstractmethod
    def get_paper_abstracts(self, **kwargs) -> List[str]:
        """Fetches the abstracts of papers based on criteria."""
        pass


class PyAlexPaperSource(AbstractPaperSource):

    def __init__(self, **kwargs):
        pass

    def get_paper_abstracts(self, country, **kwargs) -> List[str]:
        """
        Fetches paper abstracts using the pyalex library.

        :param country: The country to filter research papers by (default: "United Kingdom").
        :param kwargs: Additional filters for the search query.
        :return: A list of abstracts from the retrieved papers.
        """

        # Right now, just using the query that matches what Cesar did in his original paper
        query = Works() \
            .filter(
            authorships={"institutions": {"country_code": country}}
        ) \
            .filter(
            type="article|preprint|book-chapter|dissertation"
        ) \
            .filter(
            authorships={"is_corresponding": "true"}
        ) \
            .filter(
            authorships={"affiliations": {
                "institution_ids": "https://openalex.org/I82284825|https://openalex.org/I47508984|https://openalex"
                                   ".org/I98677209|https://openalex.org/I130828816|https://openalex.org/I241749|https"
                                   "://openalex.org/I4210092773"}}
        ) \
            .filter(
            publication_year=">1999"
        ) \
            .filter(
            primary_topic={"domain": {"id": "!2"}}
        ) \
            .filter(
            primary_topic={"domain": {"id": "!4"}}
        ).get(per_page=10)

        return query


print(PyAlexPaperSource().get_paper_abstracts("gb")[0])
