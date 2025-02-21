from abc import ABC, abstractmethod
from typing import List, Dict, Any

from pyalex import Works
from common.pinecone_store import PineconeStore


# Abstract interface for paper sources
class AbstractPaperSource(ABC):
    """
    Adapter class to allow usage of any API as a source of abstracts and papers
    """

    @abstractmethod
    def get_paper_abstracts(self, **kwargs) -> (List[str], List[Dict[str, Any]]):
        """Fetches the abstracts of papers based on criteria."""
        pass


class PyAlexPaperSource(AbstractPaperSource):

    def __init__(self, **kwargs):
        pass

    def get_paper_abstracts(self, country, **kwargs) -> (List[str], List[Dict[str, Any]]):
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
            publication_year=">2008"
        ) \
            .filter(
            primary_topic={"domain": {"id": "!2"}}
        ) \
            .filter(
            primary_topic={"domain": {"id": "!4"}}
        ).get(per_page=100)

        # We just getting the abstracts with this one, need to discuss what else is important, obv the title and stuff
        # but like any other info, can design the classes as needed
        abstracts = list(filter(lambda a: a is not None, map(lambda d: d['abstract'], query)))
        remainder = list(map(lambda d: {k: v for k, v in d.items() if k == 'id' or k == "doi" or k == "title"}, query))
        return abstracts, remainder


pcs = PineconeStore("climate-test")

ab, meta = PyAlexPaperSource().get_paper_abstracts("gb")

pcs.add_chunks(ab, meta, namespace="papers")
