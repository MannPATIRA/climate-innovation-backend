from scholarly import scholarly

from backend_server.gatherers.InformationGatherer import InformationGatherer
from backend_server.gatherers.name_matching_lib import is_name_match


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
