from typing import Generator, Dict, Any
from itertools import chain
from pyalex import Works, Topics
from .base import Fetcher


class TopicFetcher(Fetcher):
    def fetch(self) -> Generator[Dict[str, Any], None, None]:
        """
        Fetches topics and sample works from OpenAlex.
        
        Yields:
            Dict containing topic info and sample works
        """
        cursor = "*"
        while cursor:
            topics, meta = Topics().get(per_page=200, cursor=cursor, return_meta=True)
            cursor = meta["next_cursor"]
            print("number of topics: ", len(topics))
            for topic in topics:
                # Get 3 random sample works for this topic
                sample_works = Works() \
                    .filter(topics={'id': topic['id']}) \
                    .sort(cited_by_count="desc") \
                    .select(['id', 'title', 'abstract_inverted_index_v3', 'abstract_inverted_index']) \
                    .paginate(per_page=3)
                    
                # Get the first page of results (3 works)
                sample_abstracts = []
                for page in chain(sample_works):
                    for work in page:
                        if abstract := self._get_abstract(work):
                            sample_abstracts.append({
                                'title': work.get('title'),
                                'abstract': abstract
                            })
                    break # break after first page (we have already seen 3 papers)
                yield {
                    'topic_id': topic['id'],
                    'topic_name': topic['display_name'],
                    'topic_description': topic.get('description', ''),
                    'sample_works': sample_abstracts
                }
            

    def _get_abstract(self, work):
        # Reuse the abstract extraction logic from PyAlexFetcher
        inverted_index = work.get('abstract_inverted_index_v3') or work.get('abstract_inverted_index')
        
        if not inverted_index:
            return None
        
        max_position = max(pos for positions in inverted_index.values() for pos in positions)
        words = [''] * (max_position + 1)
        
        for word, positions in inverted_index.items():
            for position in positions:
                words[position] = word
        
        return ' '.join(words) 