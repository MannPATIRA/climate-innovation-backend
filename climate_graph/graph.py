import os
from backend_server.gatherers import OpenAlexInformationGatherer
from openai import OpenAI
import numpy as np
from collections import Counter
from ranking_model.ranker import RegressionRanker
import concurrent.futures

# from dotenv import load_dotenv
# load_dotenv('../.env')

sample_author_id = 'https://openalex.org/A5060519067'
sample_doi = 'https://doi.org/10.48550/arXiv.2303.11366'
sample_paper_id = "https://openalex.org/W4400454085"

# ranker = RegressionRanker()
# most_relevant = []

def get_relevant_authors(author_id, paper_id):
    current_work = OpenAlexInformationGatherer.get_work_from_paper_id(paper_id)
    current_paper_info = get_title_and_abstract(current_work)
    current_paper_embedding = np.array(embed_content(current_paper_info['content']))

    works = OpenAlexInformationGatherer.get_works_from_author_id(author_id)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(get_title_and_abstract, works))
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        embeddings_list = list(executor.map(lambda d: embed_content(d['content']), data))
    embeddings = np.squeeze(np.array(embeddings_list), axis=1)
    
    differences = np.linalg.norm(embeddings - current_paper_embedding, axis=1).flatten()
    sorted_indices = np.argsort(differences).tolist()
    most_relevant = [data[i] for i in sorted_indices][:5]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        top_authors_list = list(executor.map(
            lambda d: OpenAlexInformationGatherer.get_top_authors_from_doi(d['doi']),
            most_relevant
        ))
    
    # RANKER ADD HERE

    final = [author for sublist in top_authors_list for author in sublist][:5]
    
    return final


# def get_next_connections(author_id):

def get_abstract(work):
    # Try the v3 index first
    inverted_index = work.get('abstract_inverted_index_v3') or work.get('abstract_inverted_index')
    
    if not inverted_index:
        return None
    
    # Reconstruct the abstract from the inverted index
    # The index is a dict where keys are words and values are lists of positions
    # We need to create a list long enough to hold all words
    max_position = max(pos for positions in inverted_index.values() for pos in positions)
    words = [''] * (max_position + 1)
    
    # Place each word in its correct position(s)
    for word, positions in inverted_index.items():
        for position in positions:
            words[position] = word
    
    # Join the words to form the complete abstract
    return ' '.join(words)

def get_title_and_abstract(work):
    """
    Returns a concatenated string of the title and abstract for a work.
    If the abstract is missing, returns just the title.
    """
    title = work['title']
    abstract = get_abstract(work)
    if abstract:
        return {'abstact_existence': True, 'content': f"{title} {abstract}", 'doi' : work.get('doi')}
    return {'abstact_existence': False, 'content': title, 'doi' : work.get('doi')}

def embed_content(work):
    OPENAI_API_KEY='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'
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    response = openai_client.embeddings.create(
                    model="text-embedding-3-large",
                    input=[work]
                )
    
    embeddings = [item.embedding for item in response.data]
    return embeddings

print(get_relevant_authors(sample_author_id, sample_paper_id))
