from dotenv import load_dotenv
import numpy as np
from typing import List
from supabase import Client
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import asyncio

from .author import Author
from .paper import Paper
from .prompts import ANALYSIS_PROMPT, ResearchAnalysis
from .ranker import Ranker

load_dotenv()

class SevenQRanker(Ranker):
    """
    A Ranker implementation that uses an llm to conduct the seven-question scoring system.
    Using paper:
      Towards unearthing neglected climate innovations from scientific literature using Large Language Models
      https://arxiv.org/abs/2411.10055
    """
    def __init__(self, supabase_client: Client, model_name: str = "gpt-4o-mini", learning_rate: float = 0.01):
        super().__init__(supabase_client, model_name, learning_rate)
        # weights from Context (binary) in paper
        self.question_weights = np.array([-7, 0.211, 0.339, 0.102, -0.235, 0.663, -0.080])
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.model_with_structure = self.llm.with_structured_output(ResearchAnalysis)
        self.prompt = PromptTemplate.from_template(ANALYSIS_PROMPT)

    def rank_papers(self, papers: List[Paper]) -> List[Paper]:
        """Ranks papers based on a seven-question scoring system, conducted by an LLM"""
        async def analyze_papers():
            tasks = [self._analyze_paper_abstract(paper) for paper in papers]
            await asyncio.gather(*tasks)
        
        asyncio.run(analyze_papers())
        return sorted(papers, key=lambda p: p.score, reverse=True)

    async def _analyze_paper_abstract(self, paper: Paper):
        """Analyzes a single paper abstract and updates its score."""
        researchAnalysis = await self.model_with_structure.ainvoke(
            self.prompt.format(abstract=paper.abstract)
        )
        scores = researchAnalysis.convert_to_binary_list()
        paper.score = np.dot(scores, self.question_weights)

    def rank_authors(self, authors: List[Author]) -> List[Author]:
        """Ranks authors based on a simplified approach"""
        return authors

    def update_author_model(self, author: Author, label: int):
        """Updates the author ranking model (placeholder)."""
        pass  

    def update_paper_model(self, paper: Paper, label: int):
        """Updates the paper ranking model (placeholder)."""
        pass

    def delete_author(self, paper: Paper, author: Author):
        """Process a deletion of an author (placeholder)."""
        pass

    def accept_author(self, paper: Paper, author: Author):
        """Process an acceptance of an author (placeholder)."""
        pass

    def delete_paper(self, paper: Paper):
        """Process a deletion of a paper (placeholder)."""
        pass

    def accept_paper(self, paper: Paper):
        """Process an acceptance of a paper (placeholder)."""
        pass

    def save_model(self):
        """No changes so no need to change."""
        pass
    
    def load_model(self) -> bool:
        """No changes so no need to change."""
        return True


if __name__ == "__main__":
    # Mock Supabase client (as in testing_ranker.py)
    class MockSupabaseClient:
        def table(self, table_name):
            return self
        def upsert(self, data):
            return self
        def execute(self):
            return None

    mock_supabase_client = MockSupabaseClient()
    ranker = SevenQRanker(supabase_client=mock_supabase_client, learning_rate=0.01)

    #Import data from testing_ranker.py (you'll need to adjust imports to your project structure)
    from .testing_ranker import paper1, paper2, paper3

    papers = [paper1, paper2, paper3]

    try:
        ranked_papers = ranker.rank_papers(papers)
        print("\nRanked Papers:")
        for p in ranked_papers:
            print(f"Paper: {p.title} (Score: {p.score:.3f})")
    except Exception as e:
        print(f"An error occurred during ranking: {e}")
