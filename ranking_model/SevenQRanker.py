import numpy as np
from typing import List
from supabase import Client
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from .author import Author
from .paper import Paper
from .prompts import ANALYSIS_PROMPT, ResearchAnalysis
from .ranker import Ranker


class SevenQRanker(Ranker):
    """
    A Ranker implementation that uses an llm to conduct the seven-question scoring system.
    Using paper:
      Towards unearthing neglected climate innovations from scientific literature using Large Language Models
      https://arxiv.org/abs/2411.10055
    """
    def __init__(self, supabase_client: Client, model_name: str, learning_rate: float = 0.01):
        super().__init__(supabase_client, model_name, learning_rate)
        self.question_weights = np.ones(7) / 7
        self.llm = ChatOpenAI(model=model_name)
        self.prompt = PromptTemplate.from_template(ANALYSIS_PROMPT)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, output_parser=self.llm.with_structured_output(ResearchAnalysis))

    @Ranker.save_model_before_rank
    def rank_papers(self, papers: List[Paper]) -> List[Paper]:
        """Ranks papers based on a seven-question scoring system, conducted by an LLM"""
        for paper in papers:
            researchAnalysis = {"abstract": paper.abstract} | self.chain
            scores = researchAnalysis.convert_to_binary_list()
            paper.score = np.dot(scores, self.question_weights)
        return sorted(papers, key=lambda p: p.score, reverse=True)

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
