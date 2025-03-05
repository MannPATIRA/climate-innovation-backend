from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

ANALYSIS_PROMPT = """You are an expert in analyzing climate-related research.
Your task is to evaluate the provided research abstract based on the following criteria.
Return a boolean value (True or False) for each question. Be concise and accurate in your assessment.

Research abstract: {abstract}

MITIGATION: Could this research feasibly lead to a reduction of greenhouse gas emissions or removal of carbon dioxide from the atmosphere?
TECHNOLOGY: Does this research describe a technology with practical application?
READINESS: Does this research demonstrate that proof-of-concept has been achieved prior to commercialisation or deployment?
MARKET: Does a clear commercial market or industry need exist for this research?
TECH ENABLING: Rather than a stand-alone technology, does this research represent the fundamental science that might enable future technology development?
ECO FOCUS: Was this research conducted with an explicit climate change or sustainability application in mind?
NEGLECTEDNESS: Is this research more likely than not to be neglected by existing innovation support mechanisms in the UK?
"""

class ResearchAnalysis(BaseModel):
    """Structure for climate research analysis output"""
    mitigation: bool = Field(
        description="Could this research feasibly lead to a reduction of greenhouse gas emissions or removal of carbon dioxide from the atmosphere?"
    )
    technology: bool = Field(
        description="Does this research describe a technology with practical application?"
    )
    readiness: bool = Field(
        description="Does this research demonstrate that proof-of-concept has been achieved prior to commercialisation or deployment?"
    )
    market: bool = Field(
        description="Does a clear commercial market or industry need exist for this research?"
    )
    tech_enabling: bool = Field(
        description="Rather than a stand-alone technology, does this research represent the fundamental science that might enable future technology development?"
    )
    eco_focus: bool = Field(
        description="Was this research conducted with an explicit climate change or sustainability application in mind?"
    )
    neglectedness: bool = Field(
        description="Is this research more likely than not to be neglected by existing innovation support mechanisms in the UK?"
    )    
    
    def convert_to_binary_list(self) -> list[int]:
        """Converts the boolean values to a list of 1s and 0s."""
        return [
            int(self.mitigation),
            int(self.technology),
            int(self.readiness),
            int(self.market),
            int(self.tech_enabling),
            int(self.eco_focus),
            int(self.neglectedness),
        ]
