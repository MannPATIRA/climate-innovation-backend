from langchain.prompts import ChatPromptTemplate
from typing import Literal, List
from pydantic import BaseModel, Field

CLIMATE_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a scientific advisor helping to identify research topics that could contribute to solving climate change problems.
The idea is that we should be able to find climate solutions and research areas that can pursued and pushed forward by Imperial Climate Institute.
You will analyze research topics and their sample works to determine their relevance to climate solutions."""),
    ("human", """Please analyze this research topic and its sample works for climate relevance:

Topic: {topic_name}
Description: {topic_description}

Sample works from this topic:
{sample_works}

Evaluate whether this research topic could have direct relevance and impact in solving climate-related problems. 
Consider both mitigation and adaptation strategies.
Use the sample works to gauge the topic and what it studies more deeply. The link to climate MUST be direct and fairly clear""")
])

from typing import Literal
from pydantic import BaseModel, Field

class TopicAssessment(BaseModel):
    """Evaluation of a research topic's relevance to climate solutions."""
    analysis: str = Field(
        description="Short and focussed analysis of the topic's relevance to climate solutions",
    ) 

    is_climate_relevant: bool = Field(
        description="Whether this topic has direct relevance to climate solutions"
    )

SEARCH_QUERY_SYSTEM_PROMPT = """You are a search query generator. Generate specific search queries 
related to climate change and environmental policy documents. Return only the most relevant 
and focused queries. Don't include search operators like site: or filetype: in the queries.
Just focus on the main search terms and the most relevant keywords. Your list should contain atleast 10 queries."""

class SearchQueries(BaseModel):
    """Structure for search query generation output"""
    queries: List[str] = Field(
        description="List of specific search queries to find climate change documents"
    )

REPORT_ASSESSMENT_SYSTEM_PROMPT = """You are an expert in climate technology. 
Your task is to determine if a document is specifically about technical climate issues.
You must ensure the document details actual problems that contribute to climate change.
You should ensuret that it does not have an excessive focus on policy or politics."""

REPORT_ASSESSMENT_HUMAN_PROMPT = """Analyze the following text from a report and determine if it's about climate issues:

{text}

Provide a brief analysis explaining your decision."""

class ReportAssessment(BaseModel):
    """Assessment of whether a report is about deep tech climate issues"""
    analysis: str = Field(
        description="Explanation of why this report is or isn't about deep tech climate issues"
    )
    result: bool = Field(
        description="Whether this report is about deep tech climate issues"
    )

SUMMARY_GENERATION_PROMPT = """You are an expert summarizer of climate-related documents.
Your task is to create a concise, informative summary of the provided document.
Focus on the key findings, recommendations, and technical details that would be most relevant
for climate researchers. Maintain scientific accuracy while making the content accessible.
"""

class DocumentSummary(BaseModel):
    """Structure for document summary generation output"""
    summary: str = Field(
        description="Concise summary of the climate document that captures key points and findings"
    )