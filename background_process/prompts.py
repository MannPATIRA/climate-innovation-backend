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