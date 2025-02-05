from langchain.prompts import ChatPromptTemplate

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