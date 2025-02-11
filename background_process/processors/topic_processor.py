from typing import Dict, Any, Tuple, List
from langchain_openai import ChatOpenAI
from background_process.prompts import CLIMATE_RELEVANCE_PROMPT, TopicAssessment
from .base import Processor, ProcessingTask


class TopicProcessor(Processor):
    def __init__(self, supabase_client, model_name: str = "gpt-4o-mini"):
        super().__init__(supabase_client, None)  # No pinecone store needed
        self.evaluator = ChatOpenAI(
            model=model_name,
            temperature=0.2
        ).with_structured_output(TopicAssessment)
        self.task_id = self.create_task(ProcessingTask.TOPIC_PROCESSING)

    def format_sample_works(self, works: list) -> str:
        """Format sample works for prompt"""
        formatted = ""
        for i, work in enumerate(works, 1):
            formatted += f"\nWork {i}:\n"
            formatted += f"Title: {work['title']}\n"
            formatted += f"Abstract: {work['abstract'][:500]}...\n"  # Truncate long abstracts
        return formatted

    def get_topic_assessment(self, topic_id: str) -> Dict[str, Any]:
        """Get existing topic assessment from Supabase DB by topic_id"""
        response = self.supabase.table('openalex_topic_assessments') \
            .select("*") \
            .eq('topic_id', topic_id) \
            .execute()
        return response.data

    def save_to_db(self, assessment: TopicAssessment, topic_id: str) -> Dict[str, Any]:
        """Save assessment to Supabase"""
        data = {
            "topic_id": topic_id,
            "is_climate_relevant": assessment.is_climate_relevant,
            "analysis": assessment.analysis
        }
        response = self.supabase.table('openalex_topic_assessments').insert(data).execute()
        return response.data[0]

    def process(self, data: Dict[str, Any]) -> Tuple[TopicAssessment, Dict[str, Any]]:
        """
        Synchronous process method to satisfy abstract class.
        For single topic processing, use this.
        For batch processing, use process_batch.
        """
        # Run the async process in the event loop
        return asyncio.run(self.process_single_topic(data))

    async def process_batch(self, topics: List[Dict[str, Any]]) -> List[Tuple[TopicAssessment, Dict[str, Any]]]:
        """Process a batch of topics concurrently"""
        tasks = []
        for topic in topics:
            task = asyncio.create_task(self.process_single_topic(topic))
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

    async def process_single_topic(self, data: Dict[str, Any]) -> Tuple[TopicAssessment, Dict[str, Any]]:
        """Process a single topic asynchronously"""
        # Check if topic has already been processed
        existing_assessment = self.get_topic_assessment(data['topic_id'])
        if existing_assessment:
            print(f"Topic {data['topic_name']} already processed.")
            return None, existing_assessment[0]

        # Format the prompt
        sample_works_text = self.format_sample_works(data['sample_works'])
        chain = CLIMATE_RELEVANCE_PROMPT | self.evaluator
        
        # Get structured assessment from LLM
        assessment = await chain.ainvoke({
            "topic_name": data['topic_name'],
            "topic_description": data['topic_description'],
            "sample_works": sample_works_text
        })

        # Store in database
        record = self.save_to_db(assessment, data['topic_id'])
        
        return assessment, record 