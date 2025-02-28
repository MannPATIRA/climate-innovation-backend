import asyncio
import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from common.pinecone_store import PineconeStore

class MockQueryProcessor:
    def __init__(self):
        pass

    async def process_stream(self, query: str, chat_history: list, completion_callback):
        """Generator function that yields streaming responses"""
        # Example predefined response
        mock_response = (
            "### Climate Challenges\n\n"
            "Climate change is one of the most significant global challenges we face today. "
            "It is linked to various climate and weather extremes, leading to several problems "
            "affecting millions worldwide. Some key climate challenges include:\n\n"
            "- **Catastrophic Heat Waves**: These intense heat events lead to severe health impacts, "
            "increased mortality, and stresses on energy and water resources.\n"
            "- **Wildfires**: Often exacerbated by higher temperatures and prolonged dry conditions, "
            "wildfires destroy vast landscapes, endanger fauna and flora, and threaten human lives and properties.\n"
            "- **Floods**: Increased rainfall and rising sea levels result in severe flooding, "
            "displacing communities, damaging infrastructure, and affecting agriculture.\n\n"
            "These climate-related events underscore the need for innovative approaches and strengthened "
            "global collaboration to mitigate the impacts of climate change.\n\n"
            "{'topics': [{'topic': 'Heat Wave Impacts', 'source': 'Climate Risk Report', 'url': 'https://climate-innovation-bucket.s3.eu-north-1.amazonaws.com/SOCT_Q32024-Net-Zero-Insights-Report.pdf'}, "
            "{'topic': 'Wildfire Threats', 'source': 'Environmental Assessment', 'url': 'https://climate-innovation-bucket.s3.eu-north-1.amazonaws.com/SOCT_Q32024-Net-Zero-Insights-Report.pdf'}, "
            "{'topic': 'Flood Disasters', 'source': 'Global Climate Study', 'url': 'https://climate-innovation-bucket.s3.eu-north-1.amazonaws.com/SOCT_Q32024-Net-Zero-Insights-Report.pdf'}]}"
        )
        
        full_response = ""
        
        # Simulate streaming by yielding chunks with delays
        for chunk in mock_response:
            full_response += chunk
            print(chunk)
            await asyncio.sleep(0.01)  # Add delay to simulate real streaming
            yield chunk
        
        # After the stream is complete, call the completion callback
        await completion_callback(full_response)

class QueryProcessor:
    def __init__(self):
        # Initialize the chain by calling the helper method
        self.chain = self._create_chain()
        self.pinecone_store = PineconeStore()
    
    def _get_relevant_chunks(self, query: str) -> str:
        """Helper function to get and format relevant chunks from Pinecone"""
        # Query for similar chunks
        matches = self.pinecone_store.query_chunk(
            query_text=query,
            top_k=3,  # Adjust number of chunks as needed
            namespace="reports"
        )
        
        # Format the chunks with their metadata
        formatted_context = []
        for i, match in enumerate(matches, 1):
            if match.metadata and 'content' in match.metadata:
                chunk_text = match.metadata['content']
                chunk_title = match.metadata["report_title"]
                chunk_url = match.metadata.get("object_url", "No URL available")
                formatted_context.append(
                    f"Chunk {i}:\n{chunk_text}\n\n"
                    f"Source:\n{chunk_title}\n"
                    f"URL: {chunk_url}\n\n"
                )
        
        # Join all formatted chunks
        return "\n".join(formatted_context)

    def _create_chain(self):
        """Helper method to create and configure the LangChain processing chain"""
        model = ChatOpenAI(model="gpt-4o")

        prompt = ChatPromptTemplate.from_template(
            "Answer the following query in markdown format using the provided context\n"
            "Query: {query}\n"
            "Context \n\n: {context}\n"
            "Your response must directly start with the markdown"
            "After responding, extract 3-5 main points that answer the query from this response along with their sources.\n"
            "You must ensure that the main points focus on climate challenges that are technical, not social or health"
            "When you extract these point, you must ensure that they come from the chunks, don't use your own knowledge"
            "Format the output as a JSON list of objects, where each object has 'topic', 'source', and 'url' keys.\n"
            "Format: {{'topics': [{{'topic': 'topic 1', 'source': 'source 1', 'url': 'url 1'}}, "
            "{{'topic': 'topic 2', 'source': 'source 2', 'url': 'url 2'}}]}}"
        )
        parser = StrOutputParser()

        return (
            {
                "context": lambda x: self._get_relevant_chunks(x["query"]),
                "query": lambda x: x["query"]
            }
            | prompt 
            | model 
            | parser
        )

    async def process_stream(self, query: str, chat_history: list, completion_callback):
        """Generator function that yields streaming responses"""
        full_response = ""
        
        async for chunk in self.chain.astream({"query": query}):
            full_response += chunk
            # print(chunk)
            yield chunk
        
        # After the stream is complete, call the completion callback
        await completion_callback(full_response)

    async def generate_chat_name(self, query: str) -> str:
        """Generate a concise chat name based on the user's query"""
        prompt = ChatPromptTemplate.from_template(
            "Generate a brief (3-6 words) title for a chat that starts with this query: {query}\n"
            "The title should be descriptive but concise. Don't use quotes or punctuation.\n"
            "Just return the title directly, nothing else."
        )
        
        chain = prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()
        return await chain.ainvoke({"query": query})