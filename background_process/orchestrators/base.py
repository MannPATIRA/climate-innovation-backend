from abc import ABC, abstractmethod
from ..fetchers.base import Fetcher
from ..processors.base import Processor
from ..processors.summary_processor import SummaryProcessor
from typing import Optional


class Orchestrator(ABC):

    def __init__(
            self,
            fetcher: Fetcher,
            processor: Processor,
            summarizer: Optional[SummaryProcessor] = None
    ):
        self.fetcher = fetcher
        self.processor = processor
        self.summarizer = summarizer

    @abstractmethod
    def run(self):
        """Main process to orchestrate the ingestion and vectorising of documents""" 