from .base import Fetcher
from .report_fetcher import ReportFetcher, LocalPDFFetcher, WebScrapingReportFetcher
from .paper_fetcher import PaperFetcher, PyAlexFetcher
from .topic_fetcher import TopicFetcher
from .author_fetcher import AuthorFetcher, PyAlexAuthorFetcher
__all__ = [
    'Fetcher',
    'ReportFetcher',
    'LocalPDFFetcher',
    'PaperFetcher',
    'PyAlexFetcher',
    'TopicFetcher',
    'WebScrapingReportFetcher',
    'AuthorFetcher',
    'PyAlexAuthorFetcher'
] 