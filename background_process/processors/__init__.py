from .base import Processor, ProcessingTask
from .report_processor import ReportProcessor, PDFDocument
from .paper_processor import PaperProcessor, Paper
from .topic_processor import TopicProcessor
from ..prompts import TopicAssessment

__all__ = [
    'Processor',
    'ProcessingTask',
    'ReportProcessor',
    'PDFDocument',
    'PaperProcessor',
    'Paper',
    'TopicProcessor',   
    'TopicAssessment'
] 
