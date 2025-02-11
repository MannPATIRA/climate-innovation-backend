from abc import ABC, abstractmethod
from typing import Generator, Any


class Fetcher(ABC):

    @abstractmethod
    def fetch(self, **kwargs) -> Generator[Any, None, None]:
        """
        Yields one path / url at a time.
        This allows for processing one document at a time and cleaning up after.
        """
        pass 