from pydantic import BaseModel
from typing import Optional, Union

class Grant(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    value: Optional[Union[float, int, None]] = None  # Allow both float and int values
    funder: Optional[str] = None
    organisation: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True  # Allows more flexible type handling
        