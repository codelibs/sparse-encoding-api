from typing import List

from pydantic import BaseModel


class Docs(BaseModel):
    sentences: List[str]
