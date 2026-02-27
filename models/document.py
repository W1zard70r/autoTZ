from typing import List
from pydantic import BaseModel, Field
from .enums import TZSectionEnum

class GeneratedSection(BaseModel):
    section_id: TZSectionEnum
    title: str
    content_markdown: str = Field(description="Текст раздела в формате Markdown")
    used_node_ids: List[str] = Field(description="Какие узлы графа использовались (для traceability)", default_factory=list)

class FullTZDocument(BaseModel):
    project_name: str
    version: str
    sections: List[GeneratedSection]