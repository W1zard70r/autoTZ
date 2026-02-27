from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field
from .enums import TZSectionEnum, DataEnum

class DataSource(BaseModel):
    source_type: DataEnum
    content: Any 
    file_name: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class GeneratedSection(BaseModel):
    section_id: TZSectionEnum
    title: str
    content_markdown: str = Field(description="Текст раздела в формате Markdown")
    used_node_ids: List[str] = Field(description="ID использованных узлов", default_factory=list)

class FullTZDocument(BaseModel):
    project_name: str
    version: str
    sections: List[GeneratedSection]