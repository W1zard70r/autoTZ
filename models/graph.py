from typing import List
from pydantic import BaseModel, Field
from .enums import TZSectionEnum

class GraphNode(BaseModel):
    id: str
    label: str
    target_section: TZSectionEnum = Field(default=TZSectionEnum.GENERAL)
    content: str
    sources: List[str] = Field(default_factory=list)

class Conflict(BaseModel):
    id: str = Field(default="unknown_conflict")
    node_id: str = Field(default="")
    conflicting_values: List[str] = Field(default_factory=list)
    description: str

class UnifiedGraph(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)