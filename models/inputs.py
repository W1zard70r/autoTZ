from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field

class DataEnum(str, Enum):
    CHAT = "messages"
    DOCUMENT = "documents"
    PLAIN_TEXT = "plain_text"
    ACTION = "user_action"
    GRAPHML = "graphml"

@dataclass
class DataSource:
    source_type: DataEnum
    content: Any 
    file_name: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

# --- НОВАЯ СТРУКТУРА ДЛЯ СВОЙСТВ ---
class KeyValue(BaseModel):
    key: str = Field(description="Название свойства (например, 'status')")
    value: str = Field(description="Значение свойства (например, 'active')")

class GraphNode(BaseModel):
    id: str = Field(description="ID сущности")
    label: str = Field(description="Тип сущности")
    content: str = Field(default="", description="Текстовое описание") # <-- ДОБАВЛЕНО
    properties: List[KeyValue] = Field(default_factory=list)

class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str

class ExtractedKnowledge(BaseModel):
    summary: str = Field(description="Краткая выжимка", default="") 
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    source_window_ref: str = Field(default="")