from dataclasses import dataclass
from typing import Any, List, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field

class DataEnum(Enum):
    CHAT = "messages"
    DOCUMENT = "documents"
    PLAIN_TEXT = "plain_text"

@dataclass
class DataSource:
    source_type: DataEnum
    content: Any
    file_name: str
    metadata: Optional[Dict[str, Any]] = None

# --- СТРОГАЯ ОНТОЛОГИЯ ---
class NodeLabel(str, Enum):
    PERSON = "Person"
    COMPONENT = "Component"
    TASK = "Task"
    REQUIREMENT = "Requirement"
    CONCEPT = "Concept"

class EdgeRelation(str, Enum):
    ASSIGNED_TO = "ASSIGNED_TO"
    DEPENDS_ON = "DEPENDS_ON"
    RELATES_TO = "RELATES_TO"
    AGREES_WITH = "AGREES_WITH"
    MENTIONS = "MENTIONS"

# --- СХЕМЫ ДЛЯ LLM ---
class GlossaryItem(BaseModel):
    id: str = Field(description="Snake_case ID сущности (например: user_grisha, db_postgres)")
    name: str = Field(description="Человекочитаемое название")
    label: NodeLabel = Field(description="Тип сущности")
    description: str = Field(description="Краткое описание из контекста")

class ProjectGlossary(BaseModel):
    entities: List[GlossaryItem] = Field(default_factory=list, description="Список найденных сущностей")

class GraphNode(BaseModel):
    id: str = Field(description="ID из глоссария")
    label: NodeLabel
    properties: Dict[str, str] = Field(default_factory=dict, description="Свойства (статус, детали). Ключи и значения - строки.")

class GraphEdge(BaseModel):
    source: str = Field(description="ID исходного узла")
    target: str = Field(description="ID целевого узла")
    relation: EdgeRelation
    evidence: str = Field(description="Цитата, подтверждающая связь")

class WindowExtractionResult(BaseModel):
    summary: str = Field(description="Суммаризация текущего окна")
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)