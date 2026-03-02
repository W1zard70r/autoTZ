from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from .enums import NodeLabel, EdgeRelation, TZSectionEnum

# ─────────────────────────────────────────────
# НОВЫЕ МОДЕЛИ ДЛЯ УЛУЧШЕННОГО LAYER 1
# ─────────────────────────────────────────────

class RawEntity(BaseModel):
    """Шаг 1: сырые сущности без ограничений по ID"""
    name: str = Field(description="Человекочитаемое название")
    label: NodeLabel = Field(description="Тип сущности")
    description: str = Field(default="", description="Краткое описание")

class RawEntitiesSchema(BaseModel):
    entities: List[RawEntity] = Field(default_factory=list)

class MergeDecision(BaseModel):
    is_duplicate: bool = Field(description="Это одна и та же сущность?")
    target_global_id: Optional[str] = Field(description="ID из глобального глоссария, если дубликат")
    new_id: Optional[str] = Field(description="Новый snake_case ID, если не дубликат")

class ProjectMemory(BaseModel):
    """Структурированная память между окнами"""
    key_entities: List[str] = Field(default_factory=list, description="Важные ID, упомянутые недавно")
    open_decisions: List[str] = Field(default_factory=list, description="Открытые Decision ID")
    resolved_decisions: List[Dict[str, str]] = Field(default_factory=list)
    last_mentioned: Dict[str, str] = Field(default_factory=dict, description="id → 'окно X'")

class GraphFix(BaseModel):
    action: str = Field(description="add_node | remove_node | change_relation | fix_vote")
    node_id: Optional[str] = None
    edge_source: Optional[str] = None
    edge_target: Optional[str] = None
    new_value: Optional[str] = None
    reason: str = Field(description="Почему нужно исправить")

class FixListSchema(BaseModel):
    fixes: List[GraphFix] = Field(default_factory=list)

# ─────────────────────────────────────────────
# СТАРЫЕ МОДЕЛИ (без изменений)
# ─────────────────────────────────────────────
class KeyValue(BaseModel):
    key: str = Field(description="Название свойства")
    value: str = Field(description="Значение свойства")

class GraphNode(BaseModel):
    id: str = Field(description="Snake_case ID сущности")
    label: NodeLabel = Field(description="Тип сущности")
    name: str = Field(description="Человекочитаемое название")
    description: str = Field(default="", description="Описание сущности в контексте проекта")
    properties: List[KeyValue] = Field(default_factory=list)
    target_section: TZSectionEnum = Field(default=TZSectionEnum.UNKNOWN)

class GraphEdge(BaseModel):
    source: str = Field(description="ID исходного узла")
    target: str = Field(description="ID целевого узла")
    relation: EdgeRelation = Field(description="Тип связи")
    evidence: str = Field(default="", description="Цитата или обоснование связи")

class ExtractedKnowledge(BaseModel):
    summary: str = Field(description="Краткая выжимка чанка/окна", default="")
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    source_ref: str = Field(default="")

class Conflict(BaseModel):
    id: str = Field(default="unknown_conflict")
    node_id: str = Field(default="")
    conflicting_values: List[str] = Field(default_factory=list)
    description: str

class VoteCount(BaseModel):
    option_id: str
    option_name: str
    votes_for: int = 0
    votes_against: int = 0
    voters_for: List[str] = Field(default_factory=list)
    voters_against: List[str] = Field(default_factory=list)

    @property
    def score(self) -> int:
        return self.votes_for - self.votes_against

class DecisionResolution(BaseModel):
    decision_id: str
    decision_name: str
    winner_id: Optional[str] = None
    winner_name: Optional[str] = None
    is_tie: bool = False
    options: List[VoteCount] = Field(default_factory=list)
    conflict_description: Optional[str] = None

class UnifiedGraph(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)
    decisions: List[DecisionResolution] = Field(default_factory=list)