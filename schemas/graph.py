from typing import List, Optional
from pydantic import BaseModel, Field
from .enums import NodeLabel, EdgeRelation, TZSectionEnum


class KeyValue(BaseModel):
    key: str = Field(description="Название свойства")
    value: str = Field(description="Значение свойства")


class GraphNode(BaseModel):
    id: str = Field(description="Snake_case ID сущности")
    label: NodeLabel = Field(description="Тип сущности")
    name: str = Field(description="Человекочитаемое название")
    description: str = Field(default="", description="Описание сущности в контексте проекта")
    properties: List[KeyValue] = Field(default_factory=list, description="Доп. свойства")
    target_section: TZSectionEnum = Field(
        default=TZSectionEnum.UNKNOWN, description="К какой секции ТЗ относится узел"
    )


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
    """Итог голосования по одному варианту."""
    option_id: str
    option_name: str
    votes_for: int = 0
    votes_against: int = 0
    voters_for: List[str] = Field(default_factory=list)    # имена / ID людей
    voters_against: List[str] = Field(default_factory=list)

    @property
    def score(self) -> int:
        return self.votes_for - self.votes_against


class DecisionResolution(BaseModel):
    """Результат разрешения одного Decision-узла."""
    decision_id: str
    decision_name: str
    winner_id: Optional[str] = None          # ID победившего варианта (None = ничья)
    winner_name: Optional[str] = None
    is_tie: bool = False
    options: List[VoteCount] = Field(default_factory=list)
    conflict_description: Optional[str] = None   # заполняется при ничье или споре


class UnifiedGraph(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)
    decisions: List[DecisionResolution] = Field(default_factory=list)  # ← новое