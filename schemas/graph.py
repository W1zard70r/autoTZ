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
    target_section: TZSectionEnum = Field(default=TZSectionEnum.UNKNOWN, description="К какой секции ТЗ относится узел")

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

class UnifiedGraph(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)
    conflicts: List[Conflict] = Field(default_factory=list)

# Классы для конфликтов
class ConflictOption(BaseModel):
    id: str = Field(description="ID варианта (например, узел 'react_ui')")
    text: str = Field(description="Текст варианта (например, 'React')")
    evidence: str = Field(description="Кто и почему это предложил (например, 'Ivan: React быстрее')")

class DetectedConflict(BaseModel):
    id: str = Field(description="Уникальный ID конфликта")
    category: str = Field(description="Категория (например, 'Frontend Framework')")
    description: str = Field(description="Суть конфликта (Выбор между React и Vue)")
    options: List[ConflictOption] = Field(description="Варианты 1 и 2 (и др.)")
    ai_recommendation: str = Field(description="Вариант 3: Совет нейросети")
    
class ConflictResolution(BaseModel):
    conflict_id: str
    selected_option_id: Optional[str] = None # Если выбран один из существующих
    custom_text: Optional[str] = None # Вариант 4: Свой вариант