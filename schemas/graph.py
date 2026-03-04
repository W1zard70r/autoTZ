from typing import List, Optional
from pydantic import BaseModel, Field
from .enums import NodeLabel, EdgeRelation, TZSectionEnum

# ==========================================
# 1. СТРУКТУРЫ ДЛЯ LAYER 1 (MINER)
# ==========================================

class KnowledgeNode(BaseModel):
    """Узел, извлекаемый из сырого текста (Miner)"""
    id: str
    label: NodeLabel
    name: str
    description: str
    properties: List[str] = Field(default_factory=list)

class KnowledgeEdge(BaseModel):
    """Связь, извлекаемая из сырого текста (Miner)"""
    source: str
    target: str
    relation: EdgeRelation
    evidence: str = Field(default="", description="Цитата или обоснование связи")

class ExtractedKnowledge(BaseModel):
    """Результат обработки одного окна чата"""
    summary: str = Field(default="", description="Краткое содержание контекста")
    nodes: List[KnowledgeNode] = Field(default_factory=list)
    edges: List[KnowledgeEdge] = Field(default_factory=list)
    source_ref: Optional[str] = None

# ==========================================
# 2. СТРУКТУРЫ ДЛЯ LAYER 2 (MERGER / UNIFIED)
# ==========================================

class UnifiedNode(BaseModel):
    """Узел объединенного графа (после слияния)"""
    id: str
    label: NodeLabel
    name: str
    description: str
    properties: List[str] = Field(default_factory=list)
    # Поле target_section заполняется в конце Layer 2
    target_section: Optional[TZSectionEnum] = None 

class UnifiedEdge(BaseModel):
    """Ребро объединенного графа"""
    source: str
    target: str
    relation: EdgeRelation
    description: str = ""

class UnifiedGraph(BaseModel):
    """Итоговый граф знаний"""
    nodes: List[UnifiedNode] = Field(default_factory=list)
    edges: List[UnifiedEdge] = Field(default_factory=list)

# ==========================================
# 3. СТРУКТУРЫ ДЛЯ РАЗРЕШЕНИЯ КОНФЛИКТОВ
# ==========================================

class ConflictOption(BaseModel):
    id: str
    text: str

class ConflictSchema(BaseModel):
    """Модель конфликта, возвращаемая LLM"""
    id: str
    description: str
    category: str = "General"
    ai_recommendation: str
    options: List[ConflictOption]

class ConflictResolution(BaseModel):
    """Решение пользователя"""
    conflict_id: str
    selected_option_id: Optional[str] = None
    custom_text: Optional[str] = None