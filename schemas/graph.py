from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from .enums import NodeLabel, EdgeRelation, TZSectionEnum


class RawEntity(BaseModel):
    name: str = Field(description="Human-readable name")
    label: NodeLabel = Field(description="Entity type")
    description: str = Field(default="", description="Brief description")


class RawEntitiesSchema(BaseModel):
    entities: List[RawEntity] = Field(default_factory=list)


class LinkDecision(BaseModel):
    original_name: str = Field(description="Original entity name from extraction")
    is_duplicate: bool = Field(description="Is this entity already in glossary?")
    target_global_id: Optional[str] = Field(default=None, description="Glossary ID if duplicate")
    new_id: Optional[str] = Field(default=None, description="New snake_case ID if not duplicate")


class BatchLinkResult(BaseModel):
    decisions: List[LinkDecision] = Field(default_factory=list)


class MergeDecision(BaseModel):
    is_duplicate: bool = Field(description="Is this the same entity?")
    target_global_id: Optional[str] = Field(description="Glossary ID if duplicate")
    new_id: Optional[str] = Field(description="New snake_case ID if not duplicate")


class KeyValue(BaseModel):
    key: str = Field(description="Property name")
    value: str = Field(description="Property value")


class GraphNode(BaseModel):
    id: str = Field(description="Snake_case entity ID")
    label: NodeLabel = Field(description="Entity type")
    name: str = Field(description="Human-readable name")
    description: str = Field(default="", description="Entity description in project context")
    properties: List[KeyValue] = Field(default_factory=list, description="Additional properties")
    target_section: TZSectionEnum = Field(default=TZSectionEnum.UNKNOWN, description="Target TZ section")


class GraphEdge(BaseModel):
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    relation: EdgeRelation = Field(description="Relationship type")
    evidence: str = Field(default="", description="Quote or justification for the relationship")


class ExtractedKnowledge(BaseModel):
    summary: str = Field(description="Brief summary of the chunk/window", default="")
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


class ConflictOption(BaseModel):
    id: str = Field(description="Option ID (e.g. node 'react_ui')")
    text: str = Field(description="Option text (e.g. 'React')")
    evidence: str = Field(description="Who proposed and why")


class DetectedConflict(BaseModel):
    id: str = Field(description="Unique conflict ID")
    category: str = Field(description="Category (e.g. 'Frontend Framework')")
    description: str = Field(description="Conflict description")
    options: List[ConflictOption] = Field(description="Conflicting options")
    ai_recommendation: str = Field(description="AI recommendation")


class ConflictResolution(BaseModel):
    conflict_id: str
    selected_option_id: Optional[str] = None
    custom_text: Optional[str] = None
