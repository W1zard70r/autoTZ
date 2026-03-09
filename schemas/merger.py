from typing import List
from pydantic import BaseModel, Field

from schemas.enums import TZSectionEnum
from schemas.graph import DetectedConflict


class MergeAction(BaseModel):
    is_duplicate: bool = Field(description="Это одна и та же сущность?")
    ids_to_merge: List[str] = Field(description="Список ID, которые нужно слить в один")
    unified_id: str = Field(description="Новый ID для слитого узла")
    unified_name: str = Field(description="Общее имя")
    unified_desc: str = Field(description="Объединенное описание")


class MergeBatchResult(BaseModel):
    actions: List[MergeAction] = Field(default_factory=list)


class SectionAssignment(BaseModel):
    node_id: str
    target_section: TZSectionEnum


class SectionBatchResult(BaseModel):
    assignments: List[SectionAssignment]


class ConflictOption(BaseModel):
    id: str
    name: str
    description: str


class ConflictBatchResult(BaseModel):
    conflicts: List[DetectedConflict] = Field(default_factory=list)


class ConflictResolution(BaseModel):
    conflict_id: str
    selected_option_id: str | None = None
    custom_text: str | None = None