from .base import (
    BaseTemplate, FieldGap, FieldConflict, ValidationResult, TZResult,
)
from .gost import GostTemplate
from .household import HouseholdTemplate
from .it_project import ITProjectTemplate
from .construction import ConstructionTemplate
from .engineering import EngineeringTemplate
from schemas.enums import TemplateType

TEMPLATE_REGISTRY: dict[TemplateType, type[BaseTemplate]] = {
    TemplateType.GOST: GostTemplate,
    TemplateType.HOUSEHOLD: HouseholdTemplate,
    TemplateType.IT_PROJECT: ITProjectTemplate,
    TemplateType.CONSTRUCTION: ConstructionTemplate,
    TemplateType.ENGINEERING: EngineeringTemplate,
}


def get_template_class(template_type: TemplateType) -> type[BaseTemplate]:
    return TEMPLATE_REGISTRY[template_type]
