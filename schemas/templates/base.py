from __future__ import annotations
from typing import List, Optional, Any
from pydantic import BaseModel, Field

from schemas.enums import TemplateType


class FieldGap(BaseModel):
    """Незаполненное обязательное поле шаблона."""
    field_path: str = Field(description="Путь к полю (dot-notation)")
    field_name: str = Field(description="Название поля для пользователя")
    description: str = Field(description="Что именно нужно указать")
    section: str = Field(description="Раздел шаблона")


class FieldConflict(BaseModel):
    """Конфликтующие данные для поля."""
    field_path: str = Field(description="Путь к полю")
    field_name: str = Field(description="Название поля для пользователя")
    options: List[str] = Field(default_factory=list, description="Противоречивые значения")
    description: str = Field(description="Описание конфликта")
    section: str = Field(description="Раздел шаблона")


class ValidationResult(BaseModel):
    """Результат валидации заполненности шаблона."""
    is_complete: bool = False
    gaps: List[FieldGap] = Field(default_factory=list)
    conflicts: List[FieldConflict] = Field(default_factory=list)
    completeness_percent: float = 0.0
    total_fields: int = 0
    filled_fields: int = 0


class BaseTemplate(BaseModel):
    """Базовый класс для всех шаблонов ТЗ."""

    def validate_completeness(self) -> ValidationResult:
        """Проверяет заполненность всех обязательных полей."""
        gaps = []
        total, filled = self._walk_fields(self, "", gaps)
        pct = (filled / total * 100) if total > 0 else 0.0
        return ValidationResult(
            is_complete=len(gaps) == 0,
            gaps=gaps,
            completeness_percent=round(pct, 1),
            total_fields=total,
            filled_fields=filled,
        )

    def _walk_fields(
        self, obj: BaseModel, prefix: str, gaps: list[FieldGap],
    ) -> tuple[int, int]:
        total = 0
        filled = 0
        for name, field_info in obj.model_fields.items():
            path = f"{prefix}.{name}" if prefix else name
            value = getattr(obj, name, None)
            description = field_info.description or name

            if isinstance(value, BaseModel):
                t, f = self._walk_fields(value, path, gaps)
                total += t
                filled += f
            elif isinstance(value, list):
                total += 1
                if value:
                    filled += 1
                else:
                    if "optional" not in (field_info.description or "").lower():
                        gaps.append(FieldGap(
                            field_path=path,
                            field_name=description,
                            description=f"Необходимо заполнить: {description}",
                            section=prefix.split(".")[0] if prefix else name,
                        ))
            else:
                total += 1
                if value is not None and str(value).strip():
                    filled += 1
                else:
                    if "optional" not in (field_info.description or "").lower():
                        gaps.append(FieldGap(
                            field_path=path,
                            field_name=description,
                            description=f"Необходимо заполнить: {description}",
                            section=prefix.split(".")[0] if prefix else name,
                        ))
        return total, filled

    def to_markdown(self) -> str:
        raise NotImplementedError


class TZResult(BaseModel):
    """Итоговый результат генерации ТЗ."""
    template_type: TemplateType
    template_data: Any = Field(description="Заполненный шаблон (BaseTemplate)")
    validation: ValidationResult = Field(default_factory=ValidationResult)
    markdown: str = Field(default="", description="Отрендеренный Markdown")
