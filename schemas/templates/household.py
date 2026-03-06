"""Шаблон 2 — Свободный (для бытовых или личных задач)."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from .base import BaseTemplate


class RoomDimensions(BaseModel):
    area: Optional[str] = Field(None, description="Площадь помещения")
    ceiling_height: Optional[str] = Field(None, description="Высота потолков (optional)")
    windows: Optional[str] = Field(None, description="Количество и расположение окон (optional)")
    doors: Optional[str] = Field(None, description="Количество и расположение дверей (optional)")


class MaterialItem(BaseModel):
    name: Optional[str] = Field(None, description="Название материала")
    details: Optional[str] = Field(None, description="Цвет, марка, производитель (optional)")


class DetailsAndPreferences(BaseModel):
    """2. Детали и пожелания"""
    dimensions: RoomDimensions = Field(default_factory=RoomDimensions, description="Размеры помещения")
    style: Optional[str] = Field(None, description="Стиль/дизайн")
    materials: List[MaterialItem] = Field(default_factory=list, description="Материалы (optional)")
    budget: Optional[str] = Field(None, description="Бюджет")


class ExpectedResult(BaseModel):
    """3. Что должно быть в результате"""
    deliverables: List[str] = Field(default_factory=list, description="Конкретный список результатов")
    deadline: Optional[str] = Field(None, description="Срок выполнения")


class SpecialConditions(BaseModel):
    """4. Особые условия"""
    work_schedule: Optional[str] = Field(None, description="Время работы (optional)")
    access: Optional[str] = Field(None, description="Доступ в помещение (optional)")
    cleanup: Optional[str] = Field(None, description="Уборка после работ (optional)")


class HouseholdTemplate(BaseTemplate):
    """Шаблон ТЗ для бытовых и личных задач."""
    task_description: Optional[str] = Field(None, description="Что нужно сделать (краткое описание)")
    details: DetailsAndPreferences = Field(
        default_factory=DetailsAndPreferences, description="Детали и пожелания"
    )
    expected_result: ExpectedResult = Field(
        default_factory=ExpectedResult, description="Ожидаемый результат"
    )
    special_conditions: SpecialConditions = Field(
        default_factory=SpecialConditions, description="Особые условия"
    )
    references: List[str] = Field(default_factory=list, description="Фото/референсы (optional)")

    def to_markdown(self) -> str:
        lines = ["# Техническое задание", ""]
        lines.append("## 1. Что нужно сделать?")
        lines.append(f"{self.task_description or '_требует уточнения_'}")
        lines.append("")

        d = self.details
        lines.append("## 2. Детали и пожелания")
        dim = d.dimensions
        if dim.area:
            lines.append(f"- **Площадь:** {dim.area}")
        if dim.ceiling_height:
            lines.append(f"- **Высота потолков:** {dim.ceiling_height}")
        if dim.windows:
            lines.append(f"- **Окна:** {dim.windows}")
        if dim.doors:
            lines.append(f"- **Двери:** {dim.doors}")
        if d.style:
            lines.append(f"- **Стиль:** {d.style}")
        if d.materials:
            lines.append("- **Материалы:**")
            for m in d.materials:
                lines.append(f"  - {m.name or '—'}: {m.details or '—'}")
        if d.budget:
            lines.append(f"- **Бюджет:** {d.budget}")
        lines.append("")

        er = self.expected_result
        lines.append("## 3. Что должно быть в результате?")
        for item in er.deliverables:
            lines.append(f"- {item}")
        if er.deadline:
            lines.append(f"\n**Срок:** {er.deadline}")
        lines.append("")

        sc = self.special_conditions
        lines.append("## 4. Особые условия")
        if sc.work_schedule:
            lines.append(f"- **Время работы:** {sc.work_schedule}")
        if sc.access:
            lines.append(f"- **Доступ:** {sc.access}")
        if sc.cleanup:
            lines.append(f"- **Уборка:** {sc.cleanup}")
        lines.append("")

        if self.references:
            lines.append("## 5. Фото/референсы")
            for r in self.references:
                lines.append(f"- {r}")

        return "\n".join(lines)
