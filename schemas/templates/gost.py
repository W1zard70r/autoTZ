"""Шаблон 1 — Техническое задание (формальный по ГОСТ)."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from .base import BaseTemplate


class ContactInfo(BaseModel):
    organization: Optional[str] = Field(None, description="Организация")
    contact_person: Optional[str] = Field(None, description="Контактное лицо")
    phone: Optional[str] = Field(None, description="Телефон")
    email: Optional[str] = Field(None, description="Email")


class Introduction(BaseModel):
    """1. Введение"""
    project_name: Optional[str] = Field(None, description="Наименование проекта")
    basis: Optional[str] = Field(None, description="Основание для разработки")
    customer: ContactInfo = Field(default_factory=ContactInfo, description="Заказчик")
    contractor: ContactInfo = Field(default_factory=ContactInfo, description="Исполнитель")


class GoalAndRelevance(BaseModel):
    """2. Цель и актуальность проекта"""
    goal: Optional[str] = Field(None, description="Цель создания проекта")
    relevance: Optional[str] = Field(None, description="Актуальность проекта")


class FunctionalRequirementsList(BaseModel):
    mandatory: List[str] = Field(default_factory=list, description="Обязательные требования")
    optional: List[str] = Field(default_factory=list, description="Желательные требования (optional)")
    deferred: List[str] = Field(default_factory=list, description="Отложенные требования (optional)")


class ProjectRequirements(BaseModel):
    """3. Требования к проекту"""
    functional: FunctionalRequirementsList = Field(
        default_factory=FunctionalRequirementsList, description="Функциональные требования"
    )
    technical: List[str] = Field(default_factory=list, description="Технические требования")
    reliability_security: List[str] = Field(default_factory=list, description="Требования к надёжности и безопасности")
    scalability: Optional[str] = Field(None, description="Масштабируемость")


class ProjectStage(BaseModel):
    name: Optional[str] = Field(None, description="Название этапа")
    start_date: Optional[str] = Field(None, description="Дата начала (optional)")
    end_date: Optional[str] = Field(None, description="Дата окончания (optional)")
    deliverables: List[str] = Field(default_factory=list, description="Результаты этапа (optional)")


class StagesAndTimeline(BaseModel):
    """4. Стадии и этапы выполнения"""
    stages: List[ProjectStage] = Field(default_factory=list, description="Перечень этапов")
    checkpoints: List[str] = Field(default_factory=list, description="Промежуточные контрольные точки (optional)")


class Acceptance(BaseModel):
    """5. Порядок приёмки и сдачи"""
    criteria: List[str] = Field(default_factory=list, description="Критерии приёмки")
    documents: List[str] = Field(default_factory=list, description="Документы для сдачи")


class GostTemplate(BaseTemplate):
    """Шаблон ТЗ по ГОСТ (формальный)."""
    introduction: Introduction = Field(default_factory=Introduction, description="Введение")
    goal_and_relevance: GoalAndRelevance = Field(
        default_factory=GoalAndRelevance, description="Цель и актуальность"
    )
    requirements: ProjectRequirements = Field(
        default_factory=ProjectRequirements, description="Требования к проекту"
    )
    stages: StagesAndTimeline = Field(default_factory=StagesAndTimeline, description="Стадии и этапы")
    acceptance: Acceptance = Field(default_factory=Acceptance, description="Порядок приёмки")
    appendices: List[str] = Field(default_factory=list, description="Приложения (optional)")

    def to_markdown(self) -> str:
        lines = ["# Техническое задание", ""]
        intro = self.introduction
        lines.append("## 1. Введение")
        lines.append(f"- **Наименование проекта:** {intro.project_name or '_требует уточнения_'}")
        lines.append(f"- **Основание для разработки:** {intro.basis or '_требует уточнения_'}")
        if intro.customer:
            lines.append(f"- **Заказчик:** {intro.customer.organization or '—'}, "
                         f"{intro.customer.contact_person or '—'}, "
                         f"{intro.customer.phone or '—'}, {intro.customer.email or '—'}")
        if intro.contractor:
            lines.append(f"- **Исполнитель:** {intro.contractor.organization or '—'}, "
                         f"{intro.contractor.contact_person or '—'}, "
                         f"{intro.contractor.phone or '—'}, {intro.contractor.email or '—'}")
        lines.append("")

        g = self.goal_and_relevance
        lines.append("## 2. Цель и актуальность проекта")
        lines.append(f"- **Цель:** {g.goal or '_требует уточнения_'}")
        lines.append(f"- **Актуальность:** {g.relevance or '_требует уточнения_'}")
        lines.append("")

        r = self.requirements
        lines.append("## 3. Требования к проекту")
        lines.append("### Функциональные требования")
        if r.functional.mandatory:
            lines.append("**Обязательные:**")
            for item in r.functional.mandatory:
                lines.append(f"- {item}")
        if r.functional.optional:
            lines.append("**Желательные:**")
            for item in r.functional.optional:
                lines.append(f"- {item}")
        if r.functional.deferred:
            lines.append("**Отложенные:**")
            for item in r.functional.deferred:
                lines.append(f"- {item}")
        if r.technical:
            lines.append("### Технические требования")
            for item in r.technical:
                lines.append(f"- {item}")
        if r.reliability_security:
            lines.append("### Надёжность и безопасность")
            for item in r.reliability_security:
                lines.append(f"- {item}")
        lines.append(f"### Масштабируемость\n{r.scalability or '_требует уточнения_'}")
        lines.append("")

        s = self.stages
        lines.append("## 4. Стадии и этапы выполнения")
        for stage in s.stages:
            dates = ""
            if stage.start_date or stage.end_date:
                dates = f" ({stage.start_date or '?'} — {stage.end_date or '?'})"
            lines.append(f"- **{stage.name or 'Этап'}**{dates}")
            for d in stage.deliverables:
                lines.append(f"  - {d}")
        if s.checkpoints:
            lines.append(f"**Контрольные точки:** {', '.join(s.checkpoints)}")
        lines.append("")

        a = self.acceptance
        lines.append("## 5. Порядок приёмки и сдачи")
        if a.criteria:
            lines.append("**Критерии приёмки:**")
            for c in a.criteria:
                lines.append(f"- {c}")
        if a.documents:
            lines.append("**Документы для сдачи:**")
            for d in a.documents:
                lines.append(f"- {d}")
        lines.append("")

        if self.appendices:
            lines.append("## 6. Приложения")
            for ap in self.appendices:
                lines.append(f"- {ap}")

        return "\n".join(lines)
