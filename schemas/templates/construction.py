"""Шаблон 4 — Строительный проект."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from .base import BaseTemplate


class GeneralData(BaseModel):
    """1. Общие данные"""
    object_address: Optional[str] = Field(None, description="Объект (адрес, тип здания)")
    work_type: Optional[str] = Field(None, description="Вид работ")
    area: Optional[str] = Field(None, description="Площадь объекта")
    deadline_start: Optional[str] = Field(None, description="Срок начала (optional)")
    deadline_end: Optional[str] = Field(None, description="Срок окончания (optional)")


class ScopeOfWork(BaseModel):
    """2. Объём и перечень работ"""
    preparatory: List[str] = Field(default_factory=list, description="Подготовительные работы")
    main_works: List[str] = Field(default_factory=list, description="Основные работы")
    final_works: List[str] = Field(default_factory=list, description="Финальные работы")


class MaterialItem(BaseModel):
    name: Optional[str] = Field(None, description="Название")
    brand: Optional[str] = Field(None, description="Марка/артикул (optional)")
    quantity: Optional[str] = Field(None, description="Количество (optional)")


class Materials(BaseModel):
    """3. Материалы и оборудование"""
    customer_materials: List[MaterialItem] = Field(default_factory=list, description="Материалы заказчика (optional)")
    contractor_materials: List[MaterialItem] = Field(
        default_factory=list, description="Материалы подрядчика (optional)"
    )
    quality_requirements: List[str] = Field(default_factory=list, description="Требования к качеству (optional)")


class ScheduleAndAccess(BaseModel):
    """4. График работ и доступ"""
    work_days_time: Optional[str] = Field(None, description="Рабочие дни и время")
    holidays: Optional[str] = Field(None, description="Выходные и праздничные дни (optional)")
    access: Optional[str] = Field(None, description="Доступ на объект (optional)")
    storage_location: Optional[str] = Field(None, description="Место складирования (optional)")


class ControlAndAcceptance(BaseModel):
    """5. Контроль и приёмка"""
    intermediate_acceptances: List[str] = Field(default_factory=list, description="Промежуточные приёмки (optional)")
    photo_documentation: Optional[str] = Field(None, description="Фотофиксация процесса (optional)")
    hidden_works_acts: List[str] = Field(default_factory=list, description="Акты скрытых работ (optional)")
    final_acceptance: Optional[str] = Field(None, description="Итоговая приёмка (optional)")


class ResponsiblePerson(BaseModel):
    role: Optional[str] = Field(None, description="Роль")
    name: Optional[str] = Field(None, description="ФИО")
    contacts: Optional[str] = Field(None, description="Контакты (optional)")


class ConstructionTemplate(BaseTemplate):
    """Шаблон ТЗ для строительного проекта."""
    general_data: GeneralData = Field(default_factory=GeneralData, description="Общие данные")
    scope: ScopeOfWork = Field(default_factory=ScopeOfWork, description="Объём и перечень работ")
    materials: Materials = Field(default_factory=Materials, description="Материалы и оборудование")
    schedule: ScheduleAndAccess = Field(default_factory=ScheduleAndAccess, description="График работ и доступ")
    control: ControlAndAcceptance = Field(default_factory=ControlAndAcceptance, description="Контроль и приёмка")
    responsible_persons: List[ResponsiblePerson] = Field(
        default_factory=list, description="Ответственные лица"
    )

    def to_markdown(self) -> str:
        lines = ["# Техническое задание (Строительный проект)", ""]

        gd = self.general_data
        lines.append("## 1. Общие данные")
        lines.append(f"- **Объект:** {gd.object_address or '_требует уточнения_'}")
        lines.append(f"- **Вид работ:** {gd.work_type or '_требует уточнения_'}")
        lines.append(f"- **Площадь:** {gd.area or '_требует уточнения_'}")
        if gd.deadline_start or gd.deadline_end:
            lines.append(f"- **Сроки:** {gd.deadline_start or '?'} — {gd.deadline_end or '?'}")
        lines.append("")

        sc = self.scope
        lines.append("## 2. Объём и перечень работ")
        if sc.preparatory:
            lines.append("### Подготовительные работы")
            for w in sc.preparatory:
                lines.append(f"- {w}")
        if sc.main_works:
            lines.append("### Основные работы")
            for w in sc.main_works:
                lines.append(f"- {w}")
        if sc.final_works:
            lines.append("### Финальные работы")
            for w in sc.final_works:
                lines.append(f"- {w}")
        lines.append("")

        mt = self.materials
        lines.append("## 3. Материалы и оборудование")
        if mt.customer_materials:
            lines.append("### Материалы заказчика")
            for m in mt.customer_materials:
                lines.append(f"- {m.name or '—'} ({m.brand or '—'}) x{m.quantity or '?'}")
        if mt.contractor_materials:
            lines.append("### Материалы подрядчика")
            for m in mt.contractor_materials:
                lines.append(f"- {m.name or '—'} ({m.brand or '—'}) x{m.quantity or '?'}")
        if mt.quality_requirements:
            lines.append("### Требования к качеству")
            for q in mt.quality_requirements:
                lines.append(f"- {q}")
        lines.append("")

        sch = self.schedule
        lines.append("## 4. График работ и доступ")
        if sch.work_days_time:
            lines.append(f"- **Рабочее время:** {sch.work_days_time}")
        if sch.access:
            lines.append(f"- **Доступ:** {sch.access}")
        if sch.storage_location:
            lines.append(f"- **Складирование:** {sch.storage_location}")
        lines.append("")

        ctrl = self.control
        lines.append("## 5. Контроль и приёмка")
        if ctrl.intermediate_acceptances:
            for ia in ctrl.intermediate_acceptances:
                lines.append(f"- {ia}")
        if ctrl.final_acceptance:
            lines.append(f"- **Итоговая приёмка:** {ctrl.final_acceptance}")
        lines.append("")

        if self.responsible_persons:
            lines.append("## 6. Ответственные лица")
            for rp in self.responsible_persons:
                lines.append(f"- **{rp.role or '—'}:** {rp.name or '—'}, {rp.contacts or '—'}")

        return "\n".join(lines)
