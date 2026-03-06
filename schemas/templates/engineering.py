"""Шаблон 5 — Инженерный проект."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from .base import BaseTemplate


class BasisAndRegulations(BaseModel):
    """1. Основание и нормативная база"""
    basis: Optional[str] = Field(None, description="Основание для разработки (договор)")
    design_object: Optional[str] = Field(None, description="Объект проектирования")
    design_stages: Optional[str] = Field(None, description="Стадийность проектирования (optional)")
    regulatory_documents: List[str] = Field(default_factory=list, description="Нормативные документы (optional)")


class MainParameters(BaseModel):
    dimensions: Optional[str] = Field(None, description="Габаритные размеры")
    mass: Optional[str] = Field(None, description="Масса (optional)")
    working_environment: Optional[str] = Field(None, description="Рабочая среда (optional)")
    lifespan: Optional[str] = Field(None, description="Ресурс эксплуатации (optional)")


class TechnicalRequirements(BaseModel):
    """2. Технические требования и параметры"""
    main_parameters: MainParameters = Field(
        default_factory=MainParameters, description="Основные параметры"
    )
    construction: List[str] = Field(default_factory=list, description="Конструкционные требования")
    strength: List[str] = Field(default_factory=list, description="Прочностные требования")
    functional: List[str] = Field(default_factory=list, description="Функциональные требования")


class DrivesAndMotors(BaseModel):
    drive_type: Optional[str] = Field(None, description="Тип привода (optional)")
    power: Optional[str] = Field(None, description="Мощность (optional)")
    protection_class: Optional[str] = Field(None, description="Класс защиты (optional)")
    efficiency: Optional[str] = Field(None, description="КПД (optional)")


class ControlSystems(BaseModel):
    automation_level: Optional[str] = Field(None, description="Уровень автоматизации (optional)")
    controller: Optional[str] = Field(None, description="Контроллер (optional)")
    sensors: List[str] = Field(default_factory=list, description="Датчики (optional)")


class SystemsAndComponents(BaseModel):
    """3. Требования к системам и компонентам"""
    drives: DrivesAndMotors = Field(default_factory=DrivesAndMotors, description="Приводы и двигатели (optional)")
    control: ControlSystems = Field(default_factory=ControlSystems, description="Системы управления (optional)")
    safety: List[str] = Field(default_factory=list, description="Безопасность и защита")
    operational: List[str] = Field(default_factory=list, description="Эксплуатационные требования (optional)")


class Documentation(BaseModel):
    """4. Документация и чертежи"""
    composition: List[str] = Field(default_factory=list, description="Состав проектной документации")
    formats_and_standards: Optional[str] = Field(None, description="Форматы и стандарты (optional)")
    drawing_requirements: Optional[str] = Field(None, description="Требования к чертежам (optional)")


class TestingAndAcceptance(BaseModel):
    """5. Испытания и приёмка"""
    test_types: List[str] = Field(default_factory=list, description="Виды испытаний")
    test_methods: Optional[str] = Field(None, description="Методики испытаний (optional)")
    acceptance_criteria: List[str] = Field(default_factory=list, description="Критерии приёмки")


class ProjectStage(BaseModel):
    name: Optional[str] = Field(None, description="Название этапа")
    start_date: Optional[str] = Field(None, description="Дата начала (optional)")
    end_date: Optional[str] = Field(None, description="Дата окончания (optional)")
    description: Optional[str] = Field(None, description="Описание (optional)")


class ResponsiblePerson(BaseModel):
    role: Optional[str] = Field(None, description="Должность/роль")
    name: Optional[str] = Field(None, description="ФИО")
    contacts: Optional[str] = Field(None, description="Контакты (optional)")


class EngineeringTemplate(BaseTemplate):
    """Шаблон ТЗ для инженерного проекта."""
    basis: BasisAndRegulations = Field(
        default_factory=BasisAndRegulations, description="Основание и нормативная база"
    )
    technical: TechnicalRequirements = Field(
        default_factory=TechnicalRequirements, description="Технические требования"
    )
    systems: SystemsAndComponents = Field(
        default_factory=SystemsAndComponents, description="Системы и компоненты"
    )
    documentation: Documentation = Field(
        default_factory=Documentation, description="Документация и чертежи"
    )
    testing: TestingAndAcceptance = Field(
        default_factory=TestingAndAcceptance, description="Испытания и приёмка"
    )
    timeline: List[ProjectStage] = Field(default_factory=list, description="Сроки и этапы")
    responsible_persons: List[ResponsiblePerson] = Field(
        default_factory=list, description="Ответственные лица"
    )

    def to_markdown(self) -> str:
        lines = ["# Техническое задание (Инженерный проект)", ""]

        b = self.basis
        lines.append("## 1. Основание и нормативная база")
        lines.append(f"- **Основание:** {b.basis or '_требует уточнения_'}")
        lines.append(f"- **Объект проектирования:** {b.design_object or '_требует уточнения_'}")
        if b.design_stages:
            lines.append(f"- **Стадийность:** {b.design_stages}")
        if b.regulatory_documents:
            lines.append("- **Нормативные документы:**")
            for doc in b.regulatory_documents:
                lines.append(f"  - {doc}")
        lines.append("")

        t = self.technical
        lines.append("## 2. Технические требования и параметры")
        mp = t.main_parameters
        if mp.dimensions:
            lines.append(f"- **Габариты:** {mp.dimensions}")
        if mp.mass:
            lines.append(f"- **Масса:** {mp.mass}")
        if mp.working_environment:
            lines.append(f"- **Рабочая среда:** {mp.working_environment}")
        if mp.lifespan:
            lines.append(f"- **Ресурс:** {mp.lifespan}")
        if t.construction:
            lines.append("### Конструкционные требования")
            for c in t.construction:
                lines.append(f"- {c}")
        if t.strength:
            lines.append("### Прочностные требования")
            for s in t.strength:
                lines.append(f"- {s}")
        if t.functional:
            lines.append("### Функциональные требования")
            for f in t.functional:
                lines.append(f"- {f}")
        lines.append("")

        sys = self.systems
        lines.append("## 3. Требования к системам и компонентам")
        dr = sys.drives
        if dr.drive_type or dr.power:
            lines.append("### Приводы и двигатели")
            if dr.drive_type:
                lines.append(f"- **Тип:** {dr.drive_type}")
            if dr.power:
                lines.append(f"- **Мощность:** {dr.power}")
            if dr.protection_class:
                lines.append(f"- **Класс защиты:** {dr.protection_class}")
            if dr.efficiency:
                lines.append(f"- **КПД:** {dr.efficiency}")
        ctrl = sys.control
        if ctrl.automation_level or ctrl.controller:
            lines.append("### Системы управления")
            if ctrl.automation_level:
                lines.append(f"- **Уровень автоматизации:** {ctrl.automation_level}")
            if ctrl.controller:
                lines.append(f"- **Контроллер:** {ctrl.controller}")
            if ctrl.sensors:
                lines.append(f"- **Датчики:** {', '.join(ctrl.sensors)}")
        if sys.safety:
            lines.append("### Безопасность")
            for s in sys.safety:
                lines.append(f"- {s}")
        lines.append("")

        d = self.documentation
        lines.append("## 4. Документация и чертежи")
        if d.composition:
            for c in d.composition:
                lines.append(f"- {c}")
        if d.formats_and_standards:
            lines.append(f"- **Форматы:** {d.formats_and_standards}")
        lines.append("")

        ta = self.testing
        lines.append("## 5. Испытания и приёмка")
        if ta.test_types:
            lines.append("### Виды испытаний")
            for tt in ta.test_types:
                lines.append(f"- {tt}")
        if ta.acceptance_criteria:
            lines.append("### Критерии приёмки")
            for ac in ta.acceptance_criteria:
                lines.append(f"- {ac}")
        lines.append("")

        if self.timeline:
            lines.append("## 6. Сроки и этапы")
            for stage in self.timeline:
                dates = ""
                if stage.start_date or stage.end_date:
                    dates = f" ({stage.start_date or '?'} — {stage.end_date or '?'})"
                lines.append(f"- **{stage.name or 'Этап'}**{dates}")
                if stage.description:
                    lines.append(f"  {stage.description}")
            lines.append("")

        if self.responsible_persons:
            lines.append("## 7. Ответственные")
            for rp in self.responsible_persons:
                lines.append(f"- **{rp.role or '—'}:** {rp.name or '—'}, {rp.contacts or '—'}")

        return "\n".join(lines)
