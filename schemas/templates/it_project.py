"""Шаблон 3 — IT-проект."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from .base import BaseTemplate


class GeneralDescription(BaseModel):
    """1. Общее описание проекта"""
    purpose: Optional[str] = Field(None, description="Назначение системы")
    analogues: List[str] = Field(default_factory=list, description="Аналоги/конкуренты (optional)")
    expected_outcome: Optional[str] = Field(None, description="Что будет в итоге")


class UserRole(BaseModel):
    name: Optional[str] = Field(None, description="Название роли")
    capabilities: List[str] = Field(default_factory=list, description="Возможности роли")


class UseCase(BaseModel):
    name: Optional[str] = Field(None, description="Название сценария")
    steps: List[str] = Field(default_factory=list, description="Шаги сценария")


class FunctionalRequirements(BaseModel):
    """2. Функциональные требования"""
    roles: List[UserRole] = Field(default_factory=list, description="Пользовательские роли")
    use_cases: List[UseCase] = Field(default_factory=list, description="Основные сценарии использования")
    modules: List[str] = Field(default_factory=list, description="Список модулей/страниц")


class DesignSystem(BaseModel):
    colors: Optional[str] = Field(None, description="Цветовая схема (optional)")
    fonts: Optional[str] = Field(None, description="Шрифты (optional)")
    style: Optional[str] = Field(None, description="Стиль интерфейса (optional)")


class Responsiveness(BaseModel):
    mobile: Optional[str] = Field(None, description="Мобильные устройства (optional)")
    desktop: Optional[str] = Field(None, description="Десктоп (optional)")
    tablet: Optional[str] = Field(None, description="Планшеты (optional)")


class UIUXRequirements(BaseModel):
    """3. Требования к интерфейсу"""
    design_system: DesignSystem = Field(default_factory=DesignSystem, description="Дизайн-система")
    responsiveness: Responsiveness = Field(default_factory=Responsiveness, description="Адаптивность")


class TechStack(BaseModel):
    frontend: Optional[str] = Field(None, description="Фронтенд технологии")
    backend: Optional[str] = Field(None, description="Бэкенд технологии")
    database: Optional[str] = Field(None, description="База данных")
    other: List[str] = Field(default_factory=list, description="Прочие технологии (optional)")


class Integration(BaseModel):
    name: Optional[str] = Field(None, description="Название интеграции")
    details: Optional[str] = Field(None, description="Детали (optional)")


class TechnicalRequirements(BaseModel):
    """4. Технические требования"""
    tech_stack: TechStack = Field(default_factory=TechStack, description="Стек технологий")
    integrations: List[Integration] = Field(default_factory=list, description="Интеграции (optional)")
    performance: List[str] = Field(default_factory=list, description="Требования к производительности (optional)")


class NonFunctionalRequirements(BaseModel):
    """5. Нефункциональные требования"""
    security: List[str] = Field(default_factory=list, description="Безопасность")
    scalability: List[str] = Field(default_factory=list, description="Масштабируемость (optional)")
    response_time: List[str] = Field(default_factory=list, description="Время отклика (optional)")


class DevelopmentPhase(BaseModel):
    name: Optional[str] = Field(None, description="Название фазы")
    deadline: Optional[str] = Field(None, description="Срок (optional)")
    features: List[str] = Field(default_factory=list, description="Функции (optional)")


class DevelopmentStages(BaseModel):
    """6. Этапы разработки и сроки"""
    mvp: DevelopmentPhase = Field(default_factory=DevelopmentPhase, description="Минимальная версия")
    release: DevelopmentPhase = Field(default_factory=DevelopmentPhase, description="Основной релиз (optional)")
    support: Optional[str] = Field(None, description="Поддержка и доработки (optional)")


class AcceptanceCriteria(BaseModel):
    """7. Критерии приёмки"""
    testing: List[str] = Field(default_factory=list, description="Тестирование")
    documentation: List[str] = Field(default_factory=list, description="Документация")


class ITProjectTemplate(BaseTemplate):
    """Шаблон ТЗ для IT-проекта."""
    general: GeneralDescription = Field(
        default_factory=GeneralDescription, description="Общее описание проекта"
    )
    functional: FunctionalRequirements = Field(
        default_factory=FunctionalRequirements, description="Функциональные требования"
    )
    ui_ux: UIUXRequirements = Field(
        default_factory=UIUXRequirements, description="Требования к интерфейсу"
    )
    technical: TechnicalRequirements = Field(
        default_factory=TechnicalRequirements, description="Технические требования"
    )
    non_functional: NonFunctionalRequirements = Field(
        default_factory=NonFunctionalRequirements, description="Нефункциональные требования"
    )
    stages: DevelopmentStages = Field(
        default_factory=DevelopmentStages, description="Этапы разработки"
    )
    acceptance: AcceptanceCriteria = Field(
        default_factory=AcceptanceCriteria, description="Критерии приёмки"
    )

    def to_markdown(self) -> str:
        lines = ["# Техническое задание (IT-проект)", ""]

        g = self.general
        lines.append("## 1. Общее описание проекта")
        lines.append(f"- **Назначение системы:** {g.purpose or '_требует уточнения_'}")
        if g.analogues:
            lines.append(f"- **Аналоги/конкуренты:** {', '.join(g.analogues)}")
        lines.append(f"- **Итоговый продукт:** {g.expected_outcome or '_требует уточнения_'}")
        lines.append("")

        f = self.functional
        lines.append("## 2. Функциональные требования")
        if f.roles:
            lines.append("### Пользовательские роли")
            for role in f.roles:
                lines.append(f"**{role.name or 'Роль'}:**")
                for cap in role.capabilities:
                    lines.append(f"- {cap}")
        if f.use_cases:
            lines.append("### Основные сценарии использования")
            for uc in f.use_cases:
                lines.append(f"**{uc.name or 'Сценарий'}:**")
                for i, step in enumerate(uc.steps, 1):
                    lines.append(f"{i}. {step}")
        if f.modules:
            lines.append("### Модули и страницы")
            for m in f.modules:
                lines.append(f"- {m}")
        lines.append("")

        u = self.ui_ux
        lines.append("## 3. Требования к интерфейсу (UI/UX)")
        ds = u.design_system
        if ds.colors:
            lines.append(f"- **Цвета:** {ds.colors}")
        if ds.fonts:
            lines.append(f"- **Шрифты:** {ds.fonts}")
        if ds.style:
            lines.append(f"- **Стиль:** {ds.style}")
        resp = u.responsiveness
        if resp.mobile:
            lines.append(f"- **Мобильные:** {resp.mobile}")
        if resp.desktop:
            lines.append(f"- **Десктоп:** {resp.desktop}")
        if resp.tablet:
            lines.append(f"- **Планшеты:** {resp.tablet}")
        lines.append("")

        t = self.technical
        lines.append("## 4. Технические требования")
        ts = t.tech_stack
        lines.append("### Стек технологий")
        lines.append(f"- **Фронтенд:** {ts.frontend or '_требует уточнения_'}")
        lines.append(f"- **Бэкенд:** {ts.backend or '_требует уточнения_'}")
        lines.append(f"- **База данных:** {ts.database or '_требует уточнения_'}")
        for other in ts.other:
            lines.append(f"- {other}")
        if t.integrations:
            lines.append("### Интеграции")
            for intg in t.integrations:
                lines.append(f"- **{intg.name or '—'}:** {intg.details or '—'}")
        if t.performance:
            lines.append("### Производительность")
            for p in t.performance:
                lines.append(f"- {p}")
        lines.append("")

        nf = self.non_functional
        lines.append("## 5. Нефункциональные требования")
        if nf.security:
            lines.append("### Безопасность")
            for s in nf.security:
                lines.append(f"- {s}")
        if nf.scalability:
            lines.append("### Масштабируемость")
            for s in nf.scalability:
                lines.append(f"- {s}")
        if nf.response_time:
            lines.append("### Время отклика")
            for r in nf.response_time:
                lines.append(f"- {r}")
        lines.append("")

        st = self.stages
        lines.append("## 6. Этапы разработки и сроки")
        if st.mvp.name or st.mvp.features:
            lines.append(f"### {st.mvp.name or 'Минимальная версия (MVP)'}")
            if st.mvp.deadline:
                lines.append(f"- **Срок:** {st.mvp.deadline}")
            for feat in st.mvp.features:
                lines.append(f"- {feat}")
        if st.release.name or st.release.features:
            lines.append(f"### {st.release.name or 'Основной релиз'}")
            if st.release.deadline:
                lines.append(f"- **Срок:** {st.release.deadline}")
            for feat in st.release.features:
                lines.append(f"- {feat}")
        if st.support:
            lines.append(f"### Поддержка\n{st.support}")
        lines.append("")

        a = self.acceptance
        lines.append("## 7. Критерии приёмки")
        if a.testing:
            lines.append("### Тестирование")
            for t_item in a.testing:
                lines.append(f"- {t_item}")
        if a.documentation:
            lines.append("### Документация")
            for d in a.documentation:
                lines.append(f"- {d}")

        return "\n".join(lines)
