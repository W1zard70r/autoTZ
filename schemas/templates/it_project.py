"""Шаблон 3 — IT-проект."""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from .base import BaseTemplate


class GeneralDescription(BaseModel):
    """1. Общее описание проекта"""
    purpose: Optional[str] = Field(
        None, 
        description="Сформулируй главную бизнес-цель и назначение системы. Какую проблему решает продукт? Для кого он создается? Напиши профессиональным техническим языком, 1-3 абзаца."
    )
    analogues: List[str] = Field(
        default_factory=list, 
        description="Перечисли известные аналоги или конкурентов, упомянутых в контексте. Если есть возможность, кратко укажи их отличия от разрабатываемой системы. Если данных нет, оставь пустым."
    )
    expected_outcome: Optional[str] = Field(
        None, 
        description="Опиши итоговый результат проекта (deliverable). Что конкретно будет передано заказчику? (например: веб-приложение, мобильное приложение, API-сервис, библиотека). Укажи ключевую ценность результата."
    )


class UserRole(BaseModel):
    name: Optional[str] = Field(
        None, 
        description="Название роли пользователя (например: 'Администратор', 'Анонимный гость', 'Менеджер')."
    )
    capabilities: List[str] = Field(
        default_factory=list, 
        description="Список прав и возможностей этой роли. Используй глаголы действий (например: 'Просматривает отчеты', 'Управляет правами доступа')."
    )


class UseCase(BaseModel):
    name: Optional[str] = Field(
        None, 
        description="Понятное название пользовательского сценария (например: 'Оформление заказа', 'Регистрация в системе')."
    )
    steps: List[str] = Field(
        default_factory=list, 
        description="Пошаговое описание сценария: от действия пользователя до реакции системы. Напиши шаги последовательно (1. Пользователь нажимает... 2. Система проверяет...)."
    )


class FunctionalRequirements(BaseModel):
    """2. Функциональные требования"""
    roles: List[UserRole] = Field(
        default_factory=list, 
        description="Определи все уникальные роли пользователей в системе и их права."
    )
    use_cases: List[UseCase] = Field(
        default_factory=list, 
        description="Выдели основные пользовательские сценарии (Use Cases), описывающие ключевой функционал системы."
    )
    modules: List[str] = Field(
        default_factory=list, 
        description="Сформируй список логических модулей, подсистем или ключевых экранов (например: 'Модуль авторизации', 'Платежный шлюз', 'Панель администратора'). Кратко поясни суть каждого."
    )


class DesignSystem(BaseModel):
    colors: Optional[str] = Field(
        None, 
        description="Требования к цветовой схеме: фирменные цвета, наличие темной/светлой темы, общие пожелания (например: 'строгий корпоративный', 'пастельные тона')."
    )
    fonts: Optional[str] = Field(
        None, 
        description="Требования к типографике: семейства шрифтов, размеры, читабельность."
    )
    style: Optional[str] = Field(
        None, 
        description="Общий стиль интерфейса (например: Material Design, Flat, минимализм) или ссылки на брендбук/референсы."
    )


class Responsiveness(BaseModel):
    mobile: Optional[str] = Field(
        None, 
        description="Специфические требования к мобильной версии (Mobile-first, поддержка iOS/Android браузеров, минимальное разрешение)."
    )
    desktop: Optional[str] = Field(
        None, 
        description="Специфические требования к десктопной версии (поддерживаемые браузеры, разрешения от 1024px и выше)."
    )
    tablet: Optional[str] = Field(
        None, 
        description="Поведение интерфейса на планшетах (портретная/альбомная ориентация)."
    )


class UIUXRequirements(BaseModel):
    """3. Требования к интерфейсу"""
    design_system: DesignSystem = Field(
        default_factory=DesignSystem, 
        description="Извлеки все визуальные требования: цвета, шрифты, стилистику."
    )
    responsiveness: Responsiveness = Field(
        default_factory=Responsiveness, 
        description="Определи требования к адаптивности под разные типы устройств."
    )


class TechStack(BaseModel):
    frontend: Optional[str] = Field(
        None, 
        description="Технологии клиентской части (например: React, Vue.js, Swift, Kotlin). Укажи версии, если есть."
    )
    backend: Optional[str] = Field(
        None, 
        description="Технологии серверной части (например: Python/FastAPI, Node.js, Java/Spring, Go). Укажи архитектуру, если известна."
    )
    database: Optional[str] = Field(
        None, 
        description="Используемые СУБД (например: PostgreSQL, MongoDB, Redis). Включая брокеры сообщений."
    )
    other: List[str] = Field(
        default_factory=list, 
        description="Прочие технические инструменты: Docker, CI/CD, облачные провайдеры (AWS, Yandex Cloud), Nginx и т.д."
    )


class Integration(BaseModel):
    name: Optional[str] = Field(
        None, 
        description="Название внешней системы для интеграции (например: '1C:Предприятие', 'Telegram API', 'Stripe')."
    )
    details: Optional[str] = Field(
        None, 
        description="Детали интеграции: протоколы (REST, SOAP, gRPC), цель интеграции (зачем она нужна), форматы данных."
    )


class TechnicalRequirements(BaseModel):
    """4. Технические требования"""
    tech_stack: TechStack = Field(
        default_factory=TechStack, 
        description="Собери полный стек технологий проекта. Не выдумывай технологии, используй только упомянутые факты."
    )
    integrations: List[Integration] = Field(
        default_factory=list, 
        description="Список всех внешних систем, API и сервисов, с которыми должна интегрироваться разрабатываемая система."
    )
    performance: List[str] = Field(
        default_factory=list, 
        description="Извлеки метрики производительности: RPS (запросов в секунду), ожидаемый онлайн (DAU/MAU), лимиты по памяти/CPU."
    )


class NonFunctionalRequirements(BaseModel):
    """5. Нефункциональные требования"""
    security: List[str] = Field(
        default_factory=list, 
        description="Требования к ИБ: стандарты шифрования, авторизация (OAuth2, JWT), защита от DDoS, соответствие законам (ФЗ-152, GDPR)."
    )
    scalability: List[str] = Field(
        default_factory=list, 
        description="Требования к масштабируемости: поддержка горизонтального/вертикального масштабирования, микросервисная архитектура, кубернетис."
    )
    response_time: List[str] = Field(
        default_factory=list, 
        description="SLA и время отклика (например: 'загрузка страницы < 2 сек', 'ответ API < 200 мс')."
    )


class DevelopmentPhase(BaseModel):
    name: Optional[str] = Field(
        None, 
        description="Название этапа (например: 'MVP', 'Альфа-версия', 'Этап 1: Проектирование')."
    )
    deadline: Optional[str] = Field(
        None, 
        description="Сроки реализации этапа (конкретные даты, количество недель/спринтов)."
    )
    features: List[str] = Field(
        default_factory=list, 
        description="Список ключевого функционала, который должен быть готов к концу этого этапа."
    )


class DevelopmentStages(BaseModel):
    """6. Этапы разработки и сроки"""
    mvp: DevelopmentPhase = Field(
        default_factory=DevelopmentPhase, 
        description="Описание первой рабочей версии продукта (MVP). Что в нее входит и когда дедлайн."
    )
    release: DevelopmentPhase = Field(
        default_factory=DevelopmentPhase, 
        description="Описание полноценного релиза (V1.0). Функции, не вошедшие в MVP."
    )
    support: Optional[str] = Field(
        None, 
        description="Условия гарантийной поддержки, SLA, планируемые доработки после релиза."
    )


class AcceptanceCriteria(BaseModel):
    """7. Критерии приёмки"""
    testing: List[str] = Field(
        default_factory=list, 
        description="Требования к проверке качества: виды тестов (Unit, E2E, нагрузочное), процент покрытия кода (coverage), пайплайны CI/CD, баг-трекинг."
    )
    documentation: List[str] = Field(
        default_factory=list, 
        description="Список артефактов, которые нужно сдать вместе с кодом (Swagger API, Руководство пользователя, Архитектурный документ, Doxygen)."
    )

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
        if g:
            lines.append("## 1. Общее описание проекта")
            lines.append(f"- **Назначение системы:** {g.purpose or '_требует уточнения_'}")
            if g.analogues:
                lines.append(f"- **Аналоги/конкуренты:** {', '.join(g.analogues)}")
            lines.append(f"- **Итоговый продукт:** {g.expected_outcome or '_требует уточнения_'}")
            lines.append("")

        f = self.functional
        if f:
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
        if u:
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
        if t:
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
        if nf:
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
        if st:
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
        if a:
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
