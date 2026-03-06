"""Layer 3: Заполнение шаблона ТЗ на основе графа знаний.

Ключевые принципы:
- Заполнение по секциям (маленькая схема -> лучшее качество от LLM).
- Подробные инструкции для каждого раздела.
- Прямое применение пользовательских уточнений к полям шаблона.
"""
import logging
import typing
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from schemas.graph import UnifiedGraph, GraphNode, DecisionResolution
from schemas.enums import TZSectionEnum, TemplateType
from schemas.templates.base import BaseTemplate, TZResult, ValidationResult
from schemas.templates import get_template_class
from utils.llm_client import acall_llm_json
from utils.state_logger import log_text, log_pydantic

logger = logging.getLogger(__name__)

# ── Системные промпты ─────────────────────────────────────

SYSTEM_RU = (
    "Ты опытный технический писатель. "
    "Заполняешь раздел технического задания на основе графа знаний проекта.\n"
    "ПРАВИЛА:\n"
    "- Пиши на русском языке.\n"
    "- Каждое текстовое поле заполняй развёрнуто: минимум 1-3 предложения.\n"
    "- Для списков перечисляй ВСЕ известные элементы, не ограничивайся одним.\n"
    "- Синтезируй информацию: если из графа следует вывод — формулируй его.\n"
    "- Если данных для поля нет совсем — ставь null.\n"
    "- НЕ выдумывай факты, которых нет в графе, но РАЗВИВАЙ имеющиеся."
)

SYSTEM_EN = (
    "You are an experienced technical writer. "
    "Fill a section of a Technical Specification based on a project knowledge graph.\n"
    "RULES:\n"
    "- Write in English.\n"
    "- Each text field should be at least 1-3 sentences.\n"
    "- For list fields, enumerate ALL known items.\n"
    "- Synthesize information: if a conclusion follows from the graph — state it.\n"
    "- Leave null only when there is absolutely no data.\n"
    "- Do NOT invent facts, but DO expand on existing ones."
)

# ── Подробные инструкции по секциям ───────────────────────

SECTION_PROMPTS: Dict[str, str] = {
    # ---------- IT Project ----------
    "it_project.general": (
        "Заполни раздел «Общее описание проекта».\n"
        "- Назначение системы: подробно опиши, что система делает, для кого предназначена, "
        "какие проблемы решает (2-4 предложения).\n"
        "- Аналоги: перечисли все упомянутые конкуренты.\n"
        "- Итоговый продукт: сформулируй, что получит заказчик в результате."
    ),
    "it_project.functional": (
        "Заполни раздел «Функциональные требования».\n"
        "- Роли: выдели все роли пользователей из графа (админ, клиент, и т.д.) "
        "и для каждой подробно перечисли возможности.\n"
        "- Сценарии: опиши основные use-case пошагово.\n"
        "- Модули: перечисли все экраны, страницы, API-эндпоинты."
    ),
    "it_project.ui_ux": (
        "Заполни раздел «Требования к интерфейсу (UI/UX)».\n"
        "- Дизайн: укажи цвета, шрифты, стиль (если есть в графе).\n"
        "- Адаптивность: опиши поддержку мобильных, десктопа, планшетов.\n"
        "- Если упомянут Figma, библиотеки компонентов — укажи их."
    ),
    "it_project.technical": (
        "Заполни раздел «Технические требования».\n"
        "- Стек: укажи ВСЕ технологии из графа (фронтенд, бэкенд, БД, "
        "библиотеки, инструменты сборки и т.д.).\n"
        "- Интеграции: перечисли все API и внешние сервисы.\n"
        "- Производительность: укажи ожидания, если есть."
    ),
    "it_project.non_functional": (
        "Заполни раздел «Нефункциональные требования».\n"
        "- Безопасность: JWT, HTTPS, авторизация, и т.д.\n"
        "- Масштабируемость: ожидаемая нагрузка.\n"
        "- Время отклика: если в графе есть SLA.\n"
        "Если информации мало — выведи из контекста разумные требования."
    ),
    "it_project.stages": (
        "Заполни раздел «Этапы разработки».\n"
        "- MVP: на основе данных графа определи минимальный набор фичей.\n"
        "- Основной релиз: что будет после MVP.\n"
        "- Поддержка: обновления, техподдержка."
    ),
    "it_project.acceptance": (
        "Заполни раздел «Критерии приёмки».\n"
        "- Тестирование: виды тестов, покрытие.\n"
        "- Документация: что будет передано заказчику."
    ),
    # ---------- GOST ----------
    "gost.introduction": "Заполни «Введение»: наименование проекта, основание, заказчик и исполнитель.",
    "gost.goal_and_relevance": "Заполни «Цель и актуальность»: цель создания проекта и почему он актуален.",
    "gost.requirements": (
        "Заполни «Требования»: обязательные / желательные / отложенные функции, "
        "технические ограничения, надёжность, масштабируемость."
    ),
    "gost.stages": "Заполни «Стадии и этапы»: список этапов с датами и результатами.",
    "gost.acceptance": "Заполни «Порядок приёмки»: критерии и документы для сдачи.",
    # ---------- Household ----------
    "household.details": "Заполни «Детали и пожелания»: размеры, стиль, материалы, бюджет.",
    "household.expected_result": "Заполни «Результат»: конкретный список итогов и срок.",
    "household.special_conditions": "Заполни «Особые условия»: график, доступ, уборка.",
    # ---------- Construction ----------
    "construction.general_data": "Заполни «Общие данные»: объект, вид работ, площадь, сроки.",
    "construction.scope": "Заполни «Объём работ»: подготовительные, основные, финальные.",
    "construction.materials": "Заполни «Материалы»: заказчика, подрядчика, требования к качеству.",
    "construction.schedule": "Заполни «График и доступ»: время, выходные, ключи, склад.",
    "construction.control": "Заполни «Контроль и приёмка»: промежуточные, итоговая, акты.",
    # ---------- Engineering ----------
    "engineering.basis": "Заполни «Основание и нормативная база»: договор, объект, стадии, ГОСТ.",
    "engineering.technical": "Заполни «Технические требования»: параметры, конструкция, прочность.",
    "engineering.systems": "Заполни «Системы и компоненты»: приводы, управление, безопасность.",
    "engineering.documentation": "Заполни «Документация»: состав, форматы, чертежи.",
    "engineering.testing": "Заполни «Испытания»: виды, методики, критерии приёмки.",
}


# ── Построение контекста из графа ─────────────────────────

def _build_full_context(graph: UnifiedGraph) -> str:
    lines = []
    for n in graph.nodes:
        props = ""
        if n.properties:
            props = " | " + ", ".join(f"{p.key}={p.value}" for p in n.properties)
        lines.append(f"- [{n.label.value}] {n.name}: {n.description}{props}")
        for e in graph.edges:
            if e.source == n.id:
                other = next((nd.name for nd in graph.nodes if nd.id == e.target), e.target)
                ev = f' ("{e.evidence}")' if e.evidence else ""
                lines.append(f"    -> {e.relation.value} {other}{ev}")
            elif e.target == n.id:
                other = next((nd.name for nd in graph.nodes if nd.id == e.source), e.source)
                ev = f' ("{e.evidence}")' if e.evidence else ""
                lines.append(f"    <- {e.relation.value} {other}{ev}")
    return "\n".join(lines) if lines else "(граф пуст)"


def _build_decisions_context(decisions: List[DecisionResolution]) -> str:
    lines = []
    for d in decisions:
        status = "Не решено" if d.is_tie else f"Решение: {d.winner_name or '—'}"
        lines.append(f"- {d.decision_name}: {status}")
        for opt in d.options:
            lines.append(f"  - {opt.option_name}: за={opt.votes_for}, против={opt.votes_against}")
    return "\n".join(lines)


def _extract_basemodel_class(annotation) -> Optional[type]:
    """Извлекает BaseModel-класс из аннотации (включая Optional[Model])."""
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel) and annotation is not BaseModel:
        return annotation
    return None


def _apply_user_answers(template: BaseTemplate, answers: Dict[str, str]) -> None:
    """Применяет пользовательские ответы напрямую к полям шаблона."""
    for path, value in answers.items():
        parts = path.split(".")
        obj = template
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
            field_name = parts[-1]
            field_info = obj.model_fields.get(field_name)
            if field_info is None:
                continue
            ann = field_info.annotation
            origin = getattr(ann, "__origin__", None)
            if origin is list:
                setattr(obj, field_name, [v.strip() for v in value.split(",") if v.strip()])
            else:
                setattr(obj, field_name, value)
        except (AttributeError, TypeError):
            continue


# ── Компилятор ────────────────────────────────────────────

class TZCompiler:
    def __init__(self, language: str = "ru"):
        self.language = language
        self.system_prompt = SYSTEM_RU if language == "ru" else SYSTEM_EN

    async def compile(
        self,
        graph: UnifiedGraph,
        template_type: TemplateType,
        user_answers: Dict[str, str] | None = None,
    ) -> TZResult:
        logger.info(f"Layer 3: Заполнение шаблона {template_type.value}...")

        template_class = get_template_class(template_type)
        full_context = _build_full_context(graph)
        decisions_ctx = (
            _build_decisions_context(graph.decisions) if graph.decisions else ""
        )

        user_ctx = ""
        if user_answers:
            lines = [f"- {k}: {v}" for k, v in user_answers.items()]
            user_ctx = "\nУТОЧНЕНИЯ ПОЛЬЗОВАТЕЛЯ:\n" + "\n".join(lines)

        # --- Секция за секцией ---
        filled: Dict[str, Any] = {}
        for field_name, field_info in template_class.model_fields.items():
            model_class = _extract_basemodel_class(field_info.annotation)
            if model_class is None:
                continue

            prompt_key = f"{template_type.value}.{field_name}"
            instruction = SECTION_PROMPTS.get(
                prompt_key,
                f"Заполни раздел «{field_info.description or field_name}» максимально подробно.",
            )

            prompt = (
                f"{instruction}\n\n"
                f"ГРАФ ЗНАНИЙ ПРОЕКТА:\n{full_context}\n"
                + (f"РЕШЕНИЯ:\n{decisions_ctx}\n" if decisions_ctx else "")
                + user_ctx + "\n\n"
                "Заполни ВСЕ поля максимально подробно. "
                "Строковые поля — минимум 1-2 предложения. "
                "Списки — перечисли все известные элементы."
            )

            log_text(f"layer3_{template_type.value}_{field_name}_prompt.txt", prompt)

            try:
                section = await acall_llm_json(
                    model_class, prompt, system=self.system_prompt, max_tokens=4096,
                )
                filled[field_name] = section
                logger.info(f"  ✅ Секция '{field_name}' заполнена")
            except Exception as e:
                logger.warning(f"  Секция '{field_name}' — ошибка LLM: {e}")
                filled[field_name] = model_class()

        # --- Собираем шаблон ---
        try:
            template = template_class(**filled)
        except Exception as e:
            logger.error(f"Ошибка сборки шаблона: {e}")
            template = template_class()

        # --- Прямое применение пользовательских ответов ---
        if user_answers:
            _apply_user_answers(template, user_answers)

        validation = template.validate_completeness()
        logger.info(
            f"  Заполненность: {validation.completeness_percent}%% "
            f"({validation.filled_fields}/{validation.total_fields})"
        )

        try:
            markdown = template.to_markdown()
        except Exception as e:
            logger.error(f"Ошибка рендеринга markdown: {e}")
            markdown = "# Ошибка генерации\n\nНе удалось сформировать документ."

        result = TZResult(
            template_type=template_type,
            template_data=template,
            validation=validation,
            markdown=markdown,
        )
        log_pydantic(f"layer3_result_{template_type.value}.json", result)
        return result
