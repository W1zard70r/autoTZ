import asyncio
import logging
from typing import List
from pydantic import BaseModel, Field

from schemas.graph import UnifiedGraph, GraphNode, GraphEdge
from schemas.enums import TemplateType, NodeLabel, EdgeRelation, TZSectionEnum
from core.compiler import TZCompiler
from utils.llm_client import acall_llm_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# 1. LLM Судья для качественной оценки
# ==========================================
class CompilationAssessment(BaseModel):
    is_accurate: bool = Field(description="Верно ли скомпилировано ТЗ по фактам из графа?")
    completeness_score: int = Field(description="Оценка полноты (0-10)")
    hallucinations: List[str] = Field(description="Список найденных выдумок, которых нет в графе")
    critique: str = Field(description="Критика качества, логики и оформления")


# ==========================================
# 2. Помощники для генерации данных
# ==========================================
def n(id, name, section=TZSectionEnum.UNKNOWN, label=NodeLabel.COMPONENT, desc=""):
    """Быстрое создание узла"""
    return GraphNode(id=id, name=name, label=label, target_section=section, description=desc)


async def run_judge(graph: UnifiedGraph, markdown: str, scenario: str):
    prompt = f"""
    Сценарий теста: {scenario}
    Ты — Senior QA Engineer. Сравни Граф знаний и сгенерированное из него ТЗ.

    ГРАФ (Факты):
    {graph.model_dump_json()}

    РЕЗУЛЬТАТ (Markdown ТЗ):
    {markdown}

    КРИТЕРИИ:
    1. Фактология: Все ли сущности из графа попали в ТЗ?
    2. Галлюцинации: Есть ли в ТЗ конкретные технологии/цифры/имена, которых нет в графе?
    3. Структура: Соответствует ли текст логике разделов?
    """
    return await acall_llm_json(CompilationAssessment, prompt, system="Ты эксперт по верификации ТЗ.")


# ==========================================
# 3. ТЕСТОВЫЕ СЦЕНАРИИ
# ==========================================

async def test_smoke_minimal():
    """ТЕСТ 1: Минимальный граф (1 узел). Проверка на пустые разделы."""
    print("\n🔹 Сценарий 1: Smoke-test (Минимальные данные)")
    compiler = TZCompiler()
    graph = UnifiedGraph(nodes=[n("api", "REST API", TZSectionEnum.STACK, desc="Один единственный узел")])

    res = await compiler.compile(graph, TemplateType.IT_PROJECT)

    assert "REST API" in res.markdown
    assert res.validation.completeness_percent < 30  # ТЗ должно быть полупустым
    print(f"   [OK] Компилятор не упал, заполнено {res.validation.completeness_percent}%")


async def test_auto_sectioning():
    """ТЕСТ 2: Узлы с UNKNOWN. Проверка, может ли компилятор сам распределить факты."""
    print("\n🔹 Сценарий 2: Исправление (UNKNOWN sections)")
    compiler = TZCompiler()
    graph = UnifiedGraph(nodes=[
        n("react", "React", TZSectionEnum.UNKNOWN, desc="Библиотека для интерфейса"),
        n("auth", "OAuth2", TZSectionEnum.UNKNOWN, desc="Протокол авторизации")
    ])

    res = await compiler.compile(graph, TemplateType.IT_PROJECT)

    # Судья проверит, попали ли компоненты в нужные блоки (Интерфейс и Стек)
    assess = await run_judge(graph, res.markdown, "Auto-sectioning UNKNOWN nodes")
    print(f"   Вердикт судьи: {assess.completeness_score}/10. Галлюцинаций: {len(assess.hallucinations)}")
    assert assess.is_accurate


async def test_full_it_project():
    """ТЕСТ 3: Полный проект. Максимальное покрытие данных."""
    print("\n🔹 Сценарий 3: Полный цикл (Богатый IT-проект)")
    compiler = TZCompiler(language="ru")
    graph = UnifiedGraph(
        nodes=[
            n("frontend", "Next.js", TZSectionEnum.INTERFACE, desc="Фронтенд с SSR"),
            n("backend", "NestJS", TZSectionEnum.STACK, desc="Бэкенд на Node.js"),
            n("db", "MongoDB", TZSectionEnum.STACK, desc="NoSQL база данных"),
            n("req_1", "Real-time chat", TZSectionEnum.FUNCTIONAL, label=NodeLabel.REQUIREMENT,
              desc="Чат через WebSockets"),
            n("task_1", "Deploy to Vercel", TZSectionEnum.GENERAL, label=NodeLabel.TASK, desc="Деплой проекта"),
            n("dev_1", "Ivan Ivanov", TZSectionEnum.GENERAL, label=NodeLabel.PERSON, desc="Team Lead")
        ],
        edges=[
            GraphEdge(source="frontend", target="backend", relation=EdgeRelation.DEPENDS_ON, evidence="API calls")
        ]
    )

    res = await compiler.compile(graph, TemplateType.IT_PROJECT)

    assess = await run_judge(graph, res.markdown, "Full IT Project")
    print(f"   Полнота: {res.validation.filled_fields}/{res.validation.total_fields} полей.")
    print(f"   Анализ судьи: {assess.critique[:200]}...")

    assert assess.completeness_score >= 7
    assert len(assess.hallucinations) == 0


async def test_gost_template():
    """ТЕСТ 4: Смена шаблона (ГОСТ). Проверка формального стиля."""
    print("\n🔹 Сценарий 4: Формальный шаблон (ГОСТ 34)")
    compiler = TZCompiler()
    graph = UnifiedGraph(nodes=[
        n("sys", "АСУ ТП", TZSectionEnum.GENERAL, desc="Автоматизированная система управления"),
        n("sec", "Защита данных", TZSectionEnum.FUNCTIONAL, label=NodeLabel.REQUIREMENT, desc="Шифрование по ГОСТ")
    ])

    res = await compiler.compile(graph, TemplateType.GOST)

    assert "Наименование проекта" in res.markdown  # Специфично для ГОСТ
    assert "Основание для разработки" in res.markdown
    print(f"   [OK] Структура ГОСТ соблюдена.")


async def test_english_localization():
    """ТЕСТ 5: Локализация (EN)."""
    print("\n🔹 Сценарий 5: Локализация (Английский язык)")
    compiler = TZCompiler(language="en")
    graph = UnifiedGraph(nodes=[
        n("cloud", "AWS S3", TZSectionEnum.STACK, desc="Storage for images")
    ])

    res = await compiler.compile(graph, TemplateType.IT_PROJECT)

    assert "Technical Specification" in res.markdown or "Specification" in res.markdown
    assert "AWS S3" in res.markdown
    print(f"   [OK] Документ сгенерирован на английском.")


# ==========================================
# 4. ЗАПУСК
# ==========================================
async def main():
    try:
        await test_smoke_minimal()
        await test_auto_sectioning()
        await test_full_it_project()
        await test_gost_template()
        await test_english_localization()
        print("\n✅ Все тесты Layer 3 завершены!")
    except AssertionError as e:
        print(f"\n❌ Тест провален: {e}")
    except Exception as e:
        print(f"\n💥 Ошибка при выполнении: {e}")


if __name__ == "__main__":
    asyncio.run(main())