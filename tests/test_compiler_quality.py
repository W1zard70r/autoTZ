import asyncio
import logging
from typing import List
from pydantic import BaseModel, Field

# Импорты ваших схем и кода
from schemas.graph import UnifiedGraph, GraphNode, GraphEdge
from schemas.enums import TemplateType, NodeLabel, EdgeRelation, TZSectionEnum
from core.compiler import TZCompiler
from test_llm import acall_llm_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# 1. LLM Судья
# ==========================================
class CompilationAssessment(BaseModel):
    is_accurate: bool = Field(description="Верно ли скомпилировано ТЗ по графу?")
    completeness_score: int = Field(description="Оценка полноты (0-10)")
    hallucinations: List[str] = Field(description="Список найденных галлюцинаций")
    critique: str = Field(description="Критика качества ТЗ")


# ==========================================
# 2. Моковые данные (Вход для Компилятора)
# ==========================================
def get_mock_it_project_graph() -> UnifiedGraph:
    """Создает граф, который прошел бы через Майнер и Мерджер."""
    return UnifiedGraph(
        nodes=[
            GraphNode(
                id="react_frontend", name="React.js UI", label=NodeLabel.COMPONENT,
                description="Фронтенд на React с Vite",
                target_section=TZSectionEnum.STACK,
                properties=[{"key": "framework", "value": "React"}]
            ),
            GraphNode(
                id="fastapi_backend", name="FastAPI Backend", label=NodeLabel.COMPONENT,
                description="Бэкенд на FastAPI, Python 3.11",
                target_section=TZSectionEnum.STACK,
                properties=[{"key": "language", "value": "Python"}]
            ),
            GraphNode(
                id="auth_requirement", name="JWT Auth", label=NodeLabel.REQUIREMENT,
                description="Требуется авторизация через JWT",
                target_section=TZSectionEnum.FUNCTIONAL
            )
        ],
        edges=[
            GraphEdge(source="react_frontend", target="fastapi_backend", relation=EdgeRelation.RELATES_TO,
                      evidence="Интеграция API")
        ]
    )


# ==========================================
# 3. Функциональный тест
# ==========================================
async def test_compiler_layer():
    print("\n" + "=" * 60)
    print("🧪 ЗАПУСК ТЕСТА: Layer 3 (Compiler)")
    print("=" * 60)

    # 1. Инициализация
    compiler = TZCompiler(language="ru")
    graph = get_mock_it_project_graph()

    # 2. Запуск компиляции
    print("🚀 Компилирую ТЗ из графа...")
    tz_result = await compiler.compile(
        graph=graph,
        template_type=TemplateType.IT_PROJECT
    )

    print(f"✅ Компиляция завершена.")
    print(f"📊 Заполнено полей: {tz_result.validation.filled_fields}/{tz_result.validation.total_fields}")
    print(f"📝 Markdown результат (первые 500 символов):\n{tz_result.markdown[:500]}...")

    # 3. LLM Судья для проверки качества
    prompt = """
    Ты — Senior Tech Lead. Твоя задача проверить результат генерации ТЗ (Markdown).

    ВХОДНЫЕ ДАННЫЕ (Граф):
    {graph_data}

    РЕЗУЛЬТАТ (ТЗ):
    {tz_text}

    Проверь:
    1. Упомянуты ли все узлы из графа (React.js UI, FastAPI, JWT Auth)?
    2. Нет ли в ТЗ сущностей, которых нет в графе?
    3. Соответствует ли структура шаблону IT-проекта?
    """

    print("\n⚖️ Проверка LLM-судьей...")
    assessment: CompilationAssessment = await acall_llm_json(
        schema=CompilationAssessment,
        prompt=prompt.format(
            graph_data=graph.model_dump_json(),
            tz_text=tz_result.markdown
        ),
        system="Ты экспертный QA-инженер, проверяющий качество автоматических ТЗ."
    )

    print(f"   Вердикт: {'✅ OK' if assessment.is_accurate else '❌ FAIL'}")
    print(f"   Оценка: {assessment.completeness_score}/10")
    print(f"   Анализ: {assessment.critique}")

    if assessment.hallucinations:
        print(f"   Найденные галлюцинации: {assessment.hallucinations}")

    assert assessment.is_accurate, "LLM судья выявил ошибки в компиляции"
    print("\n🎉 Тест слоя 3 прошел успешно!")


if __name__ == "__main__":
    asyncio.run(test_compiler_layer())