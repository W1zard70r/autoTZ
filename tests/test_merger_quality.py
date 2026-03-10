import asyncio
import os
import time
import logging
from typing import List
from pydantic import BaseModel, Field

# Импортируем ваши схемы
from schemas.graph import (
    GraphNode, GraphEdge, ExtractedKnowledge, UnifiedGraph,
    DetectedConflict, ConflictResolution
)
from schemas.enums import NodeLabel, EdgeRelation, TZSectionEnum
from core.merger import SmartGraphMerger
from test_llm import acall_llm_json  # Обновлен импорт согласно вашей архитектуре

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)


# ==========================================
# 1. СХЕМА ДЛЯ LLM-СУДЬИ (LLM-as-a-Judge)
# ==========================================
class EvaluationScore(BaseModel):
    deduplication_score: int = Field(description="Оценка качества дедупликации (0-10)")
    conflict_detection_score: int = Field(
        description="Оценка качества поиска конфликтов (0-10). Нет ли ложных срабатываний?")
    hallucination_penalty: int = Field(description="Штраф за выдуманные связи или конфликты (0-10, где 0 - нет штрафа)")
    reasoning: str = Field(description="Подробный архитектурный анализ результата")


# ==========================================
# 2. ГЕНЕРАТОРЫ ТЕСТОВЫХ ДАННЫХ
# ==========================================
def create_node(id: str, name: str, label: str, desc: str) -> GraphNode:
    return GraphNode(id=id, name=name, label=NodeLabel(label), description=desc)


def create_edge(src: str, tgt: str, rel: str) -> GraphEdge:
    return GraphEdge(source=src, target=tgt, relation=EdgeRelation(rel))


def scenario_1_basic_dedup() -> List[ExtractedKnowledge]:
    """Тест 1: Слияние явных дублей и отсутствие ложных конфликтов."""
    sg1 = ExtractedKnowledge(
        summary="Документ 1",
        nodes=[
            create_node("user_auth", "User Authentication", "Component", "Модуль авторизации по JWT"),
            create_node("pg_db", "PostgreSQL", "Component", "Основная реляционная БД")
        ]
    )
    sg2 = ExtractedKnowledge(
        summary="Документ 2",
        nodes=[
            create_node("auth_module", "Auth Service", "Component", "Сервис аутентификации пользователей (JWT)"),
            create_node("redis_cache", "Redis Cache", "Component", "Кеш для сессий")
        ]
    )
    return [sg1, sg2]


def scenario_2_explicit_conflict() -> List[ExtractedKnowledge]:
    """Тест 2: Поиск архитектурного конфликта (взаимоисключающие технологии)."""
    sg1 = ExtractedKnowledge(
        nodes=[
            create_node("frontend_react", "React UI", "Component", "Фронтенд будет написан на React.js"),
            create_node("main_api", "FastAPI", "Component", "Бэкенд на Python")
        ]
    )
    sg2 = ExtractedKnowledge(
        nodes=[
            create_node("frontend_vue", "Vue.js UI", "Component", "Используем Vue 3 для пользовательского интерфейса"),
            create_node("payment_stripe", "Stripe", "Component", "Оплата через Stripe")
        ]
    )
    return [sg1, sg2]


def scenario_3_large_stress_test() -> List[ExtractedKnowledge]:
    """Тест 3: Нагрузочный тест. Проверяем скорость и работу агента на массе данных."""
    nodes_1 = []
    nodes_2 = []

    # Генерируем "базовую" архитектуру
    for i in range(50):
        nodes_1.append(create_node(f"service_{i}", f"Microservice {i}", "Component", f"Базовый микросервис номер {i}."))

    # Генерируем вторую часть с дублями и 3 конфликтами
    for i in range(50):
        if i in [10, 20, 30]:  # Дубль
            nodes_2.append(create_node(f"ms_{i}_copy", f"Service {i}", "Component", f"Микросервис {i}."))
        elif i == 5:  # Конфликт БД
            nodes_1.append(create_node("db_sql", "MySQL", "Component", "Основная монолитная база данных"))
            nodes_2.append(create_node("db_nosql", "MongoDB", "Component", "Основная база данных документального типа"))
        elif i == 15:  # Конфликт архитектуры
            nodes_1.append(create_node("arch_mono", "Monolith", "Concept", "Деплоится как единый монолит"))
            nodes_2.append(create_node("arch_micro", "Microservices", "Concept", "Микросервисная архитектура"))
        else:
            nodes_2.append(
                create_node(f"other_service_{i}", f"Integration {i}", "Component", f"Внешняя интеграция {i}."))

    return [ExtractedKnowledge(nodes=nodes_1), ExtractedKnowledge(nodes=nodes_2)]


# ==========================================
# 3. ДВИЖОК ТЕСТИРОВАНИЯ И ОЦЕНКИ
# ==========================================
async def evaluate_with_llm_judge(
        scenario_name: str,
        input_subgraphs: List[ExtractedKnowledge],
        unified_graph: UnifiedGraph,
        conflicts: List[DetectedConflict]
) -> EvaluationScore:
    """Использует мощную LLM для оценки качества работы мёрджера."""

    inputs_str = "\n".join(
        [f"Подграф {i + 1} узлы: {[n.name for n in sg.nodes]}" for i, sg in enumerate(input_subgraphs)])
    outputs_str = f"Итоговые узлы: {[n.name for n in unified_graph.nodes]}"
    conflicts_str = f"Найденные конфликты: {[c.description for c in conflicts]}" if conflicts else "Конфликтов не найдено"

    prompt = f"""
    Ты — Senior Software Architect и строгий QA-инженер. Твоя задача оценить работу алгоритма слияния графов знаний (Knowledge Graph Merger).
    Алгоритм должен был: 1. Объединить семантические дубликаты. 2. Найти архитектурные противоречия. 3. Не выдумать лишнего.

    --- ВХОДНЫЕ ДАННЫЕ (ЧТО БЫЛО ДО) ---
    {inputs_str}

    --- РЕЗУЛЬТАТ АЛГОРИТМА (ЧТО СТАЛО ПОСЛЕ) ---
    {outputs_str}

    --- ОБНАРУЖЕННЫЕ КОНФЛИКТЫ ---
    {conflicts_str}

    Оцени результат по схеме EvaluationScore.
    Учти:
    - Фронтенд и Бэкенд (React и FastAPI) - это НЕ конфликт.
    - React и Vue для одного и того же UI - это КОНФЛИКТ.
    - Разные базы данных претендующие на "основную" - это КОНФЛИКТ.
    """

    logger.info(f"🧠 Запуск LLM-судьи для оценки: {scenario_name}...")
    score: EvaluationScore = await acall_llm_json(
        schema=EvaluationScore,
        prompt=prompt,
        data=""  # Данные уже зашиты в prompt для удобства
    )
    return score


async def run_test(scenario_name: str, subgraphs: List[ExtractedKnowledge]):
    print(f"\n{'=' * 60}\n🚀 ЗАПУСК ТЕСТА: {scenario_name}\n{'=' * 60}")

    merger = SmartGraphMerger()
    captured_conflicts: List[DetectedConflict] = []

    # ---------------------------------------------------------
    # ХУК ДЛЯ АГЕНТА: Перехватываем конфликты для отчета судье
    # ---------------------------------------------------------
    async def mock_human_resolver(conflicts: List[DetectedConflict]) -> List[ConflictResolution]:
        nonlocal captured_conflicts
        captured_conflicts = conflicts

        resolutions = []

        print("\n" + "!" * 60)
        print("🛑 ТРЕБУЕТСЯ РУЧНОЕ РАЗРЕШЕНИЕ КОНФЛИКТОВ (TEST MODE)")
        print("!" * 60)

        for i, conf in enumerate(conflicts, 1):
            print(f"\n🔹 КОНФЛИКТ #{i}: {conf.description}")
            print(f"   Категория: {conf.category}")
            print(f"   🤖 AI советует: {conf.ai_recommendation}")
            print("   Варианты:")

            for idx, opt in enumerate(conf.options):
                print(f"     [{idx}] {opt.text}")

            # Запрос ввода
            user_input = input(f"\n👉 Ваш выбор (0-{len(conf.options) - 1}) [по умолчанию 0]: ").strip()

            selected_idx = 0  # Значение по умолчанию

            if user_input.isdigit():
                val = int(user_input)
                if 0 <= val < len(conf.options):
                    selected_idx = val
                else:
                    print(f"   ⚠️ Неверный индекс, выбран вариант [0]")
            elif user_input == "":
                print(f"   ✅ Выбран вариант по умолчанию [0]")
            else:
                print(f"   ⚠️ Неверный ввод, выбран вариант [0]")

            resolutions.append(ConflictResolution(
                conflict_id=conf.id,
                selected_option_id=conf.options[selected_idx].id
            ))

        print("\n" + "=" * 60)
        return resolutions
    # ЗАМЕР ВРЕМЕНИ
    start_time = time.time()

    # Запускаем единый пайплайн агента
    unified_graph = await merger.run_agentic(
        subgraphs=subgraphs,
        human_resolver=mock_human_resolver  # Передаем наш перехватчик
    )

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"⏱️  Время выполнения: {execution_time:.2f} сек.")
    print(f"📊 Исходно узлов: {sum(len(sg.nodes) for sg in subgraphs)} | Итого узлов: {len(unified_graph.nodes)}")
    print(f"⚠️ Найдено конфликтов: {len(captured_conflicts)}")

    # АНАЛИЗ КАЧЕСТВА LLM-СУДЬЕЙ
    evaluation = await evaluate_with_llm_judge(scenario_name, subgraphs, unified_graph, captured_conflicts)

    print("\n⚖️  ОЦЕНКА LLM-СУДЬИ:")
    print(f"   Дедупликация: {evaluation.deduplication_score}/10")
    print(f"   Поиск конфликтов: {evaluation.conflict_detection_score}/10")
    print(f"   Штраф за галлюцинации: -{evaluation.hallucination_penalty}")
    print(f"   Вердикт: \n   {evaluation.reasoning}\n")


async def main():
    # 1. Базовый тест (проверка на отсутствие ложных срабатываний)
    await run_test("Сценарий 1: Базовая дедупликация (Без конфликтов)", scenario_1_basic_dedup())

    # 2. Тест на конфликт (проверка качества RAG кластеризации)
    await run_test("Сценарий 2: Явный архитектурный конфликт (React vs Vue)", scenario_2_explicit_conflict())

    # 3. Большой тест (нагрузка на эмбеддинги и LangGraph)
    await run_test("Сценарий 3: Нагрузочный тест (100+ узлов, скрытые конфликты)", scenario_3_large_stress_test())


if __name__ == "__main__":
    asyncio.run(main())