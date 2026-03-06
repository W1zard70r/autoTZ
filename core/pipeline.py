"""Ядро пайплайна генерации ТЗ.

Объединяет все слои: майнинг → слияние → компиляция.
Предоставляет пошаговый API, который используется всеми интерфейсами
(CLI, API-адаптер, backend-адаптер).
"""
import asyncio
import json
import logging
from typing import List, Dict, Optional

import networkx as nx

from schemas.document import DataSource
from schemas.enums import TemplateType
from schemas.graph import (
    ExtractedKnowledge, UnifiedGraph,
    DetectedConflict, ConflictResolution,
)
from schemas.templates.base import TZResult, ValidationResult, FieldGap
from core.miner import MinerProcessor
from core.merger import SmartGraphMerger
from core.compiler import TZCompiler
from core.translator import translate_markdown
from utils.llm_client import acall_llm_text

logger = logging.getLogger(__name__)


class TZPipeline:
    """Пошаговый оркестратор генерации ТЗ.

    Каждый метод — отдельный шаг пайплайна.
    Интерфейсы (CLI / API / тесты) вызывают шаги последовательно,
    встраивая собственную логику взаимодействия с пользователем
    между шагами, если нужно.
    """

    def __init__(
        self,
        template_type: TemplateType = TemplateType.IT_PROJECT,
        language: str = "ru",
    ):
        self.template_type = template_type
        self.language = language

        self.miner = MinerProcessor()
        self.merger = SmartGraphMerger()
        self.compiler = TZCompiler(language=language)

        self.subgraphs: List[ExtractedKnowledge] = []
        self.unified_graph: Optional[UnifiedGraph] = None
        self.conflicts: List[DetectedConflict] = []
        self.result: Optional[TZResult] = None
        self.user_answers: Dict[str, str] = {}

    # ── Шаг 1: Извлечение знаний ────────────────────────────
    async def extract(self, sources: List[DataSource]) -> List[ExtractedKnowledge]:
        """Layer 1 — майнинг подграфов из источников данных."""
        logger.info(">>> Шаг 1: Извлечение знаний из источников")
        self.subgraphs = []

        for source in sources:
            logger.info(f"  Обработка: {source.file_name}")
            try:
                graphs = await self.miner.process_source(source)
                self.subgraphs.extend(graphs)
                logger.info(f"  -> Извлечено {len(graphs)} подграфов из {source.file_name}")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"  Ошибка при обработке {source.file_name}: {e}")

        logger.info(f"  Итого подграфов: {len(self.subgraphs)}")
        return self.subgraphs

    # ── Шаг 2: Слияние и дедупликация ───────────────────────
    async def merge(self) -> None:
        """Layer 2, часть 1 — слияние подграфов и дедупликация."""
        logger.info(">>> Шаг 2: Слияние подграфов и дедупликация")
        if not self.subgraphs:
            raise ValueError("Нет подграфов для слияния. Сначала вызовите extract().")
        await self.merger.merge_subgraphs_and_deduplicate(self.subgraphs)

    # ── Шаг 3: Обнаружение конфликтов ───────────────────────
    async def detect_conflicts(self) -> List[DetectedConflict]:
        """Layer 2, часть 2 — поиск логических противоречий."""
        logger.info(">>> Шаг 3: Поиск конфликтов")
        self.conflicts = await self.merger.detect_conflicts()
        return self.conflicts

    # ── Шаг 4: Применение решений пользователя ──────────────
    def apply_resolutions(self, resolutions: List[ConflictResolution]) -> None:
        """Layer 2, часть 3 — применение решений по конфликтам."""
        logger.info(">>> Шаг 4: Применение решений пользователя")
        self.merger.apply_resolutions(resolutions)

    # ── Шаг 5: Финализация графа ────────────────────────────
    async def finalize_graph(self) -> UnifiedGraph:
        """Layer 2, часть 4 — итоговый граф с голосованиями и секциями."""
        logger.info(">>> Шаг 5: Финализация графа")
        self.unified_graph = await self.merger.finalize_graph()
        logger.info(
            f"  Граф: {len(self.unified_graph.nodes)} узлов, "
            f"{len(self.unified_graph.edges)} связей"
        )
        return self.unified_graph

    # ── Шаг 6: Компиляция ТЗ ────────────────────────────────
    async def compile(self) -> TZResult:
        """Layer 3 — заполнение шаблона и генерация документа."""
        logger.info(">>> Шаг 6: Компиляция ТЗ")
        if self.unified_graph is None:
            raise ValueError("Граф не финализирован. Сначала вызовите finalize_graph().")

        self.result = await self.compiler.compile(
            graph=self.unified_graph,
            template_type=self.template_type,
            user_answers=self.user_answers or None,
        )
        logger.info(
            f"  Заполненность: {self.result.validation.completeness_percent}% "
            f"({self.result.validation.filled_fields}/{self.result.validation.total_fields})"
        )
        return self.result

    # ── Шаг 7 (опц.): Уточнение пробелов ───────────────────
    def get_gaps(self) -> List[FieldGap]:
        """Возвращает список незаполненных обязательных полей."""
        if self.result is None:
            return []
        return self.result.validation.gaps

    def add_user_answers(self, answers: Dict[str, str]) -> None:
        """Добавляет ответы пользователя для повторной компиляции."""
        self.user_answers.update(answers)

    async def recompile(self) -> TZResult:
        """Повторная компиляция с учётом ответов пользователя."""
        logger.info(">>> Повторная компиляция с уточнениями")
        return await self.compile()

    # ── Шаг 8 (опц.): Перевод ───────────────────────────────
    async def translate(self, target_language: str) -> str:
        """Переводит готовый документ на другой язык."""
        if self.result is None:
            raise ValueError("Нет результата для перевода. Сначала вызовите compile().")
        return await translate_markdown(self.result.markdown, target_language)

    # ── Полный прогон ────────────────────────────────────────
    async def run_full(
        self,
        sources: List[DataSource],
        resolutions: List[ConflictResolution] | None = None,
    ) -> TZResult:
        """Полный прогон пайплайна без интерактива (для тестов / API)."""
        await self.extract(sources)

        if not self.subgraphs:
            logger.warning("Нет подграфов после извлечения — возвращаем пустой результат.")
            return TZResult(
                template_type=self.template_type,
                template_data=None,
            )

        await self.merge()
        conflicts = await self.detect_conflicts()
        if conflicts and resolutions:
            self.apply_resolutions(resolutions)
        await self.finalize_graph()
        return await self.compile()

    # ── Генерация вариантов для конфликта ────────────────────
    async def generate_conflict_variants(
        self, conflict: DetectedConflict,
    ) -> List[str]:
        """Генерирует развёрнутые текстовые варианты для каждой опции конфликта."""
        variants: List[str] = []
        for option in conflict.options:
            prompt = (
                f"Конфликт: {conflict.description}\n"
                f"Вариант: {option.text}\n"
                f"Обоснование: {option.evidence}\n\n"
                f"Напиши краткий абзац (3-5 предложений) для технического задания, "
                f"описывающий выбор «{option.text}» с учётом контекста проекта."
            )
            try:
                text = await acall_llm_text(
                    prompt,
                    system="Ты технический писатель. Пиши на русском, кратко и по делу.",
                    max_tokens=512,
                )
                variants.append(text.strip())
            except Exception as e:
                logger.warning(f"Ошибка генерации варианта '{option.text}': {e}")
                variants.append(option.text)
        return variants

    # ── Сериализация состояния ────────────────────────────────
    def save_state(self) -> dict:
        """Сериализует состояние пайплайна в JSON-совместимый dict.

        Используется backend-адаптером для сохранения прогресса
        между вызовами taskiq-воркера.
        """
        merger_graph = None
        if self.merger.G.number_of_nodes() > 0:
            merger_graph = nx.node_link_data(self.merger.G)

        return {
            "template_type": self.template_type.value,
            "language": self.language,
            "subgraphs": [sg.model_dump(mode="json") for sg in self.subgraphs],
            "unified_graph": (
                self.unified_graph.model_dump(mode="json")
                if self.unified_graph else None
            ),
            "conflicts": [c.model_dump(mode="json") for c in self.conflicts],
            "user_answers": self.user_answers,
            "result": self.result.model_dump(mode="json") if self.result else None,
            "merger_graph": merger_graph,
            "merger_active_conflicts": [
                c.model_dump(mode="json") for c in self.merger.active_conflicts
            ],
            "miner_glossary": {
                k: v.model_dump(mode="json")
                for k, v in self.miner.global_glossary.items()
            },
            "miner_key_entities": self.miner.key_entity_ids,
        }

    @classmethod
    def load_state(cls, state: dict) -> "TZPipeline":
        """Восстанавливает пайплайн из ранее сохранённого состояния."""
        pipeline = cls(
            template_type=TemplateType(state["template_type"]),
            language=state["language"],
        )

        pipeline.subgraphs = [
            ExtractedKnowledge.model_validate(sg)
            for sg in state.get("subgraphs", [])
        ]
        if state.get("unified_graph"):
            pipeline.unified_graph = UnifiedGraph.model_validate(state["unified_graph"])
        pipeline.conflicts = [
            DetectedConflict.model_validate(c)
            for c in state.get("conflicts", [])
        ]
        pipeline.user_answers = state.get("user_answers", {})
        if state.get("result"):
            pipeline.result = TZResult.model_validate(state["result"])
        if state.get("merger_graph"):
            pipeline.merger.G = nx.node_link_graph(
                state["merger_graph"],
            )
        pipeline.merger.active_conflicts = [
            DetectedConflict.model_validate(c)
            for c in state.get("merger_active_conflicts", [])
        ]
        if state.get("miner_glossary"):
            from core.miner import GlossaryItem
            pipeline.miner.global_glossary = {
                k: GlossaryItem.model_validate(v)
                for k, v in state["miner_glossary"].items()
            }
        pipeline.miner.key_entity_ids = state.get("miner_key_entities", [])
        return pipeline
