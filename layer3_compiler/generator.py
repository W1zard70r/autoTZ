import logging
import asyncio
from typing import List
from schemas.graph import UnifiedGraph, GraphNode
from schemas.enums import TZSectionEnum
from schemas.document import FullTZDocument, GeneratedSection
from utils.llm_client import acall_llm_text
from utils.state_logger import log_text

logger = logging.getLogger(__name__)


class TZGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    async def generate_tz(self, graph: UnifiedGraph) -> FullTZDocument:
        logger.info("📝 СЛОЙ 3: Генерация документа ТЗ...")

        if graph.conflicts:
            logger.warning(f"⚠️ В графе найдено {len(graph.conflicts)} неразрешенных конфликтов!")

        sections_to_write = [
            TZSectionEnum.GENERAL,
            TZSectionEnum.FUNCTIONAL,
            TZSectionEnum.STACK,
            TZSectionEnum.INTERFACE
        ]

        tasks = []
        for sec_enum in sections_to_write:
            tasks.append(self._generate_section(sec_enum, graph))

        generated_sections = await asyncio.gather(*tasks)

        valid_sections = [sec for sec in generated_sections if sec is not None]

        return FullTZDocument(
            project_name="Техническое Задание (AI Generated)",
            version="1.0.0",
            sections=valid_sections
        )

    async def _generate_section(self, sec_enum: TZSectionEnum, graph: UnifiedGraph) -> GeneratedSection:
        relevant_nodes = [n for n in graph.nodes if n.target_section == sec_enum]

        if not relevant_nodes:
            return None

        logger.info(f"  -> Пишем раздел: {sec_enum.value} ({len(relevant_nodes)} узлов)")

        node_context = "\n".join([f"- {n.name} (ID: {n.id}): {n.description}" for n in relevant_nodes])

        prompt = f"""
Напиши раздел Технического Задания: '{sec_enum.value}'.
Используй ТОЛЬКО предоставленные факты из узлов. 
Стиль: формально-деловой. Оформление: Markdown (заголовки, списки).

ФАКТЫ ДЛЯ РАЗДЕЛА:
{node_context}
"""
        # --- LOGGING: Сохраняем переданный контекст для дебага ---
        log_text(f"layer3_prompt_{sec_enum.value}.txt", prompt)

        try:
            content_markdown = await acall_llm_text(prompt=prompt, model_name=self.model_name)
            return GeneratedSection(
                section_id=sec_enum,
                title=sec_enum.name,
                content_markdown=content_markdown
            )
        except Exception as e:
            logger.error(f"Ошибка генерации раздела {sec_enum.value}: {e}")
            return None