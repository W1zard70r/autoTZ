import logging
from typing import List
from schemas.graph import UnifiedGraph, GraphNode
from schemas.enums import TZSectionEnum
from schemas.document import FullTZDocument, GeneratedSection
from utils.llm_client import acall_llm_text

logger = logging.getLogger(__name__)

class TZGenerator:
    async def generate_tz(self, graph: UnifiedGraph) -> FullTZDocument:
        logger.info("📄 СЛОЙ 3: Компиляция Технического Задания")
        
        sections_data = []
        sections_to_generate = [
            (TZSectionEnum.GENERAL, "1. Общие сведения"),
            (TZSectionEnum.FUNCTIONAL, "2. Функциональные требования"),
            (TZSectionEnum.STACK, "3. Стек технологий"),
            (TZSectionEnum.INTERFACE, "4. Интерфейс (UI/UX)")
        ]

        # Превращаем связи в быстрый поиск
        edges_text = [f"{e.source} --[{e.relation}]--> {e.target} (Обоснование: {e.evidence})" for e in graph.edges]

        for sec_enum, sec_title in sections_to_generate:
            relevant_nodes = [n for n in graph.nodes if n.target_section == sec_enum]
            if not relevant_nodes:
                continue

            logger.info(f"  -> Генерация раздела: {sec_title} ({len(relevant_nodes)} узлов)")
            
            node_context = "\n".join([f"- [{n.label}] {n.name}: {n.description}" for n in relevant_nodes])
            edge_context = "\n".join(edges_text) # Отдаем связи для понимания контекста
            
            prompt = f"""Ты Технический Писатель. Напиши раздел ТЗ: '{sec_title}'.
            Используй ТОЛЬКО факты из предоставленных узлов и связей.
            Стиль: формально-деловой, структурированный (ГОСТ).
            Пиши только текст самого раздела, используй Markdown. Не пиши введения от себя."""
            
            data_str = f"УЗЛЫ РАЗДЕЛА:\n{node_context}\n\nСВЯЗИ ПРОЕКТА:\n{edge_context}"
            
            content = await acall_llm_text(prompt=prompt, data=data_str)
            
            sections_data.append(GeneratedSection(
                section_id=sec_enum,
                title=sec_title,
                content_markdown=content,
                used_node_ids=[n.id for n in relevant_nodes]
            ))

        return FullTZDocument(
            project_name="Генерируемый Проект",
            version="1.0.0",
            sections=sections_data
        )