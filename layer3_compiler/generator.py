import logging
import asyncio
from typing import List, Dict
from schemas.graph import UnifiedGraph
from schemas.enums import TZSectionEnum, TZStandardEnum
from schemas.document import (
    FinalExportDocument, 
    TitlePageData, 
    DocumentSection
)
from utils.llm_client import acall_llm_text

logger = logging.getLogger(__name__)

class TZGenerator:
    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        self.model_name = model_name

    async def generate_tz(self, graph: UnifiedGraph, standard: TZStandardEnum = TZStandardEnum.GOST_34) -> FinalExportDocument:
        logger.info(f"📝 СЛОЙ 3: Генерация документа ТЗ ({standard.value})...")

        # Последовательная генерация
        raw_sections_map = await self._generate_raw_content(graph)

        if standard == TZStandardEnum.GOST_34:
            return self._map_to_gost_34(raw_sections_map)
        else:
            return self._map_to_simple(raw_sections_map)

    async def _generate_raw_content(self, graph: UnifiedGraph) -> Dict[str, str]:
        results_map = {}
        
        sections = [
            TZSectionEnum.GENERAL,
            TZSectionEnum.FUNCTIONAL,
            TZSectionEnum.STACK,
            TZSectionEnum.INTERFACE
        ]

        # Генерируем по очереди с паузами
        for i, sec_enum in enumerate(sections):
            logger.info(f"  -> Генерация секции [{i+1}/{len(sections)}]: {sec_enum.value}")
            
            # --- ПАУЗА 5 СЕКУНД ---
            await asyncio.sleep(5)
            
            content = await self._generate_section_text(sec_enum, graph)
            if content:
                results_map[sec_enum] = content

        # Назначение (отдельно)
        logger.info("  -> Генерация назначения...")
        await asyncio.sleep(5)
        results_map["PURPOSE"] = await self._generate_purpose_text(graph)

        return results_map

    async def _generate_section_text(self, sec_enum: TZSectionEnum, graph: UnifiedGraph) -> str:
        # Фильтруем узлы для этой секции
        nodes = [n for n in graph.nodes if n.target_section == sec_enum]
        
        context = "\n".join([f"- {n.name}: {n.description}" for n in nodes])
        if not context:
            context = "Данных нет. Напиши стандартный текст для этого раздела."

        prompt = f"""
        Напиши раздел ТЗ (ГОСТ 34) на тему: '{sec_enum.value}'.
        Факты из проекта:
        {context}
        
        Дополни текст профессиональными формулировками, чтобы он выглядел полным.
        """
        
        try:
            return await acall_llm_text(prompt=prompt, model_name=self.model_name)
        except Exception as e:
            logger.error(f"Ошибка генерации {sec_enum}: {e}")
            return "Ошибка генерации раздела."

    async def _generate_purpose_text(self, graph: UnifiedGraph) -> str:
        context = "\n".join([f"- {n.name}" for n in graph.nodes[:20]])
        prompt = f"Напиши цель создания системы (2-3 предложения) на основе списка компонентов:\n{context}"
        try:
            return await acall_llm_text(prompt, self.model_name)
        except:
            return "Система предназначена для автоматизации процессов."

    def _map_to_gost_34(self, content_map: Dict[str, str]) -> FinalExportDocument:
        sec_1 = DocumentSection(
            number="1.", title="Общие сведения", 
            content=content_map.get(TZSectionEnum.GENERAL, "")
        )
        sec_2 = DocumentSection(
            number="2.", title="Назначение и цели", 
            content=content_map.get("PURPOSE", "")
        )
        sec_3 = DocumentSection(
            number="3.", title="Требования к системе",
            content="Система должна соответствовать требованиям:",
            subsections=[
                DocumentSection(number="3.1.", title="Функциональные требования", content=content_map.get(TZSectionEnum.FUNCTIONAL, "")),
                DocumentSection(number="3.2.", title="Стек технологий", content=content_map.get(TZSectionEnum.STACK, "")),
                DocumentSection(number="3.3.", title="Интерфейс", content=content_map.get(TZSectionEnum.INTERFACE, ""))
            ]
        )

        return FinalExportDocument(
            standard=TZStandardEnum.GOST_34,
            title_page=TitlePageData(project_name="AI Project"),
            structure=[sec_1, sec_2, sec_3]
        )

    def _map_to_simple(self, content_map):
        return FinalExportDocument(
            standard=TZStandardEnum.SIMPLE_MD,
            title_page=TitlePageData(project_name="Simple"),
            structure=[]
        )