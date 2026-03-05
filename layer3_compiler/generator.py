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
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
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

        # Генерируем по очереди
        for i, sec_enum in enumerate(sections):
            logger.info(f"  -> Генерация секции [{i+1}/{len(sections)}]: {sec_enum.value}")
            
            # --- ЗАДЕРЖКА (Rate Limits) ---
            await asyncio.sleep(5)
            
            content = await self._generate_section_text(sec_enum, graph)
            if content:
                results_map[sec_enum] = content

        # Назначение (отдельно)
        logger.info("  -> Генерация назначения (Purpose)...")
        await asyncio.sleep(5)
        results_map["PURPOSE"] = await self._generate_purpose_text(graph)

        return results_map

    async def _generate_section_text(self, sec_enum: TZSectionEnum, graph: UnifiedGraph) -> str:
        # 1. Собираем узлы для этой секции
        nodes = [n for n in graph.nodes if n.target_section == sec_enum]
        
        # ХАК: Если для "General Info" пусто, берем вообще все Requirement и Task
        if sec_enum == TZSectionEnum.GENERAL and len(nodes) < 2:
            logger.info("    (Мало данных для General, берем глобальный контекст)")
            nodes = [n for n in graph.nodes if n.label in ["Requirement", "Task", "Concept"]]

        # Логируем, что мы нашли
        logger.info(f"    Найдено {len(nodes)} фактов для {sec_enum.value}")
        if len(nodes) > 0:
            logger.info(f"    Примеры: {[n.name for n in nodes[:3]]}...")

        # Формируем контекст
        context_lines = []
        for n in nodes:
            line = f"- {n.name} ({n.label}): {n.description}"
            if n.properties:
                line += f" (Детали: {n.properties})"
            context_lines.append(line)
        
        context_str = "\n".join(context_lines)

        if not context_str:
            return "Нет данных для формирования раздела."

        # МОЩНЫЙ ПРОМПТ
        prompt = f"""
        ТЫ: Профессиональный Технический Писатель. Твоя задача - написать раздел Технического Задания (ГОСТ 34).
        
        РАЗДЕЛ: {sec_enum.value}
        
        ВХОДНЫЕ ФАКТЫ (ИЗ ГРАФА ЗНАНИЙ):
        {context_str}
        
        СТРОГИЕ ПРАВИЛА:
        1. ИСПОЛЬЗУЙ ТОЛЬКО ФАКТЫ ВЫШЕ. Не выдумывай функционал, которого нет в списке.
        2. ЗАПРЕЩЕНО писать фразы типа "Раздел в разработке", "Нет данных", "Ниже приведен шаблон".
        3. Если фактов мало - напиши коротко, но КОНКРЕТНО по этим фактам.
        4. Стиль: формально-деловой, сухой, технический. Без маркетинга.
        5. Используй Markdown (заголовки, списки).
        
        Пример хорошего ответа:
        "Система должна обеспечивать авторизацию через JWT токены. Для хранения данных использовать PostgreSQL 15."
        """
        
        try:
            return await acall_llm_text(prompt=prompt, model_name=self.model_name)
        except Exception as e:
            logger.error(f"Ошибка генерации {sec_enum}: {e}")
            return "Ошибка генерации раздела."

    async def _generate_purpose_text(self, graph: UnifiedGraph) -> str:
        # Для "Цели создания" берем Задачи и Требования
        target_nodes = [n for n in graph.nodes if n.label in ["Task", "Requirement"]]
        if not target_nodes:
            target_nodes = graph.nodes[:10] # Fallback

        context = "\n".join([f"- {n.name}: {n.description}" for n in target_nodes])
        
        prompt = f"""
        Напиши раздел "Назначение и цели создания системы" (2-3 абзаца).
        Основывайся на этих требованиях:
        {context}
        
        Пиши конкретно. Не используй общие фразы про "эффективное управление".
        Пиши: "Система предназначена для [конкретная задача из списка] с использованием [технология из списка]."
        """
        try:
            return await acall_llm_text(prompt, self.model_name)
        except:
            return "Система предназначена для автоматизации процессов заказчика."

    def _map_to_gost_34(self, content_map: Dict[str, str]) -> FinalExportDocument:
        sec_1 = DocumentSection(
            number="1.", title="Общие сведения", 
            content=content_map.get(TZSectionEnum.GENERAL, "Описание системы отсутствует.")
        )
        sec_2 = DocumentSection(
            number="2.", title="Назначение и цели", 
            content=content_map.get("PURPOSE", "")
        )
        sec_3 = DocumentSection(
            number="3.", title="Требования к системе",
            content="Система должна соответствовать следующим требованиям:",
            subsections=[
                DocumentSection(number="3.1.", title="Функциональные требования", content=content_map.get(TZSectionEnum.FUNCTIONAL, "")),
                DocumentSection(number="3.2.", title="Стек технологий", content=content_map.get(TZSectionEnum.STACK, "")),
                DocumentSection(number="3.3.", title="Интерфейс", content=content_map.get(TZSectionEnum.INTERFACE, ""))
            ]
        )

        return FinalExportDocument(
            standard=TZStandardEnum.GOST_34,
            title_page=TitlePageData(project_name="AI Generated Project"),
            structure=[sec_1, sec_2, sec_3]
        )

    def _map_to_simple(self, content_map):
        return FinalExportDocument(
            standard=TZStandardEnum.SIMPLE_MD,
            title_page=TitlePageData(project_name="Simple"),
            structure=[]
        )