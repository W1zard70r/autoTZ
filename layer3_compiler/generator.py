import logging
import asyncio
from typing import List, Dict
from schemas.graph import UnifiedGraph
from schemas.enums import TZSectionEnum, TZStandardEnum, NodeLabel
from schemas.document import FinalExportDocument, TitlePageData, DocumentSection
from utils.llm_client import acall_llm_text

logger = logging.getLogger(__name__)

class TZGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.model_name = model_name

    async def generate_tz(self, graph: UnifiedGraph, standard: TZStandardEnum = TZStandardEnum.GOST_34) -> FinalExportDocument:
        logger.info("📝 СЛОЙ 3: Заполнение Pydantic-схемы данными от LLM...")
        
        content_map = {}
        sections_to_generate = [
            TZSectionEnum.GENERAL, 
            TZSectionEnum.FUNCTIONAL, 
            TZSectionEnum.STACK, 
            TZSectionEnum.INTERFACE
        ]
        
        for sec in sections_to_generate:
            # Увеличили паузу между разделами для стабильности
            await asyncio.sleep(8) 
            
            nodes = [n for n in graph.nodes if n.target_section == sec]
            
            # Fallback: если классификатор пуст
            if not nodes:
                if sec == TZSectionEnum.FUNCTIONAL:
                    nodes = [n for n in graph.nodes if n.label in [NodeLabel.REQUIREMENT, NodeLabel.TASK]]
                elif sec == TZSectionEnum.STACK:
                    nodes = [n for n in graph.nodes if n.label == NodeLabel.COMPONENT]

            if not nodes:
                content_map[sec] = "Требования по данному разделу не зафиксированы в обсуждении."
                continue

            logger.info(f"  -> Генерация контента для {sec.value} ({len(nodes)} фактов)...")
            facts = "\n".join([f"- {n.name}: {n.description}" for n in nodes])
            
            prompt = f"""
            ЗАДАЧА: Напиши содержательную часть раздела ТЗ на тему: "{sec.value}".
            ДАННЫЕ:
            {facts}
            
            ИНСТРУКЦИЯ:
            1. Пиши ТОЛЬКО основной текст (абзацы, списки). 
            2. ЗАПРЕЩЕНО писать заголовки типа "# Раздел" или "1.1. Название".
            3. ЗАПРЕЩЕНО использовать нумерацию строк типа "2.1.1".
            4. Стиль: формальный, технический (ГОСТ 34).
            5. Язык: Русский.
            """
            
            # Мы убрали внутренний try-except. 
            # Теперь ошибка пробросится в acall_llm_text, где сработает 20 попыток ретрая.
            content_map[sec] = await acall_llm_text(prompt, model_name=self.model_name)
            logger.info(f"  -> Секция {sec.value} успешно заполнена.")

        return self._build_gost_document(content_map)

    def _build_gost_document(self, content_map: Dict) -> FinalExportDocument:
        """Сборка итогового Pydantic-объекта"""
        
        s1 = DocumentSection(
            number="1.", title="Общие сведения", 
            content=content_map.get(TZSectionEnum.GENERAL, "")
        )
        
        s2 = DocumentSection(
            number="2.", title="Функциональные требования",
            content=content_map.get(TZSectionEnum.FUNCTIONAL, "")
        )
        
        s3 = DocumentSection(
            number="3.", title="Технический стек", 
            content=content_map.get(TZSectionEnum.STACK, "")
        )
        
        s4 = DocumentSection(
            number="4.", title="Интерфейс и UX", 
            content=content_map.get(TZSectionEnum.INTERFACE, "")
        )

        return FinalExportDocument(
            standard=TZStandardEnum.GOST_34,
            title_page=TitlePageData(project_name="EduPlatform - Система корпоративного обучения"),
            structure=[s1, s2, s3, s4]
        )