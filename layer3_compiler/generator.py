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
            await asyncio.sleep(8) 
            
            nodes = [n for n in graph.nodes if n.target_section == sec]
            
            if not nodes:
                if sec == TZSectionEnum.FUNCTIONAL:
                    nodes = [n for n in graph.nodes if n.label in [NodeLabel.REQUIREMENT, NodeLabel.TASK]]
                elif sec == TZSectionEnum.STACK:
                    nodes = [n for n in graph.nodes if n.label == NodeLabel.COMPONENT]
                nodes = [n for n in nodes if n.label != NodeLabel.PERSON]

            if not nodes:
                content_map[sec] = "Данные не зафиксированы."
                continue

            logger.info(f"  -> Секция {sec.value}: {len(nodes)} фактов.")
            facts = "\n".join([f"- {n.name}: {n.description}" for n in nodes])
            
            # --- ИДЕАЛЬНЫЙ ПРОМПТ ---
            prompt = f"""
            ЗАДАЧА: Напиши ТЕКСТОВОЕ СОДЕРЖАНИЕ для раздела ТЗ: "{sec.value}".
            Твой ответ будет вставлен в готовое поле Pydantic-модели.
            
            ДАННЫЕ ДЛЯ ОПИСАНИЯ:
            {facts}
            
            СТРОЖАЙШИЕ ПРАВИЛА ОФОРМЛЕНИЯ:
            1. ЗАПРЕЩЕНО использовать заголовки (символы #, ##, ###).
            2. ЗАПРЕЩЕНО использовать нумерацию типа "1.1.", "2.1.3" и так далее. 
            3. Используй только обычные абзацы и маркированные списки (через символ -).
            4. Пиши СРАЗУ ПО СУТИ. Не пиши вводных фраз типа "В этом разделе мы рассмотрим...".
            
            ПРИМЕР ПЛОХОГО ОТВЕТА:
            "## 2. Функции. 2.1. Система должна..."
            
            ПРИМЕР ХОРОШЕГО ОТВЕТА:
            "Система обеспечивает автоматизацию обучения. Основные возможности включают:
            - Назначение курсов сотрудникам.
            - Просмотр аналитики."

            Стиль: Технический ГОСТ. Язык: Русский.
            """
            
            content_map[sec] = await acall_llm_text(prompt, model_name=self.model_name)
            logger.info(f"  -> Секция {sec.value} заполнена успешно.")

        return self._build_gost_document(content_map)

    def _build_gost_document(self, content_map: Dict) -> FinalExportDocument:
        """Сборка Pydantic-объекта со строгой структурой"""
        
        # Разделы создаются кодом. Номера 1, 2, 3... уже здесь.
        return FinalExportDocument(
            standard=TZStandardEnum.GOST_34,
            title_page=TitlePageData(project_name="EduPlatform - Система корпоративного обучения"),
            structure=[
                DocumentSection(number="1.", title="Общие сведения", content=content_map.get(TZSectionEnum.GENERAL, "")),
                DocumentSection(number="2.", title="Функциональные требования", content=content_map.get(TZSectionEnum.FUNCTIONAL, "")),
                DocumentSection(number="3.", title="Технический стек", content=content_map.get(TZSectionEnum.STACK, "")),
                DocumentSection(number="4.", title="Интерфейс и UX", content=content_map.get(TZSectionEnum.INTERFACE, ""))
            ]
        )