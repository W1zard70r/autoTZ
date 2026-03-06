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
        logger.info("📝 СЛОЙ 3: Генерация документа на основе семантического анализа графа...")
        content_map = {}
        sections = [TZSectionEnum.GENERAL, TZSectionEnum.FUNCTIONAL, TZSectionEnum.STACK, TZSectionEnum.INTERFACE]
        
        # Подготавливаем полный список фактов (технических)
        all_tech_nodes = [n for n in graph.nodes if n.label != NodeLabel.PERSON]
        full_context = "\n".join([f"ID: {n.id} | Сущность: {n.name} | Описание: {n.description}" for n in all_tech_nodes])

        for sec in sections:
            await asyncio.sleep(10)
            
            # Вместо жесткой фильтрации, мы просим LLM: 
            # "Вот весь граф. Выбери из него то, что подходит для этого раздела и опиши".
            
            prompt = f"""
            ЗАДАЧА: Напиши развернутый текст для раздела ТЗ: "{sec.value}".
            
            ВЕСЬ СПИСОК ФАКТОВ ПРОЕКТА:
            {full_context}
            
            ИНСТРУКЦИЯ:
            1. Выбери из списка выше только те факты, которые СЕМАНТИЧЕСКИ относятся к теме "{sec.value}".
            2. Сгруппируй их в 3-4 логических блока.
            3. Формат: **[Название блока]**: [Развернутое техническое описание].
            4. Пиши утвердительно, в стиле ГОСТ. 
            5. Если в списке нет фактов для этой темы, напиши: "Данные не зафиксированы".
            6. ЗАПРЕЩЕНО использовать заголовки (#) и нумерацию.
            
            ЯЗЫК: РУССКИЙ.
            """
            
            try:
                result = await acall_llm_text(prompt, model_name=self.model_name)
                # Если LLM выдала "Данные не зафиксированы", попробуем поискать хоть что-то по тегам
                if "не зафиксированы" in result.lower():
                    tagged_nodes = [n for n in graph.nodes if n.target_section == sec]
                    if tagged_nodes:
                        logger.info(f"   -> Режим Fallback для {sec.value} (по тегам)")
                        retry_facts = "\n".join([f"- {n.name}: {n.description}" for n in tagged_nodes])
                        result = await acall_llm_text(f"Опиши эти факты для раздела {sec.value}:\n{retry_facts}", model_name=self.model_name)
                
                content_map[sec] = result
                logger.info(f"  -> Раздел {sec.value} сформирован.")
            except:
                content_map[sec] = "Ошибка генерации раздела."

        return self._build_gost_document(content_map)

    def _build_gost_document(self, content_map: Dict) -> FinalExportDocument:
        return FinalExportDocument(
            standard=TZStandardEnum.GOST_34,
            title_page=TitlePageData(project_name="EduPlatform"),
            structure=[
                DocumentSection(number="1.", title="Общие сведения", content=content_map.get(TZSectionEnum.GENERAL, "")),
                DocumentSection(number="2.", title="Функциональные требования", content=content_map.get(TZSectionEnum.FUNCTIONAL, "")),
                DocumentSection(number="3.", title="Технический стек", content=content_map.get(TZSectionEnum.STACK, "")),
                DocumentSection(number="4.", title="Интерфейс и UX", content=content_map.get(TZSectionEnum.INTERFACE, ""))
            ]
        )