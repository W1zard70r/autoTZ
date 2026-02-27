import logging
import asyncio
from typing import List, Dict
from pydantic import BaseModel, Field

from schemas.document import DataSource
from schemas.enums import NodeLabel
from schemas.graph import ExtractedKnowledge
from utils.preprocessing import format_chat_message
from utils.llm_client import acall_llm_json
from utils.state_logger import log_pydantic, log_dict
from .windowing import asplit_chat_into_semantic_threads

logger = logging.getLogger(__name__)


class GlossaryItem(BaseModel):
    id: str = Field(description="Snake_case ID")
    name: str = Field(description="Человекочитаемое название")
    label: NodeLabel = Field(description="Тип сущности")
    description: str = Field(description="Краткое описание")


class ProjectGlossary(BaseModel):
    entities: List[GlossaryItem] = Field(default_factory=list)


class MinerProcessor:
    def __init__(self):
        self.global_glossary_dict: Dict[str, GlossaryItem] = {}

    def _format_glossary(self) -> str:
        return "\n".join([f"- {e.id} ({e.label.value}): {e.name}" for e in self.global_glossary_dict.values()])

    async def process_source(self, source: DataSource) -> List[ExtractedKnowledge]:
        logger.info(f"⛏️ СЛОЙ 1: Начинаем извлечение из {source.file_name}")
        extracted_graphs = []
        previous_summary = ""

        if source.source_type == "chat":
            windows = await asplit_chat_into_semantic_threads(source.content)
            msg_lookup = {m["id"]: m for m in source.content if m.get("type") == "message"}

            for ref, msgs in windows:
                text_chunk = "\n".join([format_chat_message(m, msg_lookup) for m in msgs])
                logger.info(f"  -> Анализ окна {ref} ({len(msgs)} сообщений)")

                graph = await self._extract_subgraph_2pass(text_chunk, ref, previous_summary)
                previous_summary = graph.summary
                extracted_graphs.append(graph)
                
                # --- LOGGING: Сохраняем каждый извлеченный подграф ---
                safe_ref = ref.replace(":", "_").replace("/", "_")
                log_pydantic(f"layer1_subgraph_{source.file_name}_{safe_ref}.json", graph)
                
                await asyncio.sleep(2)
        else:
            graph = await self._extract_subgraph_2pass(str(source.content), source.file_name, previous_summary)
            extracted_graphs.append(graph)
            
            # --- LOGGING: Сохраняем подграф ---
            safe_ref = source.file_name.replace(":", "_").replace("/", "_")
            log_pydantic(f"layer1_subgraph_{safe_ref}.json", graph)

        # --- LOGGING: Дамп полного глоссария ---
        glossary_dump = {k: v.model_dump() for k, v in self.global_glossary_dict.items()}
        log_dict("layer1_global_glossary.json", glossary_dump)

        return extracted_graphs

    async def _extract_subgraph_2pass(self, text: str, source_ref: str, prev_summary: str) -> ExtractedKnowledge:
        glossary_prompt = f"""Найди все ключевые сущности в тексте (Люди, Компоненты, Задачи, Требования).
Если сущность уже есть в ГЛОССАРИИ ниже, используй ЕЕ СТАРЫЙ ID.
Если это новая сущность — создай новый snake_case ID.

СУЩЕСТВУЮЩИЙ ГЛОССАРИЙ:
{self._format_glossary() or 'Пока пусто.'}"""

        local_glossary = await acall_llm_json(schema=ProjectGlossary, prompt=glossary_prompt, data=text)

        for entity in local_glossary.entities:
            if entity.id not in self.global_glossary_dict:
                self.global_glossary_dict[entity.id] = entity

        graph_prompt = f"""Ты Архитектор. Извлеки граф знаний (узлы и связи).
СТРОГИЕ ПРАВИЛА:
1. Используй ТОЛЬКО ID из Глоссария проекта.
2. Обращай внимание на [FLAG: CONFIRMATION] (означает AGREES_WITH).
3. Добавляй evidence для каждой связи (почему ты их связал).

ГЛОССАРИЙ ПРОЕКТА:
{self._format_glossary()}

ПАМЯТЬ ПРОШЛЫХ ОКОН:
{prev_summary or 'Начало диалога.'}"""

        result = await acall_llm_json(schema=ExtractedKnowledge, prompt=graph_prompt, data=text)
        result.source_ref = source_ref

        for node in result.nodes:
            if node.id in self.global_glossary_dict:
                g_item = self.global_glossary_dict[node.id]
                if not node.name: node.name = g_item.name
                if not node.description: node.description = g_item.description
                if not node.label: node.label = g_item.label

        return result