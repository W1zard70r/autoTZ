import logging
import asyncio
from typing import List
from schemas.inputs import DataSource
from schemas.graph import ExtractedKnowledge
from utils.preprocessing import format_chat_message
from utils.llm_client import acall_llm_json
from .windowing import asplit_chat_into_semantic_threads

logger = logging.getLogger(__name__)

class MinerProcessor:
    async def process_source(self, source: DataSource) -> List[ExtractedKnowledge]:
        logger.info(f"⛏️ СЛОЙ 1: Начинаем извлечение из {source.file_name}")
        extracted_graphs = []

        if source.source_type == "chat":
            windows = await asplit_chat_into_semantic_threads(source.content)
            msg_lookup = {m["id"]: m for m in source.content if m.get("type") == "message"}
            
            # Выполняем извлечение последовательно, чтобы не словить Rate Limit
            for ref, msgs in windows:
                text_chunk = "\n".join([format_chat_message(m, msg_lookup) for m in msgs])
                logger.info(f"  -> Анализ окна {ref} ({len(msgs)} сообщений)")
                
                graph = await self._extract_subgraph(text_chunk, ref)
                extracted_graphs.append(graph)
                await asyncio.sleep(2) # Защита от спама API
        else:
            # Для обычных документов логика нарезки текста (упрощенно)
            graph = await self._extract_subgraph(str(source.content), source.file_name)
            extracted_graphs.append(graph)

        return extracted_graphs

    async def _extract_subgraph(self, text: str, source_ref: str) -> ExtractedKnowledge:
        prompt = """Ты Аналитик Данных и Архитектор. Извлеки из текста граф знаний.
        Сущности могут быть: Person, Component (БД, Либы), Task, Requirement, Concept.
        1. Для узлов придумай понятный snake_case id (напр. comp_postgres).
        2. Извлеки связи между ними. Выбирай relation из Enum.
        3. [FLAG: CONFIRMATION] означает AGREES_WITH.
        Обязательно добавь evidence (цитату) для каждой связи."""
        
        result = await acall_llm_json(schema=ExtractedKnowledge, prompt=prompt, data=text)
        result.source_ref = source_ref
        return result