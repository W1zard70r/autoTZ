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
        logger.info(f"⛏️ СЛОЙ 1: Извлечение из {source.file_name}")
        extracted_graphs = []
        previous_summary = ""

        if source.source_type == "chat":
            windows = await asplit_chat_into_semantic_threads(source.content)
            msg_lookup = {m["id"]: m for m in source.content if m.get("type") == "message"}

            logger.info(f"  -> Найдено {len(windows)} окон.")

            for i, (ref, msgs) in enumerate(windows):
                text_chunk = "\n".join([format_chat_message(m, msg_lookup) for m in msgs])
                logger.info(f"  -> [{i+1}/{len(windows)}] Анализ окна {ref}")

                # ВАЖНО: Мы убрали ручной цикл while. 
                # Теперь всю работу делает декоратор @retry в llm_client.py
                graph = await self._extract_subgraph_2pass(text_chunk, ref, previous_summary)
                previous_summary = graph.summary
                extracted_graphs.append(graph)
                
                safe_ref = ref.replace(":", "_").replace("/", "_")
                log_pydantic(f"layer1_subgraph_{source.file_name}_{safe_ref}.json", graph)
                
                await asyncio.sleep(4) # Базовая задержка между окнами

        else:
            graph = await self._extract_subgraph_2pass(str(source.content), source.file_name, previous_summary)
            extracted_graphs.append(graph)
            log_pydantic(f"layer1_subgraph_{source.file_name}.json", graph)

        log_dict("layer1_global_glossary.json", {k: v.model_dump() for k, v in self.global_glossary_dict.items()})
        return extracted_graphs

    async def _extract_subgraph_2pass(self, text: str, source_ref: str, prev_summary: str) -> ExtractedKnowledge:
        # Пасс 1: Глоссарий
        glossary_prompt = f"Найди ключевые сущности. Используй существующие ID или создай новые snake_case.\nГЛОССАРИЙ:\n{self._format_glossary()}"
        local_glossary = await acall_llm_json(ProjectGlossary, glossary_prompt, data=text)

        for entity in local_glossary.entities:
            if entity.id not in self.global_glossary_dict:
                self.global_glossary_dict[entity.id] = entity

        # Пасс 2: Граф
        graph_prompt = f"Извлеки граф знаний. Используй ID из глоссария.\nГЛОССАРИЙ:\n{self._format_glossary()}\nПАМЯТЬ:\n{prev_summary}"
        result = await acall_llm_json(ExtractedKnowledge, graph_prompt, data=text)
        result.source_ref = source_ref

        # Обогащение данных из глоссария
        for node in result.nodes:
            if node.id in self.global_glossary_dict:
                g = self.global_glossary_dict[node.id]
                node.name, node.description, node.label = g.name, g.description, g.label

        return result