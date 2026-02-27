import asyncio
import logging

import networkx as nx

from models import DataSource, DataEnum
from windowing import asplit_chat_into_semantic_threads, split_text_into_chunks
from preprocessing import format_chat_message_for_llm
from extractor import AsyncGraphExtractor
from graph_manager import KnowledgeGraphManager
from global_glossary import GlobalGlossary

logger = logging.getLogger(__name__)

class Layer1Processor:
    def __init__(self):
        self.global_glossary = GlobalGlossary()
        self.extractor = AsyncGraphExtractor(self.global_glossary)

    async def process_source(self, source: DataSource) -> KnowledgeGraphManager:
        logger.info(f"🚀 Запуск обработки: {source.file_name}")
        graph_manager = KnowledgeGraphManager(source.file_name)

        if source.source_type == DataEnum.CHAT:
            await self._process_chat_stateful(source, graph_manager)
        else:
            await self._process_text_stateful(source, graph_manager)

        # NEW: Вывод сообществ
        communities = nx.get_node_attributes(graph_manager.graph, "community")
        logger.info(f"Найдено сообществ: {len(set(communities.values()))}")

        return graph_manager

    async def _process_chat_stateful(self, source: DataSource, graph_manager: KnowledgeGraphManager):
        windows = await asplit_chat_into_semantic_threads(source.content)
        msg_lookup = {m["id"]: m for m in source.content if m.get("type") == "message"}
        previous_summary = ""

        for window_ref, window_msgs, _ in windows:
            logger.info(f" -> Окно: {window_ref} ({len(window_msgs)} сообщений)")
            lines = [format_chat_message_for_llm(m, msg_lookup) for m in window_msgs]
            text_content = "\n".join(lines)

            # Глоссарий → глобальный мёрдж
            glossary = await self.extractor.agenerate_glossary(text_content)
            glossary = self.global_glossary.merge(glossary)

            # Граф
            result = await self.extractor.aextract_graph(text_content, glossary, previous_summary)
            previous_summary = result.summary

            graph_manager.apply_extraction(result, f"{source.file_name}::{window_ref}")