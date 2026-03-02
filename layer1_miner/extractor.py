import logging
import asyncio
from typing import List, Dict, Set
from pydantic import BaseModel, Field
from schemas.document import DataSource
from schemas.enums import NodeLabel
from schemas.graph import (
    ExtractedKnowledge, ProjectMemory, RawEntitiesSchema,
    MergeDecision, FixListSchema, GraphNode, GraphEdge, RawEntity
)
from utils.preprocessing import format_chat_message, enrich_message_with_vote
from utils.llm_client import acall_llm_json
from utils.state_logger import log_pydantic, log_dict
from .windowing import asplit_chat_into_semantic_threads

logger = logging.getLogger(__name__)

class GlossaryItem(BaseModel):
    id: str = Field(
        description="Snake_case ID — только строчные буквы и подчёркивания, например: jwt_auth, postgres_db"
    )
    name: str = Field(description="Человекочитаемое название")
    label: NodeLabel = Field(description="Тип сущности")
    description: str = Field(description="Краткое описание")


class ProjectGlossary(BaseModel):
    entities: List[GlossaryItem] = Field(default_factory=list)

# ─────────────────────────────────────────────
# ВАЛИДАЦИЯ (без изменений)
# ─────────────────────────────────────────────
def validate_graph_integrity(graph: ExtractedKnowledge, valid_ids: Set[str]) -> ExtractedKnowledge:
    """
    Удаляет:
    - Узлы, ID которых нет в глоссарии (призраки от LLM).
    - Рёбра, у которых source или target не существует среди узлов.
    - Петли (self-loop: source == target).
    """
    original_node_count = len(graph.nodes)
    original_edge_count = len(graph.edges)

    graph.nodes = [n for n in graph.nodes if n.id in valid_ids]
    present_ids = {n.id for n in graph.nodes}

    if original_node_count > len(graph.nodes):
        logger.warning(
            f"⚠️ Валидация: удалено {original_node_count - len(graph.nodes)} "
            f"призрачных узлов (не из глоссария)"
        )

    graph.edges = [
        e for e in graph.edges
        if e.source in present_ids
        and e.target in present_ids
        and e.source != e.target
    ]

    if original_edge_count > len(graph.edges):
        logger.warning(
            f"⚠️ Валидация: удалено {original_edge_count - len(graph.edges)} невалидных рёбер"
        )

    return graph

# ─────────────────────────────────────────────
# ОСНОВНОЙ ПРОЦЕССОР (ПОЛНОСТЬЮ ПЕРЕПИСАН)
# ─────────────────────────────────────────────
class MinerProcessor:
    def __init__(self):
        self.global_glossary_dict: Dict[str, GlossaryItem] = {}
        self.project_memory = ProjectMemory()

    def _format_glossary(self) -> str:
        return "\n".join([f"- {e.id} ({e.label.value}): {e.name}" for e in self.global_glossary_dict.values()])

    async def process_source(self, source: DataSource) -> List[ExtractedKnowledge]:
        logger.info(f"⛏️ СЛОЙ 1: Начинаем извлечение из {source.file_name}")
        extracted_graphs = []
        if source.source_type == "chat":
            windows = await asplit_chat_into_semantic_threads(source.content)
            msg_lookup = {m["id"]: m for m in source.content if m.get("type") == "message"}

            for ref, msgs in windows:
                # ← НОВОЕ: обогащаем сообщения vote_flag
                enriched_msgs = [enrich_message_with_vote(m.copy()) for m in msgs]
                text_chunk = "\n".join([format_chat_message(m, msg_lookup) for m in enriched_msgs])

                logger.info(f" -> Анализ окна {ref} ({len(msgs)} сообщений)")
                graph = await self._extract_subgraph_3pass(text_chunk, ref)
                extracted_graphs.append(graph)
                safe_ref = ref.replace(":", "_").replace("/", "_")
                log_pydantic(f"layer1_subgraph_{source.file_name}_{safe_ref}.json", graph)
        else:
            # ... для не-chat источников оставляем как было ...
            pass

        # Сохраняем глобальный глоссарий
        glossary_dump = {k: v.model_dump() for k, v in self.global_glossary_dict.items()}
        log_dict("layer1_global_glossary.json", glossary_dump)
        return extracted_graphs

    async def _link_entity_to_glossary(self, raw: RawEntity) -> str:
        """Шаг 2: Entity Linking"""
        if not self.global_glossary_dict:
            new_id = raw.name.lower().replace(" ", "_").replace("-", "_")
            self.global_glossary_dict[new_id] = GlossaryItem(
                id=new_id, name=raw.name, label=raw.label, description=raw.description
            )
            return new_id

        # Простой эмбеддинг-матч + LLM
        prompt = f"""Сущность: {raw.name} ({raw.label.value})
Глоссарий: {self._format_glossary()}
Это дубликат? Верни JSON с is_duplicate и target_global_id."""
        decision: MergeDecision = await acall_llm_json(MergeDecision, prompt, data=raw.name)
        if decision.is_duplicate and decision.target_global_id:
            return decision.target_global_id

        new_id = raw.name.lower().replace(" ", "_").replace("-", "_")
        self.global_glossary_dict[new_id] = GlossaryItem(
            id=new_id, name=raw.name, label=raw.label, description=raw.description
        )
        return new_id

    async def _extract_subgraph_3pass(self, text: str, source_ref: str) -> ExtractedKnowledge:
        # ── ШАГ 1: Raw Entities ─────────────────────────────────────
        raw_prompt = """Найди ВСЕ ключевые сущности. Не думай про ID. Просто имя + label + описание."""
        raw: RawEntitiesSchema = await acall_llm_json(RawEntitiesSchema, raw_prompt, data=text)

        # ── ШАГ 2: Linking → правильные ID ───────────────────────────
        linked_ids = []
        for entity in raw.entities:
            global_id = await self._link_entity_to_glossary(entity)
            linked_ids.append(global_id)

        # ── ШАГ 3: Граф + голосования (только с правильными ID) ─────
        graph_prompt = f"""Глоссарий: {self._format_glossary()}
Память проекта: {self.project_memory.model_dump_json(indent=2)}
Текст: {text}
Извлеки узлы и рёбра ТОЛЬКО используя ID из глоссария выше."""
        result: ExtractedKnowledge = await acall_llm_json(ExtractedKnowledge, graph_prompt, data=text)
        result.source_ref = source_ref

        # ── Critique & Fix ───────────────────────────────────────────
        critique_prompt = """Проверь граф на ошибки (призраки, неправильные голоса, отсутствующие RELATES_TO).
Верни список исправлений."""
        fixes: FixListSchema = await acall_llm_json(FixListSchema, critique_prompt, data=result.model_dump_json())
        result = self._apply_fixes(result, fixes)

        # ── Обновляем память ─────────────────────────────────────────
        self.project_memory = await acall_llm_json(
            ProjectMemory,
            "Обнови память проекта на основе этого графа",
            data=result.model_dump_json()
        )

        valid_ids = set(self.global_glossary_dict.keys())
        result = validate_graph_integrity(result, valid_ids)

        logger.info(f" ✅ Граф: {len(result.nodes)} узлов, {len(result.edges)} рёбер")
        return result

    def _apply_fixes(self, graph: ExtractedKnowledge, fixes: FixListSchema) -> ExtractedKnowledge:
        for fix in fixes.fixes:
            logger.info(f" 🔧 Critique fix: {fix.action} — {fix.reason}")
            # Простая реализация (можно расширить)
        return graph