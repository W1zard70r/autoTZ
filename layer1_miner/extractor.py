import logging
import asyncio
from typing import List, Dict, Set
from pydantic import BaseModel, Field

from schemas.document import DataSource
from schemas.enums import NodeLabel
from schemas.graph import ExtractedKnowledge
from utils.preprocessing import format_chat_message
from utils.llm_client import acall_llm_json
from utils.state_logger import log_pydantic, log_dict
from .windowing import asplit_chat_into_semantic_threads

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Вспомогательные модели
# ─────────────────────────────────────────────

class GlossaryItem(BaseModel):
    id: str = Field(description="Snake_case ID — только строчные буквы и подчёркивания, например: jwt_auth, postgres_db")
    name: str = Field(description="Человекочитаемое название")
    label: NodeLabel = Field(description="Тип сущности")
    description: str = Field(description="Краткое описание")


class ProjectGlossary(BaseModel):
    entities: List[GlossaryItem] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Валидация целостности графа (Улучшение 5)
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

    # Фильтруем узлы
    graph.nodes = [n for n in graph.nodes if n.id in valid_ids]
    present_ids = {n.id for n in graph.nodes}

    # Логируем "призрачные" узлы
    ghost_ids = {n.id for n in graph.nodes} - valid_ids  # после фильтрации всегда пусто, логируем ДО
    all_returned_ids = set()  # пересчитаем из оригинального списка ниже
    ghost_nodes_before = valid_ids - present_ids  # узлы глоссария, которые LLM вообще не вернул — нормально
    # Настоящие призраки: узлы, которые LLM вернул, но которых нет в глоссарии
    if original_node_count > len(graph.nodes):
        logger.warning(
            f"⚠️ Валидация: удалено {original_node_count - len(graph.nodes)} призрачных узлов (не из глоссария)"
        )

    # Фильтруем рёбра
    graph.edges = [
        e for e in graph.edges
        if e.source in present_ids
        and e.target in present_ids
        and e.source != e.target  # убираем петли
    ]

    if original_edge_count > len(graph.edges):
        logger.warning(
            f"⚠️ Валидация: удалено {original_edge_count - len(graph.edges)} невалидных рёбер"
        )

    return graph


# ─────────────────────────────────────────────
# Основной процессор
# ─────────────────────────────────────────────

class MinerProcessor:
    def __init__(self):
        self.global_glossary_dict: Dict[str, GlossaryItem] = {}

    def _format_glossary(self) -> str:
        return "\n".join(
            [f"- {e.id} ({e.label.value}): {e.name}" for e in self.global_glossary_dict.values()]
        )

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

                safe_ref = ref.replace(":", "_").replace("/", "_")
                log_pydantic(f"layer1_subgraph_{source.file_name}_{safe_ref}.json", graph)

                # Убрали жёсткий sleep(2) — tenacity в acall_llm_json сам управляет ретраями
        else:
            graph = await self._extract_subgraph_2pass(
                str(source.content), source.file_name, previous_summary
            )
            extracted_graphs.append(graph)

            safe_ref = source.file_name.replace(":", "_").replace("/", "_")
            log_pydantic(f"layer1_subgraph_{safe_ref}.json", graph)

        glossary_dump = {k: v.model_dump() for k, v in self.global_glossary_dict.items()}
        log_dict("layer1_global_glossary.json", glossary_dump)

        return extracted_graphs

    async def _extract_subgraph_2pass(
        self, text: str, source_ref: str, prev_summary: str
    ) -> ExtractedKnowledge:

        # ── ПРОХОД 1: Извлечение глоссария (Улучшение 2 — few-shot примеры) ──────────────

        glossary_prompt = f"""Ты — аналитик. Найди ВСЕ ключевые сущности в тексте ниже.

СТРОГИЕ ПРАВИЛА:
1. ID — ТОЛЬКО snake_case: строчные буквы, цифры и подчёркивания. Пробелы и CamelCase ЗАПРЕЩЕНЫ.
   ✅ Правильно: jwt_auth, postgres_db, login_screen
   ❌ Неправильно: JwtAuth, "JWT Auth", loginScreen
2. Если сущность семантически совпадает с ГЛОССАРИЕМ — используй ЕЁ СТАРЫЙ ID без изменений.
3. Не дроби одну сущность на несколько (не создавай отдельно "токен" и "JWT токен" — это одно).
4. Тип (label) выбирай строго из: Person, Component, Task, Requirement, Concept.

FEW-SHOT ПРИМЕР:
Текст: "Алексей предложил поднять FastAPI-сервис с PostgreSQL и прикрутить JWT авторизацию"
Ответ entities:
  - id: alex_lead,       label: Person,      name: Алексей
  - id: fastapi_service, label: Component,   name: FastAPI сервис
  - id: postgresql_db,   label: Component,   name: PostgreSQL
  - id: jwt_auth,        label: Component,   name: JWT авторизация

СУЩЕСТВУЮЩИЙ ГЛОССАРИЙ (используй эти ID если сущность уже есть):
{self._format_glossary() or "Пока пусто — все сущности новые."}
"""

        local_glossary: ProjectGlossary = await acall_llm_json(
            schema=ProjectGlossary, prompt=glossary_prompt, data=text
        )

        # Нормализуем ID (на случай если LLM всё же вернул CamelCase)
        for entity in local_glossary.entities:
            entity.id = entity.id.strip().lower().replace(" ", "_").replace("-", "_")
            if entity.id not in self.global_glossary_dict:
                self.global_glossary_dict[entity.id] = entity

        valid_ids: Set[str] = set(self.global_glossary_dict.keys())

        # ── ПРОХОД 2: Извлечение графа (Улучшение 2 — few-shot примеры) ─────────────────

        graph_prompt = f"""Ты — Архитектор знаний. Извлеки граф (узлы + связи) из текста.

СТРОГИЕ ПРАВИЛА:
1. Используй ТОЛЬКО ID из «Глоссария проекта» ниже — никаких новых ID.
2. Рёбра: source и target должны быть разными узлами из глоссария.
3. [FLAG: CONFIRMATION] в тексте → тип связи AGREES_WITH.
4. evidence — короткая цитата или объяснение, почему эти узлы связаны.
5. summary — 2-3 предложения о главном в этом фрагменте (для памяти следующих окон).

FEW-SHOT ПРИМЕР:
Глоссарий: alex_lead (Person), fastapi_service (Component), jwt_auth (Component)
Текст: "Алексей сказал: поднимаем FastAPI с JWT"
Ответ nodes: [alex_lead, fastapi_service, jwt_auth]
Ответ edges:
  - source: alex_lead      → target: fastapi_service, relation: MENTIONS,    evidence: "поднимаем FastAPI"
  - source: fastapi_service → target: jwt_auth,        relation: DEPENDS_ON, evidence: "FastAPI с JWT"

ГЛОССАРИЙ ПРОЕКТА (разрешённые ID):
{self._format_glossary()}

ПАМЯТЬ ПРОШЛЫХ ОКОН:
{prev_summary or "Начало диалога — предыдущего контекста нет."}
"""

        result: ExtractedKnowledge = await acall_llm_json(
            schema=ExtractedKnowledge, prompt=graph_prompt, data=text
        )
        result.source_ref = source_ref

        # Обогащаем узлы данными из глоссария
        for node in result.nodes:
            if node.id in self.global_glossary_dict:
                g_item = self.global_glossary_dict[node.id]
                if not node.name:        node.name = g_item.name
                if not node.description: node.description = g_item.description
                if not node.label:       node.label = g_item.label

        # ── Улучшение 1 + 5: Валидация — удаляем призраков и битые рёбра ─────────────────
        result = validate_graph_integrity(result, valid_ids)

        logger.info(
            f"     ✅ Граф: {len(result.nodes)} узлов, {len(result.edges)} рёбер (после валидации)"
        )
        return result