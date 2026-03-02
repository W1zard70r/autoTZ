import logging
import asyncio
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel, Field

from schemas.graph import ExtractedKnowledge, UnifiedGraph, GraphNode, GraphEdge, Conflict
from schemas.enums import TZSectionEnum, NodeLabel
from utils.llm_client import acall_llm_json
from utils.state_logger import log_graphml, log_pydantic

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Вспомогательные модели
# ─────

class MergeAction(BaseModel):
    is_duplicate: bool = Field(description="Это одна и та же сущность?")
    ids_to_merge: List[str] = Field(description="Список ID, которые нужно слить в один")
    unified_id: str = Field(description="Новый ID для слитого узла")
    unified_name: str = Field(description="Общее имя")
    unified_desc: str = Field(description="Объединённое описание")


class MergeBatchResult(BaseModel):
    actions: List[MergeAction] = Field(default_factory=list)


class SectionAssignment(BaseModel):
    node_id: str
    target_section: TZSectionEnum


class SectionBatchResult(BaseModel):
    assignments: List[SectionAssignment]


# ─────────────────────────────────────────────
# Embedding utils (Улучшение 4)
# ─────────────────────────────────────────────

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


async def _get_embeddings(texts: List[str]) -> List[List[float]]:
    """Получает эмбеддинги батчами с задержкой."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    embeddings: List[List[float]] = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            result = await model.aembed_documents(batch)
            embeddings.extend(result)
        except Exception as e:
            logger.warning(f"Ошибка эмбеддингов (батч {i}): {e}")
            embeddings.extend([[0.0] * 768] * len(batch))
        await asyncio.sleep(0.5)

    return embeddings


async def _find_duplicate_candidates(
    nodes: List[Dict[str, Any]],
    similarity_threshold: float = 0.88,
) -> List[Tuple[Dict, Dict, float]]:
    """
    Улучшение 4: Embedding-based pre-filter для дедупликации.

    Сначала вычисляем эмбеддинги для всех узлов, затем попарно находим
    кандидатов с cosine similarity > threshold. В LLM отправляем ТОЛЬКО
    эти пары — это решает проблему cross-batch дубликатов и резко
    сокращает число LLM-запросов.
    """
    if len(nodes) < 2:
        return []

    texts = [f"{n['name']} {n.get('desc', '')}".strip() for n in nodes]
    embeddings = await _get_embeddings(texts)

    candidates: List[Tuple[Dict, Dict, float]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= similarity_threshold:
                candidates.append((nodes[i], nodes[j], sim))

    logger.info(
        f"  -> Embedding pre-filter: {len(nodes)} узлов → "
        f"{len(candidates)} пар-кандидатов на дедупликацию (threshold={similarity_threshold})"
    )
    return candidates


# ─────────────────────────────────────────────
# Основной мержер
# ─────────────────────────────────────────────

class SmartGraphMerger:
    def __init__(self):
        self.G = nx.DiGraph()
        self.conflicts: List[Conflict] = []
        self.logged_merge_actions: List[MergeAction] = []

    async def smart_merge(self, subgraphs: List[ExtractedKnowledge]) -> UnifiedGraph:
        logger.info("🔗 СЛОЙ 2: Загрузка подграфов в единый граф NetworkX")

        for sg in subgraphs:
            for node in sg.nodes:
                if not self.G.has_node(node.id):
                    self.G.add_node(node.id, **node.model_dump(mode="json"))
            for edge in sg.edges:
                edge_data = edge.model_dump(mode="json", exclude={"source", "target"})
                self.G.add_edge(edge.source, edge.target, **edge_data)

        logger.info(
            f"  -> Исходный размер: {self.G.number_of_nodes()} узлов, "
            f"{self.G.number_of_edges()} связей."
        )
        log_graphml("layer2_step1_initial_combined.graphml", self.G)

        # ── Дедупликация с embedding pre-filter (Улучшение 4) ────────────────────────────
        await self._deduplicate_with_embeddings()

        log_pydantic(
            "layer2_step2_merge_actions.json",
            MergeBatchResult(actions=self.logged_merge_actions),
        )

        await self._assign_sections()

        # ── Сборка финального графа ───────────────────────────────────────────────────────
        final_nodes: List[GraphNode] = []
        for nid, data in self.G.nodes(data=True):
            node_data = data.copy()
            if "id" not in node_data:
                node_data["id"] = nid

            if "target_section" in node_data and isinstance(node_data["target_section"], str):
                try:
                    node_data["target_section"] = TZSectionEnum(node_data["target_section"])
                except ValueError:
                    node_data["target_section"] = TZSectionEnum.UNKNOWN

            final_nodes.append(GraphNode(**node_data))

        final_edges: List[GraphEdge] = []
        for u, v, data in self.G.edges(data=True):
            clean_data = {k: val for k, val in data.items() if k not in {"source", "target"}}
            final_edges.append(GraphEdge(source=u, target=v, **clean_data))

        unified_graph = UnifiedGraph(
            nodes=final_nodes, edges=final_edges, conflicts=self.conflicts
        )

        log_graphml("layer2_step3_final_unified.graphml", self.G)
        log_pydantic("layer2_step3_final_unified.json", unified_graph)

        return unified_graph

    async def _deduplicate_with_embeddings(self):
        """
        Улучшение 4: двухэтапная дедупликация.

        Этап A — Embedding pre-filter: быстро находим похожие пары по косинусному расстоянию.
        Этап B — LLM верификация: отправляем только пары-кандидаты, не случайные батчи.

        Это решает проблему оригинального кода, где батчи по 15 узлов нарезались
        произвольно и дубликаты из разных батчей никогда не встречались.
        """
        nodes_by_label: Dict[str, List[Dict[str, Any]]] = {}
        for nid, data in self.G.nodes(data=True):
            label = data.get("label", "unknown")
            nodes_by_label.setdefault(label, []).append(
                {"id": nid, "name": data.get("name", ""), "desc": data.get("description", "")}
            )

        for label, nodes in nodes_by_label.items():
            if len(nodes) < 2:
                continue

            logger.info(f"  -> Дедупликация группы '{label}' ({len(nodes)} узлов)...")

            # Этап A: Embedding pre-filter
            candidates = await _find_duplicate_candidates(nodes, similarity_threshold=0.88)

            if not candidates:
                logger.info(f"     Дубликатов не найдено в '{label}'")
                continue

            # Этап B: LLM верификация кандидатов
            # Группируем попарные кандидаты в кластеры (чтобы отправить в LLM компактно)
            # Простой подход: объединяем пары через union-find
            id_to_cluster: Dict[str, int] = {}
            clusters: Dict[int, List[str]] = {}
            cluster_counter = 0

            for node_a, node_b, sim in candidates:
                id_a, id_b = node_a["id"], node_b["id"]
                ca = id_to_cluster.get(id_a)
                cb = id_to_cluster.get(id_b)

                if ca is None and cb is None:
                    clusters[cluster_counter] = [id_a, id_b]
                    id_to_cluster[id_a] = cluster_counter
                    id_to_cluster[id_b] = cluster_counter
                    cluster_counter += 1
                elif ca is None:
                    clusters[cb].append(id_a)
                    id_to_cluster[id_a] = cb
                elif cb is None:
                    clusters[ca].append(id_b)
                    id_to_cluster[id_b] = ca
                elif ca != cb:
                    # Объединяем два кластера
                    for nid in clusters[cb]:
                        id_to_cluster[nid] = ca
                    clusters[ca].extend(clusters.pop(cb))

            # LLM верифицирует каждый кластер
            node_lookup = {n["id"]: n for n in nodes}
            for cluster_ids in clusters.values():
                cluster_nodes = [node_lookup[nid] for nid in cluster_ids if nid in node_lookup]
                if len(cluster_nodes) < 2:
                    continue

                data_str = "\n".join(
                    [f"ID: {n['id']} | Имя: {n['name']} | Описание: {n['desc']}"
                     for n in cluster_nodes]
                )
                prompt = """Ты Архитектор. Перед тобой узлы, которые ПРЕДПОЛОЖИТЕЛЬНО являются дубликатами (отобраны по семантическому сходству).

Проверь: это действительно одна и та же сущность или разные?
- Если дубликаты → is_duplicate=true, верни MergeAction.
- Если разные сущности → is_duplicate=false, верни пустой список actions.

Выбирай unified_id из существующих ID (предпочитай более короткий и понятный)."""

                try:
                    result: MergeBatchResult = await acall_llm_json(
                        schema=MergeBatchResult, prompt=prompt, data=data_str
                    )
                    for action in result.actions:
                        if action.is_duplicate and len(action.ids_to_merge) > 1:
                            self.logged_merge_actions.append(action)
                            self._merge_nodes_in_graph(action)
                            logger.info(
                                f"     🔗 Слито: {action.ids_to_merge} → {action.unified_id}"
                            )
                except Exception as e:
                    logger.error(f"Ошибка LLM верификации кластера: {e}")

    def _merge_nodes_in_graph(self, action: MergeAction):
        valid_ids = [nid for nid in action.ids_to_merge if self.G.has_node(nid)]
        if not valid_ids:
            return

        primary_id = action.unified_id
        if not self.G.has_node(primary_id):
            base_data = self.G.nodes[valid_ids[0]].copy()
            base_data.update(
                {"id": primary_id, "name": action.unified_name, "description": action.unified_desc}
            )
            self.G.add_node(primary_id, **base_data)

        for old_id in valid_ids:
            if old_id == primary_id:
                continue
            for u, v, data in list(self.G.edges(old_id, data=True)):
                if u == old_id:
                    self.G.add_edge(primary_id, v, **data)
            for u, v, data in list(self.G.in_edges(old_id, data=True)):
                if v == old_id:
                    self.G.add_edge(u, primary_id, **data)
            self.G.remove_node(old_id)

    async def _assign_sections(self):
        logger.info("  -> Распределение узлов по секциям ТЗ...")
        nodes_to_assign = [
            {"id": n, "name": d.get("name"), "label": d.get("label")}
            for n, d in self.G.nodes(data=True)
            if d.get("label") != NodeLabel.PERSON
        ]

        if not nodes_to_assign:
            return

        prompt = """Распредели каждый узел в одну из секций ТЗ:
- GENERAL    (general_info)   — общая информация, цели, задачи
- STACK      (tech_stack)     — компоненты, БД, библиотеки, инфраструктура
- FUNCTIONAL (functional_req) — требования, фичи, бизнес-логика
- INTERFACE  (ui_ux)          — всё про UI/UX, экраны, формы

Верни assignments для КАЖДОГО узла без пропусков."""

        batch_size = 20
        for i in range(0, len(nodes_to_assign), batch_size):
            batch = nodes_to_assign[i : i + batch_size]
            data_str = "\n".join(
                [f"ID:{n['id']} | {n['label']} | {n['name']}" for n in batch]
            )
            try:
                result: SectionBatchResult = await acall_llm_json(
                    schema=SectionBatchResult, prompt=prompt, data=data_str
                )
                for assignment in result.assignments:
                    if self.G.has_node(assignment.node_id):
                        self.G.nodes[assignment.node_id]["target_section"] = (
                            assignment.target_section
                        )
            except Exception as e:
                logger.error(f"Ошибка назначения секций: {e}")