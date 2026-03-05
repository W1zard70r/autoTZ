import logging
import asyncio
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

from schemas.graph import (
    ExtractedKnowledge, UnifiedGraph, GraphNode, GraphEdge,
    Conflict, VoteCount, DecisionResolution, DetectedConflict, ConflictResolution,
)
from schemas.enums import TZSectionEnum, NodeLabel, EdgeRelation
from utils.llm_client import acall_llm_json
from utils.state_logger import log_graphml, log_pydantic
from utils.embeddings import aget_embeddings_safe

logger = logging.getLogger(__name__)


class MergeAction(BaseModel):
    is_duplicate: bool = Field(description="Это одна и та же сущность?")
    ids_to_merge: List[str] = Field(description="Список ID, которые нужно слить в один")
    unified_id: str = Field(description="Новый ID для слитого узла")
    unified_name: str = Field(description="Общее имя")
    unified_desc: str = Field(description="Объединенное описание")


class MergeBatchResult(BaseModel):
    actions: List[MergeAction] = Field(default_factory=list)


class SectionAssignment(BaseModel):
    node_id: str
    target_section: TZSectionEnum


class SectionBatchResult(BaseModel):
    assignments: List[SectionAssignment]


class ConflictBatchResult(BaseModel):
    conflicts: List[DetectedConflict] = Field(default_factory=list)


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a, b = np.array(v1), np.array(v2)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm > 0 else 0.0


async def _find_duplicate_candidates(
        nodes: List[Dict[str, Any]],
        similarity_threshold: float = 0.88,
) -> List[Tuple[Dict, Dict, float]]:
    if len(nodes) < 2:
        return []
    texts = [f"{n['name']} {n.get('desc', '')}".strip() for n in nodes]

    embeddings = await aget_embeddings_safe(texts, batch_size=20, delay=0.5)

    candidates = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= similarity_threshold:
                candidates.append((nodes[i], nodes[j], sim))
    logger.info(
        f"  -> Embedding pre-filter: {len(nodes)} узлов → "
        f"{len(candidates)} пар-кандидатов (threshold={similarity_threshold})"
    )
    return candidates


def resolve_decisions(G: nx.MultiDiGraph) -> List[DecisionResolution]:
    resolutions: List[DecisionResolution] = []
    decision_nodes = [
        (nid, data) for nid, data in G.nodes(data=True)
        if data.get("label") in (NodeLabel.DECISION.value, NodeLabel.DECISION)
    ]

    for decision_id, decision_data in decision_nodes:
        decision_name = decision_data.get("name", decision_id)

        option_ids = [
            v for _, v, edata in G.out_edges(decision_id, data=True)
            if edata.get("relation") in (EdgeRelation.RELATES_TO.value, EdgeRelation.RELATES_TO)
        ]

        if not option_ids:
            logger.warning(f"⚠️ Decision '{decision_id}' не имеет вариантов (RELATES_TO рёбра не найдены)")
            continue

        vote_counts: Dict[str, VoteCount] = {}
        for opt_id in option_ids:
            opt_name = G.nodes[opt_id].get("name", opt_id) if G.has_node(opt_id) else opt_id
            vote_counts[opt_id] = VoteCount(option_id=opt_id, option_name=opt_name)

        for src, tgt, edata in G.edges(data=True):
            relation = edata.get("relation", "")
            if isinstance(relation, EdgeRelation):
                relation = relation.value

            if tgt not in vote_counts:
                continue

            voter_name = G.nodes[src].get("name", src) if G.has_node(src) else src

            if relation == EdgeRelation.VOTED_FOR.value:
                vote_counts[tgt].votes_for += 1
                vote_counts[tgt].voters_for.append(voter_name)
            elif relation == EdgeRelation.VOTED_AGAINST.value:
                vote_counts[tgt].votes_against += 1
                vote_counts[tgt].voters_against.append(voter_name)

        options_list = list(vote_counts.values())

        if not any(vc.votes_for + vc.votes_against > 0 for vc in options_list):
            resolution = DecisionResolution(
                decision_id=decision_id,
                decision_name=decision_name,
                is_tie=True,
                options=options_list,
                conflict_description="Голосов не обнаружено. Решение не принято.",
            )
        else:
            sorted_options = sorted(options_list, key=lambda x: x.score, reverse=True)
            top = sorted_options[0]
            second = sorted_options[1] if len(sorted_options) > 1 else None
            is_tie = second is not None and top.score == second.score

            if is_tie:
                tied_names = [o.option_name for o in sorted_options if o.score == top.score]
                resolution = DecisionResolution(
                    decision_id=decision_id,
                    decision_name=decision_name,
                    is_tie=True,
                    options=options_list,
                    conflict_description=(
                        f"Ничья между: {', '.join(tied_names)} (счёт {top.score}). "
                        f"Требуется ручное решение."
                    ),
                )
            else:
                resolution = DecisionResolution(
                    decision_id=decision_id,
                    decision_name=decision_name,
                    winner_id=top.option_id,
                    winner_name=top.option_name,
                    is_tie=False,
                    options=options_list,
                )
                G.add_edge(
                    decision_id,
                    top.option_id,
                    relation=EdgeRelation.RESOLVED_TO.value,
                    evidence=f"Победитель голосования: {top.votes_for} за, {top.votes_against} против",
                )
                logger.info(
                    f"  🗳️  '{decision_name}': победил '{top.option_name}' "
                    f"({top.votes_for}✅ / {top.votes_against}❌)"
                )

        resolutions.append(resolution)

    return resolutions


def format_merge_report(
        resolutions: List[DecisionResolution],
        conflicts: List[Conflict],
) -> str:
    lines: List[str] = []

    if resolutions:
        lines.append("=" * 60)
        lines.append("🗳️  ИТОГИ ГОЛОСОВАНИЙ")
        lines.append("=" * 60)
        for res in resolutions:
            lines.append(f"\n📌 {res.decision_name}")
            lines.append("-" * 40)
            for opt in sorted(res.options, key=lambda x: x.score, reverse=True):
                bar_for = "✅" * opt.votes_for
                bar_against = "❌" * opt.votes_against
                score_str = f"[{opt.score:+d}]"
                voters_for_str = f"  За: {', '.join(opt.voters_for)}" if opt.voters_for else ""
                voters_against_str = f"  Против: {', '.join(opt.voters_against)}" if opt.voters_against else ""

                lines.append(
                    f"  {'👑 ' if res.winner_id == opt.option_id else '   '}"
                    f"{opt.option_name:<25} {bar_for}{bar_against}  {score_str}"
                )
                if voters_for_str:     lines.append(f"           {voters_for_str}")
                if voters_against_str: lines.append(f"           {voters_against_str}")

            if res.is_tie:
                lines.append(f"\n  ⚠️  КОНФЛИКТ: {res.conflict_description}")
            elif res.winner_name:
                lines.append(f"\n  ✅ ПРИНЯТО: {res.winner_name}")

    if conflicts:
        lines.append("\n" + "=" * 60)
        lines.append("⚡ КОНФЛИКТЫ ГРАФА (требуют внимания)")
        lines.append("=" * 60)
        for conflict in conflicts:
            lines.append(f"\n  •[{conflict.node_id}] {conflict.description}")
            if conflict.conflicting_values:
                for val in conflict.conflicting_values:
                    lines.append(f"      ↳ {val}")

    if not resolutions and not conflicts:
        lines.append("✅ Голосований и конфликтов не обнаружено.")

    return "\n".join(lines)


class SmartGraphMerger:
    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.conflicts: List[Conflict] = []
        self.active_conflicts: List[DetectedConflict] = []
        self.logged_merge_actions: List[MergeAction] = []

    async def merge_subgraphs_and_deduplicate(self, subgraphs: List[ExtractedKnowledge]):
        """
        ЭТАП 1: Загрузка всех подграфов в единый граф и устранение полных дубликатов.
        """
        logger.info("🔗 СЛОЙ 2 (Шаг 1): Загрузка подграфов и дедупликация...")

        for sg in subgraphs:
            for node in sg.nodes:
                if not self.G.has_node(node.id):
                    self.G.add_node(node.id, **node.model_dump(mode='json'))
            for edge in sg.edges:
                edge_data = edge.model_dump(mode='json', exclude={'source', 'target'})
                self.G.add_edge(edge.source, edge.target, **edge_data)

        logger.info(f"  -> Исходный размер графа: {self.G.number_of_nodes()} узлов, {self.G.number_of_edges()} связей.")
        log_graphml("layer2_step1_initial_combined.graphml", self.G)

        await self._deduplicate_with_embeddings()

        log_pydantic("layer2_step2_merge_actions.json", MergeBatchResult(actions=self.logged_merge_actions))
        logger.info("✅ Этап 1 завершен. Граф очищен от явных дублей.")

    async def detect_conflicts(self) -> List[DetectedConflict]:
        """
        ЭТАП 2: Поиск логических противоречий.
        """
        logger.info("🔍 СЛОЙ 2 (Шаг 2): Поиск логических конфликтов...")

        tech_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("label") in ["Component", "Concept", "Requirement"]
        ]

        if not tech_nodes:
            logger.info("  -> Нет технических узлов для анализа конфликтов.")
            return []

        nodes_context = []
        for nid in tech_nodes[:60]:
            data = self.G.nodes[nid]
            nodes_context.append(f"ID: {nid} | Имя: {data.get('name')} | Описание: {data.get('description')}")

        nodes_desc = "\n".join(nodes_context)

        prompt = """
        Ты Главный Системный Архитектор. Твоя задача — найти ВЗАИМОИСКЛЮЧАЮЩИЕ (конфликтующие) технологические решения или бизнес-требования в списке узлов.

        ⚠️ СТРОГИЕ ПРАВИЛА (ЧТО НЕ ЯВЛЯЕТСЯ КОНФЛИКТОМ):
        1. Взаимодополняющие технологии (например, "Форма логина (Email/Password)" и "JWT-токен" РАБОТАЮТ ВМЕСТЕ — это НЕ конфликт).
        2. Фронтенд и Бэкенд технологии (например, "React" и "FastAPI" — это НЕ конфликт, они из разных миров).
        3. База данных и Кеш (например, "PostgreSQL" и "Redis" — это НЕ конфликт).

        🚨 ЧТО ЯВЛЯЕТСЯ КОНФЛИКТОМ (ТОЛЬКО ЭТО):
        1. Две технологии претендуют на ОДНУ И ТУ ЖЕ роль (например, "Фронтенд на React" VS "Фронтенд на Vue").
        2. Две основные базы данных (например, "Основная БД PostgreSQL" VS "Основная БД MongoDB", если не указано микросервисное разделение).
        3. Два разных платежных шлюза для одной страны (например, "Stripe" VS "ЮKassa" для оплаты в РФ).
        4. Явные противоречия в бизнес-требованиях (например, "Темная тема" VS "Только светлая тема").

        Если нашел НАСТОЯЩИЙ конфликт:
        1. Опиши его суть (description).
        2. Укажи категорию (category).
        3. Выдели конфликтующие варианты (options).
        4. Дай профессиональную рекомендацию (ai_recommendation).
        """

        try:
            result = await acall_llm_json(
                schema=ConflictBatchResult,
                prompt=prompt,
                data=nodes_desc
            )
            self.active_conflicts = result.conflicts

            if self.active_conflicts:
                logger.warning(f"⚠️ Найдено {len(self.active_conflicts)} настоящих конфликтов!")
            else:
                logger.info("✅ Логических конфликтов не обнаружено (или найдены только взаимодополняющие технологии).")

            return result.conflicts

        except Exception as e:
            logger.error(f"Ошибка поиска конфликтов: {e}")
            return []

    def apply_resolutions(self, resolutions: List[ConflictResolution]):
        """
        ЭТАП 3: Применение решений пользователя к графу.
        """
        logger.info("🛠️ СЛОЙ 2 (Шаг 3): Применение решений пользователя...")

        for res in resolutions:
            conflict = next((c for c in self.active_conflicts if c.id == res.conflict_id), None)
            if not conflict:
                continue

            all_option_ids = [opt.id for opt in conflict.options]

            if res.selected_option_id:
                winner_id = res.selected_option_id
                logger.info(f"  -> Победил: {winner_id}")
                for loose_id in all_option_ids:
                    if loose_id != winner_id and self.G.has_node(loose_id):
                        self.G.remove_node(loose_id)

            elif res.custom_text:
                logger.info(f"  -> Свой вариант: {res.custom_text}")
                for old_id in all_option_ids:
                    if self.G.has_node(old_id):
                        self.G.remove_node(old_id)

                new_id = f"custom_{res.conflict_id}"[:30]
                self.G.add_node(
                    new_id,
                    name=res.custom_text,
                    label="Component",
                    description="Выбор пользователя",
                    target_section="tech_stack"
                )

        self.active_conflicts = []

    async def finalize_graph(self) -> UnifiedGraph:
        """
        ЭТАП 4: Финальная обработка и распределение по секциям.
        """
        logger.info("🏁 СЛОЙ 2 (Шаг 4): Финализация графа...")

        logger.info("  -> Разрешение голосований...")
        resolutions = resolve_decisions(self.G)

        await self._assign_sections()

        final_nodes: List[GraphNode] = []
        for nid, data in self.G.nodes(data=True):
            node_data = data.copy()
            if "id" not in node_data: node_data["id"] = nid
            if "target_section" in node_data and isinstance(node_data["target_section"], str):
                try:
                    node_data["target_section"] = TZSectionEnum(node_data["target_section"])
                except ValueError:
                    node_data["target_section"] = TZSectionEnum.UNKNOWN
            try:
                final_nodes.append(GraphNode(**node_data))
            except Exception as e:
                logger.warning(f"Пропуск узла {nid}: {e}")

        final_edges: List[GraphEdge] = []
        for u, v, data in self.G.edges(data=True):
            clean_data = {k: val for k, val in data.items() if k not in {"source", "target"}}
            try:
                final_edges.append(GraphEdge(source=u, target=v, **clean_data))
            except Exception as e:
                logger.warning(f"Пропуск ребра {u}→{v}: {e}")


        unified_graph = UnifiedGraph(
            nodes=final_nodes,
            edges=final_edges,
            conflicts=self.conflicts,
            decisions=resolutions,
        )

        # --- LOGGING ---
        log_graphml("layer2_step3_final_unified.graphml", self.G)
        log_pydantic("layer2_step3_final_unified.json", unified_graph)

        report = format_merge_report(resolutions, self.conflicts)
        print("\n" + report + "\n")
        logger.info("📋 Отчёт о голосованиях и конфликтах выведен.")

        return unified_graph

    async def _deduplicate_with_embeddings(self):
        nodes_by_label: Dict[str, List[Dict[str, Any]]] = {}
        for nid, data in self.G.nodes(data=True):
            label = data.get("label", "unknown")
            nodes_by_label.setdefault(label, []).append(
                {"id": nid, "name": data.get("name", ""), "desc": data.get("description", "")}
            )

        for label, nodes in nodes_by_label.items():
            if len(nodes) < 2:
                continue

            if label in (NodeLabel.DECISION.value, NodeLabel.DECISION):
                continue

            logger.info(f"  -> Дедупликация группы '{label}' ({len(nodes)} узлов)...")
            candidates = await _find_duplicate_candidates(nodes, similarity_threshold=0.88)

            if not candidates:
                logger.info(f"     Дубликатов не найдено в '{label}'")
                continue

            id_to_cluster: Dict[str, int] = {}
            clusters: Dict[int, List[str]] = {}
            counter = 0

            for node_a, node_b, _ in candidates:
                id_a, id_b = node_a["id"], node_b["id"]
                ca, cb = id_to_cluster.get(id_a), id_to_cluster.get(id_b)

                if ca is None and cb is None:
                    clusters[counter] = [id_a, id_b]
                    id_to_cluster[id_a] = id_to_cluster[id_b] = counter
                    counter += 1
                elif ca is None:
                    clusters[cb].append(id_a);
                    id_to_cluster[id_a] = cb
                elif cb is None:
                    clusters[ca].append(id_b);
                    id_to_cluster[id_b] = ca
                elif ca != cb:
                    for nid in clusters[cb]: id_to_cluster[nid] = ca
                    clusters[ca].extend(clusters.pop(cb))

            node_lookup = {n["id"]: n for n in nodes}
            for cluster_ids in clusters.values():
                cluster_nodes = [node_lookup[nid] for nid in cluster_ids if nid in node_lookup]
                if len(cluster_nodes) < 2:
                    continue

                data_str = "\n".join(
                    [f"ID: {n['id']} | Имя: {n['name']} | Описание: {n['desc']}"
                     for n in cluster_nodes]
                )
                prompt = """Ты Архитектор. Перед тобой узлы, отобранные по семантическому сходству.
Проверь: это действительно одна и та же сущность или разные?
- Дубликаты → is_duplicate=true, верни MergeAction.
- Разные → is_duplicate=false, верни пустой список.
Выбирай unified_id из существующих (предпочти более короткий)."""

                try:
                    result: MergeBatchResult = await acall_llm_json(
                        schema=MergeBatchResult, prompt=prompt, data=data_str
                    )
                    for action in result.actions:
                        if action.is_duplicate and len(action.ids_to_merge) > 1:
                            self.logged_merge_actions.append(action)
                            self._merge_nodes_in_graph(action)
                            logger.info(f"     🔗 Слито: {action.ids_to_merge} → {action.unified_id}")
                except Exception as e:
                    logger.error(f"Ошибка LLM верификации кластера: {e}")

    def _merge_nodes_in_graph(self, action: MergeAction):
        valid_ids = [nid for nid in action.ids_to_merge if self.G.has_node(nid)]
        if not valid_ids:
            return

        primary_id = action.unified_id
        if not self.G.has_node(primary_id):
            base_data = self.G.nodes[valid_ids[0]].copy()
            base_data.update({
                "id": primary_id,
                "name": action.unified_name,
                "description": action.unified_desc,
            })
            self.G.add_node(primary_id, **base_data)

        for old_id in valid_ids:
            if old_id == primary_id: continue
# В MultiDiGraph правильно перенаправляем ВСЕ рёбра
            for u, v, data in list(self.G.out_edges(old_id, data=True)):
                target = primary_id if v == old_id else v
                self.G.add_edge(primary_id, target, **data)
            for u, v, data in list(self.G.in_edges(old_id, data=True)):
                if u == old_id:
                    continue  # self-loops уже скопированы в out_edges
                self.G.add_edge(u, primary_id, **data)
            self.G.remove_node(old_id)

    async def _assign_sections(self):
        logger.info("  -> Распределение узлов по секциям ТЗ...")

        nodes_to_assign = [
            {"id": n, "name": d.get("name"), "label": d.get("label")}
            for n, d in self.G.nodes(data=True)
            if d.get("label") not in (NodeLabel.PERSON.value, NodeLabel.PERSON)
               and d.get("target_section", "uncategorized") in (
               "uncategorized", TZSectionEnum.UNKNOWN.value, TZSectionEnum.UNKNOWN)
        ]

        if not nodes_to_assign:
            return

        prompt = """Распредели каждый узел в одну из секций ТЗ:
    - GENERAL    (general_info)   — общая информация, цели, задачи
    - STACK      (tech_stack)     — компоненты, БД, библиотеки, Decision-узлы выбора технологий
    - FUNCTIONAL (functional_req) — требования, фичи, бизнес-логика
    - INTERFACE  (ui_ux)          — UI/UX, экраны, формы
    
    Верни assignments для КАЖДОГО узла без пропусков."""

        for i in range(0, len(nodes_to_assign), 20):
            batch = nodes_to_assign[i: i + 20]
            data_str = "\n".join([f"ID:{n['id']} | {n['name']}" for n in batch])
            try:
                result: SectionBatchResult = await acall_llm_json(
                    schema=SectionBatchResult, prompt=prompt, data=data_str
                )
                for assignment in result.assignments:
                    if self.G.has_node(assignment.node_id):
                        self.G.nodes[assignment.node_id]["target_section"] = assignment.target_section
            except Exception as e:
                logger.error(f"Ошибка назначения секций: {e}")