import logging
import asyncio
import math
import networkx as nx
import numpy as np
from typing import List, Dict, Any

from schemas.graph import (
    ExtractedKnowledge, UnifiedGraph, GraphNode, GraphEdge,
    Conflict, VoteCount, DecisionResolution, DetectedConflict, ConflictResolution,
)
from schemas.enums import TZSectionEnum, NodeLabel, EdgeRelation
from schemas.merger import SectionBatchResult, MergeBatchResult, MergeAction, ConflictBatchResult
from utils.llm_client import acall_llm_json
from utils.state_logger import log_graphml, log_pydantic
from utils.embeddings import aget_embeddings_safe

logger = logging.getLogger(__name__)



def _format_nodes_for_dedup(nodes: List[Dict[str, Any]]) -> str:
    """Format nodes as a string for LLM deduplication."""
    return "\n".join(
        [f"ID: {n['id']} | Имя: {n['name']} | Описание: {n.get('desc', '')}"
         for n in nodes]
    )


def calculate_cosine_similarity_matrix(embeddings_list: List[List[float]]) -> np.ndarray:
    if not embeddings_list:
        return np.array([])
    vecs = np.array(embeddings_list)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    vecs_normalized = vecs / norms
    return np.dot(vecs_normalized, vecs_normalized.T)


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
        self.node_embeddings: Dict[str, List[float]] = {}

    async def _populate_embeddings_cache(self):
        """Пакетно получает эмбеддинги для всех узлов графа, у которых их еще нет."""
        nodes_to_embed = []
        texts_to_embed = []

        for nid, data in self.G.nodes(data=True):
            if nid not in self.node_embeddings:
                text = f"Тип: {data.get('label', '')}. Название: {data.get('name', '')}. Описание: {data.get('description', '')}"
                nodes_to_embed.append(nid)
                texts_to_embed.append(text)

        if not nodes_to_embed:
            return

        logger.info(f"🧠 Векторизация {len(nodes_to_embed)} новых узлов...")
        embeddings = await aget_embeddings_safe(texts_to_embed, batch_size=20)

        for nid, emb in zip(nodes_to_embed, embeddings):
            self.node_embeddings[nid] = emb

    async def merge_subgraphs_and_deduplicate(self, subgraphs: List[ExtractedKnowledge]):
        """ЭТАП 1: Загрузка подграфов и УМНАЯ векторная дедупликация."""
        logger.info("🔗 СЛОЙ 2 (Шаг 1): Загрузка подграфов и дедупликация...")

        for sg in subgraphs:
            for node in sg.nodes:
                if not self.G.has_node(node.id):
                    self.G.add_node(node.id, **node.model_dump(mode='json'))
            for edge in sg.edges:
                edge_data = edge.model_dump(mode='json', exclude={'source', 'target'})
                self.G.add_edge(edge.source, edge.target, **edge_data)

        logger.info(f"  -> Исходный размер графа: {self.G.number_of_nodes()} узлов.")
        log_graphml("layer2_step1_initial_combined.graphml", self.G)

        await self._populate_embeddings_cache()

        await self._deduplicate_with_embeddings()

        log_pydantic("layer2_step2_merge_actions.json", MergeBatchResult(actions=self.logged_merge_actions))
        logger.info("✅ Этап 1 завершен. Граф очищен от семантических дублей.")

    async def detect_conflicts(self) -> List[DetectedConflict]:
        """ЭТАП 2: Поиск логических противоречий на базе семантических кластеров (RAG)."""
        logger.info("🔍 СЛОЙ 2 (Шаг 2): Векторный RAG-поиск логических конфликтов...")

        tech_nodes = [
            nid for nid, d in self.G.nodes(data=True)
            if d.get("label") in ["Component", "Concept", "Requirement"]
        ]

        if len(tech_nodes) < 2:
            return []

        tech_embeddings = [self.node_embeddings[nid] for nid in tech_nodes]
        sim_matrix = calculate_cosine_similarity_matrix(tech_embeddings)

        conflict_graph = nx.Graph()
        for i in range(len(tech_nodes)):
            conflict_graph.add_node(tech_nodes[i])
            for j in range(i + 1, len(tech_nodes)):
                if 0.75 <= sim_matrix[i, j] < 0.92:
                    conflict_graph.add_edge(tech_nodes[i], tech_nodes[j])

        clusters = [list(c) for c in nx.connected_components(conflict_graph) if len(c) > 1]

        if not clusters:
            logger.info("  -> Потенциальных конкурентных технологий не найдено.")
            return []

        nodes_context = []
        for idx, cluster in enumerate(clusters[:10]):
            nodes_context.append(f"\n--- Кластер подозрения {idx + 1} ---")
            for nid in cluster:
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
        3. Выдели конфликтующие варианты (options) по их ID.
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
                logger.warning(f"⚠️ RAG нашел {len(self.active_conflicts)} архитектурных конфликтов!")
            else:
                logger.info("✅ Логических конфликтов в подозрительных кластерах не обнаружено.")

            return result.conflicts

        except Exception as e:
            logger.error(f"Ошибка RAG-поиска конфликтов: {e}")
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

        log_graphml("layer2_step3_final_unified.graphml", self.G)
        log_pydantic("layer2_step3_final_unified.json", unified_graph)

        report = format_merge_report(resolutions, self.conflicts)
        print("\n" + report + "\n")
        logger.info("📋 Отчёт о голосованиях и конфликтах выведен.")

        return unified_graph

    async def _deduplicate_with_embeddings(self):
        """
        УМНАЯ ДЕДУПЛИКАЦИЯ:
        Сначала математически находим кластеры очень похожих узлов (sim > 0.88),
        а затем отправляем в LLM только эти кластеры.
        Это экономит 90% токенов и времени!
        """
        nodes_by_label: Dict[str, List[str]] = {}
        for nid, data in self.G.nodes(data=True):
            label = data.get("label", "unknown")
            if label not in (NodeLabel.DECISION.value, NodeLabel.DECISION):
                nodes_by_label.setdefault(label, []).append(nid)

        for label, nids in nodes_by_label.items():
            if len(nids) < 2:
                continue

            # 1. Получаем векторы только для текущего лейбла
            embs = [self.node_embeddings[nid] for nid in nids]

            sim_matrix = calculate_cosine_similarity_matrix(embs)

            sim_graph = nx.Graph()
            sim_graph.add_nodes_from(nids)

            for i in range(len(nids)):
                for j in range(i + 1, len(nids)):
                    if sim_matrix[i, j] >= 0.88:
                        sim_graph.add_edge(nids[i], nids[j])

            duplicate_clusters = [list(c) for c in nx.connected_components(sim_graph) if len(c) > 1]

            if not duplicate_clusters:
                continue

            logger.info(
                f"  -> Найдено {len(duplicate_clusters)} кластеров-кандидатов на слияние для '{label}'. Отправка в LLM...")

            # 5. Скармливаем LLM только реальных кандидатов на слияние
            for cluster in duplicate_clusters:
                cluster_data = [{"id": nid, "name": self.G.nodes[nid].get("name", ""),
                                 "desc": self.G.nodes[nid].get("description", "")} for nid in cluster]
                data_str = _format_nodes_for_dedup(cluster_data)

                prompt = """Ты Архитектор. Перед тобой список узлов одного типа из графа знаний.
                            Найди группы дубликатов — узлы, которые описывают одну и ту же сущность.
                            - Для каждой группы дубликатов верни MergeAction с is_duplicate=true.
                            - Выбирай unified_id из существующих (предпочти более короткий и понятный).
                            - Если дубликатов нет, верни пустой список actions."""

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
                    logger.error(f"Ошибка LLM дедупликации кластера '{label}': {e}")

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