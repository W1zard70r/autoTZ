import logging
import networkx as nx
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Импортируем схемы
from schemas.graph import (
    ExtractedKnowledge, 
    UnifiedGraph, 
    GraphNode, 
    GraphEdge, 
    DetectedConflict, 
    ConflictResolution,
)
from schemas.enums import TZSectionEnum, NodeLabel
from utils.llm_client import acall_llm_json
from utils.state_logger import log_graphml, log_pydantic

logger = logging.getLogger(__name__)


# --- Внутренние модели для LLM ---
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


class SmartGraphMerger:
    def __init__(self):
        self.G = nx.DiGraph()
        self.active_conflicts: List[DetectedConflict] = []
        self.logged_merge_actions: List[MergeAction] = [] 

    async def merge_subgraphs_and_deduplicate(self, subgraphs: List[ExtractedKnowledge]):
        """
        ЭТАП 1: Загрузка всех подграфов в единый граф и устранение полных дубликатов.
        """
        logger.info("🔗 СЛОЙ 2 (Шаг 1): Загрузка подграфов и дедупликация...")

        # 1. Физическое объединение графов
        for sg in subgraphs:
            for node in sg.nodes:
                if not self.G.has_node(node.id):
                    self.G.add_node(node.id, **node.model_dump(mode='json'))
            for edge in sg.edges:
                edge_data = edge.model_dump(mode='json', exclude={'source', 'target'})
                self.G.add_edge(edge.source, edge.target, **edge_data)

        logger.info(f"  -> Исходный размер графа: {self.G.number_of_nodes()} узлов, {self.G.number_of_edges()} связей.")
        log_graphml("layer2_step1_initial_combined.graphml", self.G)

        # 2. Группировка узлов по Label
        nodes_by_label: Dict[str, List[Dict[str, Any]]] = {}
        for nid, data in self.G.nodes(data=True):
            label = data.get("label")
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append({
                "id": nid, 
                "name": data.get("name"), 
                "desc": data.get("description")
            })

        # 3. Дедупликация (поиск синонимов)
        for label, nodes in nodes_by_label.items():
            if len(nodes) < 2: 
                continue

            logger.info(f"  -> Дедупликация группы '{label}' ({len(nodes)} узлов)...")
            
            # Разбиваем на батчи
            batch_size = 15
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                prompt = """Ты Архитектор. Найди дубликаты среди этих узлов (синонимы, одно и то же понятие).
                Например: 'Postgres' и 'PostgreSQL' - это дубликаты.
                'React' и 'Vue' - это НЕ дубликаты.
                
                Если находишь дубликаты, верни MergeAction с is_duplicate=true.
                Если дубликатов нет, верни пустой список actions."""
                
                data_str = "\n".join([f"ID: {n['id']} | Имя: {n['name']} | Описание: {n['desc']}" for n in batch])
                
                try:
                    result = await acall_llm_json(schema=MergeBatchResult, prompt=prompt, data=data_str)
                    
                    for action in result.actions:
                        if action.is_duplicate and len(action.ids_to_merge) > 1:
                            self.logged_merge_actions.append(action)
                            self._merge_nodes_in_graph(action)
                            
                except Exception as e:
                    logger.error(f"Ошибка при дедупликации батча в группе {label}: {e}")

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

        await self._assign_sections()

        final_nodes = []
        for nid, data in self.G.nodes(data=True):
            node_data = data.copy()
            if "id" not in node_data: node_data["id"] = nid

            if "target_section" in node_data and isinstance(node_data["target_section"], str):
                try:
                    node_data["target_section"] = TZSectionEnum(node_data["target_section"])
                except ValueError:
                    node_data["target_section"] = TZSectionEnum.UNKNOWN
            
            final_nodes.append(GraphNode(**node_data))

        final_edges = []
        for u, v, data in self.G.edges(data=True):
            clean_data = {k: val for k, val in data.items() if k not in {'source', 'target'}}
            final_edges.append(GraphEdge(source=u, target=v, **clean_data))

        unified_graph = UnifiedGraph(
            nodes=final_nodes, 
            edges=final_edges, 
            conflicts=self.active_conflicts
        )

        log_graphml("layer2_step3_final_unified.graphml", self.G)
        log_pydantic("layer2_step3_final_unified.json", unified_graph)

        return unified_graph

    # --- Внутренние методы ---

    def _merge_nodes_in_graph(self, action: MergeAction):
        valid_ids = [nid for nid in action.ids_to_merge if self.G.has_node(nid)]
        if not valid_ids: return

        primary_id = action.unified_id
        if not self.G.has_node(primary_id):
            base_data = self.G.nodes[valid_ids[0]].copy()
            base_data.update({
                "id": primary_id,
                "name": action.unified_name,
                "description": action.unified_desc
            })
            self.G.add_node(primary_id, **base_data)

        for old_id in valid_ids:
            if old_id == primary_id: continue
            for u, v, data in list(self.G.edges(old_id, data=True)):
                if u == old_id: self.G.add_edge(primary_id, v, **data)
            for u, v, data in list(self.G.in_edges(old_id, data=True)):
                if v == old_id: self.G.add_edge(u, primary_id, **data)
            self.G.remove_node(old_id)

    async def _assign_sections(self):
        logger.info("  -> Распределение узлов по секциям ТЗ...")
        nodes_to_assign = [
            {"id": n, "name": d.get("name"), "label": d.get("label")}
            for n, d in self.G.nodes(data=True) 
            if d.get("label") != NodeLabel.PERSON and d.get("target_section", "uncategorized") == "uncategorized"
        ]

        if not nodes_to_assign: return

        prompt = """Распредели каждый узел в одну из секций ТЗ (GENERAL, STACK, FUNCTIONAL, INTERFACE)."""

        batch_size = 20
        for i in range(0, len(nodes_to_assign), batch_size):
            batch = nodes_to_assign[i:i + batch_size]
            data_str = "\n".join([f"ID:{n['id']} | {n['name']}" for n in batch])
            try:
                result = await acall_llm_json(schema=SectionBatchResult, prompt=prompt, data=data_str)
                for assignment in result.assignments:
                    if self.G.has_node(assignment.node_id):
                        self.G.nodes[assignment.node_id]["target_section"] = assignment.target_section
            except Exception as e:
                logger.warning(f"Ошибка распределения секций: {e}")