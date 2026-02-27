import logging
import networkx as nx
from typing import List
from pydantic import BaseModel, Field
from schemas.graph import ExtractedKnowledge, UnifiedGraph, GraphNode, GraphEdge
from schemas.enums import TZSectionEnum
from utils.llm_client import acall_llm_json

logger = logging.getLogger(__name__)

# Схемы для LLM Слияния
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

class SmartGraphMerger:
    def __init__(self):
        self.G = nx.DiGraph()

    async def smart_merge(self, subgraphs: List[ExtractedKnowledge]) -> UnifiedGraph:
        logger.info("🔗 СЛОЙ 2: Загрузка подграфов в единый граф NetworkX")
        
        # 1. Складываем всё в один граф
        for sg in subgraphs:
            for node in sg.nodes:
                if not self.G.has_node(node.id):
                    self.G.add_node(node.id, **node.model_dump())
            for edge in sg.edges:
                self.G.add_edge(edge.source, edge.target, **edge.model_dump())

        logger.info(f"  -> Исходный размер: {self.G.number_of_nodes()} узлов, {self.G.number_of_edges()} связей.")

        # 2. Группируем узлы по типам для батч-слияния
        nodes_by_label = {}
        for nid, data in self.G.nodes(data=True):
            label = data.get("label")
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append({"id": nid, "name": data.get("name"), "desc": data.get("description")})

        # 3. Дедупликация через LLM
        for label, nodes in nodes_by_label.items():
            if len(nodes) < 2: continue
            
            logger.info(f"  -> Дедупликация группы '{label}' ({len(nodes)} узлов)...")
            # Бьем на батчи по 15 узлов, чтобы не перегрузить контекст
            batch_size = 15
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i+batch_size]
                await self._resolve_duplicates_batch(batch)

        # 4. Распределение по секциям ТЗ
        await self._assign_sections()

        # 5. Экспорт в UnifiedGraph
        final_nodes = [GraphNode(**data) for _, data in self.G.nodes(data=True)]
        final_edges = [GraphEdge(source=u, target=v, **data) for u, v, data in self.G.edges(data=True)]
        
        return UnifiedGraph(nodes=final_nodes, edges=final_edges)

    async def _resolve_duplicates_batch(self, nodes_batch: List[dict]):
        prompt = """Ты Архитектор. Найди дубликаты среди этих узлов (синонимы, одно и то же понятие).
        Если находишь дубликаты, верни MergeAction с is_duplicate=true.
        Если дубликатов нет, верни пустой список actions."""
        
        data_str = "\n".join([f"ID: {n['id']} | Имя: {n['name']} | Описание: {n['desc']}" for n in nodes_batch])
        
        try:
            result = await acall_llm_json(schema=MergeBatchResult, prompt=prompt, data=data_str)
            for action in result.actions:
                if action.is_duplicate and len(action.ids_to_merge) > 1:
                    self._merge_nodes_in_graph(action)
        except Exception as e:
            logger.error(f"Ошибка при дедупликации: {e}")

    def _merge_nodes_in_graph(self, action: MergeAction):
        valid_ids = [nid for nid in action.ids_to_merge if self.G.has_node(nid)]
        if not valid_ids: return

        primary_id = action.unified_id
        if not self.G.has_node(primary_id):
            # Создаем новый узел на основе первого старого
            base_data = self.G.nodes[valid_ids[0]].copy()
            base_data.update({
                "id": primary_id,
                "name": action.unified_name,
                "description": action.unified_desc
            })
            self.G.add_node(primary_id, **base_data)

        # Перенаправляем связи
        for old_id in valid_ids:
            if old_id == primary_id: continue
            
            for u, v, data in list(self.G.edges(old_id, data=True)):
                if u == old_id: self.G.add_edge(primary_id, v, **data)
            
            for u, v, data in list(self.G.in_edges(old_id, data=True)):
                if v == old_id: self.G.add_edge(u, primary_id, **data)
                
            self.G.remove_node(old_id)

    async def _assign_sections(self):
        logger.info("  -> Распределение узлов по секциям ТЗ...")
        nodes_to_assign = [{"id": n, "name": d.get("name"), "label": d.get("label")} 
                           for n, d in self.G.nodes(data=True) if d.get("label") != "Person"]
        
        if not nodes_to_assign: return

        prompt = """Распредели каждый узел в одну из секций ТЗ:
        - GENERAL (общая инфа, задачи)
        - STACK (компоненты, БД, либы)
        - FUNCTIONAL (требования, фичи)
        - INTERFACE (всё про UI/UX)"""

        batch_size = 20
        for i in range(0, len(nodes_to_assign), batch_size):
            batch = nodes_to_assign[i:i+batch_size]
            data_str = "\n".join([f"ID:{n['id']} | {n['label']} | {n['name']}" for n in batch])
            try:
                result = await acall_llm_json(schema=SectionBatchResult, prompt=prompt, data=data_str)
                for assignment in result.assignments:
                    if self.G.has_node(assignment.node_id):
                        self.G.nodes[assignment.node_id]["target_section"] = assignment.target_section
            except Exception:
                pass