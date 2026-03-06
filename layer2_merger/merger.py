import logging
import asyncio
import random
import networkx as nx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from schemas.graph import (
    UnifiedGraph, UnifiedNode, UnifiedEdge, 
    ConflictSchema, ConflictResolution, ConflictOption
)
from schemas.enums import TZSectionEnum, NodeLabel
from utils.llm_client import acall_llm_json, DEFAULT_MODEL

logger = logging.getLogger(__name__)

class SectionAssignment(BaseModel):
    node_id: str
    target_section: TZSectionEnum

class SectionBatchResult(BaseModel):
    assignments: List[SectionAssignment]

class ConflictBatchResult(BaseModel):
    conflicts: List[ConflictSchema] = Field(default_factory=list)

class SmartGraphMerger:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.G = nx.DiGraph()
        self.active_conflicts: List[ConflictSchema] = []

    async def merge_subgraphs_and_deduplicate(self, subgraphs: List[Any]):
        """Сборка графа из подграфов"""
        for sg in subgraphs:
            for node in sg.nodes:
                if not self.G.has_node(node.id):
                    self.G.add_node(node.id, **node.model_dump())
            for edge in sg.edges:
                edge_data = edge.model_dump(exclude={'source', 'target'})
                self.G.add_edge(edge.source, edge.target, **edge_data)
        logger.info(f"✅ Граф загружен: {self.G.number_of_nodes()} узлов.")

    async def detect_conflicts(self) -> List[ConflictSchema]:
        """Поиск конфликтов. Теперь БЕЗ подавления ошибок."""
        logger.info("🔍 СЛОЙ 2: Поиск логических конфликтов...")
        
        nodes_list = [f"ID: {nid} | Name: {d.get('name')}" for nid, d in self.G.nodes(data=True) 
                      if d.get("label") in [NodeLabel.COMPONENT, NodeLabel.REQUIREMENT, NodeLabel.TASK]]
        
        if not nodes_list: return []
        
        prompt = """
        ЗАДАЧА: Найти технические противоречия.
        ПРАВИЛА: 
        1. Создавай ОТДЕЛЬНЫЕ конфликты для разных ролей: Бэкенд, Фронтенд, БД.
        2. В поле 'options.id' пиши ТОЛЬКО реальные ID из списка.
        3. ЯЗЫК: РУССКИЙ.
        """

        # Убран try-except. Ретрай идет внутри acall_llm_json
        result = await acall_llm_json(ConflictBatchResult, prompt, "\n".join(nodes_list))
        self.active_conflicts = result.conflicts
        return self.active_conflicts

    def apply_resolutions(self, resolutions: List[ConflictResolution]):
        """Удаление отвергнутых технологий"""
        logger.info("🛠️ СЛОЙ 2: Применение решений...")
        for res in resolutions:
            conflict = next((c for c in self.active_conflicts if c.id == res.conflict_id), None)
            if not conflict: continue
            
            selected_id = res.selected_option_id
            
            if not selected_id and res.custom_text:
                text = res.custom_text.lower().strip()
                for opt in conflict.options:
                    if text in opt.text.lower() or text in opt.id.lower():
                        selected_id = opt.id
                        break

            if selected_id:
                for opt in conflict.options:
                    if opt.id != selected_id and self.G.has_node(opt.id):
                        logger.info(f"  -> 🗑️ Удалена отвергнутая технология: {opt.id}")
                        self.G.remove_node(opt.id)

    async def finalize_graph(self) -> UnifiedGraph:
        """Финализация графа"""
        logger.info("🏁 СЛОЙ 2: Финализация...")
        await self._assign_sections()
        
        final_nodes = []
        for nid, data in self.G.nodes(data=True):
            section = data.get("target_section")
            if not section or section == "uncategorized":
                lbl = data.get("label")
                if lbl == NodeLabel.COMPONENT: section = TZSectionEnum.STACK
                elif lbl in [NodeLabel.REQUIREMENT, NodeLabel.TASK]: section = TZSectionEnum.FUNCTIONAL
                else: section = TZSectionEnum.GENERAL

            final_nodes.append(UnifiedNode(
                id=nid, label=data.get("label", NodeLabel.CONCEPT),
                name=data.get("name", nid), description=data.get("description", ""),
                target_section=section
            ))
        
        final_edges = [UnifiedEdge(source=u, target=v, relation=d.get("relation", "MENTIONS")) 
                       for u, v, d in self.G.edges(data=True)]
        
        return UnifiedGraph(nodes=final_nodes, edges=final_edges)

    async def _assign_sections(self):
        """Распределение узлов по секциям. БЕЗ подавления ошибок."""
        nodes = [{"id": n, "name": d.get("name")} for n, d in self.G.nodes(data=True) if d.get('label') != NodeLabel.PERSON]
        if not nodes: return
        
        for i in range(0, len(nodes), 10):
            batch = nodes[i:i+10]
            data_str = "\n".join([f"{n['id']}: {n['name']}" for n in batch])
            prompt = "Распредели по секциям: general_info, functional_req, tech_stack, ui_ux. Верни JSON."
            
            # Убран try-except. Если батч упадет - он будет ретраиться.
            result = await acall_llm_json(SectionBatchResult, prompt, data_str)
            for asn in result.assignments:
                if self.G.has_node(asn.node_id):
                    self.G.nodes[asn.node_id]["target_section"] = asn.target_section