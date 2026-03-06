import logging
import asyncio
import random
import networkx as nx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from schemas.graph import UnifiedGraph, UnifiedNode, UnifiedEdge, ConflictSchema, ConflictResolution
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
        for sg in subgraphs:
            for node in sg.nodes:
                if not self.G.has_node(node.id):
                    self.G.add_node(node.id, **node.model_dump())
            for edge in sg.edges:
                self.G.add_edge(edge.source, edge.target, **edge.model_dump(exclude={'source', 'target'}))
        logger.info(f"✅ Граф: {self.G.number_of_nodes()} узлов.")

    async def detect_conflicts(self) -> List[ConflictSchema]:
        nodes_list = [f"ID: {nid} | Name: {d.get('name')}" for nid, d in self.G.nodes(data=True) 
                      if d.get("label") in [NodeLabel.COMPONENT, NodeLabel.REQUIREMENT, NodeLabel.TASK]]
        if not nodes_list: return []
        
        prompt = """ЗАДАЧА: Найти технические противоречия. Группируй их по ролям (Бэкенд, Фронтенд, БД). 
        В опциях используй только реальные ID из списка. ЯЗЫК: РУССКИЙ."""
        
        result = await acall_llm_json(ConflictBatchResult, prompt, "\n".join(nodes_list))
        self.active_conflicts = result.conflicts
        return self.active_conflicts

    async def apply_resolutions(self, resolutions: List[ConflictResolution]):
        for res in resolutions:
            conflict = next((c for c in self.active_conflicts if c.id == res.conflict_id), None)
            if not conflict: continue
            
            selected_ids = []
            if res.selected_option_id:
                selected_ids = [res.selected_option_id]
            elif res.custom_text:
                parts = res.custom_text.replace(" и ", ",").split(",")
                for part in parts:
                    clean = part.strip().lower()
                    if clean.isdigit() and int(clean) < len(conflict.options):
                        selected_ids.append(conflict.options[int(clean)].id)
                        continue
                    for opt in conflict.options:
                        if clean in opt.text.lower() or clean in opt.id.lower():
                            selected_ids.append(opt.id)
                            break
            
            if selected_ids:
                for opt in conflict.options:
                    if opt.id not in selected_ids and self.G.has_node(opt.id):
                        self.G.remove_node(opt.id)

    async def finalize_graph(self) -> UnifiedGraph:
        # Сначала запускаем классификацию
        await self._assign_sections()
        
        final_nodes = []
        for nid, data in self.G.nodes(data=True):
            # Базовый структурный Fallback (без ключевых слов)
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
        """Усиленный классификатор"""
        nodes = [{"id": n, "name": d.get("name"), "desc": d.get("description")} 
                 for n, d in self.G.nodes(data=True) if d.get('label') != NodeLabel.PERSON]
        if not nodes: return
        
        for i in range(0, len(nodes), 10):
            data_str = "\n".join([f"{n['id']}: {n['name']} ({n['desc']})" for n in nodes[i:i+10]])
            prompt = """Для каждого узла выбери раздел ТЗ:
            - general_info: концепции, роли, цели проекта.
            - functional_req: действия, бизнес-логика, эндпоинты API.
            - tech_stack: языки, фреймворки, базы данных, библиотеки.
            - ui_ux: визуальные части, экраны, формы, элементы интерфейса.
            Верни JSON SectionBatchResult."""
            
            result = await acall_llm_json(SectionBatchResult, prompt, data_str)
            for asn in result.assignments:
                if self.G.has_node(asn.node_id):
                    self.G.nodes[asn.node_id]["target_section"] = asn.target_section