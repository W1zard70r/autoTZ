import logging
import asyncio
import networkx as nx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from schemas.graph import (
    UnifiedGraph, UnifiedNode, UnifiedEdge, 
    KnowledgeNode, 
    ConflictSchema, ConflictResolution, ConflictOption
)
from schemas.enums import NodeLabel, EdgeRelation, TZSectionEnum
from utils.llm_client import acall_llm_json
from utils.state_logger import log_graphml, log_pydantic

logger = logging.getLogger(__name__)

# === ВНУТРЕННИЕ МОДЕЛИ ДЛЯ LLM (Чтобы Pydantic не ругался) ===

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
    target_section: str # Строка, которую потом приведем к Enum

class SectionBatchResult(BaseModel):
    assignments: List[SectionAssignment]

class DetectedConflict(BaseModel):
    id: str
    category: str
    description: str
    options: List[ConflictOption]
    ai_recommendation: str

class ConflictBatchResult(BaseModel):
    conflicts: List[DetectedConflict] = Field(default_factory=list)

# ============================================================

class SmartGraphMerger:
    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        self.model_name = model_name
        # Используем NetworkX для удобной работы с графом (как в старой версии)
        self.G = nx.DiGraph()
        self.active_conflicts: List[ConflictSchema] = []

    async def merge_subgraphs_and_deduplicate(self, subgraphs: List[Any]):
        """
        ЭТАП 1: Загрузка в NetworkX и дедупликация.
        """
        logger.info("🔗 СЛОЙ 2 (Шаг 1): Загрузка подграфов и дедупликация...")

        # 1. Загрузка данных в плоский граф NetworkX
        for sg in subgraphs:
            for node in sg.nodes:
                # Превращаем KnowledgeNode в словарь для NetworkX
                if not self.G.has_node(node.id):
                    # Безопасная конвертация
                    node_data = node.model_dump() if hasattr(node, 'model_dump') else node.__dict__
                    self.G.add_node(node.id, **node_data)
            
            for edge in sg.edges:
                edge_data = edge.model_dump() if hasattr(edge, 'model_dump') else edge.__dict__
                # Убираем source/target из атрибутов ребра, они уже есть в структуре графа
                clean_attrs = {k: v for k, v in edge_data.items() if k not in ['source', 'target']}
                self.G.add_edge(edge.source, edge.target, **clean_attrs)

        logger.info(f"  -> Исходный размер графа: {self.G.number_of_nodes()} узлов, {self.G.number_of_edges()} связей.")
        log_graphml("layer2_step1_initial_combined.graphml", self.G)

        # 2. Группировка узлов по Label
        nodes_by_label: Dict[str, List[Dict[str, Any]]] = {}
        for nid, data in self.G.nodes(data=True):
            label = data.get("label", "Unknown")
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

            logger.info(f"  -> Обработка группы '{label}' ({len(nodes)} узлов)...")
            
            # Разбиваем на батчи по 15 штук
            batch_size = 15
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                # --- ВАЖНО: ЗАДЕРЖКА ---
                await asyncio.sleep(4) 
                
                await self._process_deduplication_batch(label, batch)

        logger.info("✅ Этап 1 завершен. Граф очищен от явных дублей.")

    async def _process_deduplication_batch(self, label: str, batch: List[Dict]):
        prompt = """Ты Архитектор. Найди дубликаты среди этих узлов (синонимы, одно и то же понятие).
        Например: 'Postgres' и 'PostgreSQL' - это дубликаты.
        
        Если находишь дубликаты, верни MergeAction с is_duplicate=true.
        В ids_to_merge перечисли ВСЕ ID, которые относятся к одной сущности.
        Unified ID должен быть snake_case.
        """
        
        data_str = "\n".join([f"ID: {n['id']} | Имя: {n['name']} | Описание: {n['desc']}" for n in batch])
        
        try:
            result = await acall_llm_json(schema=MergeBatchResult, prompt=prompt, data=data_str, model_name=self.model_name)
            
            for action in result.actions:
                if action.is_duplicate and len(action.ids_to_merge) > 1:
                    self._merge_nodes_in_graph(action)
                    
        except Exception as e:
            logger.error(f"Ошибка при дедупликации батча в группе {label}: {e}")

    def _merge_nodes_in_graph(self, action: MergeAction):
        """Физическое слияние узлов в NetworkX графе"""
        valid_ids = [nid for nid in action.ids_to_merge if self.G.has_node(nid)]
        if not valid_ids: return

        primary_id = action.unified_id
        
        # Если целевого узла еще нет (новый ID), создаем его на основе первого
        if not self.G.has_node(primary_id):
            base_data = self.G.nodes[valid_ids[0]].copy()
            base_data.update({
                "id": primary_id,
                "name": action.unified_name,
                "description": action.unified_desc
            })
            self.G.add_node(primary_id, **base_data)
        else:
            # Если уже есть, обновляем описание
            self.G.nodes[primary_id]["name"] = action.unified_name
            self.G.nodes[primary_id]["description"] = action.unified_desc

        # Переносим связи
        for old_id in valid_ids:
            if old_id == primary_id: continue
            
            # Исходящие
            for u, v, data in list(self.G.edges(old_id, data=True)):
                if u == old_id: self.G.add_edge(primary_id, v, **data)
            
            # Входящие
            for u, v, data in list(self.G.in_edges(old_id, data=True)):
                if v == old_id: self.G.add_edge(u, primary_id, **data)
            
            # Удаляем старый узел
            self.G.remove_node(old_id)

    async def detect_conflicts(self) -> List[ConflictSchema]:
        """
        ЭТАП 2: Поиск логических противоречий.
        """
        logger.info("🔍 СЛОЙ 2 (Шаг 2): Поиск логических конфликтов...")
        
        # Собираем контекст из графа
        nodes_context = []
        # Берем топ-50 узлов, чтобы не забить контекст
        tech_nodes = [n for n, d in self.G.nodes(data=True) if d.get("label") in ["Component", "Requirement", "Concept"]]
        
        for nid in tech_nodes[:50]: 
            data = self.G.nodes[nid]
            nodes_context.append(f"ID: {nid} | Имя: {data.get('name')} | Описание: {data.get('description')}")
        
        nodes_desc = "\n".join(nodes_context)

        prompt = """
        Ты Главный Системный Архитектор. Найди ВЗАИМОИСКЛЮЧАЮЩИЕ технологические решения в списке.
        Например: "React" VS "Vue" (если они оба предложены как основной фреймворк).
        
        Игнорируй:
        - Фронтенд vs Бэкенд (React + Python = ОК).
        - БД + Кэш (Postgres + Redis = ОК).
        
        Если есть конфликт, опиши его и предложи варианты.
        """

        try:
            result = await acall_llm_json(
                schema=ConflictBatchResult, 
                prompt=prompt, 
                data=nodes_desc,
                model_name=self.model_name
            )
            
            # Конвертируем DetectedConflict (Internal) -> ConflictSchema (Public)
            self.active_conflicts = []
            for c in result.conflicts:
                self.active_conflicts.append(ConflictSchema(
                    id=c.id,
                    description=c.description,
                    category=c.category,
                    ai_recommendation=c.ai_recommendation,
                    options=c.options
                ))
            
            return self.active_conflicts
            
        except Exception as e:
            logger.error(f"Ошибка поиска конфликтов: {e}")
            return []

    def apply_resolutions(self, resolutions: List[ConflictResolution]):
        """
        ЭТАП 3: Применение решений пользователя.
        """
        logger.info("🛠️ СЛОЙ 2 (Шаг 3): Применение решений пользователя...")
        
        for res in resolutions:
            conflict = next((c for c in self.active_conflicts if c.id == res.conflict_id), None)
            if not conflict: continue
            
            all_options = [opt.id for opt in conflict.options]

            if res.selected_option_id:
                logger.info(f"  -> Победил: {res.selected_option_id}")
                # Удаляем проигравших
                for loose_id in all_options:
                    if loose_id != res.selected_option_id and self.G.has_node(loose_id):
                        self.G.remove_node(loose_id)

            elif res.custom_text:
                logger.info(f"  -> Свой вариант: {res.custom_text}")
                # Удаляем все варианты конфликта
                for loose_id in all_options:
                    if self.G.has_node(loose_id):
                        self.G.remove_node(loose_id)
                
                # Добавляем новый узел
                new_id = f"custom_{res.conflict_id[:5]}"
                self.G.add_node(new_id, name=res.custom_text, label="Component", description="Custom User Choice", target_section="tech_stack")

    async def finalize_graph(self) -> UnifiedGraph:
        """
        ЭТАП 4: Конвертация NetworkX -> UnifiedGraph и распределение по секциям.
        """
        logger.info("🏁 СЛОЙ 2 (Шаг 4): Финализация графа...")

        # 1. Распределение по секциям
        await self._assign_sections()

        # 2. Конвертация в Pydantic
        final_nodes = []
        for nid, data in self.G.nodes(data=True):
            # Подготовка данных
            node_data = {
                "id": nid,
                "label": data.get("label", "Concept"),
                "name": data.get("name", nid),
                "description": data.get("description", ""),
                "properties": data.get("properties", []),
                "target_section": data.get("target_section")
            }
            
            # Валидация target_section
            if node_data["target_section"]:
                try:
                    # Проверяем, валидный ли это enum
                    TZSectionEnum(node_data["target_section"])
                except ValueError:
                    node_data["target_section"] = None

            final_nodes.append(UnifiedNode(**node_data))

        final_edges = []
        for u, v, data in self.G.edges(data=True):
            edge_data = {
                "source": u,
                "target": v,
                "relation": data.get("relation", "RELATES_TO"),
                "description": data.get("evidence", "") or data.get("description", "")
            }
            final_edges.append(UnifiedEdge(**edge_data))

        unified_graph = UnifiedGraph(nodes=final_nodes, edges=final_edges)
        
        # Логируем результат
        log_graphml("layer2_step3_final_unified.graphml", self.G)
        
        return unified_graph

    async def _assign_sections(self):
        logger.info("  -> Распределение узлов по секциям ТЗ...")
        
        # Берем узлы без секции
        nodes_to_assign = []
        for n, d in self.G.nodes(data=True):
            if not d.get("target_section") or d.get("target_section") == "uncategorized":
                nodes_to_assign.append({"id": n, "name": d.get("name", "")})

        if not nodes_to_assign: return

        # Батчами по 20
        batch_size = 20
        for i in range(0, len(nodes_to_assign), batch_size):
            batch = nodes_to_assign[i:i + batch_size]
            data_str = "\n".join([f"ID:{n['id']} | {n['name']}" for n in batch])
            
            prompt = """Распредели узлы по секциям ТЗ:
            - general_info (Сущности, БД)
            - tech_stack (Технологии, языки)
            - functional_req (Функции, задачи)
            - ui_ux (Дизайн)
            """
            
            # --- ВАЖНО: ЗАДЕРЖКА ---
            await asyncio.sleep(4) 

            try:
                result = await acall_llm_json(schema=SectionBatchResult, prompt=prompt, data=data_str, model_name=self.model_name)
                for assignment in result.assignments:
                    if self.G.has_node(assignment.node_id):
                        self.G.nodes[assignment.node_id]["target_section"] = assignment.section
            except Exception as e:
                logger.warning(f"Ошибка распределения секций: {e}")