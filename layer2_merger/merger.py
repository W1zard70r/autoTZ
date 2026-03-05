import logging
import asyncio
import random
import networkx as nx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from schemas.graph import (
    UnifiedGraph, UnifiedNode, UnifiedEdge, 
    KnowledgeNode, 
    ConflictSchema, ConflictResolution, ConflictOption
)
from schemas.enums import NodeLabel, EdgeRelation, TZSectionEnum
from utils.llm_client import acall_llm_json, DEFAULT_MODEL, acall_llm_text
from utils.state_logger import log_graphml, log_pydantic

logger = logging.getLogger(__name__)

# === ВНУТРЕННИЕ МОДЕЛИ ===

class MergeAction(BaseModel):
    is_duplicate: bool
    ids_to_merge: List[str]
    unified_id: str
    unified_name: str
    unified_desc: str

class MergeBatchResult(BaseModel):
    actions: List[MergeAction] = Field(default_factory=list)

class SectionAssignment(BaseModel):
    node_id: str
    target_section: str 

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

class NormalizedEntity(BaseModel):
    name: str
    label: str
    description: str

# ============================

class SmartGraphMerger:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.G = nx.DiGraph()
        self.active_conflicts: List[ConflictSchema] = []

    async def merge_subgraphs_and_deduplicate(self, subgraphs: List[Any]):
        """ЭТАП 1: Загрузка и дедупликация"""
        logger.info("🔗 СЛОЙ 2 (Шаг 1): Загрузка подграфов и дедупликация...")

        for sg in subgraphs:
            for node in sg.nodes:
                if not self.G.has_node(node.id):
                    node_data = node.model_dump() if hasattr(node, 'model_dump') else node.__dict__
                    self.G.add_node(node.id, **node_data)
            
            for edge in sg.edges:
                edge_data = edge.model_dump() if hasattr(edge, 'model_dump') else edge.__dict__
                clean_attrs = {k: v for k, v in edge_data.items() if k not in ['source', 'target']}
                self.G.add_edge(edge.source, edge.target, **clean_attrs)

        logger.info(f"  -> Исходный размер графа: {self.G.number_of_nodes()} узлов, {self.G.number_of_edges()} связей.")
        log_graphml("layer2_step1_initial_combined.graphml", self.G)

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

        for label, nodes in nodes_by_label.items():
            if len(nodes) < 2: continue
            logger.info(f"  -> Обработка группы '{label}' ({len(nodes)} узлов)...")
            
            batch_size = 15
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                await self._process_deduplication_batch(label, batch)

        logger.info("✅ Этап 1 завершен. Граф очищен от явных дублей.")

    async def _process_deduplication_batch(self, label: str, batch: List[Dict]):
        # Улучшенный промпт, чтобы не терять данные
        prompt = """Ты Архитектор Баз Знаний. Твоя задача - найти ПОЛНЫЕ СИНОНИМЫ.
        
        Пример дубликатов: "Postgres" и "PostgreSQL".
        Пример РАЗНЫХ сущностей: "User" и "Admin" (НЕ объединять!).
        
        Если нашел дубликаты:
        1. is_duplicate = true
        2. ids_to_merge = [список всех ID синонимов]
        3. unified_id = выбери самый короткий и понятный ID (snake_case)
        """
        data_str = "\n".join([f"ID: {n['id']} | Имя: {n['name']} | Desc: {n['desc']}" for n in batch])
        try:
            result = await acall_llm_json(schema=MergeBatchResult, prompt=prompt, data=data_str, model_name=self.model_name)
            for action in result.actions:
                if action.is_duplicate and len(action.ids_to_merge) > 1:
                    self._merge_nodes_in_graph(action)
        except Exception as e:
            logger.error(f"Ошибка при дедупликации батча в группе {label}: {e}")

    def _merge_nodes_in_graph(self, action: MergeAction):
        valid_ids = [nid for nid in action.ids_to_merge if self.G.has_node(nid)]
        if not valid_ids: return

        primary_id = action.unified_id
        if not self.G.has_node(primary_id):
            base_data = self.G.nodes[valid_ids[0]].copy()
            base_data.update({"id": primary_id, "name": action.unified_name, "description": action.unified_desc})
            self.G.add_node(primary_id, **base_data)
        else:
            self.G.nodes[primary_id].update({"name": action.unified_name, "description": action.unified_desc})

        for old_id in valid_ids:
            if old_id == primary_id: continue
            for u, v, data in list(self.G.edges(old_id, data=True)):
                if u == old_id: self.G.add_edge(primary_id, v, **data)
            for u, v, data in list(self.G.in_edges(old_id, data=True)):
                if v == old_id: self.G.add_edge(u, primary_id, **data)
            self.G.remove_node(old_id)

    async def detect_conflicts(self) -> List[ConflictSchema]:
        """ЭТАП 2: Поиск конфликтов"""
        logger.info("🔍 СЛОЙ 2 (Шаг 2): Поиск логических конфликтов...")
        
        nodes_list = []
        for nid, d in self.G.nodes(data=True):
            if d.get("label") in ["Component", "Requirement", "Concept", "Task"]:
                nodes_list.append(f"ID: {nid} | Name: {d.get('name')} | Desc: {d.get('description')}")
        
        nodes_desc = "\n".join(nodes_list)

        prompt = """
        РОЛЬ: Ты Главный Системный Архитектор (Russian speaking).
        ЗАДАЧА: Найти технические противоречия и споры.
        
        ВАЖНО ПО ID:
        В поле `options.id` ты ДОЛЖЕН использовать ТОЛЬКО существующие ID из списка ниже! 
        НЕ ПРИДУМЫВАЙ НОВЫЕ ID (типа "option_1"), если есть реальный узел (например "react").
        Если реального узла для опции нет, используй "none".

        ЯЗЫК: ВЕСЬ вывод (description, ai_recommendation, текст опций) СТРОГО НА РУССКОМ.
        """

        try:
            result = await acall_llm_json(
                schema=ConflictBatchResult, 
                prompt=prompt, 
                data=nodes_desc,
                model_name=self.model_name
            )
            
            self.active_conflicts = []
            for c in result.conflicts:
                self.active_conflicts.append(ConflictSchema(
                    id=c.id, description=c.description, category=c.category,
                    ai_recommendation=c.ai_recommendation, options=c.options
                ))
            return self.active_conflicts
        except Exception as e:
            logger.error(f"Ошибка поиска конфликтов: {e}")
            return []

    def _smart_remove_node(self, node_id: str):
        """
        Умное удаление: удаляет узел И связанные с ним 'Requirements'/'Tasks',
        если они больше ни к чему не привязаны (сироты).
        Это решает проблему 'Backend на Django' при удалении 'Django'.
        """
        if not self.G.has_node(node_id): return
        
        # 1. Находим соседей, которые зависят от этого узла
        neighbors_to_check = []
        for neighbor in self.G.neighbors(node_id):
            n_data = self.G.nodes[neighbor]
            # Если сосед - это Требование или Задача
            if n_data.get("label") in ["Requirement", "Task"]:
                neighbors_to_check.append(neighbor)
        
        # Также проверяем входящие связи (кто ссылается на удаляемый узел)
        for u, v in self.G.in_edges(node_id):
            n_data = self.G.nodes[u]
            if n_data.get("label") in ["Requirement", "Task"]:
                neighbors_to_check.append(u)

        # 2. Удаляем основной узел
        self.G.remove_node(node_id)
        logger.info(f"  -> 🗑️ Удален узел: {node_id}")

        # 3. Проверяем "сирот"
        for nid in set(neighbors_to_check):
            if not self.G.has_node(nid): continue
            
            # Если у требования осталось 0 или 1 связь (которая могла вести к людям),
            # и нет связей с другими Компонентами - удаляем его.
            # Упрощенная логика: если степень узла стала 0 (изолирован) -> удаляем
            if self.G.degree(nid) == 0:
                self.G.remove_node(nid)
                logger.info(f"  -> 🗑️ Каскадно удалено 'сиротское' требование: {nid}")
            else:
                # Если связи есть, проверяем, не ведут ли они только к "Людям" (авторам)
                # Если требование "Django Req" связано только с "Alex Lead", но не с компонентами - оно бесполезно
                connected_labels = [self.G.nodes[n].get("label") for n in self.G.neighbors(nid)]
                connected_labels += [self.G.nodes[u].get("label") for u, _ in self.G.in_edges(nid)]
                
                if "Component" not in connected_labels:
                     self.G.remove_node(nid)
                     logger.info(f"  -> 🗑️ Каскадно удалено требование без компонентов: {nid}")


    async def apply_resolutions(self, resolutions: List[ConflictResolution]):
        """ЭТАП 3: Применение решений"""
        logger.info("🛠️ СЛОЙ 2 (Шаг 3): Применение решений пользователя...")
        
        for res in resolutions:
            conflict = next((c for c in self.active_conflicts if c.id == res.conflict_id), None)
            if not conflict: continue
            
            all_options = [opt.id for opt in conflict.options]
            selected_id = res.selected_option_id

            # 1. Попытка найти по тексту (fuzzy match)
            if not selected_id and res.custom_text:
                clean_text = res.custom_text.lower().strip()
                for opt in conflict.options:
                    if clean_text in opt.text.lower() or opt.text.lower() in clean_text:
                        selected_id = opt.id
                        logger.info(f"  -> Авто-выбор по тексту: '{res.custom_text}' -> ID {selected_id}")
                        break

            # 2. Если выбран ID (или найден по тексту)
            if selected_id:
                # Удаляем ВСЕХ проигравших с помощью _smart_remove_node
                for loose_id in all_options:
                    # ВАЖНО: Не удаляем победителя
                    if loose_id != selected_id:
                        self._smart_remove_node(loose_id)
            
            # 3. Если кастомный ввод (и не совпал с опциями)
            elif res.custom_text:
                logger.info(f"  -> Обработка кастомного ввода: {res.custom_text}")
                
                # Удаляем ВСЕ опции конфликта (так как юзер выбрал третий путь)
                for loose_id in all_options:
                    self._smart_remove_node(loose_id)

                # Нормализуем ввод через LLM
                norm_entity = await self._normalize_user_choice(res.custom_text)
                
                new_id = f"custom_{res.conflict_id[:5]}_{random.randint(100,999)}"
                self.G.add_node(
                    new_id,
                    name=norm_entity.name,
                    label=norm_entity.label,
                    description=norm_entity.description,
                    target_section="tech_stack" # Дефолт
                )
                logger.info(f"  -> ✨ Создан новый узел: {norm_entity.name} ({new_id})")

    async def _normalize_user_choice(self, text: str) -> NormalizedEntity:
        prompt = f"""Преобразуй пожелание пользователя в формальную сущность для ТЗ.
        Вход: "{text}"
        Примеры:
        "хочу питон" -> Name="Python", Label="Component", Description="Язык программирования"
        "не надо js" -> Name="No JavaScript", Label="Requirement", Description="Запрет на использование JS"
        """
        try:
            return await acall_llm_json(NormalizedEntity, prompt, model_name=self.model_name)
        except:
            return NormalizedEntity(name=text, label="Requirement", description=text)

    async def finalize_graph(self) -> UnifiedGraph:
        """ЭТАП 4: Финализация"""
        logger.info("🏁 СЛОЙ 2 (Шаг 4): Финализация графа...")
        await self._assign_sections()

        final_nodes = []
        for nid, data in self.G.nodes(data=True):
            node_data = {
                "id": nid,
                "label": data.get("label", "Concept"),
                "name": data.get("name", nid),
                "description": data.get("description", ""),
                "properties": data.get("properties", []),
                "target_section": data.get("target_section")
            }
            if node_data["target_section"]:
                try:
                    TZSectionEnum(node_data["target_section"])
                except ValueError:
                    node_data["target_section"] = None
            final_nodes.append(UnifiedNode(**node_data))

        final_edges = []
        for u, v, data in self.G.edges(data=True):
            edge_data = {
                "source": u, "target": v,
                "relation": data.get("relation", "RELATES_TO"),
                "description": data.get("evidence", "") or data.get("description", "")
            }
            final_edges.append(UnifiedEdge(**edge_data))

        ug = UnifiedGraph(nodes=final_nodes, edges=final_edges)
        log_graphml("layer2_step3_final_unified.graphml", self.G)
        log_pydantic("layer2_step3_final_unified.json", ug)
        
        return ug

    async def _assign_sections(self):
        logger.info("  -> Распределение узлов по секциям ТЗ...")
        nodes_to_assign = []
        for n, d in self.G.nodes(data=True):
            # Перераспределяем ВСЕ технические узлы, чтобы исправить ошибки прошлого
            if d.get("label") in ["Component", "Requirement", "Concept"]:
                nodes_to_assign.append({"id": n, "name": d.get("name", "")})

        if not nodes_to_assign: return

        batch_size = 20
        for i in range(0, len(nodes_to_assign), batch_size):
            batch = nodes_to_assign[i:i + batch_size]
            data_str = "\n".join([f"ID:{n['id']} | {n['name']}" for n in batch])
            
            prompt = """Для каждого узла выбери target_section ИЗ СПИСКА:
            - general_info (Сущности, Схема БД, Бизнес-цели)
            - tech_stack (Языки, Фреймворки, БД)
            - functional_req (Авторизация, API, Валидация)
            - ui_ux (Дизайн, Экраны, UI-кит)
            """
            
            await asyncio.sleep(2) 
            try:
                result = await acall_llm_json(schema=SectionBatchResult, prompt=prompt, data=data_str, model_name=self.model_name)
                for assignment in result.assignments:
                    if self.G.has_node(assignment.node_id):
                        self.G.nodes[assignment.node_id]["target_section"] = assignment.target_section
            except Exception as e:
                logger.warning(f"Ошибка распределения секций: {e}")