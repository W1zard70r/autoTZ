import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, TypedDict, Literal, Annotated, Union, get_origin, get_args
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from schemas.graph import UnifiedGraph, GraphNode, GraphEdge
from schemas.enums import TemplateType
from schemas.templates.base import BaseTemplate, TZResult
from schemas.templates import get_template_class
from utils.llm_client import acall_llm_json

logger = logging.getLogger(__name__)


# ============================================================================
# TOOLS FOR GRAPH NAVIGATION
# ============================================================================

@dataclass
class GraphContext:
    graph: UnifiedGraph
    template_type: TemplateType
    user_answers: Optional[Dict] = None


class GraphTools:
    def __init__(self, context: GraphContext):
        self.context = context
        self._id_cache: Dict[str, GraphNode] = {}
        self._name_cache: Dict[str, List[GraphNode]] = {}
        self._build_cache()

    def _build_cache(self):
        for node in self.context.graph.nodes:
            self._id_cache[node.id] = node
            name_key = node.name.lower()
            if name_key not in self._name_cache:
                self._name_cache[name_key] = []
            self._name_cache[name_key].append(node)

    def _get_node_by_id_or_name(self, key: str) -> Optional[GraphNode]:
        if key in self._id_cache:
            return self._id_cache[key]
        matches = self._name_cache.get(key.lower(), [])
        return matches[0] if matches else None

    def search_nodes(self, query: str, node_type: Optional[str] = None, limit: int = 10) -> List[Dict]:
        results =[]
        query_lower = query.lower()

        for node in self.context.graph.nodes:
            score = 0

            # 1. Высший приоритет: совпадение по target_section
            if node.target_section:
                sec_val = getattr(node.target_section, 'value', str(node.target_section)).lower()
                if query_lower in sec_val:
                    score += 10

            # 2. Высокий приоритет: совпадение в названии
            if query_lower in node.name.lower():
                score += 5

            # 3. Средний приоритет: совпадение в описании
            if node.description and query_lower in node.description.lower():
                score += 3

            # 4. Низкий приоритет: поиск внутри properties
            properties_str = " ".join([f"{p.key}:{p.value}" for p in node.properties])
            if query_lower in properties_str.lower():
                score += 1

            # 5. Бонус за совпадение типа (если запрошен)
            if node_type:
                lbl_val = getattr(node.label, 'value', str(node.label))
                if lbl_val == node_type:
                    score += 2

            if score > 0:
                results.append({
                    "id": node.id,
                    "name": node.name,
                    "label": getattr(node.label, 'value', str(node.label)),
                    "description": node.description,
                    "score": score,
                    "properties":[p.model_dump() for p in node.properties]
                })

        # Сортируем по убыванию релевантности
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_node_details(self, node_id: str) -> Optional[Dict]:
        node = self._get_node_by_id_or_name(node_id)
        if not node:
            return None
        incoming =[e for e in self.context.graph.edges if e.target == node.id]
        outgoing =[e for e in self.context.graph.edges if e.source == node.id]
        return {
            "id": node.id,
            "name": node.name,
            "label": getattr(node.label, 'value', str(node.label)),
            "description": node.description,
            "properties":[p.model_dump() for p in node.properties],
            "target_section": getattr(node.target_section, 'value', str(node.target_section)) if node.target_section else None,
            "incoming":[{"from": e.source, "relation": getattr(e.relation, 'value', str(e.relation))} for e in incoming],
            "outgoing":[{"to": e.target, "relation": getattr(e.relation, 'value', str(e.relation))} for e in outgoing]
        }

    def get_related_nodes(self, node_id: str, relation_type: Optional[str] = None) -> List[Dict]:
        edges =[e for e in self.context.graph.edges if e.source == node_id or e.target == node_id]
        if relation_type:
            edges =[e for e in edges if getattr(e.relation, 'value', str(e.relation)) == relation_type]

        results =[]
        for edge in edges:
            is_outgoing = edge.source == node_id
            related_id = edge.target if is_outgoing else edge.source
            related_node = self._id_cache.get(related_id)
            if related_node:
                results.append({
                    "node": {"id": related_node.id, "name": related_node.name, "label": getattr(related_node.label, 'value', str(related_node.label))},
                    "relation": getattr(edge.relation, 'value', str(edge.relation)),
                    "direction": "outgoing" if is_outgoing else "incoming",
                    "evidence": edge.evidence
                })
        return results

    def get_graph_summary(self) -> Dict:
        types: Dict[str, int] = {}
        for node in self.context.graph.nodes:
            t = getattr(node.label, 'value', str(node.label))
            types[t] = types.get(t, 0) + 1
        return {
            "total_nodes": len(self.context.graph.nodes),
            "total_edges": len(self.context.graph.edges),
            "node_labels": types,
            "node_names":[n.name for n in self.context.graph.nodes]
        }


# ============================================================================
# STATE & HELPERS
# ============================================================================

class SectionDraft(BaseModel):
    content: Dict[str, Any]
    sources: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""

class ReviewResult(BaseModel):
    is_approved: bool
    issues: List[Dict[str, str]] = Field(default_factory=list)
    missing_facts: List[str] = Field(default_factory=list)
    suggestions: str = ""
    confidence: float = 0.0

class EditRequest(BaseModel):
    section_name: str
    feedback: str
    preserve_aspects: List[str] = Field(default_factory=list)

_RUN_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_run(run_id: str, schema_class: type, context: GraphContext) -> None:
    _RUN_REGISTRY[run_id] = {"schema_class": schema_class, "context": context}

def get_run_meta(run_id: str) -> Dict[str, Any]:
    return _RUN_REGISTRY[run_id]

class AgentState(TypedDict):
    run_id: str
    section_name: str
    instruction: str
    draft: Optional[SectionDraft]
    review: Optional[ReviewResult]
    iterations: int
    max_iterations: int
    edit_mode: bool
    edit_request: Optional[EditRequest]
    final_result: Optional[Dict[str, Any]]
    status: Literal["drafting", "reviewing", "editing", "approved", "rejected"]
    logs: List[str]

def get_model_class(field_info) -> Optional[type]:
    ann = field_info.annotation
    origin = get_origin(ann)

    if origin is Union:
        for arg in get_args(ann):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    if origin in (list,):
        for arg in get_args(ann):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    if origin is Annotated:
        for arg in get_args(ann):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann

    return None

def make_thread_id(prefix: str, graph: UnifiedGraph) -> str:
    graph_fingerprint = hashlib.md5(
        f"{len(graph.nodes)}-{len(graph.edges)}".encode()
    ).hexdigest()[:8]
    return f"{prefix}_{graph_fingerprint}"


# ============================================================================
# AGENT PROMPTS
# ============================================================================

RESEARCHER_SYSTEM = """Ты Researcher Agent. Твоя задача — собрать факты из графа и создать черновик секции ТЗ.
ПРАВИЛА:
1. Используй ТОЛЬКО факты из предоставленного контекста графа.
2. Каждый факт должен основываться на узлах (node ID).
3. НЕ выдумывай технические детали или имена, которых нет в контексте.
4. Если информации мало — заполни то, что есть, и отметь нехватку в notes."""

REVIEWER_SYSTEM = """Ты Reviewer Agent. Проверяй черновик строго по графу.
Отмечай hallucination, missing_source, inconsistency.
is_approved=True только если черновик полностью опирается на предоставленные источники."""

EDITOR_SYSTEM = """Ты Editor Agent. Исправляй секцию строго по feedback пользователя.
Обязательно сохраняй preserve_aspects без изменений."""


# ============================================================================
# AGENTS
# ============================================================================

class ResearcherAgent:

    # Маппинг названий секций из Pydantic в названия категорий в Knowledge Graph
    SECTION_MAP = {
        "general": ["general_info", "general", "overview", "project"],
        "functional":["functional_req", "functional", "features", "requirements"],
        "tech":["tech_stack", "tech", "technologies", "architecture"],
        "ui":["ui_ux", "ui", "ux", "interface", "design"],
        "non_functional": ["non_functional", "performance", "security"]
    }

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or acall_llm_json

    async def run(self, state: AgentState) -> Command:
        logger.info(f"[Researcher] Starting for: {state['section_name']}")
        meta = get_run_meta(state['run_id'])
        tools = GraphTools(meta['context'])

        review = state.get('review')
        extra_instruction = ""
        if review and not review.is_approved:
            issues_text = "\n".join([f"- {i.get('type')}: {i.get('detail', '')}" for i in review.issues])
            suggestions = review.suggestions or ""
            extra_instruction = (
                f"\n\nПРЕДЫДУЩИЕ ПРОБЛЕМЫ (ИСПРАВЬ ИХ):\n{issues_text}\n{suggestions}"
            )

        exploration = await self._explore_graph(tools, state)
        draft = await self._create_draft(meta['schema_class'], tools, state, exploration, extra_instruction)

        return Command(
            update={
                "draft": draft,
                "status": "reviewing",
                "logs": state.get("logs", []) +[f"Researcher: explored {len(exploration)} nodes"]
            },
            goto="reviewer"
        )

    async def _explore_graph(self, tools: GraphTools, state: AgentState) -> List[Dict]:
        section_name = state['section_name']

        # 1. Поиск по маппингу (самый надежный способ найти узлы для секции)
        search_terms = self.SECTION_MAP.get(section_name.lower(), [section_name])
        results =[]
        for term in search_terms:
            results.extend(tools.search_nodes(term, limit=10))

        # 2. Поиск по ключевым словам из инструкции
        key_terms = self._extract_key_terms(state['instruction'])
        for term in key_terms[:3]:
            results.extend(tools.search_nodes(term, limit=5))

        # 3. Дедупликация и сбор полных данных
        unique_results = {r['id']: r for r in results}
        sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)

        detailed =[]
        seen: set = set()

        for r in sorted_results[:10]:  # Берем топ-10 узлов
            if r['id'] not in seen:
                details = tools.get_node_details(r['id'])
                if details:
                    detailed.append(details)
                    seen.add(r['id'])
                    # Захватываем связанных соседей
                    for rel in tools.get_related_nodes(r['id'])[:2]:
                        rid = rel['node']['id']
                        if rid not in seen:
                            d = tools.get_node_details(rid)
                            if d:
                                detailed.append(d)
                                seen.add(rid)

        # 4. Fallback: если ничего не найдено
        if not detailed:
            logger.warning(f"[Researcher] No nodes found for '{section_name}'. Using limited fallback.")
            summary = tools.get_graph_summary()
            # Берем максимум 3 узла, чтобы не засорять контекст
            for node_name in summary.get("node_names", [])[:3]:
                for r in tools.search_nodes(node_name, limit=1):
                    if r['id'] not in seen:
                        d = tools.get_node_details(r['id'])
                        if d:
                            detailed.append(d)
                            seen.add(r['id'])

        return detailed

    def _extract_key_terms(self, instruction: str) -> List[str]:
        words = instruction.lower().replace(',', ' ').replace('.', ' ').split()
        stop = {
            'и', 'в', 'на', 'с', 'по', 'для', 'из', 'к', 'а', 'не', 'что', 'как', 'это',
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are'
        }
        return[w for w in words if len(w) > 3 and w not in stop][:5]

    async def _create_draft(
            self, schema_class: type, tools: GraphTools, state: AgentState, exploration: List[Dict], extra: str = ""
    ) -> SectionDraft:
        context_parts = []
        sources =[]
        for node in exploration:
            context_parts.append(
                f"Узел: {node['name']} (ID: {node['id']})\n"
                f"Описание: {node.get('description', 'N/A')}\n"
                f"Данные: {node.get('properties',[])}\n"
                f"Входящие связи: {node.get('incoming',[])}\n"
                f"Исходящие связи: {node.get('outgoing',[])}"
            )
            sources.append(node['id'])

        if not context_parts:
            summary = tools.get_graph_summary()
            graph_data = f"[Детальные данные не найдены]\nСводка графа: {summary}"
            logger.error(f"[Researcher] Exploration empty for section '{state['section_name']}'")
        else:
            graph_data = "\n---\n".join(context_parts)

        prompt = (
            f"Создай черновик секции ТЗ.\n\n"
            f"НАЗВАНИЕ СЕКЦИИ: {state['section_name']}\n"
            f"ИНСТРУКЦИЯ К СЕКЦИИ: {state['instruction']}{extra}\n\n"
            f"Верни JSON строго по схеме. Если данных в графе не хватает для полного ответа, "
            f"заполни поля насколько возможно и не выдумывай лишнего."
        )

        try:
            draft_data = await self.llm_client(
                schema=schema_class,
                prompt=prompt,
                data=graph_data,
                system=RESEARCHER_SYSTEM,
                max_tokens=16384
            )
            return SectionDraft(
                content=draft_data.model_dump(),
                sources=list(set(sources)),
                confidence=0.85 if len(sources) > 2 else 0.5,
                notes=f"Based on {len(sources)} nodes"
            )
        except Exception as e:
            logger.error(f"[Researcher] _create_draft failed: {e}")
            return SectionDraft(content={}, sources=[], confidence=0.0, notes=str(e))


class ReviewerAgent:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or acall_llm_json

    async def run(self, state: AgentState) -> Command:
        logger.info(f"[Reviewer] Reviewing: {state['section_name']}")

        if not state.get('draft'):
            return Command(
                update={"status": "rejected", "logs": state.get("logs", []) + ["Reviewer: no draft → rejected"]},
                goto=END
            )

        meta = get_run_meta(state['run_id'])
        review = await self._verify_draft(state, meta)

        iterations = state.get('iterations', 0)
        max_iter = state.get('max_iterations', 3)

        if review.is_approved or iterations >= max_iter:
            if not review.is_approved:
                logger.warning(
                    f"[Reviewer] Section '{state['section_name']}' не прошла проверку "
                    f"за {max_iter} итераций — принимаем as-is."
                )

            final: Optional[Dict[str, Any]] = None
            draft = state.get('draft')
            if draft and draft.content:
                try:
                    validated = meta['schema_class'].model_validate(draft.content)
                    final = validated.model_dump()
                except Exception as e:
                    logger.warning(f"[Reviewer] model_validate failed: {e}")
                    final = draft.content

            status: Literal["approved", "rejected"] = "approved" if review.is_approved else "rejected"
            log_msg = f"Reviewer: {status}" + ("" if review.is_approved else " (max iterations reached)")

            return Command(
                update={
                    "review": review,
                    "final_result": final,
                    "status": status,
                    "logs": state.get("logs", []) + [log_msg]
                },
                goto=END
            )

        return Command(
            update={
                "review": review,
                "iterations": iterations + 1,
                "status": "drafting",
                "logs": state.get("logs", []) + ["Reviewer: issues found → back to researcher"]
            },
            goto="researcher"
        )

    async def _verify_draft(self, state: AgentState, meta: Dict[str, Any]) -> ReviewResult:
        draft = state['draft']
        tools = GraphTools(meta['context'])

        # ДОБАВЛЕНО: Собираем тексты источников, на которые сослался Researcher
        source_parts = []
        for src_id in draft.sources:
            node = tools.get_node_details(src_id)
            if node:
                source_parts.append(
                    f"Узел: {node['name']} (ID: {node['id']})\n"
                    f"Описание: {node.get('description', 'N/A')}\n"
                    f"Данные: {node.get('properties', [])}"
                )

        sources_data = "\n---\n".join(source_parts) if source_parts else "Нет детализированных данных по источникам."

        prompt = (
            f"Проверь черновик:\n\n"
            f"ЧЕРНОВИК: {draft.content}\n"
            f"ИСТОЧНИКИ (ID УЗЛОВ): {draft.sources}\n\n"
            f"Убедись, что нет галлюцинаций (hallucination) и противоречий (inconsistency). "
            f"Одобряй (is_approved=True) только если все факты из черновика есть в текстах переданных источников."
        )
        try:
            return await self.llm_client(
                schema=ReviewResult,
                prompt=prompt,
                data=sources_data,  # <-- ПЕРЕДАЕМ ТЕКСТЫ ИСТОЧНИКОВ В LLM
                system=REVIEWER_SYSTEM,
                max_tokens=16384
            )
        except Exception as e:
            logger.error(f"[Reviewer] _verify_draft failed: {e}")
            return ReviewResult(is_approved=False, issues=[{"type": "error", "detail": str(e)}])


class EditorAgent:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client or acall_llm_json

    async def run(self, state: AgentState) -> Command:
        logger.info(f"[Editor] Editing: {state['section_name']}")
        edit_request = state.get('edit_request')
        if not edit_request or not state.get('draft'):
            return Command(
                update={"status": "approved", "logs": state.get("logs", []) +["Editor: nothing to edit"]},
                goto=END
            )

        meta = get_run_meta(state['run_id'])
        tools = GraphTools(meta['context'])

        additional =[]
        seen: set = set()
        for term in self._extract_feedback_terms(edit_request.feedback):
            for r in tools.search_nodes(term, limit=3):
                if r['id'] not in seen:
                    d = tools.get_node_details(r['id'])
                    if d:
                        additional.append(d)
                        seen.add(r['id'])

        edited = await self._apply_edits(state, meta['schema_class'], state['draft'], edit_request, additional)

        return Command(
            update={
                "draft": edited,
                "edit_mode": False,
                "edit_request": None,
                "status": "reviewing",
                "logs": state.get("logs", []) + ["Editor: feedback applied"]
            },
            goto="reviewer"
        )

    def _extract_feedback_terms(self, feedback: str) -> List[str]:
        words = feedback.lower().replace(',', ' ').replace('.', ' ').split()
        stop = {'и', 'в', 'на', 'с', 'по', 'для', 'из', 'это', 'что', 'как', 'не'}
        return [w for w in words if len(w) > 3 and w not in stop][:5]

    async def _apply_edits(
        self, state: AgentState, schema_class: type, draft: SectionDraft, request: EditRequest, context: List[Dict]
    ) -> SectionDraft:
        prompt = (
            f"Перепиши секцию по feedback пользователя:\n\n"
            f"ТЕКУЩАЯ ВЕРСИЯ: {draft.content}\n"
            f"FEEDBACK ПОЛЬЗОВАТЕЛЯ: {request.feedback}\n"
            f"СОХРАНИ (НЕ УДАЛЯЙ): {request.preserve_aspects}\n"
            f"ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ: {context[:5]}"
        )
        try:
            edited_data = await self.llm_client(
                schema=schema_class,
                prompt=prompt,
                system=EDITOR_SYSTEM,
                max_tokens=16384
            )
            return SectionDraft(
                content=edited_data.model_dump(),
                sources=draft.sources + [c.get('id') for c in context if 'id' in c],
                confidence=draft.confidence
            )
        except Exception as e:
            logger.error(f"[Editor] _apply_edits failed: {e}")
            return draft


# ============================================================================
# WORKFLOW
# ============================================================================

def create_compiler_workflow(checkpointer=None) -> StateGraph:
    workflow = StateGraph(AgentState)

    researcher = ResearcherAgent()
    reviewer = ReviewerAgent()
    editor = EditorAgent()

    def router_node(state: AgentState) -> Command:
        target = "editor" if state.get("edit_mode") else "researcher"
        return Command(goto=target)

    workflow.add_node("router", router_node)
    workflow.add_node("researcher", researcher.run)
    workflow.add_node("reviewer", reviewer.run)
    workflow.add_node("editor", editor.run)

    workflow.add_edge(START, "router")
    workflow.add_edge("editor", "reviewer")

    return workflow.compile(checkpointer=checkpointer)


# ============================================================================
# COMPILER
# ============================================================================

class TZCompiler:
    def __init__(self, language: str = "ru", max_iterations: int = 3):
        self.language = language
        self.max_iterations = max_iterations
        self.checkpointer = MemorySaver()
        self.workflow = create_compiler_workflow(self.checkpointer)

    async def compile(
        self, graph: UnifiedGraph, template_type: TemplateType, user_answers: Optional[Dict] = None
    ) -> TZResult:
        context = GraphContext(graph=graph, template_type=template_type, user_answers=user_answers)
        template_class = get_template_class(template_type)

        filled_sections: Dict[str, Any] = {}
        section_logs: Dict[str, List[str]] = {}

        for field_name, field_info in template_class.model_fields.items():
            model_class = get_model_class(field_info)
            if not model_class:
                continue

            logger.info(f"[Compiler] Processing section: {field_name}")
            run_id = f"{field_name}_{make_thread_id('compile', graph)}"
            register_run(run_id, model_class, context)

            initial_state: AgentState = {
                "run_id": run_id, "section_name": field_name,
                "instruction": field_info.description or f"Заполни секцию {field_name}",
                "draft": None, "review": None, "iterations": 0,
                "max_iterations": self.max_iterations, "edit_mode": False,
                "edit_request": None, "final_result": None,
                "status": "drafting", "logs":[]
            }

            result = await self.workflow.ainvoke(
                initial_state, config={"configurable": {"thread_id": run_id}}
            )

            if result.get('final_result'):
                try:
                    filled_sections[field_name] = model_class.model_validate(result['final_result'])
                except Exception as e:
                    logger.warning(f"[Compiler] final_result validate failed for {field_name}: {e}")
                    filled_sections[field_name] = model_class()  # Безопасный дефолт
            elif result.get('draft') and result['draft'].content:
                try:
                    filled_sections[field_name] = model_class.model_validate(result['draft'].content)
                except Exception as e:
                    logger.warning(f"[Compiler] draft validate failed for {field_name}: {e}")
                    filled_sections[field_name] = model_class()  # Безопасный дефолт
            else:
                logger.error(f"[Compiler] Section '{field_name}' completely failed. Creating empty.")
                filled_sections[field_name] = model_class()

            section_logs[field_name] = result.get('logs',[])

        try:
            template = template_class.model_validate(filled_sections)
        except Exception as e:
            logger.warning(f"[Compiler] template model_validate failed: {e}, using model_construct")
            template = template_class.model_construct(**filled_sections)

        return TZResult(
            template_type=template_type, template_data=template,
            validation=getattr(template, 'validate_completeness', lambda: {})(),
            markdown=getattr(template, 'to_markdown', lambda: "")(),
            logs=section_logs
        )

    async def edit_section(
        self, graph: UnifiedGraph, template_type: TemplateType, section_name: str,
        current_value: BaseModel, feedback: str, preserve_aspects: Optional[List[str]] = None
    ) -> BaseModel:
        context = GraphContext(graph=graph, template_type=template_type)
        template_class = get_template_class(template_type)
        field_info = template_class.model_fields.get(section_name)
        if not field_info:
            raise ValueError(f"Unknown section: {section_name}")

        model_class = get_model_class(field_info)
        if not model_class:
            raise ValueError(f"Cannot determine model class for section: {section_name}")

        run_id = f"edit_{section_name}_{make_thread_id('edit', graph)}"
        register_run(run_id, model_class, context)

        initial_state: AgentState = {
            "run_id": run_id, "section_name": section_name,
            "instruction": field_info.description or "",
            "draft": SectionDraft(content=current_value.model_dump(), sources=[], confidence=1.0),
            "review": None, "iterations": 0, "max_iterations": 2, "edit_mode": True,
            "edit_request": EditRequest(
                section_name=section_name, feedback=feedback,
                preserve_aspects=preserve_aspects or[]
            ),
            "final_result": None, "status": "editing", "logs": ["Starting edit mode"]
        }

        result = await self.workflow.ainvoke(
            initial_state, config={"configurable": {"thread_id": run_id}}
        )

        if result.get('final_result'):
            try:
                return model_class.model_validate(result['final_result'])
            except Exception as e:
                logger.warning(f"[Compiler] edit final_result validate failed: {e}")

        if result.get('draft') and result['draft'].content:
            try:
                return model_class.model_validate(result['draft'].content)
            except Exception as e:
                logger.warning(f"[Compiler] edit draft validate failed: {e}")

        raise RuntimeError(f"Edit failed for section '{section_name}': no valid result produced.")