"""Layer 3: Graph-Navigator Multi-Agent Compiler."""
import logging
import typing
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool

from schemas.graph import UnifiedGraph
from schemas.enums import TemplateType
from schemas.templates.base import BaseTemplate, TZResult
from schemas.templates import get_template_class
from utils.llm_client import acall_llm_json

logger = logging.getLogger(__name__)

# ── 1. Системные промпты (Контроль качества) ───────────────

WRITER_SYSTEM = (
    "Ты — технический писатель. Пиши строго по предоставленным данным из графа. "
    "Запрещено: придумывать сроки, модули, роли или функционал, которого нет в списке узлов. "
    "Если информации недостаточно — ставь null или 'Не указано'. Пиши только на русском."
)

REVIEWER_SYSTEM = (
    "Ты — аудитор ТЗ. Проверь текст на наличие выдумок. "
    "Если Писатель добавил функционал (например, 'Отзывы', 'Оплаты', 'Поставщики'), "
    "которого нет в списке узлов графа — ОТКЛОНЯЙ черновик. "
    "Будь безжалостным к галлюцинациям."
)

# ── 2. Инструменты для Агента ──────────────────────────────

def get_graph_tools(graph: UnifiedGraph):
    nodes_map = {n.id: n for n in graph.nodes}
    edges = graph.edges

    @tool
    def search_neighbors(node_id: str) -> str:
        """Позволяет агенту исследовать граф связей вокруг узла."""
        if node_id not in nodes_map: return "Узел не найден."
        node = nodes_map[node_id]
        results = [f"Узел: {node.name} - {node.description}"]
        for e in edges:
            if e.source == node_id:
                target = nodes_map.get(e.target)
                results.append(f"-> {e.relation.value} -> {target.name if target else e.target}")
            elif e.target == node_id:
                source = nodes_map.get(e.source)
                results.append(f"<- {e.relation.value} <- {source.name if source else e.source}")
        return "\n".join(results)
    return [search_neighbors]

# ── 3. Агентская логика ────────────────────────────────────

class AgentState(TypedDict):
    section_name: str
    nodes_list: str
    instruction: str
    schema_class: type[BaseModel]
    draft: Optional[BaseModel]
    feedback: str
    iterations: int

async def writer_node(state: AgentState) -> AgentState:
    prompt = f"Раздел: {state['section_name']}\nИнструкция: {state['instruction']}\nГраф: {state['nodes_list']}"
    if state["feedback"]: prompt += f"\n\nПравки Рецензента: {state['feedback']}"
    
    draft = await acall_llm_json(state["schema_class"], prompt, system=WRITER_SYSTEM)
    return {"draft": draft, "iterations": state["iterations"] + 1, "feedback": ""}

async def reviewer_node(state: AgentState) -> AgentState:
    draft_json = state["draft"].model_dump_json() if state["draft"] else "{}"
    prompt = f"Факты: {state['nodes_list']}\nЧерновик: {draft_json}\nВерни is_approved=True, если нет выдумок."
    
    class ReviewerOutput(BaseModel): is_approved: bool; feedback: str
    res = await acall_llm_json(ReviewerOutput, prompt, system=REVIEWER_SYSTEM)
    return {"feedback": res.feedback if not res.is_approved else "APPROVED"}

def reviewer_router(state: AgentState):
    return END if state["feedback"] == "APPROVED" or state["iterations"] >= 3 else "writer"

# ── 4. Компилятор ──────────────────────────────────────────

class TZCompiler:
    def __init__(self, language: str = "ru"): self.language = language

    def _extract_class(self, ann):
        if isinstance(ann, type) and issubclass(ann, BaseModel): return ann
        origin = getattr(ann, "__origin__", None)
        if origin is typing.Union:
            for arg in typing.get_args(ann):
                if isinstance(arg, type) and issubclass(arg, BaseModel): return arg
        return None

    def _apply_user_answers(self, template: BaseTemplate, answers: Dict[str, str]):
        for path, value in answers.items():
            parts = path.split("."); obj = template
            try:
                for p in parts[:-1]: obj = getattr(obj, p)
                setattr(obj, parts[-1], value)
            except: continue

    async def compile(self, graph: UnifiedGraph, template_type: TemplateType, user_answers=None) -> TZResult:
        workflow = StateGraph(AgentState)
        workflow.add_node("writer", writer_node); workflow.add_node("reviewer", reviewer_node)
        workflow.set_entry_point("writer"); workflow.add_edge("writer", "reviewer")
        workflow.add_conditional_edges("reviewer", reviewer_router)
        app = workflow.compile()

        template_class = get_template_class(template_type)
        filled = {}; nodes_list = ", ".join([f"{n.name}({n.id})" for n in graph.nodes])
        
        for field_name, field_info in template_class.model_fields.items():
            model_class = self._extract_class(field_info.annotation)
            if not model_class: continue

            final_state = await app.ainvoke({
                "section_name": field_name, "nodes_list": nodes_list,
                "instruction": field_info.description or "Заполни.",
                "schema_class": model_class, "draft": None, "feedback": "", "iterations": 0
            })
            filled[field_name] = final_state["draft"]

        template = template_class(**filled)
        if user_answers: self._apply_user_answers(template, user_answers)
        return TZResult(template_type=template_type, template_data=template, validation=template.validate_completeness(), markdown=template.to_markdown())