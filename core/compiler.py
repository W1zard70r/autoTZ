"""Layer 3: Graph-Navigator Multi-Agent Compiler."""
import logging
import typing
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from schemas.graph import UnifiedGraph
from schemas.enums import TemplateType
from schemas.templates.base import BaseTemplate, TZResult
from schemas.templates import get_template_class
from utils.llm_client import acall_llm_json
from utils.tools import get_graph_tools # Импорт из нового файла

logger = logging.getLogger(__name__)

WRITER_SYSTEM = "Ты техписатель. Пиши строго на основе фактов из графа. Не выдумывай функционал."
REVIEWER_SYSTEM = "Ты аудитор. Отклоняй черновик, если там есть выдумки (функционал, сроки, роли), которых нет в графе."

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
    if state["feedback"]: prompt += f"\n\nПравки: {state['feedback']}"
    draft = await acall_llm_json(state["schema_class"], prompt, system=WRITER_SYSTEM)
    return {"draft": draft, "iterations": state["iterations"] + 1, "feedback": ""}

async def reviewer_node(state: AgentState) -> AgentState:
    draft_json = state["draft"].model_dump_json() if state["draft"] else "{}"
    prompt = f"Факты: {state['nodes_list']}\nЧерновик: {draft_json}\nВерни is_approved=True, если нет выдумок."
    class R(BaseModel): is_approved: bool; feedback: str
    res = await acall_llm_json(R, prompt, system=REVIEWER_SYSTEM)
    return {"feedback": res.feedback if not res.is_approved else "APPROVED"}

class TZCompiler:
    def __init__(self, language: str = "ru"): self.language = language

    async def compile(self, graph: UnifiedGraph, template_type: TemplateType, user_answers=None) -> TZResult:
        workflow = StateGraph(AgentState)
        workflow.add_node("writer", writer_node); workflow.add_node("reviewer", reviewer_node)
        workflow.set_entry_point("writer"); workflow.add_edge("writer", "reviewer")
        workflow.add_conditional_edges("reviewer", lambda s: END if s["feedback"]=="APPROVED" or s["iterations"]>=3 else "writer")
        app = workflow.compile()

        template_class = get_template_class(template_type)
        filled = {}; nodes_list = ", ".join([f"{n.name}({n.id})" for n in graph.nodes])
        
        for field_name, field_info in template_class.model_fields.items():
            model_class = field_info.annotation if isinstance(field_info.annotation, type) else None
            if not model_class or not issubclass(model_class, BaseModel): continue
            final_state = await app.ainvoke({
                "section_name": field_name, "nodes_list": nodes_list,
                "instruction": field_info.description or "Заполни.",
                "schema_class": model_class, "draft": None, "feedback": "", "iterations": 0
            })
            filled[field_name] = final_state["draft"]

        template = template_class(**filled)
        return TZResult(template_type=template_type, template_data=template, validation=template.validate_completeness(), markdown=template.to_markdown())