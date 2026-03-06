import logging
import asyncio
from typing import List
from schemas.graph import UnifiedGraph, GraphNode, GraphEdge, DecisionResolution
from schemas.enums import TZSectionEnum
from schemas.document import FullTZDocument, GeneratedSection
from utils.llm_client import acall_llm_text
from utils.state_logger import log_text

logger = logging.getLogger(__name__)

COMPILER_SYSTEM = """You are a technical writer generating a formal Technical Specification (TZ) document.
Write in a formal business style using Markdown formatting (headers, lists, tables where appropriate).
Use ONLY the provided facts from the knowledge graph. Do NOT invent information."""


class TZGenerator:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name

    async def generate_tz(self, graph: UnifiedGraph) -> FullTZDocument:
        logger.info("Layer 3: Generating TZ document...")

        sections_to_write = [
            TZSectionEnum.GENERAL,
            TZSectionEnum.FUNCTIONAL,
            TZSectionEnum.STACK,
            TZSectionEnum.INTERFACE,
        ]

        # Check for UNKNOWN nodes and add them to a misc section
        unknown_nodes = [n for n in graph.nodes if n.target_section == TZSectionEnum.UNKNOWN]

        tasks = []
        for sec_enum in sections_to_write:
            tasks.append(self._generate_section(sec_enum, graph))

        generated_sections = await asyncio.gather(*tasks)
        valid_sections = [sec for sec in generated_sections if sec is not None]

        # Handle UNKNOWN nodes: add to a "Miscellaneous" section
        if unknown_nodes:
            misc_section = await self._generate_misc_section(unknown_nodes, graph)
            if misc_section:
                valid_sections.append(misc_section)

        # Add decisions section if there are any
        if graph.decisions:
            decision_section = self._generate_decisions_section(graph.decisions)
            if decision_section:
                valid_sections.append(decision_section)

        return FullTZDocument(
            project_name="Technical Specification (AI Generated)",
            version="1.0.0",
            sections=valid_sections
        )

    def _build_node_context(self, nodes: List[GraphNode], graph: UnifiedGraph) -> str:
        lines = []
        node_ids = {n.id for n in nodes}

        for n in nodes:
            # Node info
            props_str = ""
            if n.properties:
                props_str = " | Properties: " + ", ".join([f"{p.key}={p.value}" for p in n.properties])
            lines.append(f"- [{n.label.value}] {n.name} (ID: {n.id}): {n.description}{props_str}")

            # Related edges
            related_edges = [
                e for e in graph.edges
                if e.source == n.id or e.target == n.id
            ]
            for e in related_edges:
                other_id = e.target if e.source == n.id else e.source
                other_node = next((nd for nd in graph.nodes if nd.id == other_id), None)
                other_name = other_node.name if other_node else other_id
                direction = "->" if e.source == n.id else "<-"
                evidence_str = f' ("{e.evidence}")' if e.evidence else ""
                lines.append(f"    {direction} {e.relation.value} {other_name}{evidence_str}")

        return "\n".join(lines)

    async def _generate_section(self, sec_enum: TZSectionEnum, graph: UnifiedGraph) -> GeneratedSection:
        relevant_nodes = [n for n in graph.nodes if n.target_section == sec_enum]

        if not relevant_nodes:
            return None

        logger.info(f"  -> Writing section: {sec_enum.value} ({len(relevant_nodes)} nodes)")

        node_context = self._build_node_context(relevant_nodes, graph)

        section_descriptions = {
            TZSectionEnum.GENERAL: "General Information - project goals, objectives, stakeholders, and high-level overview",
            TZSectionEnum.FUNCTIONAL: "Functional Requirements - features, business logic, user stories, and acceptance criteria",
            TZSectionEnum.STACK: "Technology Stack - components, databases, libraries, frameworks, and architecture decisions",
            TZSectionEnum.INTERFACE: "UI/UX Requirements - screens, forms, user interactions, and design specifications",
        }

        section_desc = section_descriptions.get(sec_enum, sec_enum.value)

        prompt = f"""Write the "{section_desc}" section of the Technical Specification.

Use ONLY the following facts from the knowledge graph (nodes and their relationships):

{node_context}

Format: Markdown with appropriate headers (##, ###), bullet lists, and tables if needed.
Be comprehensive but stick to the facts. Group related items logically."""

        log_text(f"layer3_prompt_{sec_enum.value}.txt", prompt)

        try:
            content_markdown = await acall_llm_text(
                prompt=prompt, model_name=self.model_name, system=COMPILER_SYSTEM
            )
            return GeneratedSection(
                section_id=sec_enum,
                title=sec_enum.name,
                content_markdown=content_markdown
            )
        except Exception as e:
            logger.error(f"Error generating section {sec_enum.value}: {e}")
            return None

    async def _generate_misc_section(self, nodes: List[GraphNode], graph: UnifiedGraph) -> GeneratedSection:
        if not nodes:
            return None

        logger.info(f"  -> Writing miscellaneous section ({len(nodes)} uncategorized nodes)")
        node_context = self._build_node_context(nodes, graph)

        prompt = f"""Write an "Additional Information" section covering uncategorized but important items.

Facts:
{node_context}

Format: Markdown. Group related items logically."""

        try:
            content_markdown = await acall_llm_text(
                prompt=prompt, model_name=self.model_name, system=COMPILER_SYSTEM
            )
            return GeneratedSection(
                section_id=TZSectionEnum.UNKNOWN,
                title="ADDITIONAL",
                content_markdown=content_markdown
            )
        except Exception as e:
            logger.error(f"Error generating misc section: {e}")
            return None

    def _generate_decisions_section(self, decisions: List[DecisionResolution]) -> GeneratedSection:
        lines = ["The following decisions were identified and resolved during project discussions:\n"]

        for d in decisions:
            lines.append(f"### {d.decision_name}")
            if d.is_tie:
                lines.append(f"**Status:** Unresolved (tie)")
                if d.conflict_description:
                    lines.append(f"**Note:** {d.conflict_description}")
            elif d.winner_name:
                lines.append(f"**Decision:** {d.winner_name}")
            else:
                lines.append(f"**Status:** Pending")

            if d.options:
                lines.append("\n| Option | Votes For | Votes Against | Score |")
                lines.append("|--------|-----------|---------------|-------|")
                for opt in sorted(d.options, key=lambda x: x.score, reverse=True):
                    winner_mark = " (winner)" if opt.option_id == d.winner_id else ""
                    lines.append(
                        f"| {opt.option_name}{winner_mark} | {opt.votes_for} | {opt.votes_against} | {opt.score:+d} |"
                    )
            lines.append("")

        content = "\n".join(lines)
        return GeneratedSection(
            section_id=TZSectionEnum.GENERAL,
            title="DECISIONS",
            content_markdown=content
        )
