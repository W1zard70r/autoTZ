import logging
from typing import List, Dict, Set
from pydantic import BaseModel, Field

from schemas.document import DataSource
from schemas.enums import NodeLabel
from schemas.graph import (
    ExtractedKnowledge, RawEntity,
    BatchLinkResult, GraphNode, GraphEdge,
)
from utils.preprocessing import format_chat_message, enrich_message_with_vote
from utils.llm_client import acall_llm_json
from utils.state_logger import log_pydantic, log_dict
from core.windowing import split_chat_into_windows, split_text_into_windows

logger = logging.getLogger(__name__)

SCHEMA_DESCRIPTION = """ALLOWED SCHEMAS:

NodeLabel (type of entity):
  Person     - person, team member, stakeholder
  Component  - technology, library, service, database, API endpoint
  Task       - task, user story, work item
  Requirement - functional or non-functional requirement
  Concept    - abstract idea, architecture pattern, approach
  Decision   - open question, vote, choice between alternatives

EdgeRelation (type of relationship):
  ASSIGNED_TO   - Person -> Task/Component (who is responsible)
  DEPENDS_ON    - Component/Task -> Component/Task (dependency)
  RELATES_TO    - any -> any (general relationship, also Decision -> options)
  AGREES_WITH   - Person -> Requirement/Concept (person agrees)
  MENTIONS      - Person -> any (person mentioned something)
  VOTED_FOR     - Person -> Component/Concept (person voted for this option)
  VOTED_AGAINST - Person -> Component/Concept (person voted against)
  RESOLVED_TO   - Decision -> Component/Concept (decision was resolved to this)

TZSectionEnum (target section in the technical specification):
  general_info    - general info, project goals
  tech_stack      - technologies, DBs, libraries, frameworks
  functional_req  - functional requirements, features, business logic
  ui_ux           - UI/UX, screens, forms
  uncategorized   - cannot determine"""

SYSTEM_PROMPT = f"""You are a knowledge graph extraction specialist for software project documentation.
Your job is to extract structured entities and relationships from team conversations.

{SCHEMA_DESCRIPTION}

RULES:
- Use snake_case for all IDs (e.g. postgresql_db, login_screen, jwt_auth)
- Every node MUST have a meaningful description
- Every edge MUST have evidence (quote or reasoning from the text)
- For votes/decisions: create a Decision node and connect Person nodes with VOTED_FOR/VOTED_AGAINST
- Assign target_section to each node based on its nature
- Do NOT invent entities not present in the text"""


class GlossaryItem(BaseModel):
    id: str = Field(description="Snake_case ID")
    name: str = Field(description="Human-readable name")
    label: NodeLabel = Field(description="Entity type")
    description: str = Field(description="Brief description")


def validate_graph_integrity(graph: ExtractedKnowledge, valid_ids: Set[str]) -> ExtractedKnowledge:
    original_node_count = len(graph.nodes)
    original_edge_count = len(graph.edges)

    graph.nodes = [n for n in graph.nodes if n.id in valid_ids]
    present_ids = {n.id for n in graph.nodes}

    if original_node_count > len(graph.nodes):
        logger.warning(
            f"Validation: removed {original_node_count - len(graph.nodes)} "
            f"phantom nodes (not in glossary)"
        )

    graph.edges = [
        e for e in graph.edges
        if e.source in present_ids
        and e.target in present_ids
        and e.source != e.target
    ]

    if original_edge_count > len(graph.edges):
        logger.warning(
            f"Validation: removed {original_edge_count - len(graph.edges)} invalid edges"
        )

    return graph


class MinerProcessor:
    def __init__(self):
        self.global_glossary: Dict[str, GlossaryItem] = {}
        self.key_entity_ids: List[str] = []

    def _format_glossary(self) -> str:
        if not self.global_glossary:
            return "(empty)"
        return "\n".join(
            [f"- {e.id} ({e.label.value}): {e.name} -- {e.description}"
             for e in self.global_glossary.values()]
        )

    async def process_source(self, source: DataSource) -> List[ExtractedKnowledge]:
        logger.info(f"Layer 1: extracting from {source.file_name}")
        extracted_graphs = []

        if source.source_type == "chat":
            windows = split_chat_into_windows(source.content)
            msg_lookup = {m["id"]: m for m in source.content if m.get("type") == "message"}

            logger.info(f"  -> Found {len(windows)} semantic windows")

            for i, (ref, msgs) in enumerate(windows):
                enriched_msgs = [enrich_message_with_vote(m.copy()) for m in msgs]
                text_chunk = "\n".join([format_chat_message(m, msg_lookup) for m in enriched_msgs])

                logger.info(f"  -> [{i+1}/{len(windows)}] Window {ref} ({len(msgs)} messages)")
                graph = await self._extract_subgraph(text_chunk, ref)
                extracted_graphs.append(graph)
                safe_ref = ref.replace(":", "_").replace("/", "_")
                log_pydantic(f"layer1_subgraph_{source.file_name}_{safe_ref}.json", graph)
        else:
            text_content = str(source.content)
            text_windows = split_text_into_windows(text_content, max_chars=50000)
            logger.info(f"  -> Found {len(text_windows)} text windows")

            for i, (ref, chunk) in enumerate(text_windows):
                logger.info(f"  -> [{i+1}/{len(text_windows)}] Window {ref} ({len(chunk)} chars)")
                try:
                    graph = await self._extract_subgraph(chunk, ref)
                    extracted_graphs.append(graph)
                    safe_ref = ref.replace(":", "_").replace("/", "_")
                    log_pydantic(f"layer1_subgraph_{source.file_name}_{safe_ref}.json", graph)
                except Exception as e:
                    logger.error(f"  Error processing window {ref}: {e}")

        glossary_dump = {k: v.model_dump() for k, v in self.global_glossary.items()}
        log_dict("layer1_global_glossary.json", glossary_dump)

        return extracted_graphs

    async def _batch_link_entities(self, entities: List[RawEntity]) -> List[str]:
        if not entities:
            return []

        if not self.global_glossary:
            ids = []
            for e in entities:
                new_id = e.name.lower().replace(" ", "_").replace("-", "_")
                self.global_glossary[new_id] = GlossaryItem(
                    id=new_id, name=e.name, label=e.label, description=e.description
                )
                ids.append(new_id)
            return ids

        data_lines = []
        for i, entity in enumerate(entities):
            data_lines.append(
                f"{i}. \"{entity.name}\" ({entity.label.value}) -- {entity.description}"
            )
        data_str = "\n".join(data_lines)

        prompt = f"""For each entity below, decide: is it already present in the glossary (duplicate), or is it a new entity?
If duplicate: set is_duplicate=true, target_global_id=<glossary id>.
If new: set is_duplicate=false, new_id=<snake_case_id>.
Return one decision per entity, in the same order.

Current glossary:
{self._format_glossary()}"""

        try:
            result: BatchLinkResult = await acall_llm_json(
                BatchLinkResult, prompt, data=data_str, system=SYSTEM_PROMPT
            )
            ids = []
            for i, entity in enumerate(entities):
                if i < len(result.decisions):
                    decision = result.decisions[i]
                    if decision.is_duplicate and decision.target_global_id and decision.target_global_id in self.global_glossary:
                        ids.append(decision.target_global_id)
                    else:
                        new_id = (decision.new_id or entity.name).lower().replace(" ", "_").replace("-", "_")
                        self.global_glossary[new_id] = GlossaryItem(
                            id=new_id, name=entity.name, label=entity.label, description=entity.description
                        )
                        ids.append(new_id)
                else:
                    new_id = entity.name.lower().replace(" ", "_").replace("-", "_")
                    self.global_glossary[new_id] = GlossaryItem(
                        id=new_id, name=entity.name, label=entity.label, description=entity.description
                    )
                    ids.append(new_id)
            return ids
        except Exception as e:
            logger.error(f"Batch linking error: {e}")
            ids = []
            for entity in entities:
                new_id = entity.name.lower().replace(" ", "_").replace("-", "_")
                self.global_glossary[new_id] = GlossaryItem(
                    id=new_id, name=entity.name, label=entity.label, description=entity.description
                )
                ids.append(new_id)
            return ids

    async def _extract_subgraph(self, text: str, source_ref: str) -> ExtractedKnowledge:
        glossary_context = self._format_glossary()
        key_entities_context = ", ".join(self.key_entity_ids[-20:]) if self.key_entity_ids else "(none yet)"

        extract_prompt = f"""Analyze the following team conversation and extract a knowledge graph.

EXISTING GLOSSARY (reuse these IDs where applicable):
{glossary_context}

RECENTLY MENTIONED ENTITIES: {key_entities_context}

INSTRUCTIONS:
1. Extract ALL entities: people, components, tasks, requirements, concepts, decisions
2. Extract ALL relationships between entities
3. For votes/opinions: use VOTED_FOR / VOTED_AGAINST edges from Person to the option
4. For decisions/choices: create a Decision node and link options with RELATES_TO
5. Use IDs from glossary for entities already known. Create new snake_case IDs for new entities.
6. Set target_section for each node.

EXAMPLE INPUT:
"[2024-01-01] Alex: Let's use PostgreSQL for the main DB
[2024-01-01] Ivan: Agreed. And add Redis for caching."

EXAMPLE OUTPUT:
nodes:
  - id: postgresql_db, label: Component, name: "PostgreSQL", description: "Main database", target_section: tech_stack
  - id: redis_cache, label: Component, name: "Redis", description: "Caching layer", target_section: tech_stack
  - id: alex, label: Person, name: "Alex", description: "Team member"
  - id: ivan, label: Person, name: "Ivan", description: "Team member"
edges:
  - source: alex, target: postgresql_db, relation: MENTIONS, evidence: "Alex: Let's use PostgreSQL"
  - source: ivan, target: postgresql_db, relation: AGREES_WITH, evidence: "Ivan: Agreed"
  - source: ivan, target: redis_cache, relation: MENTIONS, evidence: "Ivan: And add Redis for caching"
  - source: redis_cache, target: postgresql_db, relation: RELATES_TO, evidence: "Both part of data layer"

NOW EXTRACT FROM THIS TEXT:"""

        result: ExtractedKnowledge = await acall_llm_json(
            ExtractedKnowledge, extract_prompt, data=text, system=SYSTEM_PROMPT,
            max_tokens=32768,
        )
        result.source_ref = source_ref

        raw_entities = [
            RawEntity(name=n.name, label=n.label, description=n.description)
            for n in result.nodes
        ]
        linked_ids = await self._batch_link_entities(raw_entities)

        id_remap = {}
        for i, node in enumerate(result.nodes):
            if i < len(linked_ids):
                old_id = node.id
                new_id = linked_ids[i]
                id_remap[old_id] = new_id
                node.id = new_id

        for edge in result.edges:
            edge.source = id_remap.get(edge.source, edge.source)
            edge.target = id_remap.get(edge.target, edge.target)

        valid_ids = set(self.global_glossary.keys())
        result = validate_graph_integrity(result, valid_ids)

        self.key_entity_ids.extend([n.id for n in result.nodes])
        self.key_entity_ids = self.key_entity_ids[-50:]

        logger.info(f" Graph: {len(result.nodes)} nodes, {len(result.edges)} edges")
        return result
