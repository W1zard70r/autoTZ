import networkx as nx
import json
import logging
from datetime import datetime
from models import WindowExtractionResult

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    def __init__(self, project_name: str):
        self.graph = nx.DiGraph(name=project_name)
        logger.info(f"Создан граф проекта: {project_name}")

    def smart_merge_node(self, node_id: str, new_properties: dict, source_ref: str):
        """Умное слияние свойств + история"""
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **new_properties)
            return

        current = self.graph.nodes[node_id]
        if "history" not in current:
            current["history"] = {}

        for k, new_val in new_properties.items():
            old_val = current.get(k)
            if old_val is None or (isinstance(new_val, str) and len(str(new_val)) > len(str(old_val)) + 3):
                current[k] = new_val

            # История
            current["history"][k] = {
                "value": new_val,
                "timestamp": datetime.now().isoformat(),
                "source": source_ref
            }

    def apply_extraction(self, result: WindowExtractionResult, source_ref: str):
        logger.info(f"Применяем извлечение из {source_ref} ({len(result.nodes)} узлов, {len(result.edges)} связей)")

        # Узлы
        for node in result.nodes:
            self.smart_merge_node(node.id, node.properties, source_ref)
            self.graph.nodes[node.id]["label"] = node.label.value

        # Рёбра
        for edge in result.edges:
            if not self.graph.has_node(edge.source):
                self.graph.add_node(edge.source, label="Unknown")
            if not self.graph.has_node(edge.target):
                self.graph.add_node(edge.target, label="Unknown")

            self.graph.add_edge(
                edge.source, edge.target,
                relation=edge.relation.value,
                evidence=edge.evidence,
                source_ref=source_ref,
                added_at=datetime.now().isoformat()
            )

    def export_to_graphml(self, filepath: str):
        export_g = self.graph.copy()
        for _, data in export_g.nodes(data=True):
            for k, v in list(data.items()):
                if isinstance(v, (dict, list)):
                    data[k] = json.dumps(v, ensure_ascii=False)
        for _, _, data in export_g.edges(data=True):
            for k, v in list(data.items()):
                if isinstance(v, (dict, list)):
                    data[k] = json.dumps(v, ensure_ascii=False)
        nx.write_graphml(export_g, filepath)
        logger.info(f"Граф экспортирован в {filepath}")