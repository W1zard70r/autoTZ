from langchain_core.tools import tool
from schemas.graph import UnifiedGraph

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