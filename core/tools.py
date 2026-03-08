import networkx as nx

def get_graph_navigator(G: nx.MultiDiGraph):
    
    def search_neighbors(node_id: str, depth: int = 1) -> str:
        """Позволяет агенту 'ходить' по графу: найти всё связанное с узлом."""
        try:
            # Получаем подграф из соседей
            subgraph = nx.ego_graph(G, node_id, radius=depth)
            results = []
            for n, d in subgraph.nodes(data=True):
                results.append(f"Node: {d.get('name')} ({d.get('label')})")
            for u, v, d in subgraph.edges(data=True):
                results.append(f"{u} --{d.get('relation')}--> {v}")
            return "\n".join(results)
        except:
            return "Узел не найден."

    return search_neighbors