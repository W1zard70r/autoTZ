import networkx as nx
from pyvis.network import Network


def visualize_interactive_html(graphml_path: str, output_html: str):
    # 1. Читаем готовый граф
    G = nx.read_graphml(graphml_path)

    # 2. Настраиваем интерактивную сеть
    # select_menu добавляет выпадающий список для поиска узлов
    net = Network(notebook=False, directed=True, height="800px", width="100%", select_menu=True)

    # Добавляем цвета в зависимости от типа сущности (label)
    color_map = {
        "Person": "#ff9999",
        "Component": "#99ccff",
        "Task": "#99ff99",
        "Requirement": "#ffcc99",
        "Concept": "#e5ccff"
    }

    # Настраиваем визуальное отображение узлов
    for node_id, node_data in G.nodes(data=True):
        label_type = node_data.get("label", "Unknown")
        node_data["color"] = color_map.get(label_type, "#cccccc")
        node_data["title"] = f"Type: {label_type}\nID: {node_id}"  # Текст при наведении (tooltip)

        # Заменяем технический ID на красивое имя, если оно есть в свойствах
        if "name" in node_data:
            node_data["label"] = node_data["name"]

    # 3. Загружаем данные из NetworkX
    net.from_nx(G)

    # 4. Включаем физику, чтобы граф красиво "расправился"
    net.repulsion(node_distance=150, central_gravity=0.2, spring_length=200)

    # 5. Сохраняем в HTML
    net.show(output_html, notebook=False)
    print(f"✅ Интерактивный граф сохранен в: {output_html}")

# Пример использования:
visualize_interactive_html("output/telegram_backend_team.graphml", "output/graph_view.html")