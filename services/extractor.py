import networkx as nx
import os
import tempfile
from typing import List
from models.inputs import DataSource, DataEnum, ExtractedKnowledge, GraphNode, GraphEdge, KeyValue

class DataExtractorService:
    def extract(self, source: DataSource) -> ExtractedKnowledge:
        print(f"🔍 [Extractor] Парсинг GraphML: {source.file_name}")

        if source.source_type != DataEnum.GRAPHML:
            print("❌ Ошибка: поддерживается только GRAPHML")
            return ExtractedKnowledge()

        try:
            # Создаем временный файл для networkx
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.graphml', delete=False, encoding='utf-8') as tmp:
                tmp.write(str(source.content))
                tmp_path = tmp.name

            G = nx.read_graphml(tmp_path)
            os.remove(tmp_path)

            nodes = []
            edges = []

            # 1. Парсим Узлы
            for node_id, attrs in G.nodes(data=True):
                # Ищем атрибуты (разные сервисы могут называть их по-разному)
                label = attrs.get('label', attrs.get('d2', 'Unknown'))
                name = attrs.get('name', attrs.get('d1', node_id))
                desc = attrs.get('description', attrs.get('d4', ''))

                # Собираем все остальное в properties
                props = []
                for k, v in attrs.items():
                    if k not in ['label', 'name', 'description', 'd1', 'd2', 'd4']:
                        props.append(KeyValue(key=k, value=str(v)[:100]))

                # Формируем читабельный контент
                content = name
                if desc and desc != name:
                    content += f": {desc}"

                nodes.append(GraphNode(
                    id=node_id,
                    label=label,
                    content=content, # <-- Важно: теперь мы заполняем content сразу
                    properties=props
                ))

            # 2. Парсим Ребра
            for u, v, attrs in G.edges(data=True):
                relation = attrs.get('relation', attrs.get('d10', 'RELATED_TO'))
                edges.append(GraphEdge(
                    source=u,
                    target=v,
                    relation=relation
                ))

            return ExtractedKnowledge(
                summary=f"Граф из файла {source.file_name}",
                nodes=nodes,
                edges=edges,
                source_window_ref=source.file_name
            )

        except Exception as e:
            print(f"❌ Ошибка парсинга: {e}")
            return ExtractedKnowledge(summary=f"Error: {e}")