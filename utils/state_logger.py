import os
import json
import networkx as nx
from enum import Enum
from pydantic import BaseModel

LOGS_DIR = "logs"


def init_logs_dir():
    """Создает директорию для логов, если её нет."""
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)


def log_pydantic(filename, model):
    """Сохраняет Pydantic модель в JSON."""
    filepath = os.path.join(LOGS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(model.model_dump_json(indent=2))


def log_dict(filename, data):
    """Сохраняет обычный словарь в JSON."""
    filepath = os.path.join(LOGS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def log_text(filename, text):
    """Сохраняет простой текст (например, промпты)."""
    filepath = os.path.join(LOGS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def sanitize_for_graphml(val):
    """Приводит любые сложные объекты к базовым типам (str, int, float, bool) для GraphML."""
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, BaseModel):
        return val.model_dump_json()
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def log_graphml(filename, G):
    """Безопасно сохраняет NetworkX граф в формат GraphML с поддержкой MultiGraphs."""
    export_g = G.copy()

    # 1. Очищаем атрибуты узлов
    for n, data in export_g.nodes(data=True):
        for k, val in list(data.items()):
            export_g.nodes[n][k] = sanitize_for_graphml(val)

    # 2. Очищаем атрибуты рёбер (проверяем тип графа)
    if export_g.is_multigraph():
        # Для MultiDiGraph нам нужен уникальный 'key' каждого ребра
        for u, v, key, data in export_g.edges(data=True, keys=True):
            for k, val in list(data.items()):
                export_g[u][v][key][k] = sanitize_for_graphml(val)
    else:
        # Для обычного DiGraph
        for u, v, data in export_g.edges(data=True):
            for k, val in list(data.items()):
                export_g[u][v][k] = sanitize_for_graphml(val)

    # 3. Сохраняем файл
    filepath = os.path.join(LOGS_DIR, filename)
    try:
        nx.write_graphml(export_g, filepath)
    except Exception as e:
        print(f"Ошибка при сохранении GraphML ({filename}): {e}")