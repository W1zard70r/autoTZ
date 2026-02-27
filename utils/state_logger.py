import os
import json
import networkx as nx
from enum import Enum
from pydantic import BaseModel

LOG_DIR = "logs"

def init_logs_dir():
    """Создает директорию для логов, если её нет"""
    os.makedirs(LOG_DIR, exist_ok=True)

def log_pydantic(filename: str, model: BaseModel):
    """Сохраняет Pydantic модель в JSON"""
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(model.model_dump_json(indent=2))

def log_dict(filename: str, data: dict):
    """Сохраняет обычный словарь в JSON"""
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def log_text(filename: str, text: str):
    """Сохраняет простой текст (например, промпты)"""
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

def sanitize_for_graphml(val):
    """Приводит любые сложные объекты к базовым типам (str, int, float, bool) для NetworkX"""
    if val is None:
        return ""
    if isinstance(val, Enum):
        return str(val.value)
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    if not isinstance(val, (int, float, str, bool)):
        return str(val)
    return val

def log_graphml(filename: str, G: nx.DiGraph):
    """Безопасно сохраняет NetworkX граф в формат GraphML."""
    filepath = os.path.join(LOG_DIR, filename)
    export_g = G.copy()
    
    # Очистка атрибутов узлов
    for nid, data in export_g.nodes(data=True):
        for k, v in list(data.items()):
            export_g.nodes[nid][k] = sanitize_for_graphml(v)
                
    # Очистка атрибутов ребер
    for u, v, data in export_g.edges(data=True):
        for k, val in list(data.items()):
            export_g.edges[u, v][k] = sanitize_for_graphml(val)
                
    nx.write_graphml(export_g, filepath)