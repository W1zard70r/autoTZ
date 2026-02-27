from typing import Dict, Optional
from models import ProjectGlossary, GlossaryItem

class GlobalGlossary:
    """Глобальный глоссарий проекта — защищает от дубликатов сущностей"""
    def __init__(self):
        self.entities: Dict[str, GlossaryItem] = {}
        self.name_to_id: Dict[str, str] = {}  # для fuzzy matching

    def _find_similar(self, name: str) -> Optional[str]:
        """Простой fuzzy match по имени"""
        name_lower = name.lower().strip()
        for stored_name, eid in self.name_to_id.items():
            if (name_lower in stored_name or stored_name in name_lower or
                len(set(name_lower) & set(stored_name)) / len(name_lower) > 0.7):
                return eid
        return None

    def merge(self, new_glossary: ProjectGlossary) -> ProjectGlossary:
        """Мёрджит новые сущности в глобальный глоссарий"""
        for entity in new_glossary.entities:
            existing_id = self._find_similar(entity.name) or self.entities.get(entity.id)
            if existing_id:
                entity.id = existing_id  # используем уже существующий ID
            else:
                # Добавляем новую
                self.entities[entity.id] = entity
                self.name_to_id[entity.name.lower()] = entity.id
        return new_glossary