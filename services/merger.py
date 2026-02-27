from typing import List
from models.graph import UnifiedGraph, GraphNode, Conflict
from models.inputs import ExtractedKnowledge
from utils.llm_client import call_llm_json

class GraphMergerService:
    def merge(self, chunks: List[ExtractedKnowledge]) -> UnifiedGraph:
        print(f"🔗 [Merger] Объединение {len(chunks)} графов...")
        
        # Упрощенная сериализация
        context_data = "\n\n".join([
            f"SOURCE {chunk.source_window_ref}:\nNODES: {chunk.model_dump_json(include={'nodes', 'edges'})}"
            for chunk in chunks
        ])

        system_prompt = """
        Ты Системный Архитектор. Объедини графы знаний в одну структуру UnifiedGraph.
        
        1. Объединяй синонимы (Auth = Login).
        2. Если есть противоречия (MySQL vs Postgres), создай Conflict.
        3. target_section выбери из: general_info, tech_stack, functional_req, ui_ux.
        """
        
        try:
            return call_llm_json(
                schema=UnifiedGraph,
                prompt=system_prompt,
                data=context_data
            )
        except Exception as e:
            print(f"❌ Ошибка Merger: {e}")
            return UnifiedGraph()