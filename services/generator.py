from models.graph import UnifiedGraph
from models.document import FullTZDocument, GeneratedSection
from models.enums import TZSectionEnum
from utils.llm_client import call_llm_text

class TZGeneratorService:
    def generate(self, graph: UnifiedGraph, template: dict) -> FullTZDocument:
        print(f"📝 [Generator] Генерация документа...")

        if graph.conflicts:
            print(f"⚠️ Найдено {len(graph.conflicts)} конфликтов. Генерация продолжается, но конфликты будут отмечены.")

        sections = []
        # Определяем порядок разделов
        sections_to_write = [
            TZSectionEnum.GENERAL,
            TZSectionEnum.FUNCTIONAL,
            TZSectionEnum.STACK,
            TZSectionEnum.INTERFACE
        ]

        for sec_enum in sections_to_write:
            # Фильтруем узлы для текущего раздела
            relevant_nodes = [n for n in graph.nodes if n.target_section == sec_enum]
            
            if not relevant_nodes:
                continue

            print(f"  > Пишем раздел: {sec_enum.value} ({len(relevant_nodes)} узлов)")
            
            node_context = "\n".join([f"- {n.label} ({n.id}): {n.content}" for n in relevant_nodes])
            
            prompt = f"""
            Напиши раздел Технического Задания: '{sec_enum.value}'.
            Используй ТОЛЬКО предоставленные факты. Стиль: формально-деловой, ГОСТ.
            Используй Markdown заголовки и списки.
            """
            
            content = call_llm_text(prompt, data=node_context)
            
            sections.append(GeneratedSection(
                section_id=sec_enum,
                title=sec_enum.name,
                content_markdown=content,
                used_node_ids=[n.id for n in relevant_nodes]
            ))

        return FullTZDocument(
            project_name="Online Course Platform",
            version="1.0.0",
            sections=sections
        )