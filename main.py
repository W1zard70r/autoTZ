import os
from typing import List

from models.inputs import DataSource, DataEnum
from models.graph import UnifiedGraph
from models.document import FullTZDocument
from services.extractor import DataExtractorService
from services.merger import GraphMergerService
from services.generator import TZGeneratorService

def load_graphml(filepath: str) -> DataSource:
    with open(filepath, 'r', encoding='utf-8') as f:
        return DataSource(
            source_type=DataEnum.GRAPHML,
            content=f.read(),
            file_name=os.path.basename(filepath)
        )

def main():
    print("==========================================")
    print("🚀 ГЕНЕРАТОР ТЗ (GRAPHML EDITION)")
    print("==========================================\n")

    extractor = DataExtractorService()
    merger = GraphMergerService()
    generator = TZGeneratorService()

    # Берем 3 графа: твой backend и 2 новых от друга
    files = [
        "data/telegram_backend_team.graphml",
        "data/frontend_app.graphml",
        "data/deploy_infra.graphml"
    ]
    
    inputs = []
    print("📂 Загрузка графов:")
    for f in files:
        if os.path.exists(f):
            inputs.append(load_graphml(f))
            print(f"  - {f} (ok)")
        else:
            print(f"  - {f} (НЕ НАЙДЕН)")

    # 1. PARSING (Без LLM)
    print("\n--- ЭТАП 1: ПАРСИНГ ГРАФОВ ---")
    chunks = []
    for src in inputs:
        chunk = extractor.extract(src)
        chunks.append(chunk)
        print(f"  ✅ {src.file_name}: узлов={len(chunk.nodes)}, связей={len(chunk.edges)}")

    # 2. MERGING (С LLM)
    print("\n--- ЭТАП 2: СЛИЯНИЕ (LLM) ---")
    unified_graph = merger.merge(chunks)
    print(f"  ✅ Граф объединен. Узлов: {len(unified_graph.nodes)}")
    
    # 3. GENERATION (С LLM)
    print("\n--- ЭТАП 3: ГЕНЕРАЦИЯ ТЗ ---")
    try:
        doc = generator.generate(unified_graph, template={})
        with open("FINAL_TZ.md", "w", encoding="utf-8") as f:
            f.write(f"# {doc.project_name}\n\n")
            for sec in doc.sections:
                f.write(f"## {sec.title}\n{sec.content_markdown}\n\n")
        print(f"\n🎉 ГОТОВО! Файл: FINAL_TZ.md")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()