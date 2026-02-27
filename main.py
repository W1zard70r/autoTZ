import os
import asyncio
import logging
from dotenv import load_dotenv

from schemas.document import DataSource
from schemas.enums import DataEnum
from layer1_miner.extractor import MinerProcessor
from layer2_merger.merger import SmartGraphMerger
from layer3_compiler.generator import TZGenerator
from utils.test_data_gen import get_backend_chat_dataset, get_frontend_chat_dataset

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


async def main():
    print("==================================================")
    print("🚀 ГЕНЕРАТОР ТЗ (3-LAYER GRAPH PIPELINE)")
    print("==================================================\n")

    # Инициализация слоев
    miner = MinerProcessor()
    merger = SmartGraphMerger()
    compiler = TZGenerator()

    # Входные данные (имитация чата)
    sources = [
        DataSource(
            source_type=DataEnum.CHAT,
            content=get_backend_chat_dataset(),
            file_name="chat_backend_team"
        ),
        DataSource(
            source_type=DataEnum.CHAT,
            content=get_frontend_chat_dataset(),
            file_name="chat_frontend_team"
        )
    ]

    # ---------------------------------------------------------
    # ЭТАП 1: MINER (Извлечение подграфов из всех источников)
    # ---------------------------------------------------------
    logger.info(">>> СТАРТ ЭТАПА 1: Майнинг знаний")

    all_extracted_subgraphs = []

    for source in sources:
        logger.info(f"📂 Обработка источника: {source.file_name}")
        # Майнер использует накопленный глоссарий для улучшения связности
        subgraphs = await miner.process_source(source)
        all_extracted_subgraphs.extend(subgraphs)
        logger.info(f"   -> Извлечено {len(subgraphs)} чанков из {source.file_name}")

    # ---------------------------------------------------------
    # ЭТАП 2: MERGER (Дедупликация и Слияние)
    # ---------------------------------------------------------
    logger.info(">>> СТАРТ ЭТАПА 2")
    unified_graph = await merger.smart_merge(all_extracted_subgraphs)
    logger.info(f"✅ Граф объединен. Итоговых узлов: {len(unified_graph.nodes)}")
    print("-" * 50)

    # ---------------------------------------------------------
    # ЭТАП 3: COMPILER (Генерация Markdown)
    # ---------------------------------------------------------
    logger.info(">>> СТАРТ ЭТАПА 3")
    doc = await compiler.generate_tz(unified_graph)

    # Сохранение результата
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "FINAL_TZ.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# {doc.project_name}\n")
        f.write(f"**Версия:** {doc.version}\n\n")
        f.write("---\n\n")
        for sec in doc.sections:
            f.write(f"## {sec.title}\n\n")
            f.write(f"{sec.content_markdown}\n\n")
            f.write("---\n\n")

    logger.info(f"🎉 ГОТОВО! Техническое задание сохранено: {output_path}")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())