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
from utils.state_logger import init_logs_dir

load_dotenv()

# Инициализируем папку логов
init_logs_dir()

# Настраиваем двойное логирование (в консоль и в файл app.log)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def main():
    print("==================================================")
    print("🚀 ГЕНЕРАТОР ТЗ (3-LAYER GRAPH PIPELINE С ЛОГАМИ)")
    print("==================================================\n")

    miner = MinerProcessor()
    merger = SmartGraphMerger()
    compiler = TZGenerator()

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

    # --- ЭТАП 1: MINER ---
    logger.info(">>> СТАРТ ЭТАПА 1: Майнинг знаний")
    all_extracted_subgraphs = []

    for source in sources:
        logger.info(f"📂 Обработка источника: {source.file_name}")
        subgraphs = await miner.process_source(source)
        all_extracted_subgraphs.extend(subgraphs)
        logger.info(f"   -> Извлечено {len(subgraphs)} чанков из {source.file_name}")

    print("-" * 50)

    # --- ЭТАП 2: MERGER ---
    logger.info(">>> СТАРТ ЭТАПА 2: Слияние")
    unified_graph = await merger.smart_merge(all_extracted_subgraphs)
    logger.info(f"✅ Граф объединен. Итоговых узлов: {len(unified_graph.nodes)}")
    
    print("-" * 50)

    # --- ЭТАП 3: COMPILER ---
    logger.info(">>> СТАРТ ЭТАПА 3: Генерация")
    doc = await compiler.generate_tz(unified_graph)

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