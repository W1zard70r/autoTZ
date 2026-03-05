import os
import asyncio
import logging
from dotenv import load_dotenv

from schemas.document import DataSource, DocumentSection
from schemas.enums import DataEnum, TZStandardEnum
from schemas.graph import ConflictResolution
from layer1_miner.extractor import MinerProcessor
from layer2_merger.merger import SmartGraphMerger
from layer3_compiler.generator import TZGenerator
from utils.test_data_gen import get_backend_chat_dataset, get_frontend_chat_dataset, get_product_chat_dataset
from utils.state_logger import init_logs_dir

load_dotenv()
init_logs_dir()

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
    print("🚀 ГЕНЕРАТОР ТЗ (Ultra-Stable Pipeline)")

    miner = MinerProcessor()
    merger = SmartGraphMerger()
    compiler = TZGenerator()

    sources = [
        DataSource(source_type=DataEnum.CHAT, content=get_product_chat_dataset(), file_name="product_chat"),
        DataSource(source_type=DataEnum.CHAT, content=get_backend_chat_dataset(), file_name="backend_chat"),
        DataSource(source_type=DataEnum.CHAT, content=get_frontend_chat_dataset(), file_name="frontend_chat")
    ]

    # --- ЭТАП 1 ---
    logger.info(">>> ЭТАП 1: Майнинг")
    all_subgraphs = []
    
    for src in sources:
        try:
            res = await miner.process_source(src)
            all_subgraphs.extend(res)
            logger.info("⏳ Пауза 10с между файлами...")
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Ошибка майнера {src.file_name}: {e}")

    unified_graph = None

    # --- ЭТАП 2 ---
    if all_subgraphs:
        logger.info(">>> ЭТАП 2: Мерджинг")
        try:
            await merger.merge_subgraphs_and_deduplicate(all_subgraphs)
            
            logger.info("⏳ Пауза 20с перед поиском конфликтов...")
            await asyncio.sleep(20)
            
            conflicts = await merger.detect_conflicts()
            
            if conflicts:
                print(f"\n🛑 НАЙДЕНО {len(conflicts)} КОНФЛИКТОВ")
                resolutions = []
                for i, conf in enumerate(conflicts):
                    print(f"\n🔹 #{i+1}: {conf.description}")
                    print(f"   Совет AI: {conf.ai_recommendation}")
                    u_in = input("   👉 Выбор (0-авто, текст-свой): ").strip()
                    if u_in.isdigit() and len(conf.options) > 0:
                        resolutions.append(ConflictResolution(conflict_id=conf.id, selected_option_id=conf.options[0].id))
                    else:
                        resolutions.append(ConflictResolution(conflict_id=conf.id, custom_text=u_in or "Manual fix"))
                merger.apply_resolutions(resolutions)
            
            unified_graph = await merger.finalize_graph()
            
        except Exception as e:
            logger.error(f"Ошибка мерджинга: {e}")
            return

    # --- ЭТАП 3 ---
    if unified_graph:
        logger.info(">>> ЭТАП 3: Генерация")
        logger.info("⏳ Пауза 45с перед генерацией (охлаждение)...")
        await asyncio.sleep(45)

        try:
            final_doc = await compiler.generate_tz(unified_graph, standard=TZStandardEnum.GOST_34)
            
            # Исправлена ошибка с output_dir
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            # JSON
            with open(os.path.join(output_dir, "FINAL_TZ_PAYLOAD.json"), "w", encoding="utf-8") as f:
                f.write(final_doc.model_dump_json(indent=2, ensure_ascii=False))
            
            # MD
            with open(os.path.join(output_dir, "PREVIEW_GOST.md"), "w", encoding="utf-8") as f:
                f.write(f"# {final_doc.title_page.project_name}\n\n")
                def write_md(sec, lvl=1):
                    indent = "#" * lvl
                    t = f"{indent} {sec.number} {sec.title}\n\n{sec.content or ''}\n\n"
                    for s in sec.subsections: t += write_md(s, lvl+1)
                    return t
                for sec in final_doc.structure:
                    f.write(write_md(sec, 2))
                    
            logger.info(f"✅ Готово! См. папку {output_dir}")
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}", exc_info=True)
    else:
        logger.error("❌ Граф не создан!")

if __name__ == "__main__":
    asyncio.run(main())