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
from utils.test_data_gen import get_backend_chat_dataset, get_frontend_chat_dataset
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
    print("🚀 ГЕНЕРАТОР ТЗ (Stable Pipeline)")

    miner = MinerProcessor()
    merger = SmartGraphMerger()
    compiler = TZGenerator()

    sources = [
        DataSource(source_type=DataEnum.CHAT, content=get_backend_chat_dataset(), file_name="backend"),
        DataSource(source_type=DataEnum.CHAT, content=get_frontend_chat_dataset(), file_name="frontend")
    ]

    # --- ЭТАП 1 ---
    logger.info(">>> ЭТАП 1: Майнинг")
    all_subgraphs = []
    for src in sources:
        try:
            res = await miner.process_source(src)
            all_subgraphs.extend(res)
            logger.info("⏳ Пауза 5с...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Ошибка майнера {src.file_name}: {e}")

    # --- ЭТАП 2 ---
    logger.info(">>> ЭТАП 2: Мерджинг")
    if all_subgraphs:
        # 2.1 Слияние и дедупликация
        await merger.merge_subgraphs_and_deduplicate(all_subgraphs)
        
        logger.info("⏳ Пауза 10с перед поиском конфликтов...")
        await asyncio.sleep(10)
        
        # 2.2 Конфликты
        conflicts = await merger.detect_conflicts()
        
        # 2.3 Human-in-the-loop (Решение конфликтов)
        if conflicts:
            print("\n" + "!" * 60)
            print("🛑 ОБНАРУЖЕНЫ КОНФЛИКТЫ! Требуется решение.")
            resolutions = []
            
            for i, conf in enumerate(conflicts, 1):
                print(f"\n🔹 КОНФЛИКТ #{i}: {conf.description}")
                print(f"   AI Совет: {conf.ai_recommendation}")
                print("   Варианты:")
                for idx, opt in enumerate(conf.options):
                    print(f"     [{idx}] {opt.text}")
                print("     [Любой текст] -> Свой вариант")

                while True:
                    user_input = input(f"   👉 Выбор #{i}: ").strip()
                    if not user_input: continue
                    
                    if user_input.isdigit():
                        opt_idx = int(user_input)
                        if 0 <= opt_idx < len(conf.options):
                            resolutions.append(ConflictResolution(
                                conflict_id=conf.id, 
                                selected_option_id=conf.options[opt_idx].id
                            ))
                            break
                    else:
                        resolutions.append(ConflictResolution(
                            conflict_id=conf.id, 
                            custom_text=user_input
                        ))
                        break
            
            logger.info("Применяем решения...")
            merger.apply_resolutions(resolutions)
        else:
            logger.info("✅ Конфликтов нет.")
        
        # 2.4 Финализация
        unified_graph = await merger.finalize_graph()

    # --- ЭТАП 3 ---
    logger.info(">>> ЭТАП 3: Генерация")
    logger.info("⏳ Пауза 15с перед генерацией (защита от 429)...")
    await asyncio.sleep(15)

    try:
        final_doc = await compiler.generate_tz(unified_graph, standard=TZStandardEnum.GOST_34)
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, "FINAL_TZ_PAYLOAD.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(final_doc.model_dump_json(indent=2, ensure_ascii=False))
            
        md_path = os.path.join(output_dir, "PREVIEW_GOST.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {final_doc.title_page.project_name}\n\n")
            
            def write_md(sec, lvl=1):
                indent = "#" * lvl
                res = f"{indent} {sec.number} {sec.title}\n\n{sec.content or ''}\n\n"
                for sub in sec.subsections:
                    res += write_md(sub, lvl+1)
                return res

            for sec in final_doc.structure:
                f.write(write_md(sec, 2))
                
        logger.info(f"✅ Готово! Результаты в {output_dir}")
        
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())