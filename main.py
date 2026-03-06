import os
import asyncio
import logging
from dotenv import load_dotenv

from schemas.document import DataSource, FinalExportDocument
from schemas.enums import DataEnum, TZStandardEnum
from schemas.graph import ConflictResolution

from layer1_miner.extractor import MinerProcessor
from layer2_merger.merger import SmartGraphMerger
from layer3_compiler.generator import TZGenerator

from utils.test_data_gen import get_product_chat_dataset, get_backend_chat_dataset, get_frontend_chat_dataset
from utils.state_logger import init_logs_dir

load_dotenv()
init_logs_dir()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("logs/app.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def render_to_markdown(doc: FinalExportDocument) -> str:
    """Программная сборка MD из Pydantic объекта"""
    lines = [
        f"# ТЗ: {doc.title_page.project_name}",
        f"**Организация:** {doc.title_page.organization_name}",
        f"**Версия:** {doc.version}\n",
        "---\n"
    ]
    for sec in doc.structure:
        lines.append(f"## {sec.number} {sec.title}\n\n{sec.content.strip()}\n")
        for sub in sec.subsections:
            lines.append(f"### {sub.number} {sub.title}\n\n{sub.content.strip()}\n")
    return "\n".join(lines)

async def main():
    logger.info("🚀 СТАРТ ПАЙПЛАЙНА")
    miner, merger, compiler = MinerProcessor(), SmartGraphMerger(), TZGenerator()

    sources = [
        DataSource(source_type=DataEnum.CHAT, content=get_product_chat_dataset(), file_name="product_chat"),
        DataSource(source_type=DataEnum.CHAT, content=get_backend_chat_dataset(), file_name="backend_chat"),
        DataSource(source_type=DataEnum.CHAT, content=get_frontend_chat_dataset(), file_name="frontend_chat")
    ]

    # --- ЭТАП 1: Майнинг ---
    all_subgraphs = []
    for src in sources:
        try:
            res = await miner.process_source(src)
            all_subgraphs.extend(res)
            logger.info(f"⏳ Охлаждение после {src.file_name} (30 сек)...")
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"❌ Ошибка майнера {src.file_name}: {e}")

    if not all_subgraphs:
        logger.error("❌ Данные не собраны!")
        return

    # --- ЭТАП 2: Мерджинг и Конфликты ---
    await merger.merge_subgraphs_and_deduplicate(all_subgraphs)
    conflicts = await merger.detect_conflicts()
    
    if conflicts:
        print(f"\n🛑 ОБНАРУЖЕНО {len(conflicts)} ТЕХНИЧЕСКИХ ПРОТИВОРЕЧИЙ")
        resolutions = []
        for i, conf in enumerate(conflicts):
            print(f"\n🔹 Конфликт #{i+1}: {conf.description}")
            # ТЕПЕРЬ СОВЕТ БУДЕТ ВИДЕН:
            print(f"   💡 СОВЕТ АРХИТЕКТОРА: {conf.ai_recommendation}") 
            print("   Варианты:")
            for idx, opt in enumerate(conf.options):
                print(f"     [{idx}] {opt.text} (ID: {opt.id})")
            
            u_in = input("\n   👉 Ваш выбор (номер или ID): ").strip()
            
            # Логика: если ввели цифру - берем ID этой опции. Если текст - передаем текст.
            if u_in.isdigit():
                idx = int(u_in)
                if idx < len(conf.options):
                    resolutions.append(ConflictResolution(
                        conflict_id=conf.id, 
                        selected_option_id=conf.options[idx].id
                    ))
                    print(f"   ✅ Принят вариант: {conf.options[idx].text}")
                else:
                    resolutions.append(ConflictResolution(conflict_id=conf.id, custom_text=u_in))
            else:
                resolutions.append(ConflictResolution(conflict_id=conf.id, custom_text=u_in))
        
        merger.apply_resolutions(resolutions)

    unified_graph = await merger.finalize_graph()

    # --- ЭТАП 3: Генерация ---
    logger.info(">>> ЭТАП 3: Генерация контента")
    final_doc = await compiler.generate_tz(unified_graph)

    # Экспорт
    os.makedirs("output", exist_ok=True)
    with open("output/FINAL_TZ_PAYLOAD.json", "w", encoding="utf-8") as f:
        f.write(final_doc.model_dump_json(indent=2, ensure_ascii=False))
    
    with open("output/PREVIEW_GOST.md", "w", encoding="utf-8") as f:
        f.write(render_to_markdown(final_doc))
            
    logger.info("✅ ГОТОВО! Папка output.")

if __name__ == "__main__":
    asyncio.run(main())