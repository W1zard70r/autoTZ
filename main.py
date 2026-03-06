import os
import asyncio
import logging
from dotenv import load_dotenv

# Схемы данных
from schemas.document import DataSource, FinalExportDocument
from schemas.enums import DataEnum, TZStandardEnum
from schemas.graph import ConflictResolution

# Слои пайплайна
from layer1_miner.extractor import MinerProcessor
from layer2_merger.merger import SmartGraphMerger
from layer3_compiler.generator import TZGenerator

# Генераторы тестовых данных
from utils.test_data_gen import (
    get_product_chat_dataset, 
    get_backend_chat_dataset, 
    get_frontend_chat_dataset
)
from utils.state_logger import init_logs_dir

# Инициализация
load_dotenv()
init_logs_dir()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def render_to_markdown(doc: FinalExportDocument) -> str:
    """
    ЧИСТЫЙ ЭКСПОРТ: Преобразует Pydantic объект в Markdown файл.
    Здесь нет LLM, только чтение данных из полей.
    """
    lines = [
        f"# ТЗ: {doc.title_page.project_name}",
        f"**Организация:** {doc.title_page.organization_name}",
        f"**Стандарт:** {doc.standard.value}",
        f"**Версия:** {doc.version}\n",
        "---\n"
    ]

    def process_sections(sections, level=2):
        for sec in sections:
            prefix = "#" * level
            # Печатаем номер и заголовок из Pydantic
            lines.append(f"{prefix} {sec.number} {sec.title}\n")
            
            # Печатаем контент, который сгенерировала LLM (только текст и списки)
            if sec.content:
                lines.append(f"{sec.content.strip()}\n")
            
            # Если есть подразделы, заходим в них
            if sec.subsections:
                process_sections(sec.subsections, level + 1)
            
            lines.append("")

    process_sections(doc.structure)
    return "\n".join(lines)

async def main():
    logger.info("🚀 ЗАПУСК ULTRA-STABLE PIPELINE")
    
    miner = MinerProcessor()
    merger = SmartGraphMerger()
    compiler = TZGenerator()

    sources = [
        DataSource(source_type=DataEnum.CHAT, content=get_product_chat_dataset(), file_name="product_chat"),
        DataSource(source_type=DataEnum.CHAT, content=get_backend_chat_dataset(), file_name="backend_chat"),
        DataSource(source_type=DataEnum.CHAT, content=get_frontend_chat_dataset(), file_name="frontend_chat")
    ]

    # --- ЭТАП 1: МАЙНИНГ ---
    all_subgraphs = []
    for src in sources:
        try:
            subgraphs = await miner.process_source(src)
            all_subgraphs.extend(subgraphs)
            logger.info(f"⏳ Остывание API (30 сек) после {src.file_name}...")
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"❌ Ошибка майнера {src.file_name}: {e}")

    if not all_subgraphs:
        logger.error("❌ Данные не собраны. Выход.")
        return

    # --- ЭТАП 2: МЕРДЖИНГ ---
    await merger.merge_subgraphs_and_deduplicate(all_subgraphs)
    
    # Ищем конфликты (ретраи внутри)
    conflicts = await merger.detect_conflicts()
    
    if conflicts:
        print(f"\n" + "="*60)
        print(f"🛑 ОБНАРУЖЕНО {len(conflicts)} КОНФЛИКТОВ ТРЕБОВАНИЙ")
        print("="*60)
        
        resolutions = []
        for i, conf in enumerate(conflicts):
            print(f"\n🔹 Конфликт #{i+1}: {conf.description}")
            print(f"   💡 Совет AI: {conf.ai_recommendation}")
            print("   Варианты:")
            for idx, opt in enumerate(conf.options):
                print(f"     [{idx}] {opt.text} (ID: {opt.id})")
            
            u_in = input("\n   👉 Ваш выбор (номер, список через запятую или свой текст): ").strip()
            
            if u_in.isdigit():
                idx = int(u_in)
                if idx < len(conf.options):
                    resolutions.append(ConflictResolution(
                        conflict_id=conf.id, 
                        selected_option_id=conf.options[idx].id
                    ))
                else:
                    resolutions.append(ConflictResolution(conflict_id=conf.id, custom_text=u_in))
            else:
                # Если ввели "Django" или "0, 1" или "GraphQL"
                resolutions.append(ConflictResolution(conflict_id=conf.id, custom_text=u_in))
        
        # Применяем решения (асинхронно, так как может потребоваться нормализация текста)
        await merger.apply_resolutions(resolutions)
        logger.info("✅ Все конфликты разрешены.")

    # Финализируем граф (классификация узлов по разделам)
    unified_graph = await merger.finalize_graph()

    # --- ЭТАП 3: ГЕНЕРАЦИЯ ---
    logger.info(">>> ЭТАП 3: Заполнение ТЗ фактами...")
    final_doc = await compiler.generate_tz(unified_graph)

    # --- ЭКСПОРТ ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Сохраняем Pydantic JSON (оригинал данных)
    json_path = os.path.join(output_dir, "FINAL_TZ_PAYLOAD.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(final_doc.model_dump_json(indent=2, ensure_ascii=False))
    
    # 2. Сохраняем Markdown (программный рендер)
    md_content = render_to_markdown(final_doc)
    md_path = os.path.join(output_dir, "PREVIEW_GOST.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
            
    logger.info(f"✅ Пайплайн завершен! Результаты в папке {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())