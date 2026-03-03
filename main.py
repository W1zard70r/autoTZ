import os
import asyncio
import logging
from dotenv import load_dotenv

from schemas.document import DataSource
from schemas.enums import DataEnum
from schemas.graph import ConflictResolution
from layer1_miner.extractor import MinerProcessor
from layer2_merger.merger import SmartGraphMerger
from layer3_compiler.generator import TZGenerator
from utils.test_data_gen import get_backend_chat_dataset, get_frontend_chat_dataset
from utils.state_logger import init_logs_dir

load_dotenv()

# Инициализируем папку логов
init_logs_dir()

# Настраиваем логирование
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
    print("🚀 ГЕНЕРАТОР ТЗ (HUMAN-IN-THE-LOOP PIPELINE)")
    print("==================================================\n")

    miner = MinerProcessor()
    merger = SmartGraphMerger()
    compiler = TZGenerator()

    # 1. Подготовка источников данных
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

    # --- ЭТАП 1: MINER (Майнинг знаний) ---
    logger.info(">>> СТАРТ ЭТАПА 1: Майнинг знаний")
    all_extracted_subgraphs = []

    for source in sources:
        logger.info(f"📂 Обработка источника: {source.file_name}")
        try:
            # Майнинг подграфов из источника
            # Важно: В miner.py желательно добавить небольшую задержку между окнами,
            # но здесь мы добавим задержку между файлами.
            subgraphs = await miner.process_source(source)
            all_extracted_subgraphs.extend(subgraphs)
            logger.info(f"   -> Извлечено {len(subgraphs)} чанков из {source.file_name}")
            
            # Задержка 5 секунд, чтобы "остудить" лимит запросов Google API
            logger.info("⏳ Пауза 5с для соблюдения квоты API...")
            await asyncio.sleep(5) 

        except Exception as e:
            logger.error(f"❌ Критическая ошибка при обработке {source.file_name}: {e}")

    print("-" * 50)

    # --- ЭТАП 2: MERGER (Слияние и Разрешение конфликтов) ---
    logger.info(">>> СТАРТ ЭТАПА 2: Слияние графов")

    if not all_extracted_subgraphs:
        logger.error("❌ Нет данных для слияния. Завершение работы.")
        return

    # Шаг 2.1: Черновое слияние и дедупликация синонимов
    logger.info("...Выполняется дедупликация сущностей (Step 1)...")
    await merger.merge_subgraphs_and_deduplicate(all_extracted_subgraphs)

    # Шаг 2.2: Поиск логических конфликтов
    logger.info("...Поиск противоречий в требованиях (Step 2)...")
    conflicts = await merger.detect_conflicts()

    # Шаг 2.3: Интерактивное разрешение конфликтов (Human-in-the-loop)
    if conflicts:
        print("\n" + "!" * 60)
        print("🛑 ОБНАРУЖЕНЫ ПРОТИВОРЕЧИЯ! ПАЙПЛАЙН ПРИОСТАНОВЛЕН.")
        print("Необходимо вмешательство пользователя для принятия решений.")
        print("!" * 60 + "\n")

        resolutions = []
        
        for i, conf in enumerate(conflicts, 1):
            print(f"\n🔹 КОНФЛИКТ #{i}: {conf.description}")
            print(f"   Категория: {conf.category}")
            print(f"   🤖 AI Советует: {conf.ai_recommendation}")
            print("   Варианты выбора:")
            
            for idx, option in enumerate(conf.options):
                print(f"     [{idx}] {option.text}")
                
            print("     [Любой текст] -> Ввести свой вариант решения")

            while True:
                user_input = input(f"   👉 Ваш выбор для конфликта #{i}: ").strip()
                if not user_input:
                    continue
                
                if user_input.isdigit():
                    opt_idx = int(user_input)
                    if 0 <= opt_idx < len(conf.options):
                        selected_option = conf.options[opt_idx]
                        resolutions.append(ConflictResolution(
                            conflict_id=conf.id,
                            selected_option_id=selected_option.id
                        ))
                        print(f"   ✅ Выбрано: {selected_option.text}")
                        break
                    else:
                        print("   ❌ Неверный номер варианта.")
                else:
                    resolutions.append(ConflictResolution(
                        conflict_id=conf.id,
                        custom_text=user_input
                    ))
                    print(f"   ✍️ Принято пользовательское решение: {user_input}")
                    break
        
        print("\n🔄 Применяем решения и перестраиваем граф...")
        merger.apply_resolutions(resolutions)
    else:
        logger.info("✅ Конфликтов не обнаружено. Автоматическое продолжение.")
    
    # Шаг 2.4: Финализация графа (распределение по секциям)
    unified_graph = await merger.finalize_graph()
    logger.info(f"✅ Граф готов. Итоговых узлов: {len(unified_graph.nodes)}")

    print("-" * 50)

    # --- ЭТАП 3: COMPILER (Генерация документа) ---
    logger.info(">>> СТАРТ ЭТАПА 3: Генерация ТЗ")
    
    try:
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
        
    except Exception as e:
        logger.error(f"❌ Ошибка при генерации документа: {e}")
        
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(main())