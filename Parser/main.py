import os
import json
import asyncio
from dotenv import load_dotenv
from models import DataSource, DataEnum
from processor import Layer1Processor
from test_data_gen import get_huge_chat_dataset

load_dotenv()
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main():
    # 1. Создаем тестовые данные (Имитация сложного диалога)
    chat_data = {
        "messages": get_huge_chat_dataset()
    }

    source = DataSource(
        source_type=DataEnum.CHAT,
        content=chat_data["messages"],
        file_name="telegram_backend_team"
    )

    # 2. Инициализация и запуск Асинхронного пайплайна
    processor = Layer1Processor()

    # Запускаем обработку
    graph_manager = await processor.process_source(source)

    # 3. Сохранение результатов
    os.makedirs("output", exist_ok=True)

    # Экспорт в GraphML (Для визуализации в Gephi / Neo4j)
    graphml_path = os.path.join("output", f"{source.file_name}.graphml")
    graph_manager.export_to_graphml(graphml_path)
    print(f"✅ Граф сохранен в формате GraphML: {graphml_path}")

    # Вывод статистики
    print("\n📊 Итоговая статистика графа:")
    print(f"Узлов: {graph_manager.graph.number_of_nodes()}")
    print(f"Связей: {graph_manager.graph.number_of_edges()}")


if __name__ == "__main__":
    # Запуск асинхронного event loop
    asyncio.run(main())