import asyncio
import json
import os
import time
from typing import List, Dict

from core import TZPipeline
from interfaces.test_runner import _load_source_from_file
from schemas.enums import TemplateType


# ... (импорты из вашего кода) ...

class CheckpointRunner:
    def __init__(self, template_type=TemplateType.IT_PROJECT, language="ru"):
        self.template_type = template_type
        self.language = language
        self.state_file = "pipeline_state.json"

    async def run_layers_1_and_2(self, file_paths: List[str]):
        """Прогоняет майнинг и слияние (Слои 1 и 2) и сохраняет граф на диск."""
        print("=== ЗАПУСК СЛОЕВ 1 и 2 (Майнинг и Слияние) ===")
        sources = []
        for fp in file_paths:
            sources.append(_load_source_from_file(fp))

        pipeline = TZPipeline(template_type=self.template_type, language=self.language)

        # Шаг 1: Извлечение
        await pipeline.extract(sources)
        if not pipeline.subgraphs:
            print("❌ Нет подграфов.")
            return

        # Шаг 2: Слияние и финализация графа
        await pipeline.merge()
        await pipeline.finalize_graph()  # Обязательно финализируем перед сохранением

        # Сохранение состояния на диск
        state = pipeline.save_state()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        print(f"✅ Граф знаний успешно собран и сохранен в {self.state_file}")

    async def run_layer_3(self, user_answers: Dict[str, str] = None, output_dir: str = "output"):
        """Загружает граф с диска и запускает только компиляцию (Слой 3)."""
        print("=== ЗАПУСК СЛОЯ 3 (Компиляция ТЗ) ===")

        if not os.path.exists(self.state_file):
            print("❌ Файл состояния не найден. Сначала запустите run_layers_1_and_2()")
            return

        # Загрузка состояния с диска
        with open(self.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

        pipeline = TZPipeline.load_state(state)

        # Добавляем ответы пользователя (если есть), чтобы закрыть пробелы (Gaps)
        if user_answers:
            pipeline.add_user_answers(user_answers)

        # Шаг 3: Компиляция
        start = time.time()
        result = await pipeline.compile()  # или pipeline.recompile()
        duration = time.time() - start

        # Сохранение результата
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "FINAL_TZ.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)

        v = result.validation
        print(f"⏱ Время компиляции: {duration:.1f}s")
        print(f"📊 Заполненность: {v.completeness_percent}% ({v.filled_fields}/{v.total_fields})")

        if v.gaps:
            print(f"⚠️ Остались незаполненные поля (Gaps):")
            for gap in v.gaps:
                print(f"  - {gap.field_name}: {gap.reason}")

        print(f"🎉 ТЗ сохранено: {output_path}")
        return pipeline

async def main():
    check = CheckpointRunner()
    # await check.run_layers_1_and_2([r"C:\Users\Danii\Downloads\AyuGram Desktop\dataset (2)\project1566\chat_2.txt",                                    r"C:\Users\Danii\Downloads\AyuGram Desktop\dataset (2)\project1566\chat1.txt"])
    runner = await check.run_layer_3()
    print(runner)
    input()

if __name__ == "__main__":
   asyncio.run(main())