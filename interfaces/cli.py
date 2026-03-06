"""Консольный интерфейс (Human-in-the-Loop)."""
import asyncio
import os
import logging
from typing import List, Optional

from schemas.document import DataSource
from schemas.enums import TemplateType
from schemas.graph import ConflictResolution
from schemas.templates.base import TZResult, FieldGap
from core.pipeline import TZPipeline
from interfaces.base import BasePipelineInterface

logger = logging.getLogger(__name__)

TEMPLATE_NAMES = {
    TemplateType.GOST: "Формальный по ГОСТ",
    TemplateType.HOUSEHOLD: "Бытовой / личная задача",
    TemplateType.IT_PROJECT: "IT-проект",
    TemplateType.CONSTRUCTION: "Строительный проект",
    TemplateType.ENGINEERING: "Инженерный проект",
}


class ConsoleInterface(BasePipelineInterface):
    """Интерактивный консольный интерфейс.

    Позволяет:
    - выбрать шаблон
    - увидеть конфликты и разрешить их
    - увидеть пробелы в ТЗ и заполнить их
    - повторить компиляцию после уточнений
    """

    def __init__(
        self,
        sources: List[DataSource],
        template_type: Optional[TemplateType] = None,
        language: str = "ru",
        output_dir: str = "output",
    ):
        self.sources = sources
        self.template_type = template_type
        self.language = language
        self.output_dir = output_dir

    async def get_sources(self) -> List[DataSource]:
        return self.sources

    async def run(self) -> TZResult:
        print("=" * 60)
        print("🚀 ГЕНЕРАТОР ТЗ (HUMAN-IN-THE-LOOP PIPELINE)")
        print("=" * 60 + "\n")

        # ── Выбор шаблона ────────────────────────────────
        if self.template_type is None:
            self.template_type = self._prompt_template_selection()

        pipeline = TZPipeline(
            template_type=self.template_type,
            language=self.language,
        )

        # ── Шаг 1: Извлечение ────────────────────────────
        await pipeline.extract(self.sources)
        print("-" * 50)

        if not pipeline.subgraphs:
            print("❌ Нет данных для обработки.")
            return TZResult(
                template_type=self.template_type,
                template_data=None,
            )

        # ── Шаг 2-3: Слияние + конфликты ─────────────────
        await pipeline.merge()
        conflicts = await pipeline.detect_conflicts()

        # ── Шаг 4: Разрешение конфликтов ──────────────────
        if conflicts:
            resolutions = await self._prompt_conflict_resolution(conflicts, pipeline)
            pipeline.apply_resolutions(resolutions)
        else:
            logger.info("✅ Конфликтов не обнаружено.")

        # ── Шаг 5-6: Финализация + компиляция ─────────────
        await pipeline.finalize_graph()
        result = await pipeline.compile()
        print("-" * 50)

        # ── Шаг 7: Уточнение пробелов ────────────────────
        gaps = pipeline.get_gaps()
        if gaps:
            print(f"\n⚠️  Обнаружено {len(gaps)} незаполненных полей.")
            answers = self._prompt_fill_gaps(gaps)
            if answers:
                pipeline.add_user_answers(answers)
                result = await pipeline.recompile()

        # ── Вывод результата ──────────────────────────────
        await self.present_result(result)
        return result

    async def present_result(self, result: TZResult) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "FINAL_TZ.md")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.markdown)

        v = result.validation
        print("\n" + "=" * 60)
        print(f"📊 Заполненность: {v.completeness_percent}% "
              f"({v.filled_fields}/{v.total_fields} полей)")
        if v.gaps:
            print(f"⚠️  Пробелов: {len(v.gaps)}")
        if v.conflicts:
            print(f"⚡ Конфликтов: {len(v.conflicts)}")
        print(f"\n🎉 ТЗ сохранено: {output_path}")
        print("=" * 60)

    # ── Вспомогательные интерактивные методы ──────────────

    def _prompt_template_selection(self) -> TemplateType:
        print("📋 Выберите шаблон ТЗ:\n")
        types = list(TemplateType)
        for i, tt in enumerate(types):
            print(f"  [{i}] {TEMPLATE_NAMES.get(tt, tt.value)}")
        print()

        while True:
            choice = input("👉 Номер шаблона: ").strip()
            if choice.isdigit() and 0 <= int(choice) < len(types):
                selected = types[int(choice)]
                print(f"   ✅ Выбран: {TEMPLATE_NAMES.get(selected, selected.value)}\n")
                return selected
            print("   ❌ Неверный номер, попробуйте снова.")

    async def _prompt_conflict_resolution(self, conflicts, pipeline: TZPipeline) -> List[ConflictResolution]:
        print("\n" + "!" * 60)
        print("🛑 ОБНАРУЖЕНЫ ПРОТИВОРЕЧИЯ!")
        print("!" * 60 + "\n")

        resolutions = []
        for i, conf in enumerate(conflicts, 1):
            print(f"\n🔹 КОНФЛИКТ #{i}: {conf.description}")
            print(f"   Категория: {conf.category}")
            print(f"   🤖 AI советует: {conf.ai_recommendation}")

            # Генерация развёрнутых вариантов
            print("\n   ⏳ Генерация вариантов...")
            variants = await pipeline.generate_conflict_variants(conf)

            print("   Варианты:")
            for idx, (opt, variant_text) in enumerate(zip(conf.options, variants)):
                print(f"\n     [{idx}] {opt.text}")
                print(f"         {variant_text}")
            print(f"\n     [текст] → свой вариант")

            while True:
                user_input = input(f"   👉 Ваш выбор для #{i}: ").strip()
                if not user_input:
                    continue
                if user_input.isdigit():
                    opt_idx = int(user_input)
                    if 0 <= opt_idx < len(conf.options):
                        resolutions.append(ConflictResolution(
                            conflict_id=conf.id,
                            selected_option_id=conf.options[opt_idx].id,
                        ))
                        print(f"   ✅ Выбрано: {conf.options[opt_idx].text}")
                        break
                    print("   ❌ Неверный номер.")
                else:
                    resolutions.append(ConflictResolution(
                        conflict_id=conf.id,
                        custom_text=user_input,
                    ))
                    print(f"   ✍️ Принято: {user_input}")
                    break

        print("\n🔄 Применяем решения...")
        return resolutions

    def _prompt_fill_gaps(self, gaps: List[FieldGap]) -> dict[str, str]:
        print("\n📝 Незаполненные поля (введите значение или Enter для пропуска):\n")
        answers = {}
        for gap in gaps:
            value = input(f"  {gap.field_name}: ").strip()
            if value:
                answers[gap.field_path] = value
        return answers
