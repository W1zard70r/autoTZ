"""Прогон пайплайна на указанных файлах (batch-режим).

Принимает список путей к файлам (JSON-чаты, текстовые расшифровки и т.д.),
автоматически определяет тип, загружает как DataSource и прогоняет через
полный пайплайн — без интерактива.
"""
import json
import logging
import os
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from schemas.document import DataSource
from schemas.enums import TemplateType, DataEnum
from schemas.templates.base import TZResult
from core.pipeline import TZPipeline
from interfaces.base import BasePipelineInterface

logger = logging.getLogger(__name__)


def _load_source_from_file(filepath: str) -> DataSource:
    """Загружает файл как DataSource, определяя тип по расширению."""
    name = os.path.splitext(os.path.basename(filepath))[0]
    ext = os.path.splitext(filepath)[1].lower()

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    if ext == ".json":
        content = json.loads(raw)
        if isinstance(content, dict) and "messages" in content:
            return DataSource(
                file_name=name,
                source_type=DataEnum.CHAT,
                content=content["messages"],
            )
        return DataSource(file_name=name, source_type=DataEnum.CHAT, content=content)

    # .txt, .md и прочее — как plain text
    return DataSource(file_name=name, source_type=DataEnum.PLAIN_TEXT, content=raw)


@dataclass
class BatchResult:
    """Результат прогонки пайплайна на наборе файлов."""
    sources_count: int
    subgraphs_count: int
    nodes_count: int
    edges_count: int
    completeness: float
    gaps_count: int
    conflicts_count: int
    duration_sec: float
    output_path: Optional[str] = None
    tz_result: Optional[TZResult] = None
    errors: List[str] = field(default_factory=list)


class BatchRunner(BasePipelineInterface):
    """Прогон пайплайна на списке файлов."""

    def __init__(
        self,
        file_paths: List[str],
        template_type: TemplateType = TemplateType.IT_PROJECT,
        language: str = "ru",
        output_dir: str = "output",
    ):
        self.file_paths = file_paths
        self.template_type = template_type
        self.language = language
        self.output_dir = output_dir

    async def get_sources(self) -> List[DataSource]:
        sources = []
        for fp in self.file_paths:
            try:
                sources.append(_load_source_from_file(fp))
                logger.info(f"  Загружен: {fp}")
            except Exception as e:
                logger.error(f"  Ошибка загрузки {fp}: {e}")
        return sources

    async def run(self) -> TZResult:
        print("=" * 60)
        print("📂 BATCH-ПРОГОН ПАЙПЛАЙНА")
        print("=" * 60)
        print(f"  Файлов: {len(self.file_paths)}")
        print(f"  Шаблон:  {self.template_type.value}")
        print(f"  Язык:    {self.language}\n")

        start = time.time()

        # Загрузка
        sources = await self.get_sources()
        if not sources:
            print("❌ Не удалось загрузить ни одного файла.")
            return TZResult(template_type=self.template_type, template_data=None)

        # Прогон
        pipeline = TZPipeline(
            template_type=self.template_type,
            language=self.language,
        )
        result = await pipeline.run_full(sources)
        duration = time.time() - start

        if result.template_data is None:
            print("\n❌ Не удалось извлечь данные из источников.")
            print(f"  Время: {duration:.1f}s")
            return result

        # Сохранение
        await self.present_result(result)

        graph = pipeline.unified_graph
        batch_result = BatchResult(
            sources_count=len(sources),
            subgraphs_count=len(pipeline.subgraphs),
            nodes_count=len(graph.nodes) if graph else 0,
            edges_count=len(graph.edges) if graph else 0,
            completeness=result.validation.completeness_percent,
            gaps_count=len(result.validation.gaps),
            conflicts_count=len(result.validation.conflicts),
            duration_sec=duration,
            tz_result=result,
        )

        print(f"\n  Время:        {duration:.1f}s")
        print(f"  Подграфов:    {batch_result.subgraphs_count}")
        print(f"  Узлов/связей: {batch_result.nodes_count}/{batch_result.edges_count}")
        print(f"  Заполненность: {batch_result.completeness:.1f}%")
        print(f"  Пробелов:     {batch_result.gaps_count}")

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
        print(f"\n🎉 ТЗ сохранено: {output_path}")
        print("=" * 60)
