"""Backend-адаптер для подключения к серверу через taskiq-воркеры.

Ключевые отличия от консольного интерфейса:
- Принимает ``notify`` — callable для публикации статуса в брокер.
- Сериализует / десериализует состояние пайплайна, чтобы его можно было
  восстанавливать между вызовами taskiq-тасок.
- Методы ``save`` / ``load`` позволяют сохранять прогресс в Redis/DB.

Пример использования из taskiq-воркера::

    # Таска 1: извлечение
    adapter = BackendAdapter(template_type=..., notify=publish_status)
    await adapter.step_extract(sources)
    state = adapter.save()
    await redis.set(f"pipeline:{task_id}", json.dumps(state))

    # Таска 2: слияние (после получения следующего события)
    raw = json.loads(await redis.get(f"pipeline:{task_id}"))
    adapter = BackendAdapter.load(raw, notify=publish_status)
    conflicts = await adapter.step_merge_and_detect()
    state = adapter.save()
    ...

    # Таска 3: пользователь решил конфликты
    adapter = BackendAdapter.load(raw, notify=publish_status)
    adapter.step_apply_resolutions(resolutions)
    result = await adapter.step_finalize_and_compile()
"""
import json
import logging
from typing import List, Dict, Optional, Callable, Awaitable, Any

from schemas.document import DataSource
from schemas.enums import TemplateType
from schemas.graph import (
    ExtractedKnowledge, UnifiedGraph,
    DetectedConflict, ConflictResolution,
)
from schemas.templates.base import TZResult, FieldGap
from core.pipeline import TZPipeline
from interfaces.base import BasePipelineInterface

logger = logging.getLogger(__name__)

# Тип notify-коллбэка: async (status: str, payload: dict) -> None
NotifyFn = Callable[[str, dict], Awaitable[None]]


async def _noop_notify(status: str, payload: dict) -> None:
    """Заглушка, если notify не предоставлен."""
    pass


class BackendAdapter(BasePipelineInterface):
    """Адаптер для интеграции с бэкендом (taskiq-воркеры).

    Вызывает ``notify`` на ключевых этапах для публикации статуса
    в брокер сообщений, чтобы фронтенд мог показывать прогресс.
    """

    def __init__(
        self,
        template_type: TemplateType = TemplateType.IT_PROJECT,
        language: str = "ru",
        notify: Optional[NotifyFn] = None,
    ):
        self.pipeline = TZPipeline(
            template_type=template_type,
            language=language,
        )
        self.notify: NotifyFn = notify or _noop_notify

    # ── Пошаговые методы (для taskiq-тасок) ──────────────

    async def step_extract(self, sources: List[DataSource]) -> List[ExtractedKnowledge]:
        await self.notify("extracting", {"total_sources": len(sources)})
        result = await self.pipeline.extract(sources)
        await self.notify("extracted", {"subgraphs_count": len(result)})
        return result

    async def step_merge_and_detect(self) -> List[DetectedConflict]:
        await self.notify("merging", {})
        await self.pipeline.merge()
        await self.notify("detecting_conflicts", {})
        conflicts = await self.pipeline.detect_conflicts()
        await self.notify("conflicts_ready", {
            "count": len(conflicts),
            "conflicts": [c.model_dump(mode="json") for c in conflicts],
        })
        return conflicts

    def step_apply_resolutions(self, resolutions: List[ConflictResolution]) -> None:
        self.pipeline.apply_resolutions(resolutions)

    async def step_finalize_and_compile(
        self,
        user_answers: Optional[Dict[str, str]] = None,
    ) -> TZResult:
        if user_answers:
            self.pipeline.add_user_answers(user_answers)

        await self.notify("finalizing", {})
        await self.pipeline.finalize_graph()

        await self.notify("compiling", {})
        result = await self.pipeline.compile()

        await self.notify("done", {
            "completeness": result.validation.completeness_percent,
            "gaps": len(result.validation.gaps),
        })
        return result

    async def step_recompile(
        self,
        user_answers: Optional[Dict[str, str]] = None,
    ) -> TZResult:
        if user_answers:
            self.pipeline.add_user_answers(user_answers)
        await self.notify("recompiling", {})
        result = await self.pipeline.recompile()
        await self.notify("done", {
            "completeness": result.validation.completeness_percent,
            "gaps": len(result.validation.gaps),
        })
        return result

    async def generate_conflict_variants(
        self, conflict: DetectedConflict,
    ) -> List[str]:
        return await self.pipeline.generate_conflict_variants(conflict)

    def get_gaps(self) -> List[FieldGap]:
        return self.pipeline.get_gaps()

    # ── Сериализация / десериализация ─────────────────────

    def save(self) -> dict:
        """Сохраняет состояние пайплайна в JSON-совместимый dict."""
        return self.pipeline.save_state()

    @classmethod
    def load(cls, state: dict, notify: Optional[NotifyFn] = None) -> "BackendAdapter":
        """Восстанавливает адаптер из сериализованного состояния."""
        adapter = cls.__new__(cls)
        adapter.pipeline = TZPipeline.load_state(state)
        adapter.notify = notify or _noop_notify
        return adapter

    # ── Интерфейс BasePipelineInterface ──────────────────

    async def get_sources(self) -> List[DataSource]:
        raise NotImplementedError("Backend-адаптер получает источники извне.")

    async def run(self) -> TZResult:
        raise NotImplementedError(
            "Backend-адаптер работает пошагово. "
            "Используйте step_extract() → step_merge_and_detect() → ... → step_finalize_and_compile()."
        )

    async def present_result(self, result: TZResult) -> None:
        await self.notify("result", {
            "completeness": result.validation.completeness_percent,
            "markdown_length": len(result.markdown),
        })
