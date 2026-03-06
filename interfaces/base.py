"""Абстрактный интерфейс для пайплайна генерации ТЗ."""
from abc import ABC, abstractmethod
from typing import List

from schemas.document import DataSource
from schemas.templates.base import TZResult


class BasePipelineInterface(ABC):
    """Базовый интерфейс. Все конкретные интерфейсы наследуют его."""

    @abstractmethod
    async def get_sources(self) -> List[DataSource]:
        """Получить входные источники данных."""

    @abstractmethod
    async def run(self) -> TZResult:
        """Запустить полный цикл генерации ТЗ."""

    @abstractmethod
    async def present_result(self, result: TZResult) -> None:
        """Отобразить результат пользователю / вызывающему коду."""
