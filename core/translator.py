"""Слой перевода: при необходимости переводит готовый текст ТЗ между языками.

Основной подход — генерация сразу на целевом языке (ru/en) через системный промпт компилятора.
Этот модуль используется только для случаев, когда нужен перевод уже готового документа.
"""
import logging
from utils.llm_client import acall_llm_text

logger = logging.getLogger(__name__)

TRANSLATE_TO_RU_SYSTEM = """Ты профессиональный технический переводчик.
Переведи текст технического задания на русский язык.
Сохраняй Markdown-форматирование, структуру разделов и техническую терминологию."""

TRANSLATE_TO_EN_SYSTEM = """You are a professional technical translator.
Translate the technical specification text to English.
Preserve Markdown formatting, section structure, and technical terminology."""


async def translate_markdown(text: str, target_language: str = "ru") -> str:
    """Переводит готовый Markdown-документ на целевой язык."""
    if not text.strip():
        return text

    if target_language == "ru":
        system = TRANSLATE_TO_RU_SYSTEM
        prompt = "Переведи следующий документ на русский язык:\n\n" + text
    else:
        system = TRANSLATE_TO_EN_SYSTEM
        prompt = "Translate the following document to English:\n\n" + text

    try:
        result = await acall_llm_text(prompt=prompt, system=system)
        return result
    except Exception as e:
        logger.error(f"Ошибка перевода: {e}")
        return text
