import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from models import ProjectGlossary, WindowExtractionResult
from global_glossary import GlobalGlossary

logger = logging.getLogger(__name__)

class AsyncGraphExtractor:
    def __init__(self, global_glossary: GlobalGlossary):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
        self.glossary_llm = self.llm.with_structured_output(ProjectGlossary)
        self.graph_llm = self.llm.with_structured_output(WindowExtractionResult)
        self.global_glossary = global_glossary

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=60))
    async def agenerate_glossary(self, text_content: str) -> ProjectGlossary:
        """ПРОХОД 1: Глоссарий"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты аналитик. Найди ВСЕ ключевые сущности (Люди, Компоненты, Задачи, Требования).
Верни их в snake_case ID. Используй только предоставленные типы."""),
            ("user", "{text}")
        ])
        chain = prompt | self.glossary_llm
        return await chain.ainvoke({"text": text_content})

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=4, max=60))
    async def aextract_graph(self, text_content: str, glossary: ProjectGlossary, prev_summary: str) -> WindowExtractionResult:
        """ПРОХОД 2: Извлечение графа"""
        glossary_str = "\n".join([f"- {e.id} ({e.label.value}): {e.name} — {e.description}" for e in glossary.entities])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Ты Senior Data Architect. Извлеки граф знаний.
СТРОГИЕ ПРАВИЛА:
1. Если сущность из текста АБСОЛЮТНО СОВПАДАЕТ по смыслу с сущностью из глоссария — используй старый ID.
2. Если сущность похожа, но является другой (например, Redis vs Postgres, Индекс vs База) — СОЗДАЙ НОВЫЙ ID.
3. Учитывай ПРЕДЫДУЩИЙ КОНТЕКСТ для местоимений.
4. [FLAG: CONFIRMATION] = AGREES_WITH или ASSIGNED_TO.
5. Для каждой связи добавляй evidence — точную цитату из текста.


ГЛОССАРИЙ:
{glossary}

ПРЕДЫДУЩИЙ КОНТЕКСТ:
{prev_summary}"""),
            ("user", "Текст:\n{text}")
        ])
        chain = prompt | self.graph_llm
        return await chain.ainvoke({
            "glossary": glossary_str or "Пусто",
            "prev_summary": prev_summary or "Начало диалога.",
            "text": text_content
        })