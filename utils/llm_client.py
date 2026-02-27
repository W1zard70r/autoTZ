import os
import asyncio
import logging
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
logger = logging.getLogger(__name__)

api_key = os.getenv("GOOGLE_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Проверка ключа
if not api_key:
    logger.warning("⚠️ GOOGLE_API_KEY не найден в .env. Будет ошибка при вызовах LLM.")

llm_text = ChatGoogleGenerativeAI(model=DEFAULT_MODEL, temperature=0.1, max_retries=3)

T = TypeVar("T", bound=BaseModel)

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60))
async def acall_llm_json(schema: Type[T], prompt: str, data: str = "", model_name: str = DEFAULT_MODEL) -> T:
    """Асинхронный вызов LLM со строгим возвратом Pydantic схемы"""
    try:
        llm_structured = ChatGoogleGenerativeAI(model=model_name, temperature=0.0).with_structured_output(schema)
        full_prompt = prompt
        if data:
            full_prompt += f"\n\n--- ВХОДНЫЕ ДАННЫЕ ---\n{data}"
        
        result = await llm_structured.ainvoke(full_prompt)
        return result
    except Exception as e:
        logger.error(f"❌ Ошибка LLM JSON: {e}")
        raise e

@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60))
async def acall_llm_text(prompt: str, data: str = "", model_name: str = DEFAULT_MODEL) -> str:
    """Асинхронный вызов LLM для получения обычного текста"""
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        full_prompt = prompt
        if data:
            full_prompt += f"\n\n--- ВХОДНЫЕ ДАННЫЕ ---\n{data}"
            
        result = await llm.ainvoke(full_prompt)
        return result.content
    except Exception as e:
        logger.error(f"❌ Ошибка LLM Text: {e}")
        raise e