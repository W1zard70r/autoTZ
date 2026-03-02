import os
import logging
from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import google.api_core.exceptions

load_dotenv()
logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")

def get_llm_client(model_name: str, temperature: float = 0.1):
    """
    Фабрика клиентов. Убраны лишние параметры транспорта, 
    добавлены таймауты.
    """
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
            max_retries=1, # Ретраи делаем через tenacity снаружи
            request_timeout=60
        )
    else:
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
            max_retries=1,
            request_timeout=60
        )

T = TypeVar("T", bound=BaseModel)

# === НАСТРОЙКИ ПОВТОРОВ (RETRY) ===
# Google Free Tier банит на ~60 секунд при превышении лимитов.
# Ждем экспоненциально: 4с -> 8с -> 16с ... до 120с.
GLOBAL_RETRY_CONFIG = {
    "stop": stop_after_attempt(12), 
    "wait": wait_exponential(multiplier=2, min=4, max=120),
    "retry": retry_if_exception_type((
        google.api_core.exceptions.ResourceExhausted, 
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.GoogleAPICallError,
        TimeoutError,
        ConnectionError
    )),
    "before_sleep": before_sleep_log(logger, logging.WARNING)
}

@retry(**GLOBAL_RETRY_CONFIG)
async def acall_llm_json(schema: Type[T], prompt: str, data: str = "", model_name: str = DEFAULT_MODEL) -> T:
    try:
        llm = get_llm_client(model_name=model_name, temperature=0.0)
        llm_structured = llm.with_structured_output(schema)

        full_prompt = prompt
        if data:
            full_prompt += f"\n\n--- ВХОДНЫЕ ДАННЫЕ ---\n{data}"

        return await llm_structured.ainvoke(full_prompt)
    except Exception as e:
        logger.warning(f"⚠️ Сбой LLM JSON (будет повтор): {str(e)[:200]}")
        raise e

@retry(**GLOBAL_RETRY_CONFIG)
async def acall_llm_text(prompt: str, data: str = "", model_name: str = DEFAULT_MODEL) -> str:
    try:
        llm = get_llm_client(model_name=model_name, temperature=0.2)

        full_prompt = prompt
        if data:
            full_prompt += f"\n\n--- ВХОДНЫЕ ДАННЫЕ ---\n{data}"

        result = await llm.ainvoke(full_prompt)
        return result.content
    except Exception as e:
        logger.warning(f"⚠️ Сбой LLM Text (будет повтор): {str(e)[:200]}")
        raise e