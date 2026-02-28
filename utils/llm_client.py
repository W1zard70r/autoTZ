import os
import logging
from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

load_dotenv()
logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")

if LLM_PROVIDER == "google" and not GOOGLE_API_KEY:
    logger.warning("⚠️ GOOGLE_API_KEY не найден. Google Provider не будет работать.")
if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    logger.warning("⚠️ OPENAI_API_KEY не найден. OpenAI Provider не будет работать.")


def get_llm_client(model_name: str, temperature: float = 0.1):
    """
    Фабрика для создания клиента LLM в зависимости от LLM_PROVIDER.
    """
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            max_retries=3
        )
    else:
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=GOOGLE_API_KEY,
            max_retries=3
        )


T = TypeVar("T", bound=BaseModel)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60))
async def acall_llm_json(schema: Type[T], prompt: str, data: str = "", model_name: str = DEFAULT_MODEL) -> T:
    try:
        llm = get_llm_client(model_name=model_name, temperature=0.0)
        llm_structured = llm.with_structured_output(schema)

        full_prompt = prompt
        if data:
            full_prompt += f"\n\n--- ВХОДНЫЕ ДАННЫЕ ---\n{data}"

        return await llm_structured.ainvoke(full_prompt)
    except Exception as e:
        logger.error(f"❌ Ошибка LLM JSON ({LLM_PROVIDER}): {e}")
        raise e


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60))
async def acall_llm_text(prompt: str, data: str = "", model_name: str = DEFAULT_MODEL) -> str:
    try:
        llm = get_llm_client(model_name=model_name, temperature=0.2)

        full_prompt = prompt
        if data:
            full_prompt += f"\n\n--- ВХОДНЫЕ ДАННЫЕ ---\n{data}"

        result = await llm.ainvoke(full_prompt)
        return result.content
    except Exception as e:
        logger.error(f"❌ Ошибка LLM Text ({LLM_PROVIDER}): {e}")
        raise e