import os
import logging
from typing import Type, TypeVar
from itertools import cycle

from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
import google.genai.errors 
import google.api_core.exceptions

load_dotenv()
logger = logging.getLogger(__name__)

class KeyManager:
    def __init__(self):
        keys_str = os.getenv("GOOGLE_API_KEYS", "")
        if not keys_str:
            self.keys = [os.getenv("GOOGLE_API_KEY", "DUMMY")]
        else:
            self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        self._iterator = cycle(self.keys)
        self.current_key = next(self._iterator)

    def next_key(self):
        prev_key = self.current_key
        self.current_key = next(self._iterator)
        if prev_key != self.current_key:
            logger.info(f"🔄 Ротация ключа: ...{prev_key[-4:]} -> ...{self.current_key[-4:]}")
        return self.current_key

key_manager = KeyManager()
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")

def get_llm_client(model_name: str, temperature: float = 0.1):
    api_key = key_manager.next_key()
    # УБРАНЫ safety_settings, чтобы избежать 400 ошибки на модели 2.5
    return ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=temperature, 
        google_api_key=api_key,
        max_retries=1, 
        request_timeout=60
    )

def get_embedding_client():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=key_manager.next_key()
    )

T = TypeVar("T", bound=BaseModel)

def is_retryable(exception):
    msg = str(exception).upper()
    # Ошибку 400 (неверные аргументы) НЕ ретраим
    if "400" in msg and "INVALID_ARGUMENT" in msg:
        return False
    # Ретраим лимиты (429), сервера (500, 503) и ошибки LangChain
    retryable_errors = (
        google.genai.errors.ServerError,
        google.genai.errors.ClientError,
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        ChatGoogleGenerativeAIError,
        TimeoutError
    )
    return isinstance(exception, retryable_errors) or "429" in msg or "503" in msg

GLOBAL_RETRY_CONFIG = {
    "stop": stop_after_attempt(20), 
    "wait": wait_exponential(multiplier=1, min=2, max=10),
    "retry": retry_if_exception(is_retryable),
    "before_sleep": lambda rs: logger.warning(f"🛑 Сбой API (Попытка {rs.attempt_number}). Смена ключа...")
}

@retry(**GLOBAL_RETRY_CONFIG)
async def acall_llm_json(schema: Type[T], prompt: str, data: str = "", model_name: str = DEFAULT_MODEL) -> T:
    llm = get_llm_client(model_name=model_name, temperature=0.0)
    llm_structured = llm.with_structured_output(schema)
    full_prompt = f"{prompt}\n\n--- ДАННЫЕ ---\n{data}" if data else prompt
    return await llm_structured.ainvoke(full_prompt)

@retry(**GLOBAL_RETRY_CONFIG)
async def acall_llm_text(prompt: str, data: str = "", model_name: str = DEFAULT_MODEL) -> str:
    llm = get_llm_client(model_name=model_name, temperature=0.2)
    full_prompt = f"{prompt}\n\n--- ДАННЫЕ ---\n{data}" if data else prompt
    result = await llm.ainvoke(full_prompt)
    return result.content