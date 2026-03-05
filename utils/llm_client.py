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
    retry_if_exception_type
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, HarmBlockThreshold, HarmCategory
from langchain_openai import ChatOpenAI
import google.api_core.exceptions

load_dotenv()
logger = logging.getLogger(__name__)

class KeyManager:
    def __init__(self):
        keys_str = os.getenv("GOOGLE_API_KEYS", "")
        if not keys_str:
            single_key = os.getenv("GOOGLE_API_KEY")
            self.keys = [single_key] if single_key else []
        else:
            self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        
        if not self.keys:
            logger.warning("⚠️ GOOGLE_API_KEYS не найдены в .env!")
            self.keys = ["DUMMY_KEY"]
            
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
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").lower()

def get_llm_client(model_name: str, temperature: float = 0.1):
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature, request_timeout=60)
    
    api_key = key_manager.next_key()
    
    # Минимальный набор фильтров для экспериментальных моделей (убирает 400 ошибку)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        safety_settings=safety_settings,
        max_retries=1,
        request_timeout=60 # Увеличили таймаут
    )

def get_embedding_client():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key_manager.next_key()
    )

T = TypeVar("T", bound=BaseModel)

GLOBAL_RETRY_CONFIG = {
    "stop": stop_after_attempt(10), 
    "wait": wait_exponential(multiplier=2, min=4, max=15),
    "retry": retry_if_exception_type((
        google.api_core.exceptions.ResourceExhausted, 
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.GoogleAPICallError,
        google.api_core.exceptions.DeadlineExceeded
        # 400 Bad Request НЕ ретраим, чтобы видеть ошибку сразу
    )),
    "before_sleep": lambda rs: logger.warning(f"🛑 Сбой API (Попытка {rs.attempt_number}). Ищем живой ключ...")
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