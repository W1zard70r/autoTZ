import os
import time
from typing import Type, TypeVar, Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
# Оставляем ТВОЮ модель по умолчанию (если в .env не указано иное)
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
client = None

if api_key:
    # Инициализация клиента как в твоем рабочем тесте
    client = genai.Client(api_key=api_key)

T = TypeVar("T", bound=BaseModel)

def call_llm_json(
    schema: Type[T], 
    prompt: str, 
    data: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL 
) -> T:
    if not client:
        print("⚠️ [MOCK] Нет ключа API. Возвращаю пустой объект.")
        return _create_empty_model(schema)

    full_prompt = prompt
    if data:
        data_str = data.model_dump_json() if hasattr(data, "model_dump_json") else str(data)
        full_prompt += f"\n\nDATA CONTEXT:\n{data_str}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                )
            )
            
            if response.parsed:
                return response.parsed
            return schema.model_validate_json(response.text)

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries - 1:
                    print(f"  ⚠️ Лимит API (15 запр/мин). Ждем 15 сек... (Попытка {attempt + 1}/{max_retries})")
                    time.sleep(15)
                    continue
            print(f"❌ Gemini Error ({model_name}): {e}")
            return _create_empty_model(schema)

def call_llm_text(
    prompt: str, 
    data: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL
) -> str:
    if not client:
        return "MOCK TEXT GENERATED"

    full_prompt = prompt
    if data:
        full_prompt += f"\n\nCONTEXT:\n{str(data)}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt
            )
            return response.text
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries - 1:
                    print(f"  ⚠️ Лимит API (15 запр/мин). Ждем 15 сек... (Попытка {attempt + 1}/{max_retries})")
                    time.sleep(15)
                    continue
            print(f"❌ Gemini Text Error: {e}")
            return ""

def _create_empty_model(schema: Type[T]) -> T:
    try:
        return schema.model_construct()
    except:
        return schema()