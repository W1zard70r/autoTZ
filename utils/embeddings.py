import asyncio
import logging
from typing import List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.api_core.exceptions

logger = logging.getLogger(__name__)

# Настройки повторов специально для Google Embeddings API
EMBEDDING_RETRY_CONFIG = {
    "stop": stop_after_attempt(6),
    "wait": wait_exponential(multiplier=2, min=2, max=60),
    "retry": retry_if_exception_type((
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.GoogleAPICallError,
        TimeoutError,
        ConnectionError
    )),
    "before_sleep": before_sleep_log(logger, logging.WARNING)
}

@retry(**EMBEDDING_RETRY_CONFIG)
async def _aembed_batch_with_retry(model: GoogleGenerativeAIEmbeddings, batch: List[str]) -> List[List[float]]:
    """Асинхронно получает эмбеддинги для одного батча с автоматическими ретраями."""
    return await model.aembed_documents(batch)

async def aget_embeddings_safe(texts: List[str], batch_size: int = 20, delay: float = 0.5) -> List[List[float]]:
    """
    Разбивает тексты на батчи и безопасно получает эмбеддинги.
    В случае абсолютного провала возвращает безопасный "шумовой" вектор,
    чтобы не сломать математику (косинусное сходство).
    """
    if not texts:
        return[]

    model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    embeddings: List[List[float]] =[]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            batch_result = await _aembed_batch_with_retry(model, batch)
            embeddings.extend(batch_result)
        except Exception as e:
            logger.error(f"❌ ПРОВАЛ эмбеддингов (батч {i}, {len(batch)} текстов) после всех попыток: {e}")
            embeddings.extend([[1e-5] * 768] * len(batch))

        if i + batch_size < len(texts):
            await asyncio.sleep(delay)

    return embeddings