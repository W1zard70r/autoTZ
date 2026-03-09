import asyncio
import logging
from typing import List

import numpy as np
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
    "stop": stop_after_attempt(10),
    "wait": wait_exponential(multiplier=5, min=10, max=60),
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
        for _ in range(3):
            try:
                batch_result = await _aembed_batch_with_retry(model, batch)
                embeddings.extend(batch_result)
                break
            except Exception as e:
                logger.error(f"❌ ПРОВАЛ эмбеддингов (батч {i}, {len(batch)} текстов) после всех попыток: {e}")
                await asyncio.sleep(20)

        if i + batch_size < len(texts):
            await asyncio.sleep(delay)

    return embeddings


def calculate_cosine_similarity_matrix(embeddings_list: List[List[float]]) -> np.ndarray:
    """Вычисляет матрицу косинусного сходства для списка эмбеддингов."""
    if not embeddings_list:
        return np.array([])
    vecs = np.array(embeddings_list)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    vecs_normalized = vecs / norms
    return np.dot(vecs_normalized, vecs_normalized.T)