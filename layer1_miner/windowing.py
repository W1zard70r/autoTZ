import asyncio
import logging
import numpy as np
import networkx as nx
import community as community_louvain
from datetime import datetime, timedelta
from typing import List, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Импортируем наш менеджер ключей
from utils.llm_client import key_manager

logger = logging.getLogger(__name__)

MAX_CHARS_PER_WINDOW = 6000
OVERLAP_MESSAGES = 4
SEMANTIC_THRESHOLD = 0.65
LOOKBACK_WINDOW = 20
EMBEDDING_BATCH_SIZE = 20
EMBEDDING_DELAY = 1.0


def parse_date(date_str: str) -> datetime:
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except:
        return datetime.now()


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    v1, v2 = np.array(v1), np.array(v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0


async def get_embeddings_with_retry(model, batch: List[str], retries=6) -> List[List[float]]:
    """Получение эмбеддингов с ротацией ключей. БЕЗ заглушек."""
    for attempt in range(retries):
        try:
            # Получаем ключ ИМЕННО из модели, т.к. мы его там обновляем
            return await model.aembed_documents(batch)
        except Exception as e:
            error_str = str(e)
            
            # Если 404/400 - это ошибка модели/запроса, ротация не поможет
            if "404" in error_str or "400" in error_str:
                # Но мы попробуем пробросить выше, чтобы юзер увидел, что модель не та
                logger.error(f"❌ Модель не найдена или неверный запрос: {e}")
                raise e
            
            # Если 429 (лимиты) или 503 (сервер) - пробуем другой ключ
            if "429" in error_str or "503" in error_str or "ResourceExhausted" in error_str:
                logger.warning(f"⚠️ Embeddings Rate Limit (Попытка {attempt+1}/{retries}). Ротируем ключ...")
                new_key = key_manager.rotate()
                # ВАЖНО: Обновляем ключ внутри объекта LangChain вручную
                model.google_api_key = new_key 
                model.client = None # Сброс клиента, чтобы пересоздался с новым ключом
                await asyncio.sleep(2)
            else:
                # Другие ошибки (сеть и т.д.) тоже пробуем ретраить
                await asyncio.sleep(1)
                if attempt == retries - 1:
                    raise e
                
    raise RuntimeError("Не удалось получить эмбеддинги после всех попыток.")


# Нужно обновить импорт в начале файла:
# from utils.llm_client import get_embedding_client

async def asplit_chat_into_semantic_threads(messages: List[dict]) -> List[Tuple[str, List[dict]]]:
    valid_msgs = [m for m in messages if m.get("type") == "message" and m.get("text")]
    if not valid_msgs: return []

    texts_to_embed = [str(m.get("text", "")).strip() or "empty" for m in valid_msgs]
    embeddings = []

    # Используем нашу фабрику, которая меняет ключи
    from utils.llm_client import get_embedding_client

    for i in range(0, len(texts_to_embed), EMBEDDING_BATCH_SIZE):
        batch = texts_to_embed[i: i + EMBEDDING_BATCH_SIZE]
        
        # Получаем свежего клиента с новым ключом для каждого батча
        embeddings_model = get_embedding_client()
        
        try:
            batch_result = await embeddings_model.aembed_documents(batch)
            embeddings.extend(batch_result)
        except Exception as e:
            logger.error(f"❌ Embeddings Error: {e}")
            # Пробуем еще раз с другим ключом (простая рекурсия на 1 раз)
            await asyncio.sleep(2)
            try:
                embeddings_model = get_embedding_client() # Еще раз меняем ключ
                batch_result = await embeddings_model.aembed_documents(batch)
                embeddings.extend(batch_result)
            except:
                raise RuntimeError("Не удалось получить эмбеддинги даже со сменой ключа.")
        
        await asyncio.sleep(EMBEDDING_DELAY)

    # ... далее код построения графа G без изменений ...
    G = nx.Graph()
    # ... (весь остальной код функции asplit_chat_into_semantic_threads остается таким же)
    for i, msg in enumerate(valid_msgs):
        vec = embeddings[i]
        G.add_node(msg["id"], msg=msg, vec=vec, time=parse_date(msg.get("date", "")))
    
    # ... (и так далее до return processed_windows) ...
    # Скопируйте хвост функции из предыдущего примера или оставьте как было,
    # изменилась только часть с генерацией embeddings.
    
    for i, msg in enumerate(valid_msgs):
        reply_id = msg.get("reply_to_message_id")
        if reply_id and G.has_node(reply_id):
            G.add_edge(msg["id"], reply_id, weight=1.0)
            continue
        best_sim = 0.0
        best_target_id = None
        vec_i = G.nodes[msg["id"]]["vec"]
        time_i = G.nodes[msg["id"]]["time"]
        for j in range(max(0, i - LOOKBACK_WINDOW), i):
            prev_msg_id = valid_msgs[j]["id"]
            vec_j = G.nodes[prev_msg_id]["vec"]
            time_j = G.nodes[prev_msg_id]["time"]
            if time_i - time_j > timedelta(hours=4): continue
            sim = cosine_similarity(vec_i, vec_j)
            if sim > best_sim:
                best_sim, best_target_id = sim, prev_msg_id
        if best_sim >= SEMANTIC_THRESHOLD and best_target_id:
            G.add_edge(msg["id"], best_target_id, weight=best_sim)
    
    if G.number_of_edges() > 0:
        try:
            partition = community_louvain.best_partition(G)
            for node_id, comm_id in partition.items(): G.nodes[node_id]["community"] = comm_id
        except: pass
        
    threads = []
    for component in nx.connected_components(G):
        thread_msgs = [G.nodes[node_id]["msg"] for node_id in component]
        thread_msgs.sort(key=lambda x: parse_date(x.get("date", "")))
        threads.append(thread_msgs)
    threads.sort(key=lambda t: parse_date(t[0].get("date", "")))
    
    processed_windows = []
    for thread_idx, thread in enumerate(threads):
        current_window, current_chars = [], 0
        for msg in thread:
            msg_len = len(str(msg.get("text", "")))
            if current_chars + msg_len > MAX_CHARS_PER_WINDOW and len(current_window) > OVERLAP_MESSAGES:
                ref = f"thread_{thread_idx}_msg_{current_window[0]['id']}_to_{current_window[-1]['id']}"
                processed_windows.append((ref, current_window))
                current_window = current_window[-OVERLAP_MESSAGES:]
                current_chars = sum(len(str(m.get("text", ""))) for m in current_window)
            current_window.append(msg)
            current_chars += msg_len
        if current_window:
            ref = f"thread_{thread_idx}_msg_{current_window[0]['id']}_to_{current_window[-1]['id']}"
            processed_windows.append((ref, current_window))
            
    return processed_windows