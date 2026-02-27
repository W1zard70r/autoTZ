import os
import asyncio
import numpy as np
import networkx as nx
import community as community_louvain
from datetime import datetime, timedelta
from typing import List, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_CHARS_PER_WINDOW = 12000
OVERLAP_MESSAGES = 6
SEMANTIC_THRESHOLD = 0.65
LOOKBACK_WINDOW = 30

# Настройки Rate Limiter
EMBEDDING_BATCH_SIZE = 20  # Количество текстов за 1 запрос
EMBEDDING_DELAY = 10  # Задержка в секундах (1.0 сек = макс 60 RPM, безопасно для лимита 100)


def parse_date(date_str: str) -> datetime:
    try:
        # Обработка разных форматов дат, включая те, что с 'Z'
        return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
    except:
        return datetime.now()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


async def asplit_chat_into_semantic_threads(messages: List[dict]) -> List[Tuple[str, List[dict], List[dict]]]:
    """
    Разбивает чат на семантические треды с использованием графов и эмбеддингов.
    Включает защиту от Rate Limit API Google.
    """
    valid_msgs = [m for m in messages if m.get("type") == "message" and m.get("text")]
    if not valid_msgs: return []

    # 1. Инициализация модели
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Подготовка текстов (защита от пустых строк, которые могут вызвать ошибку API)
    texts_to_embed = [str(m.get("text", "")).strip() or "empty" for m in valid_msgs]
    print(f"🧠 Векторизация {len(texts_to_embed)} сообщений (с учетом Rate Limit)...")

    # --- RATE LIMITING LOGIC START ---
    embeddings = []
    total_batches = (len(texts_to_embed) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for i in range(0, len(texts_to_embed), EMBEDDING_BATCH_SIZE):
        batch = texts_to_embed[i: i + EMBEDDING_BATCH_SIZE]
        current_batch_num = (i // EMBEDDING_BATCH_SIZE) + 1

        try:
            # print(f"   Batch {current_batch_num}/{total_batches}...") # Раскомментировать для дебага
            batch_result = await embeddings_model.aembed_documents(batch)
            embeddings.extend(batch_result)
        except Exception as e:
            print(f"⚠️ Ошибка эмбеддинга на батче {current_batch_num}: {e}")
            # Заполняем нулями, чтобы сохранить индексы и не сломать граф
            embeddings.extend([[0.0] * 768] * len(batch))

        # Ждем перед следующим запросом, чтобы не превысить 100 RPM
        # 60 RPM = 1 запрос в секунду. Это безопасно.
        await asyncio.sleep(EMBEDDING_DELAY)
    # --- RATE LIMITING LOGIC END ---

    # 2. Строим узлы графа
    G = nx.Graph()
    for i, msg in enumerate(valid_msgs):
        G.add_node(msg["id"], msg=msg, vec=embeddings[i], time=parse_date(msg.get("date", "")))

    # 3. Устанавливаем связи (Явные + Семантические)
    print("🔗 Построение связей...")
    for i, msg in enumerate(valid_msgs):
        reply_id = msg.get("reply_to_message_id")

        # А. Явная связь (Reply)
        if reply_id and G.has_node(reply_id):
            G.add_edge(msg["id"], reply_id, type="reply")
            continue

        # Б. Неявная связь (Семантика)
        best_sim = 0.0
        best_target_id = None
        start_idx = max(0, i - LOOKBACK_WINDOW)

        for j in range(start_idx, i):
            # Проверка времени (не связываем сообщения с разницей > 4 часов)
            time_diff = G.nodes[msg["id"]]["time"] - G.nodes[valid_msgs[j]["id"]]["time"]
            if time_diff > timedelta(hours=4):
                continue

            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > best_sim:
                best_sim = sim
                best_target_id = valid_msgs[j]["id"]

        if best_sim >= SEMANTIC_THRESHOLD and best_target_id:
            G.add_edge(msg["id"], best_target_id, type="semantic", weight=best_sim)

    # 4. Community Detection (Louvain)
    # ВАЖНО: Запускаем ПОСЛЕ добавления ребер, иначе граф пустой
    if G.number_of_edges() > 0:
        try:
            print("🔍 Выполняем Community Detection (Louvain)...")
            partition = community_louvain.best_partition(G)  # Louvain работает с ненаправленными графами
            for node_id, comm_id in partition.items():
                G.nodes[node_id]["community"] = comm_id
        except Exception as e:
            print(f"⚠️ Ошибка Louvain: {e}. Пропускаем этап сообществ.")

    # 5. Извлекаем треды (Связные компоненты)
    threads = []
    for component in nx.connected_components(G):
        thread_msgs = [G.nodes[node_id]["msg"] for node_id in component]
        thread_msgs.sort(key=lambda x: parse_date(x.get("date", "")))
        threads.append(thread_msgs)

    threads.sort(key=lambda t: parse_date(t[0].get("date", "")))

    # 6. Нарезка на окна
    processed_windows = []
    for thread_idx, thread in enumerate(threads):
        current_window = []
        current_chars = 0

        for msg in thread:
            msg_len = len(str(msg.get("text", "")))

            if current_chars + msg_len > MAX_CHARS_PER_WINDOW and len(current_window) > OVERLAP_MESSAGES:
                start_id = current_window[0]["id"]
                end_id = current_window[-1]["id"]
                window_ref = f"thread_{thread_idx}_msg_{start_id}_to_{end_id}"

                processed_windows.append((window_ref, current_window, []))

                current_window = current_window[-OVERLAP_MESSAGES:]
                current_chars = sum(len(str(m.get("text", ""))) for m in current_window)

            current_window.append(msg)
            current_chars += msg_len

        if current_window:
            start_id = current_window[0]["id"]
            end_id = current_window[-1]["id"]
            window_ref = f"thread_{thread_idx}_msg_{start_id}_to_{end_id}"
            processed_windows.append((window_ref, current_window, []))

    print(f"✅ Чат разбит на {len(processed_windows)} семантических окон.")
    return processed_windows


def split_text_into_chunks(text: str) -> List[Tuple[str, str]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    chunks = splitter.split_text(text)
    return [(f"chunk_{i + 1}", chunk) for i, chunk in enumerate(chunks)]