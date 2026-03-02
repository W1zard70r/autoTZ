import asyncio
import numpy as np
import networkx as nx
import community as community_louvain
from datetime import datetime, timedelta
from typing import List, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings

MAX_CHARS_PER_WINDOW = 6000
OVERLAP_MESSAGES = 4
SEMANTIC_THRESHOLD = 0.65
LOOKBACK_WINDOW = 20
EMBEDDING_BATCH_SIZE = 20
EMBEDDING_DELAY = 1.0


def parse_date(date_str: str) -> datetime:
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.now()


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    v1, v2 = np.array(v1), np.array(v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0


async def asplit_chat_into_semantic_threads(
    messages: List[dict],
) -> List[Tuple[str, List[dict]]]:
    valid_msgs =[m for m in messages if m.get("type") == "message" and m.get("text")]
    if not valid_msgs:
        return[]

    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    texts_to_embed =[str(m.get("text", "")).strip() or "empty" for m in valid_msgs]

    embeddings =[]
    for i in range(0, len(texts_to_embed), EMBEDDING_BATCH_SIZE):
        batch = texts_to_embed[i : i + EMBEDDING_BATCH_SIZE]
        try:
            batch_result = await embeddings_model.aembed_documents(batch)
            embeddings.extend(batch_result)
        except Exception:
            embeddings.extend([[0.0] * 768] * len(batch))
        await asyncio.sleep(EMBEDDING_DELAY)

    G = nx.Graph()
    for i, msg in enumerate(valid_msgs):
        G.add_node(
            msg["id"],
            msg=msg,
            vec=embeddings[i],
            time=parse_date(msg.get("date", "")),
        )

    for i, msg in enumerate(valid_msgs):
        reply_id = msg.get("reply_to_message_id")
        if reply_id and G.has_node(reply_id):
            # ЗАЩИТА: Прямой ответ гарантированно удержит сообщения в одном кластере
            G.add_edge(msg["id"], reply_id, weight=10.0)
            continue

        best_sim, best_target_id = 0.0, None
        for j in range(max(0, i - LOOKBACK_WINDOW), i):
            time_i = G.nodes[msg["id"]]["time"]
            time_j = G.nodes[valid_msgs[j]["id"]]["time"]
            if time_i - time_j > timedelta(hours=4):
                continue
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim > best_sim:
                best_sim, best_target_id = sim, valid_msgs[j]["id"]

        if best_sim >= SEMANTIC_THRESHOLD and best_target_id:
            G.add_edge(msg["id"], best_target_id, weight=best_sim)

    threads: List[List[dict]] =[]

    if G.number_of_edges() > 0:
        try:
            partition = community_louvain.best_partition(G)
            communities: dict[int, List[dict]] = {}
            for node_id, comm_id in partition.items():
                communities.setdefault(comm_id, []).append(G.nodes[node_id]["msg"])

            threads = list(communities.values())
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Louvain failed ({e}), fallback to connected_components"
            )
            for component in nx.connected_components(G):
                thread_msgs =[G.nodes[nid]["msg"] for nid in component]
                threads.append(thread_msgs)
    else:
        threads = [[G.nodes[nid]["msg"]] for nid in G.nodes()]

    for thread in threads:
        thread.sort(key=lambda x: parse_date(x.get("date", "")))

    threads.sort(key=lambda t: parse_date(t[0].get("date", "")))

    processed_windows: List[Tuple[str, List[dict]]] =[]

    for thread_idx, thread in enumerate(threads):
        current_window: List[dict] =[]
        current_chars = 0

        for msg in thread:
            msg_len = len(str(msg.get("text", "")))
            if (
                current_chars + msg_len > MAX_CHARS_PER_WINDOW
                and len(current_window) > OVERLAP_MESSAGES
            ):
                ref = (
                    f"thread_{thread_idx}_msg_"
                    f"{current_window[0]['id']}_to_{current_window[-1]['id']}"
                )
                processed_windows.append((ref, current_window))
                current_window = current_window[-OVERLAP_MESSAGES:]
                current_chars = sum(len(str(m.get("text", ""))) for m in current_window)

            current_window.append(msg)
            current_chars += msg_len

        if current_window:
            ref = (
                f"thread_{thread_idx}_msg_"
                f"{current_window[0]['id']}_to_{current_window[-1]['id']}"
            )
            processed_windows.append((ref, current_window))

    return processed_windows