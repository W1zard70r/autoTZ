import asyncio
import numpy as np
import networkx as nx
import community as community_louvain
from datetime import datetime, timedelta
from typing import List, Tuple
from utils.embeddings import aget_embeddings_safe

MAX_CHARS_PER_WINDOW = 6000
OVERLAP_MESSAGES = 4
SEMANTIC_THRESHOLD = 0.65
LOOKBACK_WINDOW = 20
EMBEDDING_BATCH_SIZE = 20
EMBEDDING_DELAY = 1.0
MAX_SEMANTIC_NEIGHBORS = 3


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
    valid_msgs = [m for m in messages if m.get("type") == "message" and m.get("text")]
    if not valid_msgs:
        return []

    texts_to_embed = [str(m.get("text", "")).strip() or "empty" for m in valid_msgs]

    embeddings = await aget_embeddings_safe(
        texts=texts_to_embed,
        batch_size=EMBEDDING_BATCH_SIZE,
        delay=EMBEDDING_DELAY
    )

    G = nx.Graph()
    for i, msg in enumerate(valid_msgs):
        G.add_node(
            msg["id"],
            msg=msg,
            vec=embeddings[i],
            time=parse_date(msg.get("date", "")),
        )

    for i, msg in enumerate(valid_msgs):
        # Reply edges: add strong edge but ALSO check semantic neighbors
        reply_id = msg.get("reply_to_message_id")
        if reply_id and G.has_node(reply_id):
            G.add_edge(msg["id"], reply_id, weight=15.0)

        # Semantic edges: connect to ALL neighbors above threshold (k-NN)
        neighbors = []
        for j in range(max(0, i - LOOKBACK_WINDOW), i):
            time_i = G.nodes[msg["id"]]["time"]
            time_j = G.nodes[valid_msgs[j]["id"]]["time"]
            if time_i - time_j > timedelta(hours=4):
                continue
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if sim >= SEMANTIC_THRESHOLD:
                neighbors.append((valid_msgs[j]["id"], sim))

        # Sort by similarity, take top-k
        neighbors.sort(key=lambda x: x[1], reverse=True)
        for target_id, sim in neighbors[:MAX_SEMANTIC_NEIGHBORS]:
            if not G.has_edge(msg["id"], target_id):
                G.add_edge(msg["id"], target_id, weight=sim)

    threads: List[List[dict]] = []

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
                thread_msgs = [G.nodes[nid]["msg"] for nid in component]
                threads.append(thread_msgs)
    else:
        threads = [[G.nodes[nid]["msg"]] for nid in G.nodes()]

    for thread in threads:
        thread.sort(key=lambda x: parse_date(x.get("date", "")))

    threads.sort(key=lambda t: parse_date(t[0].get("date", "")))

    processed_windows: List[Tuple[str, List[dict]]] = []

    for thread_idx, thread in enumerate(threads):
        current_window: List[dict] = []
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
