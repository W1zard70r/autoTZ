"""Windowing: split chats into semantic windows.

Uses reply chains + time proximity instead of embeddings.
Faster, cheaper, does not require Embedding API.
"""
import re
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

MAX_CHARS_PER_WINDOW = 6000
OVERLAP_MESSAGES = 4
OVERLAP_PARAGRAPHS = 2
TIME_GAP_MINUTES = 30


def _parse_date(date_str: str) -> datetime:
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.now()


def split_chat_into_windows(
    messages: List[dict],
) -> List[Tuple[str, List[dict]]]:
    valid_msgs = [m for m in messages if m.get("type") == "message" and m.get("text")]
    if not valid_msgs:
        return []

    valid_msgs.sort(key=lambda m: _parse_date(m.get("date", "")))

    thread_of: Dict[int | str, int] = {}
    threads: Dict[int, List[dict]] = {}
    next_tid = 0

    for msg in valid_msgs:
        msg_id = msg["id"]
        reply_id = msg.get("reply_to_message_id")

        if reply_id and reply_id in thread_of:
            tid = thread_of[reply_id]
        else:
            tid = None
            msg_time = _parse_date(msg.get("date", ""))
            for prev_msg in reversed(valid_msgs):
                if prev_msg["id"] == msg_id:
                    continue
                if prev_msg["id"] not in thread_of:
                    continue
                prev_time = _parse_date(prev_msg.get("date", ""))
                if msg_time - prev_time < timedelta(minutes=TIME_GAP_MINUTES):
                    tid = thread_of[prev_msg["id"]]
                    break
                if prev_time < msg_time - timedelta(minutes=TIME_GAP_MINUTES):
                    break

            if tid is None:
                tid = next_tid
                next_tid += 1

        thread_of[msg_id] = tid
        threads.setdefault(tid, []).append(msg)

    for tid in threads:
        threads[tid].sort(key=lambda m: _parse_date(m.get("date", "")))

    sorted_threads = sorted(
        threads.values(),
        key=lambda t: _parse_date(t[0].get("date", "")),
    )

    windows: List[Tuple[str, List[dict]]] = []

    for thread_idx, thread in enumerate(sorted_threads):
        current_window: List[dict] = []
        current_chars = 0

        for msg in thread:
            msg_len = len(str(msg.get("text", "")))
            if (
                current_chars + msg_len > MAX_CHARS_PER_WINDOW
                and len(current_window) > OVERLAP_MESSAGES
            ):
                ref = f"thread_{thread_idx}_msg_{current_window[0]['id']}_to_{current_window[-1]['id']}"
                windows.append((ref, list(current_window)))
                current_window = current_window[-OVERLAP_MESSAGES:]
                current_chars = sum(len(str(m.get("text", ""))) for m in current_window)

            current_window.append(msg)
            current_chars += msg_len

        if current_window:
            ref = f"thread_{thread_idx}_msg_{current_window[0]['id']}_to_{current_window[-1]['id']}"
            windows.append((ref, list(current_window)))

    return windows


def split_text_into_windows(
    text: str,
    max_chars: int = MAX_CHARS_PER_WINDOW,
    overlap: int = OVERLAP_PARAGRAPHS,
) -> List[Tuple[str, str]]:
    """Split plain text into overlapping windows by paragraph boundaries.

    Returns list of (window_ref, text_chunk).
    """
    paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        return []

    # If the whole text fits in one window, return as-is
    if len(text) <= max_chars:
        return [("chunk_0", text)]

    windows: List[Tuple[str, str]] = []
    current: List[str] = []
    current_chars = 0

    for para in paragraphs:
        para_len = len(para)

        # If a single paragraph exceeds max_chars, split it by sentences
        if para_len > max_chars:
            if current:
                windows.append((f"chunk_{len(windows)}", "\n\n".join(current)))
                current = current[-overlap:] if overlap else []
                current_chars = sum(len(p) for p in current)

            sentences = re.split(r"(?<=[.!?])\s+", para)
            sent_buf: List[str] = []
            sent_chars = 0
            for sent in sentences:
                if sent_chars + len(sent) > max_chars and sent_buf:
                    windows.append((f"chunk_{len(windows)}", " ".join(sent_buf)))
                    sent_buf = sent_buf[-(overlap * 2):] if overlap else []
                    sent_chars = sum(len(s) for s in sent_buf)
                sent_buf.append(sent)
                sent_chars += len(sent)
            if sent_buf:
                current.append(" ".join(sent_buf))
                current_chars += sent_chars
            continue

        if current_chars + para_len > max_chars and current:
            windows.append((f"chunk_{len(windows)}", "\n\n".join(current)))
            current = current[-overlap:] if overlap else []
            current_chars = sum(len(p) for p in current)

        current.append(para)
        current_chars += para_len

    if current:
        windows.append((f"chunk_{len(windows)}", "\n\n".join(current)))

    return windows
