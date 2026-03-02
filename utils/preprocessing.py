import re
from typing import Tuple

CONFIRMATION_WORDS = {"ок", "да", "давайте", "плюс", "+", "ага", "согласен", "добро", "принято", "хорошо"}
REJECTION_WORDS = {"не", "нет", "не надо", "минус", "-", "отмена", "против", "не согласен"}

_VOTE_FOR_PATTERNS = [
    r"(?:я\s+)?за\s+(?P<target>[\w\s\+#\.]+)",
    r"голосую\s+за\s+(?P<target>[\w\s\+#\.]+)",
    r"поддерживаю\s+(?P<target>[\w\s\+#\.]+)",
    r"тоже\s+за\s+(?P<target>[\w\s\+#\.]+)",
]

_VOTE_AGAINST_PATTERNS = [
    r"(?:я\s+)?против\s+(?P<target>[\w\s\+#\.]+)",
    r"не\s+хочу\s+(?P<target>[\w\s\+#\.]+)",
]

def normalize_short_answers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    clean = text.lower().strip().strip(",.!?:")
    if clean in CONFIRMATION_WORDS:
        return "[FLAG: CONFIRMATION]"
    if clean in REJECTION_WORDS:
        return "[FLAG: REJECTION]"
    return text

def detect_vote(text: str) -> Tuple[str | None, str | None]:
    """Возвращает (direction, target) или (None, None)"""
    if not isinstance(text, str):
        return None, None
    lower = text.lower().strip()
    for pattern in _VOTE_FOR_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            return "for", m.group("target").strip()
    for pattern in _VOTE_AGAINST_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            return "against", m.group("target").strip()
    clean = lower.strip(",.!?:")
    if clean in CONFIRMATION_WORDS:
        return "for", None
    if clean in REJECTION_WORDS:
        return "against", None
    return None, None

def enrich_message_with_vote(msg: dict) -> dict:
    """Добавляет явный vote_flag в сообщение"""
    text = str(msg.get("text", ""))
    direction, target = detect_vote(text)
    if direction:
        target_str = target or "LAST_MENTIONED"
        msg["vote_flag"] = f"[VOTE_{direction.upper()}:{target_str}]"
    return msg

def get_clean_text(text_obj) -> str:
    if isinstance(text_obj, str):
        return text_obj
    if isinstance(text_obj, list):
        return "".join([i if isinstance(i, str) else i.get("text", "") for i in text_obj])
    return ""

def format_chat_message(msg: dict, msg_lookup: dict = None) -> str:
    date = msg.get("date", "Unknown Date")
    author = msg.get("from", msg.get("author", "Unknown"))
    text = get_clean_text(msg.get("text", ""))
    reply_str = ""
    vote_flag = msg.get("vote_flag", "")

    reply_id = msg.get("reply_to_message_id")
    if reply_id and msg_lookup and reply_id in msg_lookup:
        r_msg = msg_lookup[reply_id]
        r_author = r_msg.get("from", "Unknown")
        r_text = get_clean_text(r_msg.get("text", ""))[:40].replace("\n", " ")
        reply_str = f'[в ответ {r_author}: "{r_text}..."]'

    return f"[{date}] {author}{reply_str}{vote_flag}: {text}".strip()