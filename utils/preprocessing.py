import re

# ─────────────────────────────────────────────
# Словари для определения позиции
# ─────────────────────────────────────────────

CONFIRMATION_WORDS = {
    "ок", "да", "давайте", "плюс", "+", "ага", "согласен", "добро",
    "ок, да", "поддерживаю", "за", "голосую за", "тоже за", "я за",
    "поддержим", "принято", "хорошо", "отлично", "супер", "заходит",
}

REJECTION_WORDS = {
    "не", "нет", "не надо", "минус", "-", "отмена", "против",
    "не согласен", "я против", "не поддерживаю", "плохая идея",
    "не вариант", "нет смысла",
}

# Паттерны для явного голосования: "я за React", "голосую за Vue", "я против Postgres"
_VOTE_FOR_PATTERNS = [
    r"(?:я\s+)?за\s+(?P<target>\w[\w\s\+#\.]*)",
    r"голосую\s+за\s+(?P<target>\w[\w\s\+#\.]*)",
    r"(?:я\s+)?поддерживаю\s+(?P<target>\w[\w\s\+#\.]*)",
    r"тоже\s+за\s+(?P<target>\w[\w\s\+#\.]*)",
    r"(?:я\s+)?выбира[ею]+\s+(?P<target>\w[\w\s\+#\.]*)",
]

_VOTE_AGAINST_PATTERNS = [
    r"(?:я\s+)?против\s+(?P<target>\w[\w\s\+#\.]*)",
    r"не\s+хочу\s+(?P<target>\w[\w\s\+#\.]*)",
    r"(?:я\s+)?отклоняю\s+(?P<target>\w[\w\s\+#\.]*)",
]


def normalize_short_answers(text: str) -> str:
    """Помечает короткие «ок/нет» ответы специальными флагами для LLM."""
    if not isinstance(text, str):
        return ""
    clean = text.lower().strip().strip(",.!?:")
    if clean in CONFIRMATION_WORDS:
        return "[FLAG: CONFIRMATION]"
    if clean in REJECTION_WORDS:
        return "[FLAG: REJECTION]"
    return text


def detect_vote(text: str) -> tuple[str | None, str | None]:
    """
    Пытается определить, за что или против чего проголосовал человек.

    Возвращает:
        (direction, target_hint) где direction = 'for' | 'against' | None
        target_hint — фрагмент текста с упомянутым вариантом (может быть None)

    Примеры:
        "я за React"        → ('for',     'React')
        "голосую за Vue"    → ('for',     'Vue')
        "я против Postgres" → ('against', 'Postgres')
        "давайте"           → ('for',     None)   # безадресное согласие
        "нет"               → ('against', None)
        "обсудим детали"    → (None,      None)
    """
    if not isinstance(text, str):
        return None, None

    lower = text.lower().strip()

    # Проверяем явные паттерны "за <вариант>" / "против <вариант>"
    for pattern in _VOTE_FOR_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            return "for", m.group("target").strip()

    for pattern in _VOTE_AGAINST_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            return "against", m.group("target").strip()

    # Безадресное согласие / отказ
    clean = lower.strip(",.!?:")
    if clean in CONFIRMATION_WORDS:
        return "for", None
    if clean in REJECTION_WORDS:
        return "against", None

    return None, None


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
    reply_id = msg.get("reply_to_message_id")
    if reply_id and msg_lookup and reply_id in msg_lookup:
        r_msg = msg_lookup[reply_id]
        r_author = r_msg.get("from", "Unknown")
        r_text = get_clean_text(r_msg.get("text", ""))[:40].replace("\n", " ")
        reply_str = f'[в ответ {r_author}: "{r_text}..."]'

    return f"[{date}] {author}{reply_str}: {text}".strip()