CONFIRMATION_WORDS = {"ок", "да", "давайте", "плюс", "+", "ага", "согласен", "добро", "ок, да"}
REJECTION_WORDS = {"не", "нет", "не надо", "минус", "-", "отмена"}


def normalize_short_answers(text: str) -> str:
    if not isinstance(text, str): return ""
    clean = text.lower().strip().strip(",.!?:")
    if clean in CONFIRMATION_WORDS: return "[FLAG: CONFIRMATION]"
    if clean in REJECTION_WORDS: return "[FLAG: REJECTION]"
    return text


def get_clean_text(text_obj) -> str:
    if isinstance(text_obj, str): return text_obj
    if isinstance(text_obj, list):
        return "".join([i if isinstance(i, str) else i.get("text", "") for i in text_obj])
    return ""


def format_chat_message_for_llm(msg: dict, msg_lookup: dict = None) -> str:
    date = msg.get("date", "Unknown Date")
    author = msg.get("from", msg.get("author", "Unknown"))
    text = normalize_short_answers(get_clean_text(msg.get("text", "")))

    reply_str = ""
    reply_id = msg.get("reply_to_message_id")
    if reply_id and msg_lookup and reply_id in msg_lookup:
        r_msg = msg_lookup[reply_id]
        r_author = r_msg.get("from", "Unknown")
        r_text = get_clean_text(r_msg.get("text", ""))[:40].replace('\n', ' ')
        reply_str = f"[в ответ {r_author}: \"{r_text}...\"]"

    return f"[{date}] {author}{reply_str}: {text}".strip()