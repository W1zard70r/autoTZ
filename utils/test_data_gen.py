import json
from datetime import datetime, timedelta

def get_huge_chat_dataset():
    base_date = datetime(2026, 2, 24, 10, 0, 0)
    messages = []
    msg_id = 1

    def add_msg(author, text, time_offset_minutes=0, reply_to=None):
        nonlocal msg_id, base_date
        base_date += timedelta(minutes=time_offset_minutes)
        msg = {"id": msg_id, "type": "message", "date": base_date.isoformat(), "from": author, "text": text}
        if reply_to: msg["reply_to_message_id"] = reply_to
        messages.append(msg)
        current_id = msg_id
        msg_id += 1
        return current_id

    m1 = add_msg("Александр", "Коллеги, стартуем спринт. Главная задача - OAuth авторизация.")
    m2 = add_msg("Гриша", "Принял. Будем делать через Google или GitHub?", 2, m1)
    m3 = add_msg("Мария", "На дизайне нарисованы обе кнопки.", 5, m1)
    m4 = add_msg("Александр", "Давайте начнем с Google. GitHub во второй итерации.", 2, m2)
    m5 = add_msg("Гриша", "Ок, тогда я беру либу authlib.", 1, m4)
    add_msg("Сергей", "Только не хардкодьте секреты в код, используйте ENV!", 10)
    add_msg("Гриша", "Обижаешь, Серега. Все будет в .env.", 2)

    return messages