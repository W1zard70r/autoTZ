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
    
    base_date += timedelta(hours=3)
    m8 = add_msg("Александр", "Кстати, а куда юзеров складываем? В текущую Монгу?")
    m9 = add_msg("Гриша", "Не, для юзеров и транзакций нужна реляционка. Я за Postgres 16.", 5, m8)
    m10 = add_msg("Сергей", "Зачем зоопарк? У нас уже есть MongoDB на проде.", 2, m9)
    add_msg("Гриша", "Монга не дает ACID. Если платеж отвалится, потеряем данные.", 3)
    add_msg("Мария", "Мне без разницы, главное чтобы API отдавал JSON.", 10)
    m14 = add_msg("Александр", "Фиксируем: ставим PostgreSQL 16.", 2)
    add_msg("Гриша", "+", 1, m14)
    add_msg("Сергей", "Ок", 1, m14)

    base_date += timedelta(hours=18)
    m17 = add_msg("Мария", "Гриша, я стучусь на /api/v1/login и получаю CORS error.", 0)
    add_msg("Гриша", "Странно, я разрешил localhost:3000.", 10, m17)
    add_msg("Мария", "У меня порт 8080. Поправь конфиг.", 2)
    add_msg("Сергей", "Это на уровне Nginx надо решать.", 5)
    add_msg("Александр", "Гриша, а refresh token мы храним в httpOnly куке?", 5)
    add_msg("Гриша", "Да, чтобы XSS не прошел.", 2)

    return messages