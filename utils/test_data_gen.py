import json
from datetime import datetime, timedelta


def _create_msg(msg_id, author, text, date_obj, reply_to=None):
    msg = {
        "id": msg_id,
        "type": "message",
        "date": date_obj.isoformat(),
        "from": author,
        "text": text
    }
    if reply_to:
        msg["reply_to_message_id"] = reply_to
    return msg


def get_backend_chat_dataset():
    """Чат команды бэкенда: обсуждают API, Базу, Сервер"""
    base_date = datetime.now() - timedelta(days=5)
    messages = []

    # Диалог 1: Архитектура БД
    messages.append(_create_msg(1, "Alex Lead", "Ребят, давайте утвердим схему БД. Используем PostgreSQL?", base_date))
    messages.append(_create_msg(2, "Ivan Dev", "Да, Postgres 15. Нам нужен JSONB для хранения конфигов.",
                                base_date + timedelta(minutes=2), reply_to=1))
    messages.append(
        _create_msg(3, "Alex Lead", "Ок, принято. Создаю сущность User и Order.", base_date + timedelta(minutes=5)))

    # Диалог 2: API
    base_date += timedelta(hours=3)
    messages.append(_create_msg(4, "Petr DevOps", "Надо поднять FastAPI сервис.", base_date))
    messages.append(_create_msg(5, "Ivan Dev", "Согласен. И обязательно JWT авторизацию прикрутить.",
                                base_date + timedelta(minutes=10)))
    messages.append(_create_msg(6, "Alex Lead", "Тогда эндпоинт /login будет выдавать access и refresh токены.",
                                base_date + timedelta(minutes=12)))

    return messages


def get_frontend_chat_dataset():
    """Чат команды фронтенда: обсуждают React, Дизайн и интеграцию с API"""
    base_date = datetime.now() - timedelta(days=4)  # Чуть позже начался
    messages = []

    # Диалог 1: Стек фронта
    messages.append(_create_msg(101, "Maria UI", "Дизайн в Фигме готов. Делаем на React или Vue?", base_date))
    messages.append(
        _create_msg(102, "Dmitry Front", "Давай React + Vite. Так быстрее.", base_date + timedelta(minutes=5),
                    reply_to=101))
    messages.append(_create_msg(103, "Maria UI", "Ок. Используем библиотеку компонентов Ant Design.",
                                base_date + timedelta(minutes=7)))

    # Диалог 2: Интеграция (связь с бэкендом)
    base_date += timedelta(hours=5)
    messages.append(_create_msg(104, "Dmitry Front", "Как нам авторизовываться?", base_date))
    messages.append(_create_msg(105, "Ivan Dev", "Стучитесь в POST /login, мы там JWT отдаем.",
                                base_date + timedelta(minutes=20)))  # Иван (из бэкенда) пришел к фронтам
    messages.append(_create_msg(106, "Dmitry Front", "Понял. Тогда я сделаю форму входа (LoginScreen).",
                                base_date + timedelta(minutes=22)))
    messages.append(
        _create_msg(107, "Maria UI", "И добавь валидацию полей email/password.", base_date + timedelta(minutes=25)))

    return messages