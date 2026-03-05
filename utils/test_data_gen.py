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

def get_product_chat_dataset():
    """
    Чат 1: Продуктовое обсуждение.
    Темы: Суть проекта, Роли пользователей, Монетизация, Экраны.
    """
    base_date = datetime.now() - timedelta(days=10)
    messages = []
    
    # Идея проекта
    messages.append(_create_msg(1, "Anna PM", "Коллеги, стартуем новый проект 'EduPlatform'. Это маркетплейс курсов для корпоративного обучения.", base_date))
    messages.append(_create_msg(2, "Oleg CEO", "Да, цель - автоматизировать процесс обучения сотрудников. Основные пользователи: 'Администратор компании' и 'Сотрудник'.", base_date + timedelta(minutes=5)))
    messages.append(_create_msg(3, "Anna PM", "Принято. У Администратора должен быть Личный кабинет, где он назначает курсы и смотрит аналитику.", base_date + timedelta(minutes=10)))
    
    # Функционал
    base_date += timedelta(hours=2)
    messages.append(_create_msg(4, "Maria UI", "Я накидала прототипы. Главная страница - это Дашборд с прогрессом обучения.", base_date))
    messages.append(_create_msg(5, "Anna PM", "Отлично. И обязательно нужен раздел 'Магазин курсов', где можно купить подписку.", base_date + timedelta(minutes=15)))
    messages.append(_create_msg(6, "Oleg CEO", "По монетизации: у нас будет рекуррентная подписка (Monthly Subscription). Подключим Stripe.", base_date + timedelta(minutes=20)))
    
    return messages

def get_backend_chat_dataset():
    """
    Чат 2: Бэкенд и Архитектура.
    Темы: Стек, БД, API, Конфликты технологий.
    """
    base_date = datetime.now() - timedelta(days=5)
    messages = []
    
    # Обсуждение стека (КОНФЛИКТ 1: Python vs Node)
    messages.append(_create_msg(10, "Alex Lead", "По бэкенду: предлагаю классику - Python + Django. Нам нужна админка из коробки.", base_date))
    messages.append(_create_msg(11, "Ivan Dev", "Слушай, а может Node.js + NestJS? У нас фронты на JS, будет единый стек.", base_date + timedelta(minutes=10)))
    messages.append(_create_msg(12, "Alex Lead", "Нет, Django быстрее развернуть для MVP. У нас жесткие сроки.", base_date + timedelta(minutes=15)))
    messages.append(_create_msg(13, "Ivan Dev", "Ладно, но NestJS работает быстрее в рантайме.", base_date + timedelta(minutes=17))) 
    # (Конфликт остается открытым для AI, или можно считать победой Django, но лучше пусть AI спросит)

    # Обсуждение БД (КОНФЛИКТ 2: SQL vs NoSQL)
    base_date += timedelta(hours=4)
    messages.append(_create_msg(14, "Ivan Dev", "Какую базу берем? Данные по курсам слабо структурированы, может MongoDB?", base_date))
    messages.append(_create_msg(15, "Alex Lead", "У нас биллинг и пользователи. Нужна строгая схема и транзакции. Только PostgreSQL.", base_date + timedelta(minutes=10)))
    messages.append(_create_msg(16, "Ivan Dev", "В Монге тоже есть транзакции. Зато JSON хранить удобнее.", base_date + timedelta(minutes=12)))
    
    # API
    base_date += timedelta(hours=2)
    messages.append(_create_msg(17, "Alex Lead", "Делаем REST API. Эндпоинт POST /api/v1/courses/assign назначает курс сотруднику.", base_date))
    messages.append(_create_msg(18, "Ivan Dev", "Ок. И GET /api/v1/users/me/progress для получения прогресса.", base_date + timedelta(minutes=5)))
    
    return messages

def get_frontend_chat_dataset():
    """
    Чат 3: Фронтенд и Тестирование.
    Темы: UI стек, Валидация, Авторизация.
    """
    base_date = datetime.now() - timedelta(days=2)
    messages = []
    
    # Стек фронтенда (КОНФЛИКТ 3: React vs Vue)
    messages.append(_create_msg(20, "Dmitry Front", "Начинаю фронт. Беру React + Vite + TypeScript.", base_date))
    messages.append(_create_msg(21, "Junior Dev", "Может Vue 3? Он проще, я React не знаю.", base_date + timedelta(minutes=10)))
    messages.append(_create_msg(22, "Dmitry Front", "Весь корпоративный стандарт на React. Учи React.", base_date + timedelta(minutes=12)))
    
    # UI детали
    base_date += timedelta(hours=3)
    messages.append(_create_msg(23, "Maria UI", "Дима, на форме входа (Login Screen) нужна валидация email.", base_date))
    messages.append(_create_msg(24, "Dmitry Front", "Сделаю. Использую библиотеку React Hook Form.", base_date + timedelta(minutes=5)))
    messages.append(_create_msg(25, "QA Engineer", "И не забудьте про обработку ошибок 401 и 403. Нужен красивый экран 'Доступ запрещен'.", base_date + timedelta(minutes=20)))
    
    # Интеграция
    messages.append(_create_msg(26, "Dmitry Front", "Авторизация через JWT? Где храним токены?", base_date + timedelta(hours=1)))
    messages.append(_create_msg(27, "Ivan Dev", "Да, JWT. Access токен в памяти, Refresh в httpOnly cookie.", base_date + timedelta(minutes=5)))

    return messages
