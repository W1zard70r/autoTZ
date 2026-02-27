from enum import Enum

class DataEnum(str, Enum):
    CHAT = "messages"
    DOCUMENT = "documents"
    PLAIN_TEXT = "plain_text"
    ACTION = "user_action"

class TZSectionEnum(str, Enum):
    GENERAL = "general_info"       # Общие сведения
    STACK = "tech_stack"           # Стек технологий
    FUNCTIONAL = "functional_req"  # Функциональные требования
    INTERFACE = "ui_ux"            # Интерфейс
    UNKNOWN = "uncategorized"