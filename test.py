import os
import google.generativeai as genai
from dotenv import load_dotenv

# Загружаем API ключ
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Ошибка: GOOGLE_API_KEY не найден")
else:
    # 1. Настраиваем SDK
    genai.configure(api_key=api_key)

    print(f"{'NAME':<30} | {'DISPLAY NAME':<30} | {'METHODS'}")
    print("-" * 80)

    # 2. Получаем список всех моделей
    try:
        for m in genai.list_models():
            # supported_generation_methods показывает, что умеет модель:
            # 'generateContent' — это чат/текст (то, что вам нужно для ChatGoogleGenerativeAI)
            # 'embedContent' — это для эмбеддингов

            methods = ", ".join(m.supported_generation_methods)

            # Фильтрация (опционально): показать только модели для генерации текста
            # if 'generateContent' in m.supported_generation_methods:
            print(f"{m.name:<30} | {m.display_name:<30} | {methods}")

    except Exception as e:
        print(f"Ошибка при получении списка: {e}")