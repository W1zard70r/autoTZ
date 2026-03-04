from typing import List, Any, Dict, Optional
from pydantic import BaseModel, Field
from .enums import TZSectionEnum, DataEnum, TZStandardEnum

class DataSource(BaseModel):
    source_type: DataEnum
    content: Any
    file_name: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

# === СТРУКТУРЫ ДЛЯ ЭКСПОРТА НА БЭКЕНД ===

class TitlePageData(BaseModel):
    """Данные для генерации титульного листа (особенно важно для ГОСТ)"""
    project_name: str = Field(description="Полное наименование системы")
    project_code: Optional[str] = Field(default="RU.XXXX.00001-01", description="Шифр/Децимальный номер")
    organization_name: Optional[str] = Field(default="ООО 'Рога и Копыта'", description="Заказчик/Исполнитель")
    approver_name: Optional[str] = Field(default=None, description="Кто утверждает ТЗ")
    city: str = Field(default="Москва")
    year: str = Field(default="2026")

class DocumentSection(BaseModel):
    """Рекурсивная секция документа"""
    section_enum: Optional[TZSectionEnum] = Field(default=None, description="Смысловой ID (для ML)")
    number: str = Field(description="Номер раздела (например '1.', '4.2.')")
    title: str = Field(description="Заголовок раздела")
    content: str = Field(default="", description="Текст раздела в Markdown. Если есть подразделы, может быть пустым.")
    subsections: List['DocumentSection'] = Field(default_factory=list, description="Вложенные разделы")

    class Config:
        # Разрешаем рекурсивную ссылку на самих себя
        json_schema_extra = {
            "example": {
                "number": "1.",
                "title": "Общие сведения",
                "content": "Текст...",
                "subsections": []
            }
        }

class FinalExportDocument(BaseModel):
    """Финальный артефакт, который летит на Бэкенд"""
    standard: TZStandardEnum = Field(description="Стандарт оформления (ГОСТ 34, и т.д.)")
    version: str = Field(default="1.0.0")
    title_page: TitlePageData
    structure: List[DocumentSection] = Field(description="Дерево разделов")

# Старый класс можно оставить для внутренней работы или удалить, 
# если полностью перейдете на FinalExportDocument
class GeneratedSection(BaseModel):
    section_id: TZSectionEnum
    title: str
    content_markdown: str