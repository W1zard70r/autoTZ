import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

from schemas.document import DataSource
from schemas.enums import DataEnum, TemplateType
from interfaces.cli import ConsoleInterface
from interfaces.test_runner import BatchRunner
from utils.test_data_gen import get_backend_chat_dataset, get_frontend_chat_dataset
from utils.state_logger import init_logs_dir

load_dotenv()
init_logs_dir()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("logs/app.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _default_sources() -> list[DataSource]:
    return [
        DataSource(
            source_type=DataEnum.CHAT,
            content=get_backend_chat_dataset(),
            file_name="chat_backend_team",
        ),
        DataSource(
            source_type=DataEnum.CHAT,
            content=get_frontend_chat_dataset(),
            file_name="chat_frontend_team",
        ),
    ]


async def run_cli():
    """Интерактивный консольный режим."""
    interface = ConsoleInterface(
        sources=_default_sources(),
        template_type=None,
        language="ru",
    )
    await interface.run()


async def run_batch(file_paths: list[str], template: str = "it_project"):
    """Прогон пайплайна на указанных файлах."""
    try:
        template_type = TemplateType(template)
    except ValueError:
        print(f"❌ Неизвестный шаблон: {template}")
        print(f"   Доступные: {', '.join(t.value for t in TemplateType)}")
        return

    runner = BatchRunner(
        file_paths=file_paths,
        template_type=template_type,
        language="ru",
    )
    await runner.run()


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cli"

    if mode == "batch":
        # python main.py batch [--template it_project] file1.json file2.txt ...
        args = sys.argv[2:]
        template = "it_project"
        files = []
        i = 0
        while i < len(args):
            if args[i] == "--template" and i + 1 < len(args):
                template = args[i + 1]
                i += 2
            else:
                files.append(args[i])
                i += 1
        if not files:
            print("Использование: python main.py batch [--template TYPE] file1.json file2.txt ...")
            print(f"  Шаблоны: {', '.join(t.value for t in TemplateType)}")
            return
        await run_batch(files, template)
    elif mode == "cli":
        await run_cli()
    else:
        print("Использование: python main.py [cli|batch]")
        print("  cli   — интерактивный консольный режим (по умолчанию)")
        print("  batch — прогон на файлах: python main.py batch file1.json file2.txt")
        print("          --template TYPE — шаблон (по умолчанию it_project)")


if __name__ == "__main__":
    asyncio.run(main())