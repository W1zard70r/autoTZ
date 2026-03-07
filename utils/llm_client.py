import os
import logging
import warnings
from typing import Type, TypeVar, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.messages import SystemMessage, HumanMessage
import google.api_core.exceptions
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")

HF_REPO_ID = os.getenv("HF_REPO_ID", "unsloth/Qwen3.5-9B-GGUF")
HF_FILENAME = os.getenv("HF_FILENAME", "Qwen3.5-9B-Q4_K_S.gguf")
LOCAL_MODELS_PATH = os.getenv("LOCAL_MODELS_PATH", "/kaggle/working/models")


def get_local_model_path(repo_id: str, filename: str) -> str:
    """Скачивает модель с HF если её нет и возвращает путь."""
    os.makedirs(LOCAL_MODELS_PATH, exist_ok=True)

    local_file_path = os.path.join(LOCAL_MODELS_PATH, filename)

    if os.path.exists(local_file_path):
        logger.info(f"Using existing local model: {local_file_path}")
        return local_file_path

    logger.info(f"Model not found locally. Downloading {filename} from {repo_id}...")
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=LOCAL_MODELS_PATH,
            local_dir_use_symlinks=False
        )
        logger.info(f"Download complete: {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Failed to download model from HF: {e}")
        raise e


def get_llm_client(model_name: str, temperature: float = 0.1, max_tokens: Optional[int] = None):
    if LLM_PROVIDER == "openai":
        kwargs = dict(
            model=model_name, temperature=temperature,
            api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, max_retries=3,
        )
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)

    elif LLM_PROVIDER == "local":
        model_path = get_local_model_path(HF_REPO_ID, HF_FILENAME)

        n_ctx = max_tokens if max_tokens else 4096
        n_gpu_layers = int(os.getenv("n_gpu_layers", -1))

        logger.info(f"Loading local LLM from {model_path} (ctx={n_ctx}, gpu={n_gpu_layers})")

        return ChatLlamaCpp(
            model_path=model_path,
            temperature=temperature,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            n_batch=512,
            f16_kv=True,
        )

    else:
        kwargs = dict(
            model=model_name, temperature=temperature,
            google_api_key=GOOGLE_API_KEY, max_retries=3,
        )
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**kwargs)

T = TypeVar("T", bound=BaseModel)

GLOBAL_RETRY_CONFIG = {
    "stop": stop_after_attempt(12),
    "wait": wait_exponential(multiplier=2, min=4, max=120),
    "retry": retry_if_exception_type((
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.GoogleAPICallError,
        TimeoutError, ConnectionError
    )),
    "before_sleep": before_sleep_log(logger, logging.WARNING)
}


@retry(**GLOBAL_RETRY_CONFIG)
async def acall_llm_json(schema: Type[T], prompt: str, data: str = "", model_name: str = DEFAULT_MODEL, system: Optional[str] = None, max_tokens: int = 16384) -> T:
    try:
        llm = get_llm_client(model_name=model_name, temperature=0.1, max_tokens=max_tokens)
        llm_structured = llm.with_structured_output(schema)
        messages: List = []
        if system:
            messages.append(SystemMessage(content=system))
        user_content = prompt
        if data:
            user_content += f"\n\n--- DATA ---\n{data}"
        messages.append(HumanMessage(content=user_content))
        return await llm_structured.ainvoke(messages)
    except Exception as e:
        logger.error(f"LLM JSON error ({LLM_PROVIDER}): {e}")
        raise e


@retry(**GLOBAL_RETRY_CONFIG)
async def acall_llm_text(prompt: str, data: str = "", model_name: str = DEFAULT_MODEL, system: Optional[str] = None, max_tokens: int = 4096) -> str:
    try:
        llm = get_llm_client(model_name=model_name, temperature=0.2, max_tokens=max_tokens)
        messages: List = []
        if system:
            messages.append(SystemMessage(content=system))
        user_content = prompt
        if data:
            user_content += f"\n\n--- DATA ---\n{data}"
        messages.append(HumanMessage(content=user_content))
        result = await llm.ainvoke(messages)
        return result.content
    except Exception as e:
        logger.error(f"LLM Text error ({LLM_PROVIDER}): {e}")
        raise e
