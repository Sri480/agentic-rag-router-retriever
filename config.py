import os
from dotenv import load_dotenv
from crewai import LLM

# ---------------------------------------------------------------------------
# Get environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv(override=True)

AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

_required = {
    "AZURE_API_KEY": AZURE_API_KEY,
    "AZURE_API_BASE": AZURE_API_BASE,
    "AZURE_API_VERSION": AZURE_API_VERSION,
    "AZURE_CHAT_DEPLOYMENT": AZURE_CHAT_DEPLOYMENT,
    "AZURE_EMBEDDING_DEPLOYMENT": AZURE_EMBEDDING_DEPLOYMENT,
}
_missing = [k for k, v in _required.items() if not v or v == "XXX"]
if _missing:
    raise ValueError(f"Missing or placeholder values in .env: {_missing}")

# ---------------------------------------------------------------------------
# LLM instantiation for OpenAI
# ---------------------------------------------------------------------------
def get_llm() -> LLM:
    # Azure's OpenAI-compatible v1 endpoint. "model" is your DEPLOYMENT name.
    return LLM(
        model=f"openai/{AZURE_CHAT_DEPLOYMENT}",
        base_url=f"{AZURE_API_BASE.rstrip('/')}/openai/v1/",
        api_key=AZURE_API_KEY,
    )


# ---------------------------------------------------------------------------
# setting up config for embedding model
# ---------------------------------------------------------------------------
def get_embedding_config() -> dict:
    return {
        "provider": "openai",
        "config": {
            "api_key": AZURE_API_KEY,
            "api_base": f"{AZURE_API_BASE.rstrip('/')}/openai/v1/",
            "model_name": AZURE_EMBEDDING_DEPLOYMENT,  # embedding deployment name
        },
    }

