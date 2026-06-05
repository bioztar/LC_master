"""LLM + embeddings factory. Switches on LLM_PROVIDER env."""
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src import config


def get_chat(temperature: float = 0.0, json_mode: bool = False) -> BaseChatModel:
    config.validate()
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.OPENAI_MODEL,
            "api_key": config.OPENAI_API_KEY,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {
            "model": config.GEMINI_MODEL,
            "google_api_key": config.GOOGLE_API_KEY,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_mime_type": "application/json"}
        return ChatGoogleGenerativeAI(**kwargs)

    raise RuntimeError(f"Unknown provider: {config.LLM_PROVIDER}")


def get_embeddings() -> Embeddings:
    config.validate()
    if config.LLM_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL, api_key=config.OPENAI_API_KEY)
    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model = config.GEMINI_EMBED_MODEL
        if not model.startswith("models/"):
            model = f"models/{model}"
        return GoogleGenerativeAIEmbeddings(
            model=model, google_api_key=config.GOOGLE_API_KEY,
        )
    raise RuntimeError(f"Unknown provider: {config.LLM_PROVIDER}")
