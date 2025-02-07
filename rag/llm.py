from langchain_core.language_models import BaseLanguageModel
from llama_index.core.llms.function_calling import FunctionCallingLLM
import toml

from rag.llms.langchain_llm.llm_factory import LLMFactory as LLMFactoryLC

from rag.llms.llama_index_llm.llm_factory import LLMFactory as LLMFactoryLI

# Large Language Model configuration

config = toml.load(".streamlit/secrets.toml")

llm_type = config["llm"]["PROVIDER"]

# decomment the API key that you want to use

# api_key = config["api_key"]
api_key = config["api_key"]["ANTHROPIC_KEY"]


def get_langchain_llm() -> BaseLanguageModel:
    return LLMFactoryLC.get_llm(llm_type, api_key)


def get_llama_indexllm() -> FunctionCallingLLM:
    return LLMFactoryLI.get_llm(llm_type, api_key)