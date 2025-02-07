from langchain_core.language_models import BaseLanguageModel
from llama_index.core.llms.function_calling import FunctionCallingLLM
import toml

from rag.llms.langchain_llm.llm_factory import LLMFactory as LLMFactoryLC

from rag.llms.llama_index_llm.llm_factory import LLMFactory as LLMFactoryLI

# Large Language Model configuration

'''
llama_index_llm = Anthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=st.secrets["api_key"]["ANTHROPIC_KEY"],
    max_tokens=8192,
)
'''

# Anthropic configuration
'''llms = ChatAnthropic(temperature=0,
                    anthropic_api_key=st.secrets["ANTHROPIC_KEY"],
                    model="claude-3-5-sonnet-20240620")
'''

# Ollama configuration
'''llms = OllamaLLM(
    temperature=0,
    model="llama3.2",
    base_url="http://localhost:11434"
)
'''
'''
llms = OllamaLLM(
    temperature=0,
    model="llama3.1",
    base_url="http://localhost:11434"
)
'''

config = toml.load(".streamlit/secrets.toml")

llm_type = config["llm"]["PROVIDER"]
# api_key = config["api_key"]
api_key = config["api_key"]["ANTHROPIC_KEY"]


def get_langchain_llm() -> BaseLanguageModel:
    return LLMFactoryLC.get_llm(llm_type, api_key)


def get_llama_indexllm() -> FunctionCallingLLM:
    return LLMFactoryLI.get_llm(llm_type, api_key)
