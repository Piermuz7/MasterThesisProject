import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseLanguageModel
from langchain_ollama.llms import OllamaLLM
# Large Language Model configuration
from llama_index.llms.anthropic import Anthropic

llama_index_llm = Anthropic(
    model="claude-3-5-sonnet-20241022",
    api_key=st.secrets["api_key"]["ANTHROPIC_KEY"],
    max_tokens=8192,
)

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
import toml

from rag.llms.llm_factory import LLMFactory

config = toml.load(".streamlit/secrets.toml")

llm_type = config["llm"]["PROVIDER"]
#api_key = config["api_key"]
api_key = config["api_key"]["ANTHROPIC_KEY"]


llm = LLMFactory.get_llm(llm_type, api_key)

def get_llm() -> BaseLanguageModel:
    return llm
