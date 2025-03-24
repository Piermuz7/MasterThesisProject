from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.azure_openai import AzureOpenAI

import toml
import streamlit as st

# Large Language Model configuration

config = toml.load(".streamlit/secrets.toml")

# uncomment the API key that you want to use

api_key_anthropic = config["api_key"]["ANTHROPIC_KEY"]
api_key_azure = config["api_key"]["AZURE_OPENAI_API_KEY"]

langchain_anthropic_sonnet_3_5_llm = ChatAnthropic(
    temperature=0,
    model_name="claude-3-5-sonnet-20241022",
    max_tokens_to_sample=8192,
    api_key=api_key_anthropic,
    max_retries=3
)

llama_index_anthropic_sonnet_3_5_llm = Anthropic(temperature=0,
                                                 api_key=api_key_anthropic,
                                                 model="claude-3-5-sonnet-20241022",
                                                 max_tokens=8192,
                                                 )

langchain_anthropic_haiku_llm = ChatAnthropic(
    temperature=0,
    model_name="claude-3-5-haiku-20241022",
    max_tokens_to_sample=8192,
    api_key=api_key_anthropic,
    max_retries=3
)

langchain_anthropic_sonnet_3_7_llm = ChatAnthropic(
    temperature=0,
    model_name="claude-3-7-sonnet-20250219",
    max_tokens_to_sample=8192,
    api_key=api_key_anthropic,
    max_retries=3
)

langchain_azure_openai_gpt4o_llm = AzureChatOpenAI(
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    temperature=0,
    model="gpt-4o",
    max_tokens=4096,
    api_key=api_key_azure,
    api_version="2024-08-01-preview",
    timeout=None,
    max_retries=2,
)

llama_index_azure_openai_gpt4o_llm = AzureOpenAI(
    model="gpt-4o",
    api_key=api_key_azure,
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-08-01-preview",
    max_retries=3,
    engine="my_engine",
)
