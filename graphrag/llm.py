from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI
from llama_index.llms.anthropic import Anthropic

import toml

import streamlit as st
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.gemini import Gemini

# Large Language Model configuration

config = toml.load(".streamlit/secrets.toml")

# uncomment the API key that you want to use

# Gemini is not available in the current version of the Llama Index API for agentic workflows.
# api_key = config["api_key"]["GOOGLE_KEY"]

# Switch between Anthropic and Azure OpenAI LLMs.

# api_key = config["api_key"]["ANTHROPIC_KEY"]
api_key = config["api_key"]["AZURE_OPENAI_API_KEY"]

langchain_anthropic_llm = ChatAnthropic(
    temperature=0,
    model_name="claude-3-5-sonnet-20241022",
    max_tokens_to_sample=8192,
    api_key=api_key,
    max_retries=3
)

llama_index_anthropic_llm = Anthropic(temperature=0,
                                      api_key=st.secrets["api_key"]["ANTHROPIC_KEY"],
                                      model="claude-3-5-sonnet-20241022",
                                      max_tokens=8192,
                                      )
'''

Gemini is not available in the current version of the Llama Index API for agentic workflows.

langchain_gemini_llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=5,
    google_api_key=st.secrets["api_key"]["GOOGLE_KEY"],
)

llama_index_gemini_llm = Gemini(
    model="models/gemini-2.0-flash",
    api_key=st.secrets["api_key"]["GOOGLE_KEY"],
    max_tokens=8192,
)
'''

langchain_azure_openai_llm = AzureChatOpenAI(
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    temperature=0,
    model="gpt-4o",
    max_tokens=4096,
    api_key=api_key,
    api_version="2024-08-01-preview",
    timeout=None,
    max_retries=2,
)

llama_index_azure_openai_llm = AzureOpenAI(
    model="gpt-4o",
    api_key=api_key,
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-08-01-preview",
    max_retries=3,
    engine="my_engine",
)
