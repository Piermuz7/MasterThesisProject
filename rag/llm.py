import streamlit as st
from langchain_anthropic import ChatAnthropic

# Large Language Model configuration
llm = ChatAnthropic(temperature=0,
                    anthropic_api_key=st.secrets["ANTHROPIC_KEY"],
                    model="claude-3-5-sonnet-20240620")
