import asyncio
import sys

import streamlit as st

from rag.agent_workflow import execute_sparql_agent_workflow, execute_cypher_agent_workflow
from rag.utils import write_message

if len(sys.argv) != 4:

    query_language = sys.argv[1]

    # page config
    st.set_page_config("European Projects Bot", page_icon=":movie_camera:")

    # set up session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm the European Projects Bot! How can I help you?"},
        ]


    # def handle_submit(message):
    #     with st.spinner('Thinking...'):
    #         response = generate_response(message)
    #         write_message('assistant', response)
    def handle_submit(message):
        with st.spinner('Thinking...'):
            if query_language == 'sparql':
                # print('Executing SPARQL agent workflow')
                response = asyncio.run(execute_sparql_agent_workflow(message))
            else:
                # print('Executing Cypher agent workflow')
                response = asyncio.run(execute_cypher_agent_workflow(message))
            write_message('assistant', response)


    # display messages in session state
    for message in st.session_state.messages:
        write_message(message['role'], message['content'], save=False)

    # handle any user input
    if question := st.chat_input("Message the bot..."):
        # display user message in chat message container
        write_message('user', question)

        # generate a response
        handle_submit(question)
