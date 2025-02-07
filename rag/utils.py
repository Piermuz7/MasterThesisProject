import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx


def write_message(role, content, save=True):
    """
        Helper function that saves a message to the
        session state and then writes a message to the UI

        Args:
            role (str): The role of the message
            content (str): The content of the message
            save (bool): Whether to save the message to the session state

        Returns:
            None
    """
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # write to UI
    with st.chat_message(role):
        st.markdown(content)


def get_session_id():
    return get_script_run_ctx().session_id