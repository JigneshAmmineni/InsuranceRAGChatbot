import streamlit as st
import requests

BACKEND_URL = "http://backend:8000/chat"

st.title("Insurance Chatbot")

if "response" not in st.session_state:
    st.session_state.response = None

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    with st.spinner("Thinking..."):
        try:
            res = requests.post(BACKEND_URL, json={"message": user_input})
            if not res.ok:
                detail = res.json().get("detail", "An unexpected error occurred")
                st.session_state.response = detail
            else:
                st.session_state.response = res.json()["response"]
        except Exception as e:
            st.session_state.response = f"Error: {e}"

if st.session_state.response:
    st.markdown("**Response:**")
    st.write(st.session_state.response)
