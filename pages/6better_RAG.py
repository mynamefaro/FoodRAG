import streamlit as st
import random
import time

st.title("Food RAG")
st.subheader("The RAG agents who can ask and interract with")


# Streamed response emulator
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)
        
def show_upload(state:bool):
    if not state:
        st.session_state.messages.append({"role": "assistant","content": "You have cancel file upload"})
    st.session_state["uploader_visible"] = state


if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "request_uploader" not in st.session_state: st.session_state["request_uploader"] = False

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if type(message) == dict:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
placeholder = st.empty()

if st.session_state["uploader_visible"]:
    file = placeholder.file_uploader("Upload your data")
    if file:
        with st.spinner("Processing your file"):
                time.sleep(2)
        st.session_state["uploader_visible"] = False
        with st.chat_message("user"):
            response = st.markdown("Uploaded " + file.name)
        st.session_state.messages.append({"role": "user","content": "Uploaded " + file.name})
        with st.chat_message("assistant"):
            response = st.markdown("Received uploaded file")
        st.session_state.messages.append({"role": "assistant","content": "Received uploaded file"})
        placeholder.empty()

# Accept user input
if prompt := st.chat_input("What is up?",disabled=st.session_state["uploader_visible"]):
    # Add user message to chat history
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
    if prompt == "upload":
        with st.chat_message("assistant"):
            cols= st.columns((3,1,1))
            cols[0].write("Do you want to upload a file?")
            cols[1].button("yes", use_container_width=True, on_click=show_upload, args=[True])
            cols[2].button("no", use_container_width=True, on_click=show_upload, args=[False])
    else: 
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator())
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    
