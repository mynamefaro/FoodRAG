import streamlit as st
import random
import time
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate


# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf/#pypdf-directory

@st.cache_resource
def load_rag():
    llm = ChatNVIDIA(model="aisingapore/sea-lion-7b-instruct",temperature=1.0,top_p=0.95,seed=42)
    print('hello')
    loader = PyPDFDirectoryLoader("healthy_nutrition/")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="monsoon-nlp/bert-base-thai")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    manual_prompt_template = """
    จากข้อมูลและ Context ที่กำหนดให้ จงตอบ Question พร้อมอธิบายข้อมูลในฐานะของนักโภชนาการ
    Based on the context provided, answer the following question in a detailed and informative manner 
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    # Initialize the prompt template
    prompt = PromptTemplate(template=manual_prompt_template, input_variables=["context", "question"])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = load_rag()

st.title("Thai Food RAG")
st.subheader("RAG Agents ที่คุณสามารถถามตอบได้")


# Streamed response emulator
def response_generator(response):
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
            user_input = prompt
            response_user = rag_chain.invoke(user_input)
            response = st.write_stream(response_generator(response_user))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    
