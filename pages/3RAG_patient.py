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

import streamlit as st

# https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf/#pypdf-directory

@st.cache_resource
def load_rag():
    llm = ChatNVIDIA(model="aisingapore/sea-lion-7b-instruct")
    # llm = ChatOpenAI(model="gpt-4-turbo")
    loader = PyPDFDirectoryLoader("healthy_nutrition/")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

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
# Streamlit app layout
st.title('RAG patient Chatbot')
st.write('This is a simple RAG ChatBot for patient.')
st.code("""
what are some healthy food i should eat in singapore?
what are some healthy dish i should eat in singapore?
what is Nutri-Grade?
What should singapore men take note about Prostate Cancer?
What is SCS  Walnut  Warriors?
""")

user_input = st.text_input("Type your message here:")

if user_input:    
    response = rag_chain.invoke(user_input)
    print(f"response:{response}")
    st.text_area("Response:", value=response, height=200, max_chars=None, key=None)


# # cleanup
# vectorstore.delete_collection()


# # Load, chunk and index the contents of the blog.
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever()
# prompt = hub.pull("rlm/rag-prompt")

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# response = rag_chain.invoke("What is Task Decomposition?")
# print(f"response:{response}")

# # cleanup
# vectorstore.delete_collection()