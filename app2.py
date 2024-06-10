from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, tool
import chainlit as cl

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.runnable.config import RunnableConfig

@cl.cache
def get_retriever():
  loader = PyPDFDirectoryLoader("patient_document/")
  docs = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  vectorstore = Chroma.from_documents(documents=splits,persist_directory='./chroma', embedding=OpenAIEmbeddings())
  retriever = vectorstore.as_retriever()
  return retriever

@cl.cache
def get_retriever_food():
  loader = PyPDFDirectoryLoader("healthy_nutrition/")
  docs = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  vectorstore = Chroma.from_documents(documents=splits,persist_directory='./chroma_food', embedding=OpenAIEmbeddings())
  retriever = vectorstore.as_retriever()
  return retriever

@tool()
def doctor_patient(query) -> str:
    """useful when need to get patient's history"""
    retriever = get_retriever()
    return retriever.invoke(query)
  
@tool()
def food(query) -> str:
    """useful when you need to search something about food, nutrition, or a company"""
    retriever = get_retriever_food()
    return retriever.invoke(query)
  
tools = [
    doctor_patient,
    food
]

@cl.on_chat_start
async def start():
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    agent = initialize_agent(
        tools, llm=ChatOpenAI(temperature=0.0,model="gpt-3.5-turbo"), agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,memory=memory,handle_parsing_errors=False,max_iterations=5
    )
    return cl.user_session.set("agent",agent)


@cl.on_message
async def main(message):
  agent = cl.user_session.get("agent")
  cb = cl.LangchainCallbackHandler(stream_final_answer=True)
  cb.answer_reached = True
  config = RunnableConfig(callbacks=[cb])
  res = await agent.ainvoke(message.content, config=config)
  msg = cl.Message(content="")
  for token in res["output"].split(" "):
      await msg.stream_token(token)
      await msg.stream_token(" ")
  await msg.send()