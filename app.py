from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, initialize_agent, tool
from PIL import Image, ImageDraw, ImageFont
from random import randrange
import io
import chainlit as cl

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.react.agent import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.memory.entity import SQLiteEntityStore
from langchain import hub
from langchain_core.messages import HumanMessage,AIMessage

@cl.cache
def get_retriever():
  loader = PyPDFDirectoryLoader("patient_document/")
  docs = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  splits = text_splitter.split_documents(docs)
  vectorstore = Chroma.from_documents(documents=splits,persist_directory='./chroma', embedding=OpenAIEmbeddings())
  retriever = vectorstore.as_retriever()
  return retriever

@cl.on_chat_start
async def start():
  llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
  prompt = hub.pull("hwchase17/react-chat")
  retriever = get_retriever()
  
  ### Build retriever tool ###
  tool = create_retriever_tool(
    retriever,
    "doctor_retriever",
    "Search for relevant documents about patient history",
  )
  tools = [tool]

  memory = SQLiteEntityStore()

  agent = create_react_agent(llm, tools, prompt)
  cl.user_session.set("agent", agent)
  cl.user_session.set("messages",memory)
  print("done")
  return agent

@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")
    memory = cl.user_session.get("messages")
    print(agent,memory)
    res = agent.invoke({"input":message.content,"intermediate_steps": [],"chat_history":[]})
    await cl.Message(content=res).send()
    # memory.append(HumanMessage(content=message.content))
    # memory.append(AIMessage(content=res))