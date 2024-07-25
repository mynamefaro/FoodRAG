from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain import hub
from langgraph.prebuilt import create_react_agent

FOOD_DATABASE_QA_CHAT_PROMPT_TEMPLATE = hub.pull(
    "langchain-ai/retrieval-qa-chat")


FOOD_DATABASE_SYSTEM_PROPMT_TEMPLATE = """
You are an agent designed to interact with a Food Database and Vectorstore.
The Food Database is a collection of food-related documents, images, and information. 
You can store, retrieve, and interact with the vector database using the following tools provided.
You must also be able to interact with the Vectorstore to store and retrieve documents and images.
"""


class FoodDatabase:
    def __init__(self, sql_database_url: str, llm=ChatOpenAI(temperature=0.1)):
        self.__db_url = sql_database_url
        self.__conn = SQLDatabase.from_uri(self.__db_url)
        self.__llm = llm
        self.__sql_agent = SQLDatabaseToolkit(llm=self.__llm, db=self.__conn)
        self.__tools_list = self.__sql_agent.get_tools()
        self.__agent = create_react_agent(
            model=self.__llm, tools=self.__tools_list,
            messages_modifier=SystemMessage(content=FOOD_DATABASE_SYSTEM_PROPMT_TEMPLATE))

    def invoke(self, state, config=None):
        return self.__agent.invoke(state, config)

    def get_tools(self):
        return self.__tools_list
