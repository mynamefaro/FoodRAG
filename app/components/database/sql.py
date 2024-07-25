from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent


class BaseDatabaseToolkit:
    def __init__(self, sql_database_url: str, llm=ChatOpenAI(temperature=0.1), prompt=None):
        self.__db_url = sql_database_url
        self.__conn = SQLDatabase.from_uri(self.__db_url)
        self.__llm = llm
        self.__prompt = prompt
        self.__sql_agent = SQLDatabaseToolkit(llm=self.__llm, db=self.__conn)
        self.__tools_list = self.__sql_agent.get_tools()
        self.__agent = create_react_agent(
            model=self.__llm, tools=self.__tools_list,
            messages_modifier=self.__prompt)

    def invoke(self, state, config=None):
        return self.__agent.invoke(state, config)

    def get_tools(self):
        return self.__tools_list
