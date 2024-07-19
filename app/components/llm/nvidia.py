import os
from langchain import hub
from langchain_nvidia_ai_endpoints import ChatNVIDIA as _ChatNVIDIA
from langchain_nvidia_ai_endpoints.tools import ServerToolsMixin
from langchain_core.tools import tool
from langchain_core.prompts import BasePromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str

from app.components.core import State


class TooledChatNVIDIA(ServerToolsMixin, _ChatNVIDIA):
    pass


class AgentNVIDIA:

    def __init__(
        self,
        model_name="aisingapore/sea-lion-7b-instruct",
        tools: list[tool] = None,
        prompt: BasePromptTemplate = hub.pull("hwchase17/react-json"),
        stop: list[str] = ["\nObservation"],
        temperature=0.0,
    ):
        self.__llm = _ChatNVIDIA(
            model=model_name,
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=temperature,
        )
        self.__temperature = temperature
        self.__tools = tools
        self.__prompt = prompt
        self.__agent = None
        self.__stop = stop

    def __call__(self, *args, **kwds):
        return self.invoke(*args, **kwds)

    def bind_tools(self, tools: list[tool] = None):
        if tools is not None:
            self.__tools = tools
        return self.__llm.bind(
            tools=render_text_description(self.__tools),
            tool_names=", ".join([t.name for t in self.__tools]),
        )

    def setup_agent(self):
        if self.__prompt is None:
            raise ValueError("Prompt not set")
        if self.__tools is None or len(self.__tools) == 0:
            raise ValueError("Tools not bound")
        self.bind_tools()
        self.setup_prompt()
        self.__agent = self.__llm.bind(
            stop=self.__stop,
            tools=render_text_description(self.__tools),
            tool_names=", ".join([t.name for t in self.__tools]),
        )
        return self.__agent

    def setup_prompt(self, prompt: BasePromptTemplate = None):
        if prompt is not None:
            self.__prompt = prompt
        self.__prompt.partial(
            tools=render_text_description(self.__tools),
            tool_names=[tool.name for tool in self.__tools],
        )

    def invoke(self, state: State):
        if self.__agent is None:
            self.setup_agent()
        return self.__agent.invoke(state)


if __name__ == "__main__":

    @tool()
    def add_math(a: int, b: int):
        """
        Adds two numbers together
        Args:
            a (int): The first number
            b (int): The second number
        Return:
            int: The sum of the two numbers
        """
        return a + b

    agent = AgentNVIDIA(tools=[add_math])
    agent.setup_agent()
    print(agent.invoke({"messages": []}))
