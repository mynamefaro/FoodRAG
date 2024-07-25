from langchain_core.runnables import RunnableGenerator, RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.components.core.state import GraphState
from ..prompt import TEXT_REVISER_PROMPT_TEMPLATE, TEXT_NO_PREV_CONTEXT_REVISER_PROMPT_TEMPLATE


class DocumentReviserToolNVIDIA(RunnableGenerator):
    def __init__(
        self,
        llm: ChatNVIDIA = ChatNVIDIA(temperature=0.1),
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        if prompt is None:
            prompt = PromptTemplate(
                template=TEXT_REVISER_PROMPT_TEMPLATE,
                input_variables=["previous_context", "context"],
            )
        self.__prompt = prompt
        self.__generator = self.__prompt | self.__llm | StrOutputParser()

    def invoke(self, state: GraphState | str, config: RunnableConfig = None):
        if isinstance(state, str):
            state = {"context": state}
        if "previos_context" not in state or state['previos_context'] == "":
            self.__prompt = PromptTemplate(
                template=TEXT_NO_PREV_CONTEXT_REVISER_PROMPT_TEMPLATE,
                input_variables=["context"],
            )
            generator = self.__prompt | self.__llm | StrOutputParser()
            return generator.invoke(state, config)
        return self.__generator.invoke(state, config)


class DocumentReviserToolOpenAI(RunnableGenerator):
    def __init__(
        self,
        llm: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.1),
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        if prompt is None:
            prompt = PromptTemplate(
                template=TEXT_REVISER_PROMPT_TEMPLATE,
                input_variables=["previous_context", "context"],
            )
        self.__prompt = prompt
        self.__generator = self.__prompt | self.__llm | StrOutputParser()

    def invoke(self, state: GraphState | str, config: RunnableConfig = None):
        if isinstance(state, str):
            state = {"context": state}
        if "previos_context" not in state or state['previos_context'] == "":
            self.__prompt = PromptTemplate(
                template=TEXT_NO_PREV_CONTEXT_REVISER_PROMPT_TEMPLATE,
                input_variables=["context"],
            )
            generator = self.__prompt | self.__llm | StrOutputParser()
            return generator.invoke(state, config)
        return self.__generator.invoke(state, config)
