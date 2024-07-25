from langchain_core.runnables import RunnableGenerator, RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.components.core.state import GraphState
from ..prompt import ANSWER_GRADER_PROMPT_TEMPLATE


class AnswerGraderNVIDIA(RunnableGenerator):
    def __init__(
        self,
        llm: ChatNVIDIA = ChatNVIDIA(temperature=0.1),
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        if prompt is None:
            prompt = PromptTemplate(
                template=ANSWER_GRADER_PROMPT_TEMPLATE,
                input_variables=["generation", "question"],
            )
        self.__prompt = prompt
        self.__grader = self.__prompt | self.__llm | JsonOutputParser()

    def invoke(self, state: GraphState, config: RunnableConfig = None):
        return self.__grader.invoke(state, config)


class AnswerGraderOpenAI(RunnableGenerator):
    def __init__(
        self,
        llm: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.1),
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        if prompt is None:
            prompt = PromptTemplate(
                template=ANSWER_GRADER_PROMPT_TEMPLATE,
                input_variables=["generation", "question"],
            )
        self.__prompt = prompt
        self.__grader = self.__prompt | self.__llm | JsonOutputParser()

    def invoke(self, state: GraphState, config: RunnableConfig = None):
        return self.__grader.invoke(state, config)


if __name__ == "__main__":

    print("Answer Grader")
    question = "When was the United States founded?"
    generation = "The United States was founded in 1777 by a group of British colonists who wanted to break away from British rule."
    state = {"generation": generation, "question": question}
    llm = ChatNVIDIA()
    grader = AnswerGraderNVIDIA(llm)
    result = grader.invoke(state)
    print(result)
    llm2 = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    grader_openai = AnswerGraderOpenAI(llm2)
    result_openai = grader_openai.invoke(state)
    print(result_openai)
