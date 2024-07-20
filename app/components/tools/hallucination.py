from langchain_core.runnables import RunnableGenerator, RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.components.core.state import GraphState
from app.utils.utils import load_env

load_env()

HALLUCINATION_GRADER_PROMPT_TEMPLATE = """
You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' `isHallucinate` to indicate 
whether the answer is grounded in / supported by a set of facts provided. Provide the binary `isHallucinate` as a JSON with two keys `isHallucinate` 
and `reason` and no preamble or explanation. You must provide a reason for your decision. Check if the answer is grounded in the facts provided.
Here are the facts:
\n ------- \n
{documents} 
\n ------- \n
Here is the answer: {generation}"""


class HallucinationGraderNVIDIA(RunnableGenerator):
    def __init__(
        self,
        llm: ChatNVIDIA = ChatNVIDIA(temperature=0.1),
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        if prompt is None:
            prompt = PromptTemplate(
                template=HALLUCINATION_GRADER_PROMPT_TEMPLATE,
                input_variables=["generation", "documents"],
            )
        self.__prompt = prompt
        self.__grader = self.__prompt | self.__llm | JsonOutputParser()

    def invoke(self, state: GraphState, config: RunnableConfig = None):
        return self.__grader.invoke(state, config)


class HallucinationGraderOpenAI(RunnableGenerator):
    def __init__(
        self,
        llm: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.1),
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        if prompt is None:
            prompt = PromptTemplate(
                template=HALLUCINATION_GRADER_PROMPT_TEMPLATE,
                input_variables=["generation", "documents"],
            )
        self.__prompt = prompt
        self.__grader = self.__prompt | self.__llm | JsonOutputParser()

    def invoke(self, state: GraphState, config: RunnableConfig = None):
        return self.__grader.invoke(state, config)


if __name__ == "__main__":

    print("Hallucination Grader")

    documents = [
        {
            "title": "The history of the United States",
            "text": "The United States was founded in 1776 by a group of British colonists who wanted to break away from British rule. The country was established as a constitutional republic, with a government that is divided into three branches: the executive, legislative, and judicial branches. The United States has a long history of conflict and cooperation with other nations, and has been involved in several major wars, including World War I and World War II. The country has also experienced periods of internal conflict, such as the Civil War, which was fought over the issue",
        }
    ]

    generation = "The United States was founded in 1777 by a group of British colonists who wanted to break away from British rule."
    state = {"generation": generation, "documents": documents}
    llm = ChatNVIDIA()
    grader = HallucinationGraderNVIDIA(llm)
    result = grader.invoke(state)
    print(result)
    llm2 = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    grader_openai = HallucinationGraderOpenAI(llm2)
    result_openai = grader_openai.invoke(state)
    print(result_openai)
