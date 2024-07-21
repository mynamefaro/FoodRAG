from langchain_core.runnables import RunnableGenerator, RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.components.core.state import GraphState
from ..utils import load_env

load_env()

TEXT_REVISER_PROMPT_TEMPLATE = """
As an mute assistant, your aim is to enhance text and table readability based on specific guidelines and previous context:
Reframe sentences and sections for better comprehension.
Eliminate unclear text, e.g., content with excessive symbols or gibberish.
Shorten text without losing information. Suggestion: Summarize lengthy phrases where possible.
Rectify poorly formatted tables, e.g., adjust column alignment for clarity.
Preserve clear, understandable text as is. Example: "Use direct and easily comprehensible sentences."
Refrain from responding if text is entirely unclear or ambiguous, e.g., incomprehensible or garbled content.
Remove standalone numbers or letters not associated with text, e.g., isolated digits or letters lacking context.
Remove redundant or repetitive text, e.g., content that is repeated or reiterated unnecessarily. Suggestion: Combine repetitive sentences to previous context if possible.
Exclude non-factual elements like selection marks, addresses, picture marks, or drawings.
Ensure modifications maintain the original text's clarity and information conveyed.
Remove any irrelevant or unnecessary information. Like addresses, phone numbers, or any advertisement purpose information.
Answer back the revised text without additional comments before or after, avoiding comments about how or which guidelines have been followed.
The context of previous text is as follows:
{previous_context}

-------------------------------------------------------------------------------------------------

Please revise the following text based on these guidelines:
Text: {context}

Revised text:
"""


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
            state = GraphState(context=state)
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
            state = GraphState(context=state)
        return self.__generator.invoke(state, config)


if __name__ == "__main__":

    print("Text Reformatter")
