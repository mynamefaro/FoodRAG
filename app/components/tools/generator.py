from langchain_core.runnables import RunnableGenerator, RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.components.core.state import GraphState
from ..prompt import ANSWER_GENERATOR_PROMPT_TEMPLATE


class AnswerGeneratorNVIDIA(RunnableGenerator):
    def __init__(
        self,
        llm: ChatNVIDIA = ChatNVIDIA(temperature=0.1),
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        if prompt is None:
            prompt = PromptTemplate(
                template=ANSWER_GENERATOR_PROMPT_TEMPLATE,
                input_variables=["question", "document"],
            )
        self.__prompt = prompt
        self.__generator = self.__prompt | self.__llm | StrOutputParser()

    def invoke(self, state: GraphState, config: RunnableConfig = None):
        return self.__generator.invoke(state, config)


class AnswerGeneratorOpenAI(RunnableGenerator):
    def __init__(
        self,
        llm: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.1),
        prompt: PromptTemplate | None = None,
    ):
        self.__llm = llm
        if prompt is None:
            prompt = PromptTemplate(
                template=ANSWER_GENERATOR_PROMPT_TEMPLATE,
                input_variables=["question", "document"],
            )
        self.__prompt = prompt
        self.__generator = self.__prompt | self.__llm | StrOutputParser()

    def invoke(self, state: GraphState, config: RunnableConfig = None):
        return self.__generator.invoke(state, config)


if __name__ == "__main__":

    print("Answer Generator")

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import WebBaseLoader
    from langchain_community.vectorstores import Chroma
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=NVIDIAEmbeddings(model="NV-Embed-QA"),
    )
    retriever = vectorstore.as_retriever()

    question = "agent memory"
    docs = retriever.invoke(question)
    state = {"context": docs, "question": question}

    llm = ChatNVIDIA()
    generator = AnswerGeneratorNVIDIA(llm)
    result = generator.invoke(state)
    print(result)

    vectorstore_openai = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma-open-ai",
        embedding=NVIDIAEmbeddings(model="NV-Embed-QA"),
    )
    retriever_openai = vectorstore_openai.as_retriever()
    llm2 = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    generator_openai = AnswerGeneratorOpenAI(llm2)
    docs_openai = retriever_openai.invoke(question)
    state_openai = {"context": docs_openai, "question": question}
    result_openai = generator_openai.invoke(state_openai)
    print(result_openai)
