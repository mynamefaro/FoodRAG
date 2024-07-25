from typing import List
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool, tool, BaseToolkit
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.vectorstores import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from chromadb import ClientAPI
from ..media import PDFLoader, ImageLoader
from langchain import hub
from langgraph.prebuilt import create_react_agent

FOOD_VECTOR_QA_CHAT_PROMPT_TEMPLATE = hub.pull(
    "langchain-ai/retrieval-qa-chat")


FOOD_VECTOR_SYSTEM_PROPMT_TEMPLATE = """
You are an agent designed to interact with a Food Database and Vectorstore.
The Food Database is a collection of food-related documents, images, and information. 
You can store, retrieve, and interact with the vector database using the following tools provided.
You must also be able to interact with the Vectorstore to store and retrieve documents and images.
"""


class FoodVector:
    def __init__(self, vector_database_client: ClientAPI, llm=ChatOpenAI(temperature=0.1), embedder=OpenAIEmbeddings(), collection_name: str = "food"):
        self.__vector_db_client = vector_database_client
        self.__llm = llm
        self.__embedder = embedder
        self.__pdf_loader = PDFLoader()
        self.__image_loader = ImageLoader()
        self.__vector_db = Chroma(client=self.__vector_db_client,
                                  embedding_function=self.__embedder,
                                  collection_name=collection_name)
        self.__vector_stuff_chain = create_stuff_documents_chain(
            llm, FOOD_VECTOR_QA_CHAT_PROMPT_TEMPLATE
        )
        self.__retrieval_chain = create_retrieval_chain(
            self.__vector_db.as_retriever(), self.__vector_stuff_chain)
        self.__agent = create_react_agent(
            model=self.__llm, tools=self.get_tools())

    def load_pdf(self, path: str):
        """
        Useful when you want to read and summarize a PDF document.
        Arg:
            path: (str) Path to the PDF file.
        Return:
            list: List of Document objects.
        """
        docs = self.__pdf_loader.load(path, True)
        return docs

    def load_image(self, path: str):
        """
        Useful when you want to read and scan the context of image.
        Arg:
            path: (str) Path to the image file.
        Return:
            (Document, Image): Document object and PIL Image object.
        """
        doc, image = self.__image_loader.load(path)
        return doc, image

    def store_document(self, document: Document | list[Document]):
        """
        Store a document in the vector database.
        Arg:
            document: (Document) Document object to store.
        """
        self.__vector_db.add_documents(
            document if isinstance(document, list) else [document])

    def store_image(self, image: str | list[str]):
        """
        Store an image in the vector database.
        Arg:
            image: (Image) PIL Image object.
        """
        self.__vector_db.add_images(
            image if isinstance(image, list) else [image])

    def retrieve_document(self, query: str, limit: int = 5):
        """
        Retrieve documents from the vector database.
        Arg:
            query: (str) Query to retrieve documents.
            limit: (int) Number of documents to retrieve.
        Return:
            list: List of Document objects.
        """
        return self.__retrieval_chain.invoke({"input": query, "limit": limit})

    def get_tools(self) -> List[BaseTool]:
        return [self.load_pdf, self.load_image, self.store_document, self.store_image, self.retrieve_document]

    def invoke(self, state: dict, config=None):
        return self.__agent.invoke(state, config)
