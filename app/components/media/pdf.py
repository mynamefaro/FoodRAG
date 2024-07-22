from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import fitz
import pymupdf4llm
import pytesseract
from langchain_core.language_models import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.summarize.chain import load_summarize_chain
from pdf2image import convert_from_path
from ..tools.reviser import DocumentReviserToolNVIDIA


class PDFLoader:
    def __init__(self, llm_filter: BaseChatModel = DocumentReviserToolNVIDIA(), llm_summarizer: BaseChatModel = ChatOpenAI(model="gpt-4o-mini"), chain_type="map_reduce", text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1500)):
        self.__llm_summarizer = llm_summarizer
        self.__text_splitter = text_splitter
        self.__llm_filter = llm_filter
        self.__chain_type = chain_type
        self.__summarizer_chain = load_summarize_chain(
            self.__llm_summarizer, self.__chain_type)

    def load(self, path: str, revise=False):
        pdf_file = fitz.open(path)
        md_text = pymupdf4llm.to_markdown(pdf_file)
        if len(md_text) <= 100:
            docs_ocr = self.load_ocr(path, revise)
            if len(docs_ocr) > 0:
                return docs_ocr
        if revise:
            return self.remove_unwanted_text(self.__text_splitter.split_documents([Document(page_content=md_text)]))
        return self.__text_splitter.split_documents([Document(page_content=md_text)])

    def remove_unwanted_text(self, documents):
        for i, document in enumerate(documents):
            state = {"context": document.page_content,
                     "previous_context": documents[i-1].page_content if i > 0 else ""}
            revised_text = self.__llm_filter.invoke(state)
            documents[i].page_content = revised_text
        return documents

    def load_ocr(self, path: str, revise=False):
        pages = convert_from_path(path, 600)
        text_data = ""
        for page in pages:
            text = pytesseract.image_to_string(page)
            text_data += text
        docs = self.__text_splitter.split_documents(
            [Document(page_content=text_data)])
        if revise:
            return self.remove_unwanted_text(docs)
        return docs

    def summarize(self, documents):
        return self.__summarizer_chain.invoke(documents)

    def load_and_summarize(self, path: str):
        documents = self.load(path)
        return self.summarize(documents)
