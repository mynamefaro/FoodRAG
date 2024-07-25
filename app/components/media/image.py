from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pytesseract
from langchain_core.language_models import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64
from PIL import Image
from ..tools.reviser import DocumentReviserToolNVIDIA

IMAGE_DESCRIPTOR_PROMPT = "Please describe the image below with context below:\n\n"


class ImageLoader:
    def __init__(self, llm_descriptor: BaseChatModel = ChatOpenAI(model="gpt-4-vision-preview", temperature=0.5, max_tokens=1024), llm_filter: BaseChatModel = DocumentReviserToolNVIDIA()):
        self.__llm_descriptor = llm_descriptor
        self.__llm_filter = llm_filter

    def __encode_image(self, path: str):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def load(self, path: str):
        # load to scan ocr first if the image has very little text then use descriptor
        ocred_text = self.__load_ocr(path)
        if len(ocred_text) > 200:
            doc = Document(page_content=ocred_text)
        else:
            doc = Document(page_content=self.__llm_descriptor.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": IMAGE_DESCRIPTOR_PROMPT},
                    {"type": "text", "text": ocred_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{self.__encode_image(path)}"}}
                ])
            ]).content)
        image = Image.open(path)
        return doc, self.__encode_image(path)

    def __load_ocr(self, path: str):
        text = pytesseract.image_to_string(path)
        return self.__llm_filter.invoke(text)
