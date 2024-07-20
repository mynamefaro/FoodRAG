from typing import Annotated, TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain.docstore.document import Document


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    question: str
    user_info: str
    documents: list[Document]
    translation: str
    error: list[str]
