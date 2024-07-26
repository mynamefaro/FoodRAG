import chainlit as cl
import uuid
from chromadb import HttpClient
from chromadb.config import Settings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from app.components.core import GraphState
from app.components.database import BaseDatabaseToolkit, BaseVectorDatabaseToolkit
from app.components.agent import FoodDataAgent


chat_memory = SqliteSaver.from_conn_string("database/chat_memory.db")


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    tools_called = event.get("tool_calls")
    if tools_called:
        for tool in tools_called:
            print(tool)
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


@cl.cache
def get_food_agent():
    food_chroma_client = HttpClient(
        host="chroma", settings=Settings(anonymized_telemetry=False))
    food_db = BaseDatabaseToolkit(
        sql_database_url="sqlite:///database/usda.db")
    food_vector = BaseVectorDatabaseToolkit(food_chroma_client)
    return FoodDataAgent(food_db, food_vector)


@cl.on_chat_start
def on_chat_start():
    builder = StateGraph(GraphState)
    builder.add_node("FOOD_DATA_AGENT", get_food_agent())
    builder.add_edge(START, "FOOD_DATA_AGENT")
    builder.add_edge("FOOD_DATA_AGENT", END)
    thread_id = str(uuid.uuid4())
    assistant = builder.compile(checkpointer=chat_memory)
    cl.user_session.set("assistant", assistant)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("config", {"configurable": {"thread_id": thread_id}})
    cl.user_session.set("messages", set())


@cl.on_message
async def on_message(message: cl.Message):
    assistant: CompiledGraph = cl.user_session.get("assistant")
    config = cl.user_session.get("config")

    events = assistant.stream({"messages": ("user", message.content)},
                              config, stream_mode="values")
    messages = cl.user_session.get("messages")
    print("Events started!!")
    for event in events:
        _print_event(event, messages)
    print("Events ended!!")
    snapshot = assistant.get_state(config)
    while snapshot.next:
        print(snapshot)
        snapshot = assistant.get_state(config)
    print("Snapshot ended!!")
    await cl.Message(event.get("messages")[-1].content).send()
