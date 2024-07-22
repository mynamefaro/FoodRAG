import chainlit as cl


@cl.on_chat_start
def on_chat_start():
    cl.Message("Hello! I'm a bot!")
