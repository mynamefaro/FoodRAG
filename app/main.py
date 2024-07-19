import os
import streamlit as st

from components.llm.nvidia import Model
from utils.utils import load_env


@st.cache_resource
def load_model():
    return Model()


def main():
    load_env()
    MODEL = load_model()
    print(os.getenv("NVIDIA_API_KEY"))

    def runner():
        st.title("Hello, Streamlit!")
        st.write("This is a simple Streamlit app.")
        st.write("You can write text, draw plots, and more.")
        input = st.text_input("Enter some text:")
        if st.button("Run model"):
            output = MODEL.invoke(input)
            st.write(f"Model output: {output}")

    runner()


if __name__ == "__main__":
    main()
