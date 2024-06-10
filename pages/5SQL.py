import streamlit as st
import ast
import os
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
import pandas as pd

db = SQLDatabase.from_uri("sqlite:///food.db")
print(db.dialect)
print(db.get_usable_table_names())

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

write_query = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)
answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)
answer = answer_prompt | llm | StrOutputParser()
init_chain = RunnablePassthrough.assign(query=write_query)
get_query = (init_chain | write_query)
exe_chain = init_chain.assign(result=itemgetter("query") | execute_query)
chain = (exe_chain | answer)


# Streamlit app layout
st.title('SQL powered Chatbot')
st.write('This is a simple SQL powered ChatBot.')

# displace top 5 recent records
data_string = db.run("select * from food_sample ORDER BY date DESC, time DESC LIMIT 5;")
# print(f"data:{data_string}")

columns = ['Date', 'Time', 'Calorie', 'Mass', 'Fat', 'Carb', 'Protein', 'Food Identified']
# columns = ['date', 'time', 'calorie', 'mass', 'fat', 'carb', 'protein', 'food_identified']
df = pd.DataFrame(ast.literal_eval(data_string), columns=columns)
st.table(df)

st.code("""
what do i eat least?
""")

user_input = st.text_input("Type your message here:")

if user_input:
    # Generate response
    print(f"user_input:{user_input}")
    response = chain.invoke({"question": user_input})
    print(f"response:{response}")
    st.text_area("Response:", value=response, height=200, max_chars=None, key=None)
    st.text_area("SQL:", value=(exe_chain.invoke({"question": user_input})), height=200, max_chars=None, key=None)