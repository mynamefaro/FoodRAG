import ast
import streamlit as st
import pandas as pd
from langchain_community.utilities import SQLDatabase

st.sidebar.header("Data Visualization")

@st.cache_resource
def start_sql_db():
    db = SQLDatabase.from_uri("sqlite:///food.db")
    print(db.dialect)
    print(db.get_usable_table_names())
    return db

columns = ["Date", "Time", "Calorie", "Mass", "Fat", "Carb",
           "Protein", "Food Identified"]
db = start_sql_db()

st.header("Nutrition History")
# displace top 5 recent records
data_string = db.run("select * from food_sample ORDER BY date DESC, time DESC LIMIT 5;")
# print(f"data:{data_string}")

columns = ['Date', 'Time', 'Calorie', 'Mass', 'Fat', 'Carb', 'Protein', 'Food Identified']
# columns = ['date', 'time', 'calorie', 'mass', 'fat', 'carb', 'protein', 'food_identified']
df = pd.DataFrame(ast.literal_eval(data_string), columns=columns)
st.table(df)

# get all records
data_string = db.run("select * from food_sample ORDER BY date DESC, time DESC;")
# print(f"data:{data_string}")

columns = ['Date', 'Time', 'Calorie', 'Mass', 'Fat', 'Carb', 'Protein', 'Food Identified']
chart_data = pd.DataFrame(ast.literal_eval(data_string), columns=columns)


# chart_data = pd.DataFrame(data, columns=columns[:-1])
st.header("Nutrition Trend")

st.line_chart(chart_data, x="Date", y=[
              "Calorie", "Mass", "Fat", "Carb", "Protein"])

food_counts_pandas = chart_data['Food Identified'].str.split(
    ', ').explode().value_counts()

# Convert the dictionary to a pandas Series (you can skip this if you already have a Series)
food_counts_pandas = pd.Series(food_counts_pandas)

st.header("Total Unique Food Item Count")
# Display the bar chart
st.bar_chart(food_counts_pandas)
