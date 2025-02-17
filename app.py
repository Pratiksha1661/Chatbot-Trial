import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import random

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("foodiegenie_dataset.csv")

df = load_data()

# Function to get dish recommendations based on query
def get_dish_recommendation(query, language):
    query_tokens = word_tokenize(query.lower())

    for _, row in df.iterrows():
        if any(token in row['Dish Name'].lower() for token in query_tokens):
            if language == "Marathi":
                return f"**{row['Dish Name']}**: {row['Description (Marathi)']}"
            else:
                return f"**{row['Dish Name']}**: {row['Description (English)']}"

    return "क्षमस्व! मला हा पदार्थ माहित नाही. | Sorry! I don't know this dish."

# Streamlit UI
st.set_page_config(page_title="FoodieGenie 🤖", layout="centered")

st.title("🍽️ FoodieGenie 🤖: Your Wish ✨, Our Dish")
st.write("Welcome to **FoodieGenie**, your personalized Maharashtrian cuisine assistant!")

# Language selection
language = st.radio("Select Language | भाषा निवडा", ["English", "Marathi"])

# User input
user_query = st.text_input("Enter your query | आपली चौकशी टाइप करा", "")

# Process user query
if st.button("Ask FoodieGenie!"):
    if user_query:
        response = get_dish_recommendation(user_query, language)
        st.success(response)
    else:
        st.warning("Please enter a query! | कृपया चौकशी करा!")

# Footer
st.write("🤖 Powered by AI | Created for Maharashtrian 5-star hotels")
