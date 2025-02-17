import os
import csv
import datetime
import nltk
import ssl
import pickle
import streamlit as st
import random
import json
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL context to avoid download issues
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

# Load intents files
with open('intents.json', 'r', encoding='utf-8') as file:
    intents_english = json.load(file)

with open('intents_marathi.json', 'r', encoding='utf-8') as file:
    intents_marathi = json.load(file)

# Function to get intents based on language selection
def get_intents(lang):
    return intents_marathi if lang == "Marathi" else intents_english

# Initialize vectorizers and models
vectorizer_eng, vectorizer_mar = TfidfVectorizer(), TfidfVectorizer()
clf_eng, clf_mar = LogisticRegression(max_iter=10000), LogisticRegression(max_iter=10000)

# Function to train the chatbot model
def train_model(intents, vectorizer, clf):
    patterns, tags = [], []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
    
    x = vectorizer.fit_transform(patterns)
    clf.fit(x, tags)

# Function to generate chatbot response
def chatbot_response(input_text, lang):
    intents = get_intents(lang)
    vectorizer = vectorizer_mar if lang == "Marathi" else vectorizer_eng
    clf = clf_mar if lang == "Marathi" else clf_eng

    try:
        input_text_vector = vectorizer.transform([input_text])
        tag = clf.predict(input_text_vector)[0]
        
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    except:
        return "क्षमस्व, मी समजू शकलो नाही. कृपया पुन्हा सांगा." if lang == "Marathi" else "Sorry, I couldn't understand. Can you please rephrase?"

# Train both models
train_model(intents_english, vectorizer_eng, clf_eng)
train_model(intents_marathi, vectorizer_mar, clf_mar)

# Chat log file
chat_log_file = "chat_log.csv"
if not os.path.exists(chat_log_file):
    with open(chat_log_file, "w", newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["User Input", "Chatbot Response", "Timestamp", "Language"])

# Streamlit UI
def main():
    st.title("FoodieGenie 🤖: Your Wish ✨, Our Dish 🍽️")
    
    # Sidebar Menu
    menu = ["Home 🍽", "Conversation History 📂", "About 📝"]
    choice = st.sidebar.selectbox("Menu 🧾", menu)

    # Language selection
    language = st.sidebar.radio("Select Language / भाषा निवडा", ("English", "Marathi"))

    if choice == "Home 🍽":
        st.subheader("Features of FoodieGenie 🛎️")
        if language == "English":
            st.write("- **Instant Dining Orders** 🍕\n- **Room Service Requests** 🛏️\n- **Personalized Recommendations** 🤖\n- **24/7 Availability** 🌙")
        else:
            st.write("- **त्वरित जेवणाची ऑर्डर** 🍕\n- **रूम सर्व्हिस विनंती** 🛏️\n- **वैयक्तिक शिफारसी** 🤖\n- **२४/७ उपलब्धता** 🌙")

        st.image('foodie.png', caption="FoodieGenie - Your Personal Assistant")
        
        user_input = st.text_input("You 🗣️" if language == "English" else "तुम्ही 🗣️")

        if user_input:
            with st.spinner("FoodieGenie is typing... 📝" if language == "English" else "FoodieGenie टाईप करत आहे... 📝"):
                time.sleep(1.5)
            
            response = chatbot_response(user_input, language)
            st.text_area("FoodieGenie:", value=response, height=120)

            # Save conversation to chat log
            with open(chat_log_file, "a", newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, datetime.datetime.now(), language])

    # Conversation History Section
    elif choice == "Conversation History 📂":
        st.header("📜 Previous Conversations")
        if os.path.exists(chat_log_file):
            df = pd.read_csv(chat_log_file)
            if not df.empty:
                st.dataframe(df)  # Display chat history in a table format
            else:
                st.write("No conversation history available.")
        else:
            st.write("No conversation history available.")

    # About Page
    elif choice == "About 📝":
        st.write("FoodieGenie is your personal dining assistant!")

if __name__ == '__main__':
    main()
