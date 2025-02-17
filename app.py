import os
import csv
import datetime
import nltk
import ssl
import pickle
import streamlit as st
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time

# SSL context to avoid download issues
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

# Load intents files
with open('intents.json', 'r', encoding='utf-8') as file:
    intents_english = json.load(file)

with open('intents_marathi.json', 'r', encoding='utf-8') as file:
    intents_marathi = json.load(file)

# Language selection
def get_intents(lang):
    return intents_marathi if lang == "Marathi" else intents_english

# Initialize vectorizers and models
vectorizer_eng, vectorizer_mar = TfidfVectorizer(), TfidfVectorizer()
clf_eng, clf_mar = LogisticRegression(max_iter=10000), LogisticRegression(max_iter=10000)

def train_model(intents, vectorizer, clf):
    tags, patterns = [], []
    for intent in intents:
        for pattern in intent['patterns']:
            tags.append(intent['tags'])
            patterns.append(pattern)
    x = vectorizer.fit_transform(patterns)
    y = tags
    clf.fit(x, y)

def chatbot_response(input_text, lang):
    intents = get_intents(lang)
    vectorizer = vectorizer_mar if lang == "Marathi" else vectorizer_eng
    clf = clf_mar if lang == "Marathi" else clf_eng
    try:
        input_text = vectorizer.transform([input_text])
        tag = clf.predict(input_text)[0]
        for intent in intents:
            if tag in intent['tags']:
                return random.choice(intent['responses'])
    except:
        return "क्षमस्व, मी ते समजू शकलो नाही. कृपया पुन्हा सांगा." if lang == "Marathi" else "I'm sorry, I couldn't understand that. Could you please rephrase?"

# Train both models
train_model(intents_english, vectorizer_eng, clf_eng)
train_model(intents_marathi, vectorizer_mar, clf_mar)

# Streamlit UI
def main():
    st.title("FoodieGenie 🤖: Your Wish ✨, Our Dish 🍽️")
    language = st.sidebar.radio("Select Language / भाषा निवडा", ("English", "Marathi"))
    
    st.subheader("Features of FoodieGenie 🛎️")
    if language == "English":
        st.write("- **Instant Dining Orders** 🍕\n- **Room Service Requests** 🛏️\n- **Personalized Recommendations** 🤖\n- **24/7 Availability** 🌙")
    else:
        st.write("- **त्वरित जेवणाची ऑर्डर** 🍕\n- **रूम सर्व्हिस विनंती** 🛏️\n- **वैयक्तिक शिफारसी** 🤖\n- **२४/७ उपलब्धता** 🌙")
    
    if st.button("Start Chatting with FoodieGenie 👨🏻‍🍳"):
        st.write("नमस्कार! मी FoodieGenie आहे, तुमचा व्यक्तिगत सहाय्यक. मी कशाने मदत करू?" if language == "Marathi" else "Hello! I am FoodieGenie, your personal assistant. How can I help you today?")
    
    st.image('foodie.png', caption="FoodieGenie - Your Personal Assistant")
    
    user_input = st.text_input("You 🗣️" if language == "English" else "तुम्ही 🗣️")
    
    if user_input:
        with st.empty():
            st.write("FoodieGenie is typing... 📝" if language == "English" else "FoodieGenie टाईप करत आहे... 📝")
            time.sleep(2)
        
        response = chatbot_response(user_input, language)
        st.text_area("FoodieGenie:", value=response, height=120)

if __name__ == '__main__':
    main()
