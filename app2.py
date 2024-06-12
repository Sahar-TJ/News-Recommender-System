import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import os


st.write("## Personalized News at Your Fingertips")


# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Specify the location of NLTK data explicitly
nltk.data.path.append(os.path.join(os.path.expanduser("~"), "nltk_data"))

# Load dataset
@st.cache_data
def load_data():
    combined_df = pd.read_csv('dataset.csv')
    combined_df = combined_df[~combined_df['Description'].str.contains('&lt;strong&gt;', na=False)]
    combined_df['content'] = combined_df['Title'] + ' ' + combined_df['Description']
    combined_df['Class Index'] = combined_df['Class Index'].astype(int) - 1
    return combined_df

combined_df = load_data()

# Initialize stop words, punctuation, and lemmatizer
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)
lemmatizer = WordNetLemmatizer()

# Function to clean text
def clean_text(text):
    if re.match(r'^\w+ - ', text):
        text = re.sub(r'^\w+ - ', '', text)
    text = text.strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = text.lower()
    text = ''.join([char for char in text if char not in punctuation])
    words = word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

combined_df['cleaned_content'] = combined_df['content'].apply(clean_text)

tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(combined_df['cleaned_content'])

news_classifier_tokenizer = BertTokenizer.from_pretrained("mrm8488/bert-mini-finetuned-age_news-classification")
news_classifier_model = BertForSequenceClassification.from_pretrained("mrm8488/bert-mini-finetuned-age_news-classification")

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Function to recommend articles
def recommend_articles(query, top_n=5):
    query = clean_text(query)
    query_tfidf = tfidf_vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    recommended_articles = combined_df.iloc[top_indices]
    return recommended_articles

# Function to classify news
def classify_news(text):
    inputs = news_classifier_tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    outputs = news_classifier_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    class_idx = torch.argmax(probs).item()
    class_mapping = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    return class_mapping.get(class_idx, 'Unknown')

# Function to extract keywords
def extract_keywords(text, top_n=5):
    text = clean_text(text)
    tokens = word_tokenize(text)
    fdist = nltk.FreqDist(tokens)
    keywords = [word for word, freq in fdist.most_common(top_n)]
    return keywords

st.title('News Recommender System')

query = st.text_input('Enter your query:', 'I want to know about the crude Oil.')

if 'recommended_articles' not in st.session_state:
    st.session_state['recommended_articles'] = None
if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = 0
if 'summary' not in st.session_state:
    st.session_state['summary'] = None

if st.button('Recommend'):
    st.session_state['recommended_articles'] = recommend_articles(query)
    st.session_state['selected_index'] = 0  # Reset the selected index
    st.session_state['summary'] = None

if st.session_state['recommended_articles'] is not None:
    st.write("Recommended Articles:")
    for idx, row in st.session_state['recommended_articles'].iterrows():
        st.write(f"**Title:** {row['Title']}")
        st.write(f"**Description:** {row['Description']}")
        st.write(f"**Class:** {classify_news(row['content'])}")
        st.write("-" * 80)

    selected_index = st.number_input('Select an article index to read more:', min_value=0, max_value=len(st.session_state['recommended_articles'])-1, step=1, key='selected_index')
    
    if st.button('Show Details'):
        selected_article = st.session_state['recommended_articles'].iloc[selected_index]

        keywords = extract_keywords(selected_article['content'])
        st.session_state['summary'] = summarizer(selected_article['content'], max_length=50, min_length=25, do_sample=False)[0]['summary_text']

        st.write(f"**Selected Article Title:** {selected_article['Title']}")
        st.write(f"**Description:** {selected_article['Description']}")
        st.write(f"**Keywords:** {keywords}")

    if st.session_state['summary']:
        st.write(f"**Summary:** {st.session_state['summary']}")

    # Slider to select different summaries
    summary_length = st.slider('Select summary length:', 25, 200, 50, step=25)
    if 'summary' in st.session_state:
        selected_article = st.session_state['recommended_articles'].iloc[selected_index]
        st.session_state['summary'] = summarizer(selected_article['content'], max_length=summary_length, min_length=int(summary_length/2), do_sample=False)[0]['summary_text']
        st.write(f"**Summary:** {st.session_state['summary']}")
