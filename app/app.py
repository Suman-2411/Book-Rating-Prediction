import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('../models/best_model.pkl')
model_columns = joblib.load('../models/model_columns.pkl')
author_lookup = joblib.load('../models/author_lookup.pkl')
publisher_lookup = joblib.load('../models/publisher_lookup.pkl')
author_freq_lookup = joblib.load('../models/author_freq_lookup.pkl')
publisher_freq_lookup = joblib.load('../models/publisher_freq_lookup.pkl')
overall_avg = joblib.load('../models/overall_avg.pkl')

# Converting names to lower case to handle the case sensitivity while prediction
author_lookup = {k.strip().lower(): v for k, v in author_lookup.items()}
publisher_lookup = {k.strip().lower(): v for k, v in publisher_lookup.items()}
author_freq_lookup = {k.strip().lower(): v for k, v in author_freq_lookup.items()}
publisher_freq_lookup = {k.strip().lower(): v for k, v in publisher_freq_lookup.items()}

st.markdown("""
    <style>
    .stApp {
        background-color: #0d0d0d;
        color: #ffffff;
    }
    h1 {
        color: #e63946;
    }
    h2, h3 {
        color: #4361ee;
    }
    .stButton > button {
        background-color: #e63946;
        color: white;
        border: none;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #4361ee;
        color: white;
    }
    .stExpander {
        border: 1px solid #4361ee;
        border-radius: 6px;
    }
    label {
        color: #4361ee !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Book Rating Predictor")

with st.expander("About this project"):
    st.write("""
    this predicts what rating a book might get based on stuff like the author,
    publisher, page count, how many ratings/reviews it already has, etc.

    we tried 3 models - Linear Regression, Decision Tree and Gradient Boosting.
    Gradient Boosting did the best so that's what's running here, but honestly
    none of them were amazing (R2 was only around 0.17). makes sense though,
    the EDA showed our features don't correlate much with rating, so a lot of
    what actually decides a books rating (like genre or what people actually
    say in reviews) just isnt in this data.

    also if you type in an author or publisher the model has never seen before,
    it just uses the average rating instead since it has nothing to go on.
    """)

st.write("Enter some book details below and it will predict the average rating.")

author = st.text_input("Author name")
publisher = st.text_input("Publisher name")
num_pages = st.number_input("Number of pages", min_value=1, max_value=5000, value=300)
ratings_count = st.number_input("Number of ratings", min_value=0, value=100)
text_reviews_count = st.number_input("Number of text reviews", min_value=0, value=20)
publication_year = st.number_input("Publication year", min_value=1500, max_value=2026, value=2010)
language = st.selectbox("Language", ["English", "French", "German", "Spanish", "Japanese", "Others"])

if st.button("Predict Rating"):

    author_key = author.strip().lower()
    publisher_key = publisher.strip().lower()

    author_known = author_key in author_lookup
    publisher_known = publisher_key in publisher_lookup

    # if the author ot publisher or not founf in dataset then it will return datasets average rating
    author_enc = author_lookup.get(author_key, overall_avg)
    publisher_enc = publisher_lookup.get(publisher_key, overall_avg)
    author_freq = author_freq_lookup.get(author_key, 0)
    publisher_freq = publisher_freq_lookup.get(publisher_key, 0)

    age = 2026 - publication_year
    ratings_count_log = np.log1p(ratings_count)
    text_reviews_count_log = np.log1p(text_reviews_count)

    input_dict = {
        'Num_Pages': num_pages,
        'Age': age,
        'Author_Frequency': author_freq,
        'Publisher_Frequency': publisher_freq,
        'Author_TargetEnc': author_enc,
        'Publisher_TargetEnc': publisher_enc,
        'Ratings_Count_Log': ratings_count_log,
        'Text_Reviews_Count_Log': text_reviews_count_log,
        'Language_Code_French': 1 if language == 'French' else 0,
        'Language_Code_German': 1 if language == 'German' else 0,
        'Language_Code_Japanese': 1 if language == 'Japanese' else 0,
        'Language_Code_Others': 1 if language == 'Others' else 0,
        'Language_Code_Spanish': 1 if language == 'Spanish' else 0,
    }
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_columns] 

    prediction = model.predict(input_df)[0]

    if not author_known:
        st.warning(f"Never seen \"{author}\" in the training data, using the overall average instead.")
    if not publisher_known:
        st.warning(f"Never seen \"{publisher}\" in the training data, using the overall average instead.")

    st.success(f"Predicted rating: {prediction:.2f} / 5")