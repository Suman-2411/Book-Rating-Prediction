import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and everything it needs to make a prediction.
model = joblib.load('../models/best_model.pkl')
model_columns = joblib.load('../models/model_columns.pkl')
author_lookup = joblib.load('../models/author_lookup.pkl')
publisher_lookup = joblib.load('../models/publisher_lookup.pkl')
author_freq_lookup = joblib.load('../models/author_freq_lookup.pkl')
publisher_freq_lookup = joblib.load('../models/publisher_freq_lookup.pkl')
overall_avg = joblib.load('../models/overall_avg.pkl')

# Make lowercase copies so "Stephen King" and "stephen king" both match the same entry.
author_lookup_lower = {k.strip().lower(): v for k, v in author_lookup.items()}
publisher_lookup_lower = {k.strip().lower(): v for k, v in publisher_lookup.items()}
author_freq_lookup_lower = {k.strip().lower(): v for k, v in author_freq_lookup.items()}
publisher_freq_lookup_lower = {k.strip().lower(): v for k, v in publisher_freq_lookup.items()}

# Page title and short intro.
st.set_page_config(page_title="Book Rating Predictor", page_icon="📚")
st.title("📚 Book Rating Predictor")
st.write("Enter some details about a book, and we'll predict its average rating.")

# Ask the user for the everyday details they'd actually know about a book.
author = st.text_input("Author name", placeholder="e.g. Stephen King")
publisher = st.text_input("Publisher name", placeholder="e.g. Penguin Books")
num_pages = st.number_input("Number of pages", min_value=1, max_value=5000, value=300)
ratings_count = st.number_input("Number of ratings the book has received", min_value=0, value=100)
text_reviews_count = st.number_input("Number of text reviews the book has received", min_value=0, value=20)
publication_year = st.number_input("Publication year", min_value=1500, max_value=2026, value=2010)
language = st.selectbox("Language", ["English", "French", "German", "Spanish", "Japanese", "Others"])

if st.button("Predict Rating"):

    # Clean up the typed names so casing doesn't cause a false miss.
    author_key = author.strip().lower()
    publisher_key = publisher.strip().lower()

    # Check upfront whether we actually recognize this author/publisher.
    author_known = author_key in author_lookup_lower
    publisher_known = publisher_key in publisher_lookup_lower

    # Use the real historical rating if we know them, otherwise fall back to the average.
    author_enc = author_lookup_lower.get(author_key, overall_avg)
    publisher_enc = publisher_lookup_lower.get(publisher_key, overall_avg)
    author_freq = author_freq_lookup_lower.get(author_key, 0)
    publisher_freq = publisher_freq_lookup_lower.get(publisher_key, 0)

    # Rebuild the same features the model was trained on.
    age = 2026 - publication_year
    ratings_count_log = np.log1p(ratings_count)
    text_reviews_count_log = np.log1p(text_reviews_count)

    # Put it all into one row, matching the model's expected columns.
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

    # Line up the columns exactly the way the model expects, then predict.
    input_df = input_df[model_columns]
    prediction = model.predict(input_df)[0]

    # Show clear, hard-to-miss warnings first if we didn't recognize the author/publisher,
    # so the user understands the prediction leaned on a fallback, not real data.
    if not author_known:
        st.warning(f"⚠️ Author data not available for \"{author}\" — using the dataset average instead.")
    if not publisher_known:
        st.warning(f"⚠️ Publisher data not available for \"{publisher}\" — using the dataset average instead.")

    st.success(f"Predicted Average Rating: **{prediction:.2f} / 5**")