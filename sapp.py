import streamlit as st
import pickle
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
with open("NB.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("DT.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("LR.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("RFC.pkl", "rb") as f:
    rfc_model = pickle.load(f)

# Function for text preprocessing
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Function for label output
def output_label(n):
    if n == 0:
        return "Valid"
    elif n == 1:
        return "Hoax"

# Load TfidfVectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Set Streamlit app title
st.set_page_config(page_title="Deteksi Berita Hoax")
st.title("Deteksi Berita Hoax")

# Text input for news
news = st.text_area("Masukkan Teks Berita:", height=100)

if st.button("Submit"):
    preprocessed_news = wordopt(news)

    # Prepare input data
    testing_news = {"text": [preprocessed_news]}
    new_def_test = pd.DataFrame(testing_news)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)

    # Perform predictions
    pred_nb = nb_model.predict(new_xv_test)
    pred_dt = dt_model.predict(new_xv_test)
    pred_lr = lr_model.predict(new_xv_test)
    pred_rfc = rfc_model.predict(new_xv_test)

    # Prepare output labels
    output_nb = output_label(pred_nb[0])
    output_dt = output_label(pred_dt[0])
    output_lr = output_label(pred_lr[0])
    output_rfc = output_label(pred_rfc[0])

    # Display classification results with border
    st.subheader("Hasil Klasifikasi:")
    with st.container():
        if pred_nb[0] == 0:
            st.success("**Naive Bayes**: Berita ini " + output_nb)
        else:
            st.error("**Naive Bayes**: Berita ini " + output_nb)
        if pred_dt[0] == 0:
            st.success("**Decision Tree**: Berita ini " + output_dt)
        else:
            st.error("**Decision Tree**: Berita ini " + output_dt)
        if pred_lr[0] == 0:
            st.success("**Logistic Regression**: Berita ini " + output_lr)
        else:
            st.error("**Logistic Regression**: Berita ini " + output_lr)
        if pred_rfc[0] == 0:
            st.success("**Random Forest Classifier**: Berita ini " + output_rfc)
        else:
            st.error("**Random Forest Classifier**: Berita ini " + output_rfc)

    # Display input news with border
    st.subheader("Teks Berita:")
    with st.container():
        st.info(news)
