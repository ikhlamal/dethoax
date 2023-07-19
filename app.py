from flask import Flask, render_template, request
import pickle
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

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

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Classification route
@app.route('/classify', methods=['POST'])
def classify():
    news = request.form['news']
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
    output_rfc = output_label(pred_lr[0])

    return render_template('index.html', news=news, output_nb=output_nb, output_dt=output_dt, 
        output_lr=output_lr, output_rfc=output_rfc)

if __name__ == '__main__':
    app.run()
