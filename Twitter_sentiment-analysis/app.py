from flask import Flask, render_template, request
import pickle
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# ------------------- Flask App -------------------
app = Flask(__name__)

# ------------------- NLTK Setup -------------------
# Ensure WordNet and stopwords are downloaded
nltk_data_dir = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
negation_words = ['not', "no", "never", "none", "n't"]

# ------------------- Text Cleaning -------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)               # remove mentions/hashtags
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words_processed = []
    skip_next = False

    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue
        if word in negation_words and i+1 < len(words):
            words_processed.append(word + '_' + words[i+1])
            skip_next = True
        elif word not in stop_words:
            words_processed.append(lemmatizer.lemmatize(word))

    return ' '.join(words_processed)

# ------------------- Load Model and Vectorizer -------------------
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("vectorizer.pkl or best_model.pkl not found. Run train_model.py first!")

# ------------------- Flask Routes -------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    tweet = ""
    if request.method == 'POST':
        tweet = request.form['tweet']
        if tweet.strip() != "":
            cleaned_tweet = clean_text(tweet)
            tweet_vec = vectorizer.transform([cleaned_tweet])
            pred = best_model.predict(tweet_vec)[0]

            # Normalize output
            if pred.lower() == 'positive':
                prediction = "Positive Sentiment"
            elif pred.lower() == 'negative':
                prediction = "Negative Sentiment"
            else:
                prediction = "Neutral Sentiment"
        else:
            prediction = "Please enter a valid tweet."

    return render_template('index.html', prediction=prediction, tweet=tweet)

# ------------------- Run Flask -------------------
if __name__ == '__main__':
    app.run(debug=True)
