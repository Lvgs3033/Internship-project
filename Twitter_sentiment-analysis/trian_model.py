import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pickle

# NLTK setup (download once)
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
negation_words = ['not', "no", "never", "none", "n't"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
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

# Load dataset
train_df = pd.read_csv('twitter_training.csv', header=None, names=['id','entity','sentiment','text'])
validation_df = pd.read_csv('twitter_validation.csv', header=None, names=['id','entity','sentiment','text'])
train_df.dropna(inplace=True)
validation_df.dropna(inplace=True)

# Clean text
train_df['clean_text'] = train_df['text'].apply(clean_text)
validation_df['clean_text'] = validation_df['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(train_df['clean_text'])
X_validation_vec = vectorizer.transform(validation_df['clean_text'])
y_train = train_df['sentiment']
y_val = validation_df['sentiment']

# Train models
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_vec, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Select best model based on F1-score
model_scores = {
    "Logistic Regression": f1_score(y_val, lr_model.predict(X_validation_vec), average='weighted'),
    "Random Forest": f1_score(y_val, rf_model.predict(X_validation_vec), average='weighted'),
    "Naive Bayes": f1_score(y_val, nb_model.predict(X_validation_vec), average='weighted')
}
best_model_name = max(model_scores, key=model_scores.get)
best_model = {"Logistic Regression": lr_model,
              "Random Forest": rf_model,
              "Naive Bayes": nb_model}[best_model_name]

print(f"Best model: {best_model_name}")

# Save vectorizer and best model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
