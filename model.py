import json
import random
import joblib
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class ChatbotModel:
    def __init__(self):
        self.model = None
        self.intents = None
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
        self.load_model()

    def preprocess(self, text):
        text = text.lower().strip()
        text = self.stopword_remover.remove(text)
        text = self.stemmer.stem(text)
        return text

    def load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            self.model = data["model"]
            self.intents = data["intents"]
        else:
            self.train()

    def train(self):
        intents_path = os.path.join(os.path.dirname(__file__), "datasets.json")
        with open(intents_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.intents = {intent["tag"]: intent["responses"] for intent in data["intents"]}
        
        X = []
        y = []
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                X.append(self.preprocess(pattern))
                y.append(intent["tag"])
        
        # Optimized pipeline
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1  # Include all words basically since dataset is small
            )),
            ("clf", MultinomialNB(alpha=0.1))  # Lower alpha for less smoothing
        ])
        self.model.fit(X, y)
        
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        joblib.dump({"model": self.model, "intents": self.intents}, model_path)

    def predict(self, text):
        processed = self.preprocess(text)
        intent = self.model.predict([processed])[0]
        # Get max probability
        proba = max(self.model.predict_proba([processed])[0])
        
        # Threshold logic
        if proba < 0.15: # If confidence is too low
            # Try to see if it matches any pattern exactly after preprocessing
            # Or just fall back, but for now we trust the model with better tuning
            pass

        response = random.choice(self.intents.get(intent, ["Maaf, saya tidak mengerti"]))
        return {"intent": intent, "response": response, "confidence": float(proba)}
