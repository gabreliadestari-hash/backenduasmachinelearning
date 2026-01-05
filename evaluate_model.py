import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from model import ChatbotModel

def evaluate():
    print("Mulai Evaluasi Model...")
    
    # 1. Load Data
    print("Loading Dataset...")
    intents_path = os.path.join(os.path.dirname(__file__), "datasets.json")
    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. Prepare Data
    chatbot = ChatbotModel()
    X = []
    y = []
    
    print("Preprocessing Data...")
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Use the same preprocessing as the model
            processed_text = chatbot.preprocess(pattern)
            X.append(processed_text)
            y.append(intent["tag"])
            
    # 3. Split Data (80% Train, 20% Test)
    # Stratify ensures we have examples of each intent in both sets if possible
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Total Data: {len(X)}")
    print(f"Data Training: {len(X_train)}")
    print(f"Data Testing: {len(X_test)}")
    
    # 4. Train Model
    print("Training Model pada Data Training...")
    chatbot.model.fit(X_train, y_train)
    
    # 5. Evaluate
    print("Melakukan Prediksi pada Data Testing...")
    y_pred = chatbot.model.predict(X_test)
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print("\n" + "="*50)
    print(f"HASIL EVALUASI MODEL")
    print("="*50)
    print(f"Akurasi Model: {accuracy*100:.2f}%")
    print("\nDetail Laporan Klasifikasi:")
    print(report)
    print("="*50)

if __name__ == "__main__":
    evaluate()
