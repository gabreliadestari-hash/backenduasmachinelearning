from model import ChatbotModel

def debug():
    bot = ChatbotModel()
    
    texts = [
        "saya ingin memesan kue ulang tahun",
        "saya ingin memesan ukuran 28cm dong"
    ]
    
    print("\n--- Diagnostic ---")
    for text in texts:
        print(f"\nInput: '{text}'")
        preprocessed = bot.preprocess(text)
        print(f"Preprocessed: '{preprocessed}'")
        
        result = bot.predict(text)
        print(f"Predicted Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        # Show top 3
        proba = bot.model.predict_proba([preprocessed])[0]
        classes = bot.model.classes_
        top3_idx = proba.argsort()[-3:][::-1]
        print("Top 3 Candidates:")
        for idx in top3_idx:
            print(f"  - {classes[idx]}: {proba[idx]:.4f}")

if __name__ == "__main__":
    debug()
