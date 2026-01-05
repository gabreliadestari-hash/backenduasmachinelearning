from model import ChatbotModel

if __name__ == "__main__":
    print("Memulai training model...")
    model = ChatbotModel()
    model.train()
    print("Model berhasil disimpan ke model.pkl")
