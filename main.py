from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import ChatbotModel

app = FastAPI(title="Toko Kue Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chatbot = ChatbotModel()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float

@app.get("/")
def root():
    return {"status": "Toko Kue Chatbot API aktif"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = chatbot.predict(request.message)
    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        confidence=result["confidence"]
    )

@app.get("/health")
def health():
    return {"status": "healthy"}
