"""
KisanMitra API LITE — Lightweight FastAPI server for Render Free Tier.
Uses kisanmitra_agent_lite (no heavy ML dependencies).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import kisanmitra_agent_lite as agent

app = FastAPI(
    title="KisanMitra API",
    description="AI Agriculture Advisory for Indian Farmers",
    version="1.0",
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "KisanMitra API is running 🌾", "version": "1.0-lite"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask")
def ask_question(req: QuestionRequest):
    result = agent.ask(req.question)
    return result


@app.get("/ask")
def ask_question_get(question: str):
    result = agent.ask(question)
    return result
