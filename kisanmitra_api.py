"""
KisanMitra API — FastAPI Server
Run: python kisanmitra_api.py
Then open: http://localhost:8000/docs (Swagger UI)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Import the agent
from kisanmitra_agent import ask

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="KisanMitra API",
    description="AI Agriculture Advisory for Indian Farmers — Weather + Crop Knowledge",
    version="1.0.0"
)

# Allow all origins (for Streamlit, Android, any frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================
class QuestionRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Guntur lo paddy ki neellu pettala?"
            }
        }


class AdvisoryResponse(BaseModel):
    question: str
    language: str
    location: str
    state: str
    crop: str
    intent: str
    has_knowledge: bool
    response: str


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def home():
    """Health check"""
    return {
        "app": "KisanMitra API",
        "status": "running",
        "version": "1.0.0",
        "usage": "POST /ask with {\"question\": \"your farming question\"}"
    }


@app.post("/ask", response_model=AdvisoryResponse)
def ask_question(req: QuestionRequest):
    """
    Ask KisanMitra a farming question in Telugu, Tenglish, Hindi, or English.

    Examples:
    - "Guntur lo paddy ki neellu pettala?"
    - "కర్నూలు లో పత్తి పంటకి పురుగు మందు కొట్టాలా?"
    - "nenu komarada lo sugarcane veste laabham avutunda?"
    - "అనంతపురం లో ఈ సీజన్ కి ఏం వేయాలి?"
    - "What is the weather forecast for Hyderabad this week?"
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = ask(req.question)
        return AdvisoryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/ask")
def ask_question_get(question: str):
    """
    GET version — for quick browser testing.
    Usage: http://localhost:8000/ask?question=Guntur lo paddy ki neellu pettala?
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = ask(question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# ============================================================
# Run server
# ============================================================
if __name__ == "__main__":
    print("\n🌾 KisanMitra API starting...")
    print("📍 Open http://localhost:8000/docs for Swagger UI")
    print("📍 Or test: http://localhost:8000/ask?question=Guntur+lo+paddy+ki+neellu+pettala\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
