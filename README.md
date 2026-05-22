# 🌾 KisanMitra — AI Agriculture Advisory

AI-powered agriculture advisory system for Indian farmers, supporting Telugu, Tenglish, Hindi, and English.

## Features

- 🌤️ **Real-time weather data** via Open-Meteo API
- 📊 **Crop knowledge base** with cost/profit analysis (12 crops)
- 🐛 **Pest & disease advisory** based on weather conditions
- 💧 **Irrigation recommendations** with water balance calculations
- 🌱 **Crop selection guidance** for any Indian location
- 🗣️ **Multilingual support** — Telugu, Tenglish, Hindi, English

## Tech Stack

- **AI**: Claude Opus 4.6 (Anthropic)
- **Framework**: LangGraph (agent orchestration), LangChain (LLM integration)
- **RAG**: ChromaDB + sentence-transformers embeddings
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Weather**: Open-Meteo API (free, no key required)

## Supported Crops

Paddy, Cotton, Groundnut, Chilli, Sugarcane, Maize, Turmeric, Banana, Tomato, Onion, Mango, Tobacco

## Architecture

```
Farmer Question (Telugu/Hindi/English)
    ↓
Haiku (parse intent + location + crop)
    ↓
Geocoding (Open-Meteo)
    ↓
Weather Data + RAG Knowledge Search
    ↓
Opus (generate Telugu/Hindi/English response)
    ↓
Translation fallback (if needed)
```

## Local Development

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here

# Terminal 1: Start API
python kisanmitra_api.py

# Terminal 2: Start UI
streamlit run kisanmitra_app.py
```

## Built by

Abhishek Vatsa — AI Engineer
