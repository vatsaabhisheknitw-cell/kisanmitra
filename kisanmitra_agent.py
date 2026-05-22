"""
KisanMitra Agent — v4 Final (Language Fix Complete)

All 5 fixes applied:
1. Temperature 0.0 for strict instruction following
2. Telugu-first system prompt (in Telugu when farmer speaks Telugu)
3. Few-shot example in Telugu in system prompt
4. Pre-fill assistant response with Telugu opening
5. Two-step translation fallback (if response still in English)
"""

import os
import json
import re
import requests
from typing import TypedDict, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END


# ============================================================
# MODELS — temperature 0.0 for strict language compliance
# ============================================================
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
llm_smart = ChatAnthropic(model="claude-opus-4-6", temperature=0.0, max_tokens=2048)
llm_translator = ChatAnthropic(model="claude-opus-4-6", temperature=0.0, max_tokens=2048)


# ============================================================
# GEOCODING
# ============================================================
def geocode_location(location_name: str) -> dict:
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=5&language=en&format=json"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "results" not in data or len(data["results"]) == 0:
            return {"error": f"Location '{location_name}' not found"}
        for r in data["results"]:
            if r.get("country_code") == "IN":
                return {"name": r["name"], "lat": r["latitude"], "lon": r["longitude"],
                        "state": r.get("admin1", "India"), "country": "India", "found": True}
        r = data["results"][0]
        return {"name": r["name"], "lat": r["latitude"], "lon": r["longitude"],
                "state": r.get("admin1", "Unknown"), "country": r.get("country", "Unknown"), "found": True}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# WEATHER FUNCTIONS
# ============================================================
def get_weather_by_coords(lat, lon, location_name, state):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
           f"precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m&timezone=Asia/Kolkata")
    weather_codes = {0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                     45: "Foggy", 51: "Light drizzle", 53: "Moderate drizzle",
                     61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
                     80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
                     95: "Thunderstorm", 96: "Thunderstorm with hail"}
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return {"error": f"Weather API failed: {resp.status_code}"}
    data = resp.json()
    c = data["current"]
    return {"location": location_name, "state": state, "timestamp": c["time"],
            "temperature_c": c["temperature_2m"], "feels_like_c": c["apparent_temperature"],
            "humidity_percent": c["relative_humidity_2m"], "rain_mm": c["rain"],
            "weather_condition": weather_codes.get(c["weather_code"], "Unknown"),
            "wind_speed_kmh": c["wind_speed_10m"]}


def get_agri_by_coords(lat, lon, location_name, state):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&current=temperature_2m,relative_humidity_2m,rain,soil_temperature_0cm,soil_moisture_0_to_1cm"
           f"&daily=temperature_2m_max,temperature_2m_min,rain_sum,precipitation_probability_max,"
           f"et0_fao_evapotranspiration&timezone=Asia/Kolkata&forecast_days=7")
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return {"error": f"Agri API failed: {resp.status_code}"}
    data = resp.json()
    current = data["current"]
    daily = data["daily"]
    rain_3d = sum(daily["rain_sum"][i] for i in range(min(3, len(daily["rain_sum"]))))
    et0_3d = sum(daily["et0_fao_evapotranspiration"][i] for i in range(min(3, len(daily["et0_fao_evapotranspiration"]))))
    water_balance = rain_3d - et0_3d
    max_temps = [daily["temperature_2m_max"][i] for i in range(min(3, len(daily["temperature_2m_max"])))]
    return {"location": location_name, "state": state, "timestamp": current["time"],
            "current_temp_c": current["temperature_2m"],
            "current_humidity_percent": current["relative_humidity_2m"],
            "current_rain_mm": current["rain"],
            "soil_temp_c": current.get("soil_temperature_0cm"),
            "soil_moisture": current.get("soil_moisture_0_to_1cm"),
            "rain_next_3_days_mm": round(rain_3d, 1),
            "et0_next_3_days_mm": round(et0_3d, 1),
            "water_balance_mm": round(water_balance, 1),
            "daily_forecast": [
                {"date": daily["time"][i], "temp_max_c": daily["temperature_2m_max"][i],
                 "temp_min_c": daily["temperature_2m_min"][i], "rain_mm": daily["rain_sum"][i],
                 "rain_probability": daily["precipitation_probability_max"][i],
                 "et0_mm": daily["et0_fao_evapotranspiration"][i]}
                for i in range(len(daily["time"]))],
            "alerts": {"irrigation_needed": water_balance < -5,
                       "heat_stress_risk": any(t > 40 for t in max_temps),
                       "fungal_disease_risk": current["relative_humidity_2m"] > 85,
                       "heavy_rain_expected": rain_3d > 50}}


def get_forecast_by_coords(lat, lon, location_name, state):
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&daily=temperature_2m_max,temperature_2m_min,rain_sum,precipitation_probability_max,"
           f"wind_speed_10m_max,uv_index_max&timezone=Asia/Kolkata&forecast_days=7")
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return {"error": f"Forecast API failed: {resp.status_code}"}
    data = resp.json()
    daily = data["daily"]
    return {"location": location_name, "state": state,
            "forecast": [{"date": daily["time"][i], "temp_max_c": daily["temperature_2m_max"][i],
                          "temp_min_c": daily["temperature_2m_min"][i], "rain_mm": daily["rain_sum"][i],
                          "rain_probability": daily["precipitation_probability_max"][i],
                          "wind_max_kmh": daily["wind_speed_10m_max"][i], "uv_index": daily["uv_index_max"][i]}
                         for i in range(len(daily["time"]))]}


# ============================================================
# CROP KNOWLEDGE BASE (12 crops in Telugu)
# ============================================================
CROP_KNOWLEDGE = [
    {"crop": "paddy", "content": """పంట: వరి (Paddy)\nరాష్ట్రం: ఆంధ్రప్రదేశ్, తెలంగాణ\nసీజన్లు: ఖరీఫ్ (June-July), రబీ (November-December)\nనేల: నల్లరేగడి, ఒండ్రు మట్టి. pH 5.5-7.0\nనీటి అవసరం: 1200-1400mm. 30-40 సార్లు\nMSP (2025-26): ₹2,300/quintal (common), ₹2,320/quintal (Grade A)\nదిగుబడి: ఎకరాకు 20-25 quintals\nఆదాయం: ₹46,000-57,500/ఎకరా\nపెట్టుబడి ఖర్చు: ₹20,000-28,000/ఎకరా\nనికర లాభం: ₹18,000-37,500/ఎకరా\nతెగుళ్ళు: అగ్గి తెగులు, గొడ్డు తెగులు, దోమ పోటు\nజిల్లాలు (AP): కృష్ణా, గుంటూరు, తూర్పు గోదావరి, పశ్చిమ గోదావరి, నెల్లూరు\nజిల్లాలు (Telangana): నల్గొండ, కరీంనగర్, వరంగల్"""},
    {"crop": "cotton", "content": """పంట: పత్తి (Cotton)\nసీజన్: ఖరీఫ్ (June-July)\nనేల: నల్లరేగడి (black soil), pH 6.0-8.0\nనీటి అవసరం: 700-1200mm\nMSP (2025-26): ₹7,521/quintal (medium), ₹7,121/quintal (long)\nదిగుబడి: ఎకరాకు 8-12 quintals\nఆదాయం: ₹57,000-90,000/ఎకరా\nపెట్టుబడి ఖర్చు: ₹25,000-35,000/ఎకరా\nనికర లాభం: ₹22,000-65,000/ఎకరా\nతెగుళ్ళు: బోల్ వార్మ్, తెల్ల దోమ, రసం పీల్చే పురుగులు, గులాబీ పురుగు\nజిల్లాలు (AP): కర్నూలు, గుంటూరు, ప్రకాశం, అనంతపురం\nజిల్లాలు (Telangana): ఆదిలాబాద్, వరంగల్, ఖమ్మం"""},
    {"crop": "groundnut", "content": """పంట: వేరుశనగ (Groundnut)\nసీజన్లు: ఖరీఫ్ (June-July), రబీ (November-December)\nనేల: ఎర్రమట్టి, ఇసుక మిశ్రమ మట్టి\nనీటి అవసరం: 500-700mm\nMSP (2025-26): ₹6,377/quintal\nదిగుబడి: ఎకరాకు 8-12 quintals\nఆదాయం: ₹51,000-76,500/ఎకరా\nపెట్టుబడి ఖర్చు: ₹16,000-22,000/ఎకరా\nనికర లాభం: ₹29,000-60,000/ఎకరా\nజిల్లాలు (AP): అనంతపురం (అత్యధికం), కర్నూలు, చిత్తూరు, కడప"""},
    {"crop": "chilli", "content": """పంట: మిర్చి (Chilli)\nసీజన్: ఖరీఫ్ (July-August), రబీ (October-November)\nమార్కెట్ ధర: ₹10,000-35,000/quintal (MSP లేదు)\nదిగుబడి: ఎకరాకు 8-15 quintals\nఆదాయం: ₹80,000-5,25,000/ఎకరా\nపెట్టుబడి ఖర్చు: ₹35,000-50,000/ఎకరా\nనికర లాభం: ₹30,000-4,75,000/ఎకరా\nరిస్క్: మార్కెట్ ధర చాలా అనిశ్చితం\nజిల్లాలు (AP): గుంటూరు (#1), ప్రకాశం, కృష్ణా"""},
    {"crop": "sugarcane", "content": """పంట: చెరకు (Sugarcane)\nసీజన్: January-March, 10-14 నెలల పంట\nనీటి అవసరం: 1500-2000mm (అత్యధికం)\nFRP (2025-26): ₹315/quintal (10.25% recovery)\nదిగుబడి: ఎకరాకు 350-450 quintals\nఆదాయం: ₹1,10,000-1,70,000/ఎకరా\nపెట్టుబడి ఖర్చు: ₹50,000-70,000/ఎకరా\nనికర లాభం: ₹40,000-1,00,000/ఎకరా\nముఖ్యం: 50km లోపు sugar mill ఉండాలి. Payment ఆలస్యం 2-6 నెలలు\nజిల్లాలు (AP): విశాఖపట్నం, తూర్పు గోదావరి\nజిల్లాలు (Telangana): నిజామాబాద్, మెదక్"""},
    {"crop": "maize", "content": """పంట: మొక్కజొన్న (Maize)\nసీజన్లు: ఖరీఫ్, రబీ. 90-110 రోజులు\nనీటి అవసరం: 500-800mm\nMSP (2025-26): ₹2,090/quintal\nదిగుబడి: ఎకరాకు 25-35 quintals\nఆదాయం: ₹52,000-73,000/ఎకరా\nపెట్టుబడి ఖర్చు: ₹15,000-22,000/ఎకరా\nనికర లాభం: ₹30,000-58,000/ఎకరా\nప్రయోజనాలు: తక్కువ నీరు, తక్కువ ఖర్చు\nజిల్లాలు (AP): కర్నూలు, ప్రకాశం, గుంటూరు"""},
    {"crop": "turmeric", "content": """పంట: పసుపు (Turmeric)\nసీజన్: 7-9 నెలల పంట\nమార్కెట్ ధర: ₹8,000-18,000/quintal\nదిగుబడి: ఎకరాకు 25-35 quintals\nపెట్టుబడి ఖర్చు: ₹35,000-45,000/ఎకరా\nనికర లాభం: ₹5,000-1,00,000/ఎకరా\nజిల్లాలు (AP): కడప, ప్రకాశం\nజిల్లాలు (Telangana): నిజామాబాద్, జగిత్యాల"""},
    {"crop": "banana", "content": """పంట: అరటి (Banana)\nసీజన్: 12-14 నెలల పంట\nనీటి అవసరం: 1200-1500mm. డ్రిప్ తప్పనిసరి\nమార్కెట్ ధర: ₹500-1,200/quintal\nదిగుబడి: ఎకరాకు 250-400 quintals\nఆదాయం: ₹1,25,000-4,80,000/ఎకరా\nపెట్టుబడి ఖర్చు: ₹70,000-90,000/ఎకరా\nనికర లాభం: ₹35,000-3,90,000/ఎకరా\nరిస్క్: గాలి వాన వస్తే నష్టం\nజిల్లాలు (AP): కర్నూలు, అనంతపురం, కడప"""},
    {"crop": "tomato", "content": """పంట: టమాట (Tomato)\nసీజన్: ఏడాది పొడవునా. 90-120 రోజులు\nమార్కెట్ ధర: ₹500-4,000/quintal (గెంబ్లింగ్ క్రాప్)\nదిగుబడి: ఎకరాకు 80-150 quintals\nపెట్టుబడి ఖర్చు: ₹35,000-50,000/ఎకరా\nనికర లాభం: ₹5,000-5,50,000/ఎకరా\nరిస్క్: అత్యధిక ధర హెచ్చుతగ్గులు. Perishable\nజిల్లాలు (AP): చిత్తూరు (మదనపల్లి), కర్నూలు"""},
    {"crop": "onion", "content": """పంట: ఉల్లి (Onion)\nసీజన్: ఖరీఫ్, Late ఖరీఫ్, రబీ. 120-150 రోజులు\nమార్కెట్ ధర: ₹500-5,000/quintal\nదిగుబడి: ఎకరాకు 60-100 quintals\nపెట్టుబడి ఖర్చు: ₹25,000-35,000/ఎకరా\nనికర లాభం: ₹5,000-4,65,000/ఎకరా\nప్రయోజనం: 3-4 నెలలు నిల్వ\nజిల్లాలు (AP): కర్నూలు (#1), ప్రకాశం"""},
    {"crop": "mango", "content": """పంట: మామిడి (Mango)\nరకాలు: బంగినపల్లి, సువర్ణరేఖ, తోతాపురి\nమార్కెట్ ధర: ₹2,000-8,000/quintal\nదిగుబడి: ఎకరాకు 40-80 quintals\nఆదాయం: ₹80,000-6,40,000/ఎకరా\nవార్షిక ఖర్చు: ₹25,000-40,000/ఎకరా\nనికర లాభం: ₹40,000-6,00,000/ఎకరా\nరిస్క్: పూత సమయంలో వర్షం వస్తే నష్టం\nజిల్లాలు (AP): కృష్ణా, చిత్తూరు, అనంతపురం"""},
    {"crop": "tobacco", "content": """పంట: పొగాకు (Tobacco)\nరాష్ట్రం: ఆంధ్రప్రదేశ్ ప్రధానం\nమార్కెట్ ధర: ₹15,000-25,000/quintal (Tobacco Board auction)\nదిగుబడి: ఎకరాకు 8-12 quintals\nఆదాయం: ₹1,20,000-3,00,000/ఎకరా\nపెట్టుబడి ఖర్చు: ₹40,000-60,000/ఎకరా\nనికర లాభం: ₹60,000-2,40,000/ఎకరా\nముఖ్యం: Tobacco Board అనుమతి (NTRP) అవసరం\nజిల్లాలు (AP): ప్రకాశం, గుంటూరు, కృష్ణా"""}
]


# ============================================================
# VECTOR STORE
# ============================================================
def setup_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = []
    for crop_data in CROP_KNOWLEDGE:
        chunks = splitter.split_text(crop_data["content"])
        for i, chunk in enumerate(chunks):
            documents.append(Document(page_content=chunk, metadata={"crop": crop_data["crop"], "chunk_id": i}))
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name="kisanmitra_crops")
    return vectorstore.as_retriever(search_kwargs={"k": 4})


print("⏳ Setting up vector store...")
crop_retriever = setup_vectorstore()
print("✅ Vector store ready")


# ============================================================
# STATE
# ============================================================
class KisanMitraState(TypedDict):
    user_query: str
    language: str
    location_raw: str
    location_name: str
    location_lat: float
    location_lon: float
    location_state: str
    crop: str
    intent: str
    weather_data: Optional[dict]
    agri_data: Optional[dict]
    forecast_data: Optional[dict]
    crop_knowledge: Optional[str]
    final_response: str


# ============================================================
# LANGUAGE DETECTION (for translation fallback)
# ============================================================
def is_telugu_script(text: str) -> bool:
    telugu_chars = re.findall(r'[\u0C00-\u0C7F]', text)
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False
    return len(telugu_chars) / len(non_space) > 0.3


def is_hindi_script(text: str) -> bool:
    hindi_chars = re.findall(r'[\u0900-\u097F]', text)
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False
    return len(hindi_chars) / len(non_space) > 0.3


# ============================================================
# TRANSLATION FALLBACK (only used if Opus still returns English)
# ============================================================
def translate_to_telugu(english_text: str) -> str:
    """Step 2 of two-step chain: translate English to natural Telugu."""
    system = """నీవు తెలుగు అనువాదకుడివి. ఇచ్చిన English text ని సహజమైన రైతుకి అర్థమయ్యే తెలుగు లిపిలో అనువదించు.

నియమాలు:
- పూర్తిగా తెలుగు లిపిలో మాత్రమే రాయి
- సంఖ్యలు, units (38°C, ₹2,300, 5mm, 80%) English లోనే ఉంచు
- ప్రదేశాల పేర్లు (Guntur, Kurnool) English లోనే ఉంచు
- Headings కూడా తెలుగులో అనువదించు
- Markdown formatting (**, ##, -) అలాగే ఉంచు
- సహజమైన రైతు భాష వాడు"""
    messages = [SystemMessage(content=system),
                HumanMessage(content=f"ఈ text ని తెలుగులోకి అనువదించు:\n\n{english_text}")]
    return llm_translator.invoke(messages).content


def translate_to_hindi(english_text: str) -> str:
    """Step 2 of two-step chain: translate English to natural Hindi."""
    system = """आप हिंदी अनुवादक हैं। दिए गए English text को सहज किसान को समझ आने वाली हिंदी में अनुवाद करें।

नियम:
- पूरी तरह हिंदी (देवनागरी) लिपि में ही लिखें
- संख्याएं, units (38°C, ₹2,300, 5mm) English में ही रखें
- स्थानों के नाम (Guntur, Kurnool) English में ही रखें
- Headings भी हिंदी में अनुवाद करें
- Markdown formatting (**, ##, -) वैसा ही रखें"""
    messages = [SystemMessage(content=system),
                HumanMessage(content=f"इस text को हिंदी में अनुवाद करें:\n\n{english_text}")]
    return llm_translator.invoke(messages).content


# ============================================================
# ADVISORY TEMPLATES
# ============================================================
ADVISORY_TEMPLATES = {
    "irrigation": "Should farmer irrigate today? How much? Best time? Rain outlook 3 days? Water saving tip?",
    "pest": "Pest risk level? Which pests for {crop}? Spray or not? Best time? Prevention?",
    "forecast": "Today summary? Rain when/how much? Temperature trend? Warnings? Safe activities?",
    "harvest": "Good harvest window? Rain risk? Best time of day? Drying conditions? Storage tip?",
    "sowing": "Soil temperature okay? Moisture enough? Rain after sowing? Best day? Seed prep tip?",
    "general": "Weather summary? Irrigation needed? Pest risk? Warning? Top 2 action items?"
}


# ============================================================
# NODE 1: UNDERSTAND QUERY
# ============================================================
understand_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a multilingual parser for an Indian agriculture app.
Extract: language, location, crop, intent from the farmer's question.
For LOCATION: pick most specific place name. For CROP: normalize to English.
INTENTS: irrigation, pest, forecast, harvest, sowing, profitability, crop_selection, crop_info, general.
profitability = లాభం/profit/cost/price/MSP/ధర/ఆదాయం
crop_selection = best crop/ఏ పంట/which crop/ఏం వేయాలి
crop_info = fertilizer/ఎరువు/schedule/yield/దిగుబడి
Respond ONLY in JSON: {{"language":"...","location":"...","crop":"...","intent":"..."}}"""),
    ("human", "{query}")
])
understand_chain = understand_prompt | llm | JsonOutputParser()

def understand_query(state: KisanMitraState) -> dict:
    result = understand_chain.invoke({"query": state["user_query"]})
    return {
        "language": result.get("language", "english"),
        "location_raw": result.get("location", "unknown"),
        "crop": result.get("crop", "unknown"),
        "intent": result.get("intent", "general")
    }


# ============================================================
# NODE 2: GEOCODE
# ============================================================
def geocode_location_node(state: KisanMitraState) -> dict:
    location = state["location_raw"]
    if location == "unknown":
        return {"location_name": "unknown", "location_lat": 0.0, "location_lon": 0.0, "location_state": "unknown"}
    geo = geocode_location(location)
    if "error" in geo:
        geo = geocode_location(f"{location} India")
    if "error" in geo or not geo.get("found"):
        return {"location_name": "unknown", "location_lat": 0.0, "location_lon": 0.0, "location_state": "unknown"}
    return {"location_name": geo["name"], "location_lat": geo["lat"], "location_lon": geo["lon"], "location_state": geo["state"]}


# ============================================================
# NODE 3: FETCH WEATHER
# ============================================================
def fetch_weather_data(state: KisanMitraState) -> dict:
    lat, lon = state["location_lat"], state["location_lon"]
    name, st = state["location_name"], state["location_state"]
    result = {"weather_data": get_weather_by_coords(lat, lon, name, st)}
    intent = state["intent"]
    if intent in ["irrigation", "pest", "sowing", "harvest", "general"]:
        result["agri_data"] = get_agri_by_coords(lat, lon, name, st)
    if intent in ["forecast", "sowing", "harvest", "general"]:
        result["forecast_data"] = get_forecast_by_coords(lat, lon, name, st)
    return result


# ============================================================
# NODE: SEARCH KNOWLEDGE
# ============================================================
def search_crop_knowledge(state: KisanMitraState) -> dict:
    crop = state["crop"]
    intent = state["intent"]
    location = state.get("location_name", "")
    if intent == "profitability":
        search_query = f"{crop} profit cost income expenditure laabham"
    elif intent == "crop_selection":
        search_query = f"{location} district best crop recommendation season"
    elif intent == "crop_info":
        search_query = f"{crop} fertilizer schedule yield variety season"
    else:
        search_query = f"{crop} {state['user_query'][:100]}"
    results = crop_retriever.invoke(search_query)
    knowledge_text = "\n\n".join([f"[{doc.metadata['crop']}]: {doc.page_content}" for doc in results])
    return {"crop_knowledge": knowledge_text}


# ============================================================
# NODE: FETCH WEATHER + KNOWLEDGE
# ============================================================
def fetch_weather_and_knowledge(state: KisanMitraState) -> dict:
    weather_result = fetch_weather_data(state)
    knowledge_result = search_crop_knowledge(state)
    return {**weather_result, **knowledge_result}


# ============================================================
# SYSTEM PROMPTS — Language-first with few-shot examples
# ============================================================
def build_telugu_system_prompt(location, loc_state, crop):
    """Telugu-first system prompt with few-shot example. Fix #2 + #3."""
    return f"""నీవు KisanMitra (కిసాన్ మిత్ర) అనే AI agriculture advisor. భారతీయ రైతులకు సహాయం చేస్తావు.

అత్యంత ముఖ్యమైన నియమం: నీవు పూర్తిగా తెలుగు లిపిలో మాత్రమే సమాధానం ఇస్తావు. Input language ఏదైనా (English, Tenglish, Telugu) — output మాత్రం పూర్తిగా తెలుగు లిపిలో ఉంటుంది. English headings, English sentences ఎప్పుడూ రాయవు. సంఖ్యలు మాత్రమే English లో: 38°C, ₹2,300, 5mm, 80%. ప్రదేశాల పేర్లు English లో: Guntur, Kurnool.

ఉదాహరణ (Example) — ఇలా రాయాలి:

# 🌾 గుంటూరులో వరి - నీటిపారుదల సలహా

## 📊 ప్రస్తుత పరిస్థితి
- **ఉష్ణోగ్రత:** 38°C
- **తేమ:** 45%
- **నేల తేమ:** తక్కువ

## 💧 సలహా
**అవును, నీరు పెట్టండి.** నేల చాలా పొడిగా ఉంది. మధ్యాహ్నం కాకుండా **సాయంత్రం 5 గంటల తర్వాత** నీరు పెట్టండి - బాష్పీభవనం తక్కువ.

## ⚠️ హెచ్చరిక
రాబోయే 3 రోజులు వర్షం వచ్చే అవకాశం 20% మాత్రమే - వర్షంపై ఆధారపడకండి.

💪 మీ కష్టానికి మంచి దిగుబడి తప్పక వస్తుంది!

---

Location: {location}, {loc_state}
Crop: {crop}

నియమాలు: Reference data నుండి exact numbers వాడు. Risks గురించి నిజాయితీగా చెప్పు. 250 words లోపు. Encouraging line తో ముగించు. పూర్తిగా తెలుగు లిపిలో!"""


def build_hindi_system_prompt(location, loc_state, crop):
    """Hindi-first system prompt with few-shot example."""
    return f"""आप KisanMitra (किसान मित्र) हैं — एक AI agriculture advisor। आप भारतीय किसानों की मदद करते हैं।

सबसे महत्वपूर्ण नियम: आप पूरी तरह हिंदी (देवनागरी) में ही जवाब देते हैं। Input language कुछ भी हो — output पूरी तरह हिंदी में होगा। English headings, English sentences कभी मत लिखो। केवल संख्या English में: 38°C, ₹2,300, 5mm। स्थानों के नाम English में।

उदाहरण:

# 🌾 गुंटूर में धान - सिंचाई सलाह

## 📊 वर्तमान स्थिति
- **तापमान:** 38°C
- **नमी:** 45%

## 💧 सलाह
**हाँ, पानी दीजिए।** मिट्टी सूखी है। **शाम 5 बजे के बाद** पानी दें।

💪 आपकी मेहनत का अच्छा फल मिलेगा!

---

Location: {location}, {loc_state}
Crop: {crop}

नियम: Reference data से exact numbers use करें। 250 words के अंदर।"""


def build_english_system_prompt(location, loc_state, crop):
    return f"""You are KisanMitra, AI agriculture advisor for Indian farmers.
Location: {location}, {loc_state}
Crop: {crop}
Use exact numbers from reference data. Be honest about risks. Under 250 words. End with encouragement."""


# ============================================================
# NODE 4: GENERATE ADVISORY (with all 5 language fixes)
# ============================================================
def generate_advisory(state: KisanMitraState) -> dict:
    language = state["language"]
    intent = state["intent"]
    crop = state["crop"]
    location = state["location_name"]
    loc_state = state.get("location_state", "")

    # Handle unknown location
    if location == "unknown" and intent in ["irrigation", "pest", "forecast", "harvest", "sowing"]:
        if language in ["telugu", "tenglish"]:
            system = "నీవు KisanMitra. ప్రదేశం తెలియలేదు. మర్యాదగా ప్రదేశం అడుగు. 2-3 general tips ఇవ్వు. పూర్తిగా తెలుగు లిపిలో. 150 words లోపు."
        elif language == "hindi":
            system = "आप KisanMitra हैं। स्थान अज्ञात है। शिष्टता से पूछें। 2-3 general tips दें। पूरी तरह हिंदी में। 150 शब्दों के अंदर।"
        else:
            system = "You are KisanMitra. Location unknown. Ask politely. Give 2-3 general tips. Under 150 words."
        response = llm_smart.invoke([SystemMessage(content=system), HumanMessage(content=state["user_query"])])
        return {"final_response": response.content}

    # Build reference data
    weather_str = json.dumps(state.get("weather_data", {}), indent=2) if state.get("weather_data") else "N/A"
    agri_str = json.dumps(state.get("agri_data", {}), indent=2) if state.get("agri_data") else "N/A"
    forecast_str = json.dumps(state.get("forecast_data", {}), indent=2) if state.get("forecast_data") else "N/A"
    knowledge_str = state.get("crop_knowledge", "N/A")
    reference_block = f"[REFERENCE DATA - facts only]\n\nWeather:\n{weather_str}\n\nAgriculture:\n{agri_str}\n\nForecast:\n{forecast_str}\n\nCrop Knowledge:\n{knowledge_str}"

    # Task description
    if intent == "profitability":
        task = "Total investment, yield, income, net profit per acre. Risks for the location. Is this crop suitable? Compare with 1-2 alternatives."
    elif intent == "crop_selection":
        task = "Recommend top 2-3 crops for this location and season. For each: profit, water needs, risk. Which has best ratio?"
    elif intent == "crop_info":
        task = "Answer the farmer's specific question using exact numbers."
    else:
        task = ADVISORY_TEMPLATES.get(intent, ADVISORY_TEMPLATES["general"])
        if "{crop}" in task:
            task = task.replace("{crop}", crop if crop != "unknown" else "local crops")

    farmer_question = state["user_query"]

    # Build language-specific system prompt (Fix #2 + #3)
    if language in ["telugu", "tenglish"]:
        system = build_telugu_system_prompt(location, loc_state, crop)
        human_msg = f"రైతు ప్రశ్న: \"{farmer_question}\"\n\nTask: {task}\n\n{reference_block}\n\n👉 పూర్తిగా తెలుగు లిపిలో సమాధానం ఇవ్వు."
        # Fix #4: Pre-fill assistant response with Telugu opening
        prefill = "# 🌾 "
    elif language == "hindi":
        system = build_hindi_system_prompt(location, loc_state, crop)
        human_msg = f"किसान का प्रश्न: \"{farmer_question}\"\n\nTask: {task}\n\n{reference_block}\n\n👉 पूरी तरह हिंदी में जवाब दें।"
        prefill = "# 🌾 "
    else:
        system = build_english_system_prompt(location, loc_state, crop)
        human_msg = f"Farmer's question: \"{farmer_question}\"\n\nTask: {task}\n\n{reference_block}"
        prefill = None

    # Build messages with optional pre-fill
    messages = [SystemMessage(content=system), HumanMessage(content=human_msg)]
    if prefill:
        # Pre-fill technique: start assistant message with Telugu/Hindi opening
        messages.append(AIMessage(content=prefill))

    response = llm_smart.invoke(messages)
    response_text = response.content
    # If we used pre-fill, prepend it back
    if prefill:
        response_text = prefill + response_text

    # Fix #5: Two-step translation fallback
    # If we asked for Telugu but got English, translate
    if language in ["telugu", "tenglish"] and not is_telugu_script(response_text):
        print(f"⚠️ Telugu fallback triggered — translating English response to Telugu")
        response_text = translate_to_telugu(response_text)
    elif language == "hindi" and not is_hindi_script(response_text):
        print(f"⚠️ Hindi fallback triggered — translating English response to Hindi")
        response_text = translate_to_hindi(response_text)

    return {"final_response": response_text}


# ============================================================
# ROUTING
# ============================================================
def route_after_understanding(state: KisanMitraState) -> str:
    intent = state["intent"]
    location = state["location_raw"]
    if intent in ["profitability", "crop_selection", "crop_info"]:
        return "search_knowledge" if location == "unknown" else "geocode_location"
    return "generate_advisory" if location == "unknown" else "geocode_location"


def route_after_geocoding(state: KisanMitraState) -> str:
    intent = state["intent"]
    if state["location_name"] == "unknown":
        return "search_knowledge" if intent in ["profitability", "crop_selection", "crop_info"] else "generate_advisory"
    return "fetch_weather_and_knowledge" if intent in ["profitability", "crop_selection", "crop_info"] else "fetch_weather"


# ============================================================
# BUILD GRAPH
# ============================================================
def build_agent():
    graph = StateGraph(KisanMitraState)
    graph.add_node("understand_query", understand_query)
    graph.add_node("geocode_location", geocode_location_node)
    graph.add_node("fetch_weather", fetch_weather_data)
    graph.add_node("search_knowledge", search_crop_knowledge)
    graph.add_node("fetch_weather_and_knowledge", fetch_weather_and_knowledge)
    graph.add_node("generate_advisory", generate_advisory)
    graph.set_entry_point("understand_query")
    graph.add_conditional_edges("understand_query", route_after_understanding,
        {"geocode_location": "geocode_location", "generate_advisory": "generate_advisory", "search_knowledge": "search_knowledge"})
    graph.add_conditional_edges("geocode_location", route_after_geocoding,
        {"fetch_weather": "fetch_weather", "fetch_weather_and_knowledge": "fetch_weather_and_knowledge",
         "generate_advisory": "generate_advisory", "search_knowledge": "search_knowledge"})
    graph.add_edge("fetch_weather", "generate_advisory")
    graph.add_edge("search_knowledge", "generate_advisory")
    graph.add_edge("fetch_weather_and_knowledge", "generate_advisory")
    graph.add_edge("generate_advisory", END)
    return graph.compile()


print("⏳ Building agent...")
kisanmitra_agent = build_agent()
print("✅ KisanMitra agent ready!")


# ============================================================
# PUBLIC API
# ============================================================
def ask(question: str) -> dict:
    result = kisanmitra_agent.invoke({
        "user_query": question, "language": "", "location_raw": "", "location_name": "",
        "location_lat": 0.0, "location_lon": 0.0, "location_state": "",
        "crop": "", "intent": "", "weather_data": None, "agri_data": None,
        "forecast_data": None, "crop_knowledge": None, "final_response": ""
    })
    return {
        "question": question,
        "language": result["language"],
        "location": result["location_name"],
        "state": result["location_state"],
        "crop": result["crop"],
        "intent": result["intent"],
        "has_knowledge": bool(result.get("crop_knowledge")),
        "response": result["final_response"]
    }


if __name__ == "__main__":
    # Quick test
    result = ask("Guntur lo paddy ki neellu pettala?")
    print(f"\nLanguage: {result['language']}")
    print(f"Location: {result['location']}, {result['state']}")
    print(f"Crop: {result['crop']}")
    print(f"Intent: {result['intent']}")
    print(f"\nResponse:\n{result['response']}")
