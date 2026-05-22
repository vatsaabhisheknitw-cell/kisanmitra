"""
KisanMitra Agent LITE — Lightweight version for Render Free Tier deployment.
No sentence-transformers, no ChromaDB, no LangChain/LangGraph.
Uses direct Anthropic SDK + keyword-based crop knowledge search.
Same functionality, ~50MB RAM instead of ~500MB+.
"""

import os
import requests
import anthropic

# ============================================================
# LLM SETUP
# ============================================================
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_SMART = "claude-opus-4-6"


# ============================================================
# CROP KNOWLEDGE BASE (same data as ChromaDB version, stored as dicts)
# ============================================================
CROP_KNOWLEDGE = [
    {
        "crop": "paddy",
        "keywords": ["paddy", "rice", "వరి", "ధాన్యం", "నాట్లు"],
        "text": """పంట: వరి (Paddy/Rice)
MSP (2024-25): ₹2,300/క్వింటాల్. మార్కెట్ ధర: ₹2,000-₹2,800/క్వింటాల్.
దిగుబడి: ఎకరాకు 25-30 క్వింటాళ్లు. పెట్టుబడి: ₹25,000-₹30,000/ఎకరం.
నికర లాభం: ₹30,000-₹55,000/ఎకరం.
ఎరువుల షెడ్యూల్: నాట్ల సమయంలో DAP 50kg + Potash 25kg/ఎకరం. 25 రోజులకు Urea 50kg. 45 రోజులకు Urea 25kg + Potash 25kg.
ప్రధాన తెగుళ్లు: అగ్గి తెగులు (Blast), బ్రౌన్ ప్లాంట్ హాపర్ (BPH), తాటాకు తెగులు.
ఉత్తమ జిల్లాలు (AP): కృష్ణా, గోదావరి, నెల్లూరు. (Telangana): నిజామాబాద్, కరీంనగర్, వరంగల్.
నీటి అవసరం: 1200-1500 mm. సీజన్: ఖరీఫ్ (జూన్-నవంబర్), రబీ (నవంబర్-మార్చి)."""
    },
    {
        "crop": "cotton",
        "keywords": ["cotton", "పత్తి", "kapas"],
        "text": """పంట: పత్తి (Cotton)
MSP (2024-25): ₹7,121/క్వింటాల్ (మీడియం స్టేపుల్). మార్కెట్ ధర: ₹6,500-₹8,500/క్వింటాల్.
దిగుబడి: ఎకరాకు 8-12 క్వింటాళ్లు. పెట్టుబడి: ₹20,000-₹28,000/ఎకరం.
నికర లాభం: ₹35,000-₹65,000/ఎకరం.
ఎరువుల షెడ్యూల్: విత్తనాల సమయంలో DAP 50kg + Potash 25kg. 30 రోజులకు Urea 25kg. 60 రోజులకు Urea 25kg + Potash 25kg.
ప్రధాన తెగుళ్లు: గులాబీ పురుగు (Pink Bollworm), రసం పీల్చే పురుగులు (Sucking pests), ఆకు మచ్చ తెగులు.
ఉత్తమ జిల్లాలు (AP): కర్నూల్, గుంటూరు, ప్రకాశం. (Telangana): ఆదిలాబాద్, వరంగల్, నాగర్‌కర్నూల్.
నీటి అవసరం: 700-1200 mm. సీజన్: ఖరీఫ్ (జూన్-జనవరి)."""
    },
    {
        "crop": "groundnut",
        "keywords": ["groundnut", "peanut", "వేరుశనగ", "వేరుసెనగ"],
        "text": """పంట: వేరుశనగ (Groundnut)
MSP (2024-25): ₹6,377/క్వింటాల్. మార్కెట్ ధర: ₹5,500-₹7,500/క్వింటాల్.
దిగుబడి: ఎకరాకు 8-12 క్వింటాళ్లు. పెట్టుబడి: ₹18,000-₹22,000/ఎకరం.
నికర లాభం: ₹30,000-₹55,000/ఎకరం.
ఎరువుల షెడ్యూల్: విత్తనాల సమయంలో SSP 100kg + Gypsum 200kg/ఎకరం. 25 రోజులకు Urea 25kg.
ప్రధాన తెగుళ్లు: తికా తెగులు (Tikka disease), కాండం కుళ్ళు (Stem rot), ఆకు తొలుచు పురుగు.
ఉత్తమ జిల్లాలు (AP): అనంతపురం, కర్నూల్, కడప. (Telangana): మహబూబ్‌నగర్.
నీటి అవసరం: 500-700 mm. సీజన్: ఖరీఫ్ (జూన్-అక్టోబర్), రబీ (నవంబర్-మార్చి)."""
    },
    {
        "crop": "chilli",
        "keywords": ["chilli", "chili", "mirchi", "మిర్చి", "మిరప"],
        "text": """పంట: మిర్చి (Chilli)
మార్కెట్ ధర: ₹8,000-₹25,000/క్వింటాల్ (రకం, డిమాండ్ బట్టి మారుతుంది).
దిగుబడి: ఎకరాకు 8-15 క్వింటాళ్లు (ఎండు మిర్చి). పెట్టుబడి: ₹35,000-₹50,000/ఎకరం.
నికర లాభం: ₹40,000-₹2,00,000/ఎకరం (ధర హెచ్చు తగ్గులు ఎక్కువ).
ఎరువుల షెడ్యూల్: నాట్ల సమయంలో DAP 50kg + Potash 50kg + FYM 5 టన్నులు. 30, 60, 90 రోజులకు Urea 25kg చొప్పున.
ప్రధాన తెగుళ్లు: తామర పురుగు (Thrips), పండు తొలుచు పురుగు (Fruit borer), మూర్ఛ తెగులు (Murda/Leaf curl).
ఉత్తమ జిల్లాలు (AP): గుంటూరు, ప్రకాశం, కృష్ణా. (Telangana): వరంగల్, ఖమ్మం.
నీటి అవసరం: 600-1000 mm. సీజన్: ఖరీఫ్ (జూలై-మార్చి), రబీ (అక్టోబర్-ఏప్రిల్)."""
    },
    {
        "crop": "sugarcane",
        "keywords": ["sugarcane", "చెరకు", "sugar"],
        "text": """పంట: చెరకు (Sugarcane)
FRP (2024-25): ₹315/క్వింటాల్. SAP (AP): ₹3,500-₹4,000/టన్ను.
దిగుబడి: ఎకరాకు 40-50 టన్నులు. పెట్టుబడి: ₹45,000-₹60,000/ఎకరం.
నికర లాభం: ₹80,000-₹1,40,000/ఎకరం.
ఎరువుల షెడ్యూల్: నాటే సమయంలో DAP 75kg + Potash 50kg + FYM 10 టన్నులు. 45, 90, 120 రోజులకు Urea 35kg చొప్పున.
ప్రధాన తెగుళ్లు: ఇంటర్‌నోడ్ బోరర్, స్కేల్ ఇన్సెక్ట్, రెడ్ రాట్.
ఉత్తమ జిల్లాలు (AP): విశాఖపట్నం, తూర్పు గోదావరి. (Telangana): నిజామాబాద్, మెదక్.
నీటి అవసరం: 1500-2500 mm. సీజన్: జనవరి-మార్చి (నాటడం), 12-14 నెలల పంట."""
    },
    {
        "crop": "maize",
        "keywords": ["maize", "corn", "మొక్కజొన్న", "జొన్న"],
        "text": """పంట: మొక్కజొన్న (Maize)
MSP (2024-25): ₹2,090/క్వింటాల్. మార్కెట్ ధర: ₹1,800-₹2,500/క్వింటాల్.
దిగుబడి: ఎకరాకు 25-35 క్వింటాళ్లు. పెట్టుబడి: ₹15,000-₹20,000/ఎకరం.
నికర లాభం: ₹25,000-₹50,000/ఎకరం.
ఎరువుల షెడ్యూల్: విత్తనాల సమయంలో DAP 50kg + Potash 25kg. 25 రోజులకు Urea 50kg. 45 రోజులకు Urea 25kg.
ప్రధాన తెగుళ్లు: ఫాల్ ఆర్మీ వార్మ్ (Fall Armyworm), కాండం తొలుచు పురుగు, డౌనీ మిల్డ్యూ.
ఉత్తమ జిల్లాలు (AP): గుంటూరు, ప్రకాశం, కృష్ణా. (Telangana): కరీంనగర్, మెదక్.
నీటి అవసరం: 500-800 mm. సీజన్: ఖరీఫ్ (జూన్-సెప్టెంబర్), రబీ (అక్టోబర్-జనవరి)."""
    },
    {
        "crop": "turmeric",
        "keywords": ["turmeric", "పసుపు", "haldi"],
        "text": """పంట: పసుపు (Turmeric)
మార్కెట్ ధర: ₹8,000-₹15,000/క్వింటాల్. దిగుబడి: ఎకరాకు 20-30 క్వింటాళ్లు (పచ్చి), 5-7 క్వింటాళ్లు (ఎండు).
పెట్టుబడి: ₹40,000-₹55,000/ఎకరం. నికర లాభం: ₹50,000-₹1,20,000/ఎకరం.
ఎరువుల షెడ్యూల్: నాటే సమయంలో FYM 10 టన్నులు + DAP 50kg. 60 రోజులకు Urea 25kg + Potash 25kg. 120 రోజులకు Urea 25kg.
ప్రధాన తెగుళ్లు: దుంప కుళ్ళు (Rhizome rot), ఆకు మచ్చ (Leaf blotch), షూట్ బోరర్.
ఉత్తమ జిల్లాలు (AP): కడప, కర్నూల్. (Telangana): నిజామాబాద్, జగిత్యాల.
నీటి అవసరం: 1200-1500 mm. సీజన్: జూన్-జూలై (నాటడం), 8-9 నెలల పంట."""
    },
    {
        "crop": "banana",
        "keywords": ["banana", "అరటి", "అరటిపండు"],
        "text": """పంట: అరటి (Banana)
మార్కెట్ ధర: ₹500-₹1,200/బంచ్. దిగుబడి: ఎకరాకు 700-1000 బంచ్‌లు.
పెట్టుబడి: ₹60,000-₹80,000/ఎకరం. నికర లాభం: ₹1,00,000-₹2,50,000/ఎకరం.
ఎరువుల షెడ్యూల్: నాటే సమయంలో FYM 10 టన్నులు + SSP 100kg. ప్రతి 2 నెలలకు Urea 50kg + Potash 50kg. పూత సమయంలో Sulphate of Potash 50kg.
ప్రధాన తెగుళ్లు: పనామా విల్ట్ (Fusarium wilt), సిగటోకా ఆకు మచ్చ, బంచీ టాప్ వైరస్.
ఉత్తమ జిల్లాలు (AP): కడప, అనంతపురం, కృష్ణా. (Telangana): వరంగల్, ఖమ్మం.
నీటి అవసరం: 1800-2200 mm. సీజన్: జూన్-ఆగస్టు (నాటడం), 12-14 నెలల పంట."""
    },
    {
        "crop": "tomato",
        "keywords": ["tomato", "టమాట", "టమాటా"],
        "text": """పంట: టమాట (Tomato)
మార్కెట్ ధర: ₹500-₹4,000/క్వింటాల్ (ధర హెచ్చు తగ్గులు చాలా ఎక్కువ). దిగుబడి: ఎకరాకు 150-300 క్వింటాళ్లు.
పెట్టుబడి: ₹40,000-₹60,000/ఎకరం. నికర లాభం: ₹30,000-₹3,00,000/ఎకరం (ధర బట్టి).
ఎరువుల షెడ్యూల్: నాట్ల సమయంలో DAP 50kg + Potash 50kg + FYM 5 టన్నులు. 25, 45, 65 రోజులకు 19:19:19 complex 5kg drip ద్వారా.
ప్రధాన తెగుళ్లు: తుత జిల్లెడ (Tuta absoluta), ఆకు ముడత వైరస్ (Leaf curl virus), ఎర్లీ బ్లైట్.
ఉత్తమ జిల్లాలు (AP): కర్నూల్, చిత్తూరు, అనంతపురం. (Telangana): రంగారెడ్డి, మహబూబ్‌నగర్.
నీటి అవసరం: 400-600 mm. సీజన్: ఏడాది పొడవునా (కానీ రబీ ఉత్తమం)."""
    },
    {
        "crop": "onion",
        "keywords": ["onion", "ఉల్లి", "ఉల్లిపాయ"],
        "text": """పంట: ఉల్లి (Onion)
మార్కెట్ ధర: ₹800-₹4,000/క్వింటాల్. దిగుబడి: ఎకరాకు 80-120 క్వింటాళ్లు.
పెట్టుబడి: ₹30,000-₹40,000/ఎకరం. నికర లాభం: ₹30,000-₹2,00,000/ఎకరం (ధర బట్టి).
ఎరువుల షెడ్యూల్: నాట్ల సమయంలో DAP 50kg + Potash 50kg + Sulphur 10kg. 30 రోజులకు Urea 25kg. 45 రోజులకు Urea 25kg + Potash 25kg.
ప్రధాన తెగుళ్లు: తామర పురుగు (Thrips), ఊదా మచ్చ (Purple blotch), కాండం కుళ్ళు.
ఉత్తమ జిల్లాలు (AP): కర్నూల్, ప్రకాశం. (Telangana): మహబూబ్‌నగర్, నాగర్‌కర్నూల్.
నీటి అవసరం: 350-500 mm. సీజన్: ఖరీఫ్ (జూన్-అక్టోబర్), రబీ (నవంబర్-మార్చి), లేట్ ఖరీఫ్ (ఆగస్టు-డిసెంబర్)."""
    },
    {
        "crop": "mango",
        "keywords": ["mango", "మామిడి", "మామిడిపండు"],
        "text": """పంట: మామిడి (Mango)
మార్కెట్ ధర: ₹2,000-₹8,000/క్వింటాల్ (రకం బట్టి). దిగుబడి: ఎకరాకు 40-80 క్వింటాళ్లు (బేరింగ్ చెట్లు).
పెట్టుబడి: ₹15,000-₹25,000/ఎకరం (బేరింగ్ తోట). నికర లాభం: ₹80,000-₹3,00,000/ఎకరం.
ఎరువుల షెడ్యూల్: జూన్‌లో FYM 50kg + DAP 1kg + Potash 1kg ప్రతి చెట్టుకు. పూత ముందు (డిసెంబర్) Boron + Zinc స్ప్రే.
ప్రధాన తెగుళ్లు: పూత పిండి నల్లి (Hopper), కాయ ఈగ (Fruit fly), బూడిద తెగులు (Powdery mildew).
ఉత్తమ జిల్లాలు (AP): కృష్ణా, చిత్తూరు, విశాఖపట్నం. (Telangana): రంగారెడ్డి, మహబూబ్‌నగర్.
నీటి అవసరం: 1000-1500 mm. సీజన్: శాశ్వత పంట, 5-6 సంవత్సరాల తర్వాత కాపు, ఏప్రిల్-జూన్ పండ్ల సీజన్."""
    },
    {
        "crop": "tobacco",
        "keywords": ["tobacco", "పొగాకు"],
        "text": """పంట: పొగాకు (Tobacco)
మార్కెట్ ధర: ₹15,000-₹30,000/క్వింటాల్ (FCV రకం). దిగుబడి: ఎకరాకు 6-10 క్వింటాళ్లు (క్యూర్డ్ లీఫ్).
పెట్టుబడి: ₹50,000-₹70,000/ఎకరం (barn curing ఖర్చు కలిపి). నికర లాభం: ₹60,000-₹1,50,000/ఎకరం.
ఎరువుల షెడ్యూల్: నాట్ల సమయంలో DAP 50kg + Potash 50kg. 20 రోజులకు Ammonium Sulphate 50kg. 40 రోజులకు Potash 25kg.
ప్రధాన తెగుళ్లు: బడ్ వార్మ్ (Budworm), ఆకు తినే పురుగు, బ్లాక్ షాంక్.
ఉత్తమ జిల్లాలు (AP): గుంటూరు, ప్రకాశం, కృష్ణా (FCV belt).
నీటి అవసరం: 400-600 mm. సీజన్: అక్టోబర్-మార్చి (రబీ). Tobacco Board లైసెన్స్ అవసరం."""
    },
]


# ============================================================
# KEYWORD SEARCH (replaces ChromaDB vector search)
# ============================================================
def search_crop_knowledge(query: str, crop: str = "unknown", k: int = 4) -> list[str]:
    """Simple keyword matching over crop knowledge base."""
    query_lower = query.lower()
    scores = []

    for entry in CROP_KNOWLEDGE:
        score = 0
        # Exact crop match gets highest score
        if crop != "unknown" and crop.lower() in entry["crop"].lower():
            score += 10
        # Keyword matches
        for kw in entry["keywords"]:
            if kw.lower() in query_lower:
                score += 5
        # Partial text match — check for common query terms
        text_lower = entry["text"].lower()
        query_words = query_lower.split()
        for word in query_words:
            if len(word) > 3 and word in text_lower:
                score += 1

        if score > 0:
            scores.append((score, entry["text"]))

    # Sort by score descending, return top k
    scores.sort(key=lambda x: x[0], reverse=True)
    results = [text for _, text in scores[:k]]

    # If nothing matched but crop is known, return that crop's data
    if not results and crop != "unknown":
        for entry in CROP_KNOWLEDGE:
            if crop.lower() in entry["crop"].lower() or any(crop.lower() in kw.lower() for kw in entry["keywords"]):
                results.append(entry["text"])
                break

    # If still nothing, return top 3 crops as general info
    if not results:
        results = [entry["text"] for entry in CROP_KNOWLEDGE[:3]]

    return results


# ============================================================
# WEATHER TOOLS (Open-Meteo API — free, no key needed)
# ============================================================
def geocode_location(location: str) -> dict:
    """Convert location name to lat/lon using Open-Meteo geocoding."""
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=5&language=en"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if "results" in data:
            # Prefer Indian results
            for r in data["results"]:
                if r.get("country_code") == "IN":
                    return {
                        "lat": r["latitude"],
                        "lon": r["longitude"],
                        "name": r["name"],
                        "state": r.get("admin1", ""),
                    }
            # Fallback to first result
            r = data["results"][0]
            return {"lat": r["latitude"], "lon": r["longitude"], "name": r["name"], "state": r.get("admin1", "")}
    except Exception as e:
        print(f"Geocoding error: {e}")
    return {"lat": None, "lon": None, "name": "unknown", "state": ""}


def fetch_weather(lat: float, lon: float) -> dict:
    """Fetch 7-day weather forecast from Open-Meteo."""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_max"
            f"&timezone=Asia/Kolkata&forecast_days=7"
        )
        resp = requests.get(url, timeout=10)
        data = resp.json()
        daily = data.get("daily", {})
        if daily:
            return {
                "dates": daily.get("time", []),
                "temp_max": daily.get("temperature_2m_max", []),
                "temp_min": daily.get("temperature_2m_min", []),
                "rain": daily.get("precipitation_sum", []),
                "humidity": daily.get("relative_humidity_2m_mean", []),
                "wind": daily.get("wind_speed_10m_max", []),
            }
    except Exception as e:
        print(f"Weather error: {e}")
    return {}


# ============================================================
# UNDERSTAND QUERY (using Haiku — fast & cheap)
# ============================================================
def understand_query(question: str) -> dict:
    """Parse the user's question to extract language, location, crop, intent."""
    system = """You are a query parser for an Indian agriculture advisory system.
Extract these fields from the user's question:
- language: "telugu" or "english" or "hindi"
- location: the place/district/village mentioned (or "unknown")
- crop: the crop mentioned (or "unknown")
- intent: one of [irrigation, pest, forecast, harvest, sowing, profitability, crop_selection, crop_info, general]

Respond ONLY in this exact format (no other text):
language: <value>
location: <value>
crop: <value>
intent: <value>"""

    response = client.messages.create(
        model=MODEL_HAIKU,
        max_tokens=100,
        system=system,
        messages=[{"role": "user", "content": question}],
    )

    text = response.content[0].text.strip()
    parsed = {}
    for line in text.split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            parsed[key.strip().lower()] = val.strip().lower()

    return {
        "language": parsed.get("language", "telugu"),
        "location": parsed.get("location", "unknown"),
        "crop": parsed.get("crop", "unknown"),
        "intent": parsed.get("intent", "general"),
    }


# ============================================================
# GENERATE ADVISORY (using Opus — smart, Telugu)
# ============================================================
TELUGU_SYSTEM_PROMPT = """నువ్వు కిసాన్‌మిత్ర — ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రైతులకు AI వ్యవసాయ సలహాదారు.

నీ పని: రైతు అడిగిన ప్రశ్నకు తెలుగులో సమాధానం ఇవ్వడం.

నియమాలు:
1. మొత్తం సమాధానం తెలుగు లిపిలో రాయి. ఇంగ్లీష్ వాడకు (crop names, technical terms మాత్రమే ఇంగ్లీష్‌లో ఉండవచ్చు).
2. Knowledge base నుండి ఖచ్చితమైన సంఖ్యలు (MSP, దిగుబడి, ఖర్చు, లాభం) వాడు.
3. వాతావరణ డేటా ఉంటే దాన్ని కూడా వాడు.
4. 250 పదాలలోపు సమాధానం ఇవ్వు.
5. చివరలో ఒక ప్రోత్సాహక వాక్యం రాయి.

ఉదాహరణ ప్రశ్న: "గుంటూరు లో పత్తి లాభమా?"
ఉదాహరణ సమాధానం:
# 🌾 పత్తి — గుంటూరు జిల్లా

గుంటూరు జిల్లా పత్తి సాగుకు అనువైన ప్రాంతం.

**పెట్టుబడి:** ₹20,000-₹28,000/ఎకరం
**దిగుబడి:** 8-12 క్వింటాళ్లు/ఎకరం
**MSP:** ₹7,121/క్వింటాల్
**అంచనా ఆదాయం:** ₹57,000-₹85,000/ఎకరం
**నికర లాభం:** ₹35,000-₹65,000/ఎకరం

గులాబీ పురుగు జాగ్రత్త తీసుకోండి. సకాలంలో మందు కొట్టండి.

మీ కృషి తప్పక ఫలిస్తుంది! 🌱"""


def generate_advisory(question: str, parsed: dict, weather: dict, knowledge: list[str]) -> str:
    """Generate the final advisory response."""
    language = parsed.get("language", "telugu")
    location = parsed.get("location", "unknown")
    crop = parsed.get("crop", "unknown")
    intent = parsed.get("intent", "general")

    # Build context
    context_parts = []

    if weather:
        weather_str = f"7-day forecast: "
        for i, date in enumerate(weather.get("dates", [])[:3]):
            weather_str += f"{date}: {weather['temp_max'][i]}°C/{weather['temp_min'][i]}°C, rain={weather['rain'][i]}mm. "
        context_parts.append(weather_str)

    if knowledge:
        context_parts.append("Crop Knowledge:\n" + "\n---\n".join(knowledge[:3]))

    context = "\n\n".join(context_parts) if context_parts else "No specific data available."

    # Language instruction
    if language == "telugu":
        system = TELUGU_SYSTEM_PROMPT + f"\n\nContext data:\n{context}"
    else:
        system = f"""You are KisanMitra — AI agriculture advisor for AP and Telangana farmers.
Answer in {'Hindi' if language == 'hindi' else 'English'}.
Use exact numbers from the knowledge base. Keep under 250 words.

Context data:
{context}"""

    messages = [{"role": "user", "content": question}]

    response = client.messages.create(
        model=MODEL_SMART,
        max_tokens=1024,
        temperature=0.0,
        system=system,
        messages=messages,
    )

    result = response.content[0].text

    # Telugu fallback check
    if language == "telugu" and not _is_telugu(result):
        result = _translate_to_telugu(result)

    return result


def _is_telugu(text: str) -> bool:
    """Check if text is predominantly Telugu script."""
    telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    alpha_chars = sum(1 for c in text if c.isalpha())
    if alpha_chars == 0:
        return False
    return (telugu_chars / alpha_chars) > 0.3


def _translate_to_telugu(text: str) -> str:
    """Fallback: translate English response to Telugu."""
    try:
        response = client.messages.create(
            model=MODEL_SMART,
            max_tokens=1024,
            temperature=0.0,
            system="Translate the following agricultural advisory to Telugu (తెలుగు). Keep numbers, crop names, and formatting intact.",
            messages=[{"role": "user", "content": text}],
        )
        return response.content[0].text
    except Exception:
        return text  # Return original if translation fails


# ============================================================
# MAIN ASK FUNCTION
# ============================================================
def ask(question: str) -> dict:
    """Main entry point — same interface as the full agent."""
    # Step 1: Parse query
    parsed = understand_query(question)

    # Step 2: Geocode + fetch weather (if location provided)
    weather = {}
    location_name = parsed["location"]
    if location_name != "unknown":
        geo = geocode_location(location_name)
        if geo["lat"]:
            weather = fetch_weather(geo["lat"], geo["lon"])
            location_name = geo["name"]

    # Step 3: Search crop knowledge
    knowledge = search_crop_knowledge(question, parsed["crop"])

    # Step 4: Generate advisory
    response = generate_advisory(question, parsed, weather, knowledge)

    return {
        "question": question,
        "language": parsed["language"],
        "location": location_name,
        "crop": parsed["crop"],
        "intent": parsed["intent"],
        "final_response": response,
        "response": response,
    }


# ============================================================
# HEALTH CHECK
# ============================================================
print("✅ KisanMitra Lite agent loaded!")
