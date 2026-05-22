"""
KisanMitra — Streamlit Frontend (Deployment Ready)
Run locally: streamlit run kisanmitra_app.py
On Render: API_URL env variable points to deployed API
"""

import os
import streamlit as st
import requests

# ============================================================
# CONFIG — uses env variable if available, otherwise localhost
# ============================================================
API_URL = os.environ.get("API_URL", "http://localhost:8000") + "/ask"

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="KisanMitra — కిసాన్ మిత్ర",
    page_icon="🌾",
    layout="centered"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header { text-align: center; padding: 10px 0; }
    .main-header h1 { color: #2E7D32; font-size: 2.5em; margin-bottom: 0; }
    .main-header p { color: #666; font-size: 1.1em; }
    .info-badge {
        display: inline-block; background: #E8F5E9; color: #2E7D32;
        padding: 4px 12px; border-radius: 15px; font-size: 0.85em; margin: 2px;
    }
    .meta-info {
        background: #FFF8E1; padding: 10px 15px; border-radius: 5px;
        font-size: 0.9em; margin-bottom: 10px;
    }
    .stChatMessage { font-size: 1.05em; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🌾 KisanMitra — కిసాన్ మిత్ర</h1>
    <p>AI Agriculture Advisory for Indian Farmers</p>
    <p>
        <span class="info-badge">🌤️ Real Weather Data</span>
        <span class="info-badge">📊 Crop Knowledge</span>
        <span class="info-badge">🗣️ Telugu • Tenglish • Hindi • English</span>
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# EXAMPLE QUESTIONS
# ============================================================
with st.expander("💡 Example questions — click to try", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Telugu / Tenglish:**")
        examples_telugu = [
            "Guntur lo paddy ki neellu pettala?",
            "కర్నూలు లో పత్తి పంటకి పురుగు మందు కొట్టాలా?",
            "Nalgonda lo naa paddy pantaki neellu pettala?",
            "అనంతపురం లో ఈ సీజన్ కి ఏం వేయాలి?",
            "Komarada lo sugarcane veste laabham avutunda?",
        ]
        for ex in examples_telugu:
            if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
                st.session_state.example_question = ex
    with col2:
        st.markdown("**English / Hindi:**")
        examples_other = [
            "What is the weather forecast for Hyderabad?",
            "Is groundnut profitable in Anantapur?",
            "వరి MSP ఎంత? ఎకరాకు ఎంత లాభం?",
            "पत्ति पंटकी ఎరువులు ఎప్పుడు వేయాలి?",
            "Best crop for Kurnool this season?",
        ]
        for ex in examples_other:
            if st.button(ex, key=f"ex2_{ex[:20]}", use_container_width=True):
                st.session_state.example_question = ex


# ============================================================
# CHAT HISTORY
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "example_question" not in st.session_state:
    st.session_state.example_question = None

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👨‍🌾" if message["role"] == "user" else "🌾"):
        if message["role"] == "assistant" and "meta" in message:
            meta = message["meta"]
            st.markdown(f"""<div class="meta-info">
                📍 <b>{meta.get('location', 'N/A')}</b>, {meta.get('state', '')} &nbsp;|&nbsp;
                🌾 <b>{meta.get('crop', 'N/A')}</b> &nbsp;|&nbsp;
                🎯 <b>{meta.get('intent', 'N/A')}</b> &nbsp;|&nbsp;
                🗣️ <b>{meta.get('language', 'N/A')}</b>
                {"&nbsp;|&nbsp; 📚 <b>Knowledge Used</b>" if meta.get('has_knowledge') else ""}
            </div>""", unsafe_allow_html=True)
        st.markdown(message["content"])


# ============================================================
# CALL API
# ============================================================
def call_kisanmitra(question: str) -> dict:
    try:
        response = requests.post(API_URL, json={"question": question}, timeout=120)
        if response.status_code == 200:
            return response.json()
        return {"error": f"API error: {response.status_code} — {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "❌ Cannot connect to API. The backend might be starting up (wait 60 seconds and try again)."}
    except requests.exceptions.Timeout:
        return {"error": "⏳ Request timed out. Try again."}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}


# ============================================================
# HANDLE INPUT
# ============================================================
prompt = None
if st.session_state.example_question:
    prompt = st.session_state.example_question
    st.session_state.example_question = None

if user_input := st.chat_input("మీ వ్యవసాయ ప్రశ్న అడగండి... (Ask your farming question)"):
    prompt = user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👨‍🌾"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌾"):
        with st.spinner("🌾 KisanMitra is thinking..."):
            result = call_kisanmitra(prompt)

        if "error" in result:
            st.error(result["error"])
            st.session_state.messages.append({"role": "assistant", "content": result["error"]})
        else:
            meta = {
                "location": result.get("location", "N/A"),
                "state": result.get("state", ""),
                "crop": result.get("crop", "N/A"),
                "intent": result.get("intent", "N/A"),
                "language": result.get("language", "N/A"),
                "has_knowledge": result.get("has_knowledge", False)
            }
            st.markdown(f"""<div class="meta-info">
                📍 <b>{meta['location']}</b>, {meta['state']} &nbsp;|&nbsp;
                🌾 <b>{meta['crop']}</b> &nbsp;|&nbsp;
                🎯 <b>{meta['intent']}</b> &nbsp;|&nbsp;
                🗣️ <b>{meta['language']}</b>
                {"&nbsp;|&nbsp; 📚 <b>Knowledge Used</b>" if meta['has_knowledge'] else ""}
            </div>""", unsafe_allow_html=True)
            st.markdown(result["response"])
            st.session_state.messages.append({
                "role": "assistant", "content": result["response"], "meta": meta
            })
    st.rerun()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🌾 About KisanMitra")
    st.markdown("""
    **KisanMitra** is an AI-powered agriculture advisory
    system for Indian farmers.

    **Features:**
    - 🌤️ Real-time weather data
    - 📊 Crop cost & profit analysis
    - 🐛 Pest & disease advisory
    - 💧 Irrigation recommendations
    - 🌱 Crop selection guidance
    - 🗣️ Telugu, Tenglish, Hindi, English

    **Powered by:**
    - Claude AI (Anthropic)
    - Open-Meteo Weather API
    - LangGraph Agent Framework
    """)

    st.divider()

    st.markdown("### 📊 Supported Crops")
    crops = ["🌾 Paddy", "🌿 Cotton", "🥜 Groundnut", "🌶️ Chilli",
             "🎋 Sugarcane", "🌽 Maize", "🟡 Turmeric", "🍌 Banana",
             "🍅 Tomato", "🧅 Onion", "🥭 Mango", "🍃 Tobacco"]
    for crop in crops:
        st.markdown(f"- {crop}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("<center><small>Built with ❤️ by Abhishek</small></center>", unsafe_allow_html=True)
