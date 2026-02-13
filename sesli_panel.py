import streamlit as st
import requests
from groq import Groq
from PIL import Image
from io import BytesIO
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

# --- SAYFA AYARLARI VE TASARIM ---
st.set_page_config(page_title="SEAS V2 - Sesli Panel", page_icon="🎙️", layout="wide")

# Dark Mode ve Şık Arayüz İçin CSS
st.markdown("""
    <style>
    .main { background-color: #050505; }
    .stChatInput { bottom: 20px; }
    .status-box { padding: 20px; border-radius: 15px; background: #1a1a1a; border: 1px solid #333; text-align: center; }
    .voice-glow { box-shadow: 0 0 15px #00f2fe; border-radius: 50%; }
    </style>
    """, unsafe_allow_html=True)

# API Bağlantısı
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=API_KEY)
except:
    st.error("API Anahtarı bulunamadı!")

# --- SOL PANEL (BİLGİ) ---
with st.sidebar:
    st.title("🎙️ SEAS V2 Sesli")
    st.info("Kanka bu panelde konuşmalar otomatik olarak seslendirilir. Sesini gönder, cevabı bekle!")
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

# --- ANA EKRAN TASARIMI ---
col1, col2 = st.columns([1, 1])

if "messages" not in st.session_state:
    st.session_state.messages = []

with col1:
    st.subheader("🤖 Asistan Paneli")
    # Mesajları göster
    chat_container = st.container(height=400)
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

with col2:
    st.subheader("🎙️ Ses Kontrol Merkezi")
    st.write("Aşağıdaki butona bas ve konuşmanı yap:")
    
    # Gelişmiş Mikrofon
    audio_input = mic_recorder(
        start_prompt="🎤 Dinlemeye Başla",
        stop_prompt="🛑 Konuşmayı Bitir",
        key='voice_panel'
    )

    # Sesli Yanıt Durumu
    voice_status = st.empty()

# --- İŞLEME MERKEZİ ---
final_prompt = ""

# Eğer ses gelirse
if audio_input:
    with st.spinner("Sesin çözülüyor..."):
        try:
            audio_bio = BytesIO(audio_input['bytes'])
            audio_bio.name = "audio.wav"
            transcription = client.audio.transcriptions.create(
                file=audio_bio,
                model="whisper-large-v3",
                language="tr"
            )
            final_prompt = transcription.text
        except Exception as e:
            st.error(f"Hata: {e}")

# Klavye de çalışsın
text_input = st.chat_input("Veya buraya yaz...")
if text_input: final_prompt = text_input

if final_prompt:
    # 1. Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    
    # 2. AI Cevabı Üret
    with st.spinner("SEAS V2 düşünüyor..."):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "Sen SEAS V2'sin. Çok samimi bir kankasın. Kısa ve öz cevap ver."},
                      {"role": "user", "content": final_prompt}]
        )
        cevap = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": cevap})

    # 3. OTOMATİK SESLENDİRME
    tts = gTTS(text=cevap, lang='tr')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    
    # Ekrana bas ve OYNAT
    with col2:
        st.success("✅ Cevap Hazır!")
        st.audio(audio_fp, format='audio/mp3', autoplay=True)
        st.write(f"**Asistan Diyor Ki:** {cevap}")
    
    st.rerun()