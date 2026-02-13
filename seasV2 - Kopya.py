import streamlit as st
import requests
from groq import Groq
from PIL import Image
from io import BytesIO
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SEAS V2 - Akıllı Asistan", page_icon="🤖", layout="centered")

# CSS ile Şık Bir Görünüm
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

# API Key'i Secrets'tan Çekiyoruz (Bulut Güvenliği)
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=API_KEY)
except Exception:
    st.error("Kanka, Streamlit Cloud ayarlarından GROQ_API_KEY'i eklemeyi unutma!")

st.title("🤖 SEAS V2 - Web & Sesli")
st.caption("Kanka hoş geldin! İster yaz, ister konuş, ister resim çizdir.")

# Sohbet Geçmişini Hafızada Tut
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- MİKROFON GİRİŞİ ---
audio_input = mic_recorder(
    start_prompt="🎤 Sesli Komut Ver",
    stop_prompt="🛑 Durdur ve Sor",
    key='recorder'
)

# Sesli girişi metne çevir (Whisper Large V3)
user_prompt = ""
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
            user_prompt = transcription.text
        except Exception as e:
            st.error(f"Ses anlaşılamadı kanka: {e}")

# Klavye girişi
text_input = st.chat_input("Mesajını buraya yaz kanka...")
final_prompt = user_prompt if user_prompt else text_input

# Geçmiş mesajları ekrana bas (Sohbet balonları)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ANA MOTOR (CEVAP ÜRETME) ---
if final_prompt:
    # Kullanıcının mesajını göster
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    with st.chat_message("assistant"):
        # GÖRSEL ÇİZME KOMUTU
        if any(k in final_prompt.lower() for k in ["çiz", "resim", "görsel", "foto"]):
            st.write("Hemen hayal ediyorum...")
            clean_prompt = final_prompt.lower().replace("çiz", "").replace("resim", "").strip()
            url = f"https://image.pollinations.ai/prompt/{clean_prompt.replace(' ', '%20')}?width=1024&height=1024&model=flux&nologo=true"
            st.image(url, caption=f"İsteğin: {clean_prompt}")
            st.session_state.messages.append({"role": "assistant", "content": f"Resmi çizdim kanka: {clean_prompt}"})
        
        # NORMAL SOHBET
        else:
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Sen SEAS V2'sin. Çok samimi, zeki, 'kanka' diye hitap eden bir asistansın."},
                        {"role": "user", "content": final_prompt}
                    ]
                )
                cevap = response.choices[0].message.content
                st.markdown(cevap) # Yazılı mesajı basar
                
                # İSTEĞE BAĞLI SESLİ OKUMA
                if st.button("🔊 Cevabı Sesli Dinle"):
                    with st.spinner("Ses hazırlanıyor..."):
                        tts = gTTS(text=cevap, lang='tr')
                        audio_fp = BytesIO()
                        tts.write_to_fp(audio_fp)
                        st.audio(audio_fp, format='audio/mp3', autoplay=True)
                
                st.session_state.messages.append({"role": "assistant", "content": cevap})
            except Exception as e:
                st.error(f"Bir hata oluştu kanka: {e}")