import streamlit as st
import google.generativeai as genai
from groq import Groq
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
import PIL.Image

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SEAS V2 - Kesin Çözüm", layout="centered")

# API Bağlantıları
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    # HATA ÇÖZÜMÜ: Model ismini tam yol olarak tanımlıyoruz
    # Eğer 'gemini-1.5-flash' çalışmazsa 'models/gemini-1.5-flash' deneyecek
    try:
        vision_model = genai.GenerativeModel('gemini-1.5-flash')
    except:
        vision_model = genai.GenerativeModel('models/gemini-1.5-flash')
        
except Exception as e:
    st.error(f"API Bağlantı Hatası: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎙️👁️ SEAS V2: Tam Operasyon")

with st.sidebar:
    st.header("🖼️ Görsel Analiz")
    uploaded_file = st.file_uploader("Resim yükle...", type=['png', 'jpg', 'jpeg'])
    if st.button("Sohbeti Sıfırla"):
        st.session_state.messages = []
        st.rerun()

# SES KAYIT (Groq Whisper)
audio_input = mic_recorder(start_prompt="🎤 Sesli Sor", stop_prompt="🛑 Durdur", key='mic')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = ""
if audio_input:
    audio_bio = BytesIO(audio_input['bytes'])
    audio_bio.name = "audio.wav"
    try:
        transcription = groq_client.audio.transcriptions.create(
            file=audio_bio, model="whisper-large-v3", language="tr"
        )
        prompt = transcription.text
    except Exception as e:
        st.error(f"Ses okuma hatası: {e}")

text_input = st.chat_input("Mesajını yaz...")
if text_input: prompt = text_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if uploaded_file:
                # --- GEMINI GÖRSEL İŞLEME ---
                img = PIL.Image.open(uploaded_file)
                # Yeni modellerde generate_content kullanımı bazen model ismine göre değişir
                response = vision_model.generate_content([prompt, img])
                cevap = response.text
            else:
                # --- GROQ METİN SOHBETİ ---
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": "Samimi bir kankasın."}, {"role": "user", "content": prompt}]
                )
                cevap = response.choices[0].message.content
            
            st.markdown(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
            
            # SESLENDİRME
            tts = gTTS(text=cevap[:350], lang='tr')
            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            st.audio(audio_fp, format='audio/mp3', autoplay=True)
            
        except Exception as e:
            st.error(f"Kanka şöyle bir hata çıktı: {e}")
            st.info("İpucu: Eğer model bulunamadı diyorsa API anahtarının aktifleşmesi 1-2 dakika sürebilir.")
