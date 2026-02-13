import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import mic_recorder

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SEAS V2 - Sesli", layout="centered")

# API Bağlantısı
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎙️ SEAS V2 Sesli Panel")

# --- SES KAYIT BÖLÜMÜ ---
audio_input = mic_recorder(
    start_prompt="🎤 Konuşmak için bas",
    stop_prompt="🛑 Bitirmek için bas",
    key='mic'
)

# --- SOHBET AKIŞI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- İŞLEME ---
prompt = ""
if audio_input:
    audio_bio = BytesIO(audio_input['bytes'])
    audio_bio.name = "audio.wav"
    transcription = client.audio.transcriptions.create(
        file=audio_bio,
        model="whisper-large-v3",
        language="tr"
    )
    prompt = transcription.text

# Klavye girişi (yedek)
text_input = st.chat_input("Buraya yaz...")
if text_input: prompt = text_input

if prompt:
    # Kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI Cevabı
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": "Sen samimi bir kankasın. Kısa cevap ver."},
                  {"role": "user", "content": prompt}]
    )
    cevap = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": cevap})

    with st.chat_message("assistant"):
        st.markdown(cevap)
        
        # SESLENDİRME (Burada autoplay=True ama rerun yok, o yüzden bir kez konuşur)
        tts = gTTS(text=cevap, lang='tr')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp, format='audio/mp3', autoplay=True)

# DİKKAT: BURADA st.rerun() YOK! Sayfa kendi kendine dönmeyecek.
