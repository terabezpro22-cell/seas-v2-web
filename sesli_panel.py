import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
import base64

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SEAS V2 - Vision & Voice", layout="centered")

# API Bağlantısı
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎙️👁️ SEAS V2: Sesli & Görsel")

# --- YAN PANEL: GÖRSEL YÜKLEME ---
with st.sidebar:
    st.header("🖼️ Soru/Görsel Yükle")
    uploaded_file = st.file_uploader("Bir resim seç veya çek...", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        st.image(uploaded_file, caption="Yüklenen Görsel", use_container_width=True)

# --- SES KAYIT BÖLÜMÜ ---
audio_input = mic_recorder(
    start_prompt="🎤 Konuşarak Soru Sor",
    stop_prompt="🛑 Bitirmek İçin Bas",
    key='mic'
)

# --- SOHBET AKIŞI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- YARDIMCI FONKSİYON: RESMİ OKUMA ---
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

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

text_input = st.chat_input("Veya buraya yaz...")
if text_input: prompt = text_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Eğer resim varsa Vision modelini, yoksa normal modeli kullanıyoruz
        if uploaded_file:
            base64_image = encode_image(uploaded_file)
            # Resim varken Llama 3.2 Vision modelini kullanıyoruz
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Görseli analiz et ve şu soruya cevap ver: {prompt}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]
            )
        else:
            # Resim yoksa standart hızlı modeli kullanıyoruz
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "Sen samimi bir kankasın. Soruları çözerken adım adım açıkla."},
                          {"role": "user", "content": prompt}]
            )
        
        cevap = response.choices[0].message.content
        st.markdown(cevap)
        st.session_state.messages.append({"role": "assistant", "content": cevap})
        
        # SESLENDİRME
        tts = gTTS(text=cevap, lang='tr')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp, format='audio/mp3', autoplay=True)
