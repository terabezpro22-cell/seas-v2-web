import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
import base64

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SEAS V2 - Vision & Voice", layout="centered")

# API Bağlantısı (Secrets üzerinden)
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎙️👁️ SEAS V2: Sesli & Görsel")

# --- YARDIMCI FONKSİYONLAR ---
def encode_image(image_file):
    """Dosyayı base64 formatına çevirir ve imleci başa sarar."""
    image_file.seek(0)
    return base64.b64encode(image_file.read()).decode('utf-8')

# --- YAN PANEL: GÖRSEL YÜKLEME ---
with st.sidebar:
    st.header("🖼️ Soru/Görsel Yükle")
    uploaded_file = st.file_uploader("Bir resim seç veya çek...", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        st.image(uploaded_file, caption="Yüklenen Görsel", use_container_width=True)
    
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

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

# --- İŞLEME ---
prompt = ""
if audio_input:
    audio_bio = BytesIO(audio_input['bytes'])
    audio_bio.name = "audio.wav"
    try:
        transcription = client.audio.transcriptions.create(
            file=audio_bio,
            model="whisper-large-v3",
            language="tr"
        )
        prompt = transcription.text
    except Exception as e:
        st.error(f"Ses okuma hatası: {e}")

text_input = st.chat_input("Veya buraya yaz...")
if text_input: prompt = text_input

if prompt:
    # Kullanıcı mesajını ekrana bas ve hafızaya al
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # GÖRSEL VARSA VISION MODELİ
            if uploaded_file:
                base64_image = encode_image(uploaded_file)
                response = client.chat.completions.create(
                    model="llama-3.2-11b-vision-preview", # Daha stabil olan 11B Vision modeli
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Görseli analiz et ve şu soruyu cevapla: {prompt}"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=1024
                )
            # GÖRSEL YOKSA HIZLI MODEL
            else:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "Sen samimi bir kankasın. Soruları adım adım açıkla."},
                        {"role": "user", "content": prompt}
                    ]
                )
            
            cevap = response.choices[0].message.content
            st.markdown(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
            
            # SESLENDİRME (Hata almamak için kısa metin kontrolü)
            if cevap:
                tts = gTTS(text=cevap[:500], lang='tr') # Çok uzunsa ilk 500 karakteri oku
                audio_fp = BytesIO()
                tts.write_to_fp(audio_fp)
                st.audio(audio_fp, format='audio/mp3', autoplay=True)
                
        except Exception as e:
            st.error(f"Bir hata oluştu kanka: {e}")

# Sayfanın sürekli yenilenmesini engellemek için st.rerun() eklemiyoruz.
