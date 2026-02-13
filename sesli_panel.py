import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
import base64

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SEAS V2 - Final Fix", layout="centered")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎙️👁️ SEAS V2: Gözler Açıldı")

def encode_image(image_file):
    image_file.seek(0)
    return base64.b64encode(image_file.read()).decode('utf-8')

with st.sidebar:
    st.header("🖼️ Görsel İşleme")
    uploaded_file = st.file_uploader("Resim yükle...", type=['png', 'jpg', 'jpeg'])
    if st.button("Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

audio_input = mic_recorder(start_prompt="🎤 Sesli Sor", stop_prompt="🛑 Durdur", key='mic')

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = ""
if audio_input:
    audio_bio = BytesIO(audio_input['bytes'])
    audio_bio.name = "audio.wav"
    try:
        transcription = client.audio.transcriptions.create(file=audio_bio, model="whisper-large-v3", language="tr")
        prompt = transcription.text
    except Exception as e:
        st.error(f"Ses okuma hatası: {e}")

text_input = st.chat_input("Buraya yaz...")
if text_input: prompt = text_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if uploaded_file:
                base64_image = encode_image(uploaded_file)
                
                # --- ŞU AN GROQ'DA ÇALIŞAN TEK GÖRSEL MODEL ---
                # Diğerleri bakımda veya silindiği için tek seçenek bu:
                response = client.chat.completions.create(
                    model="llava-v1.5-7b-4096-preview", 
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }]
                )
            else:
                response = client.chat.completions.create(
                    model="llama3-70b-8192", # En stabil metin modeli
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
            st.error(f"Hata: {e}")
            st.info("Kanka Groq'un ücretsiz vision modelleri şu an sınırlı olabilir.")
