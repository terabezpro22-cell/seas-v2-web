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

st.title("🎙️👁️ SEAS V2: Tam Gaz")

def encode_image(image_file):
    image_file.seek(0)
    return base64.b64encode(image_file.read()).decode('utf-8')

with st.sidebar:
    st.header("🖼️ Görsel Analiz")
    uploaded_file = st.file_uploader("Soru veya Resim Yükle", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        st.image(uploaded_file, caption="Yüklendi!", use_container_width=True)
    if st.button("Sohbeti Sıfırla"):
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
    transcription = client.audio.transcriptions.create(file=audio_bio, model="whisper-large-v3", language="tr")
    prompt = transcription.text

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
                
                # --- AKILLI MODEL SEÇİCİ ---
                # Önce 11b'yi dene, olmazsa 90b'yi dene
                model_list = ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]
                
                response = None
                for model_name in model_list:
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]
                            }]
                        )
                        break # Eğer başarılı olursa döngüden çık
                    except Exception as e:
                        continue # Hata verirse bir sonraki modeli dene
                
                if not response:
                    st.error("Groq tüm vision modellerini bakıma almış kanka. Birazdan tekrar dene.")
                    st.stop()
            else:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": "Samimi bir kankasın."}, {"role": "user", "content": prompt}]
                )
            
            cevap = response.choices[0].message.content
            st.markdown(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
            
            tts = gTTS(text=cevap[:400], lang='tr')
            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            st.audio(audio_fp, format='audio/mp3', autoplay=True)
            
        except Exception as e:
            st.error(f"Genel Hata: {e}")
