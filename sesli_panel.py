import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
import base64

st.set_page_config(page_title="SEAS V2 - Final", layout="centered")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎙️👁️ SEAS V2: Güncel Versiyon")

def encode_image(image_file):
    image_file.seek(0)
    return base64.b64encode(image_file.read()).decode('utf-8')

with st.sidebar:
    st.header("🖼️ Görsel Yükle")
    uploaded_file = st.file_uploader("Resim seç...", type=['png', 'jpg', 'jpeg'])
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
                
                # --- GÜNCEL MODELLER (DECOMMISSIONED OLMAYANLAR) ---
                models_to_try = [
                    "llama-3.2-11b-vision-pixtral",
                    "llama-3.2-90b-vision-preview", # Bazen isim geri gelir
                    "llama-3.2-11b-text-preview"    # Sadece metin için yedek
                ]
                
                response = None
                errors = []
                
                for model_name in models_to_try:
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
                        if response: break 
                    except Exception as e:
                        errors.append(f"{model_name}: {str(e)}")
                        continue
                
                if not response:
                    st.error("Kanka Groq'da şu an büyük bir geçiş var. Vision modelleri isim değiştiriyor.")
                    for err in errors: st.warning(err)
                    st.stop()
            else:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": "Samimi bir kankasın."}, {"role": "user", "content": prompt}]
                )
            
            cevap = response.choices[0].message.content
            st.markdown(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
            
            tts = gTTS(text=cevap[:350], lang='tr')
            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            st.audio(audio_fp, format='audio/mp3', autoplay=True)
            
        except Exception as e:
            st.error(f"Genel Hata: {e}")
