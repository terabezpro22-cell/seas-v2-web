import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import mic_recorder
import PIL.Image

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SEAS V2 - Sadece Groq", layout="centered")

# API Bağlantısı (Sadece Groq)
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Sistem başlatılamadı: {e}. Lütfen .streamlit/secrets.toml dosyasına GROQ_API_KEY anahtarını eklediğinizden emin olun.")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎙️👁️ SEAS V2: Tam Operasyon (Groq Edition)")

with st.sidebar:
    st.header("🖼️ Görsel Analiz")
    uploaded_file = st.file_uploader("Resim yükle...", type=['png', 'jpg', 'jpeg'])
    if st.button("Sohbeti Sıfırla"):
        st.session_state.messages = []
        st.rerun()

# SES KAYIT
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
                # EĞER RESİM YÜKLENDİYSE: Groq'un Görme (Vision) modelini kullanıyoruz
                try:
                    response = groq_client.chat.completions.create(
                        model="llama-3.2-11b-vision-preview",
                        messages=[
                            {"role": "system", "content": "Samimi bir kankasın. Yüklenen resmi ve mesajı analiz et."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    cevap = response.choices[0].message.content
                except Exception as vision_err:
                    st.error(f"Görsel analiz modeli hatası: {vision_err}")
                    cevap = "Kanka resmi analiz ederken bir sorun çıktı, kusura bakma."
            else:
                # SADECE METİN/SES VARSA: Normal sohbet modeli
                try:
                    response = groq_client.chat.completions.create(
                        model="llama-3.3-70b-specdec",
                        messages=[{"role": "system", "content": "Samimi bir kankasın."}, {"role": "user", "content": prompt}]
                    )
                    cevap = response.choices[0].message.content
                except Exception:
                    # Yedek hızlı model
                    response = groq_client.chat.completions.create(
                        model="llama3-8b-8192",
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
