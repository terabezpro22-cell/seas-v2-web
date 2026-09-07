import streamlit as st
from groq import Groq
from gtts import gTTS
from io import BytesIO
from streamlit_mic_recorder import mic_recorder

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="SEAS V2 - Final", layout="centered")

# API Bağlantısı (Doğrudan GitHub Secrets'tan Okur)
try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"Sistem başlatılamadı: {e}. Lütfen yapılandırmayı kontrol edin.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Başlık ve Sistem Bilgisi
st.title("🎙️👁️ SEAS V2: Tam Operasyon")
st.success("⚡ **Yapay Zeka ve Gelişmiş Dil Modelleri Aktif!**")

with st.sidebar:
    st.header("🖼️ Görsel Analiz")
    st.info("🤖 Görsel Motoru: Llama 4 Scout")
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
        st.error(f"Ses Okuma Hatası: {e}")

text_input = st.chat_input("Kankana mesajını yaz...")
if text_input: prompt = text_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if uploaded_file:
                # 🖼️ GÖRSEL ANALİZ MOTORU
                try:
                    response = groq_client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
                        messages=[
                            {"role": "system", "content": "Samimi bir kankasın. Resmi ve mesajı detaylıca analiz et."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    cevap = response.choices[0].message.content
                except Exception as vision_err:
                    st.error(f"Görsel Analiz Hatası: {vision_err}")
                    cevap = "Kanka resmi şu an analiz edemedim, modelde bir yoğunluk olabilir."
            else:
                # 💬 NORMAL SOHBET MOTORLARI
                try:
                    response = groq_client.chat.completions.create(
                        model="qwen/qwen3.6-27b",
                        messages=[{"role": "system", "content": "Samimi bir kankasın."}, {"role": "user", "content": prompt}]
                    )
                    cevap = response.choices[0].message.content
                except Exception:
                    response = groq_client.chat.completions.create(
                        model="openai/gpt-oss-120b",
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
            st.error(f"İşlem Hatası: {e}")
