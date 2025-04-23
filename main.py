from fastapi import FastAPI, File, UploadFile, HTTPException
import shutil
import os
import mimetypes
import requests
from datetime import datetime
from groq import Groq
from gtts import gTTS
from io import BytesIO

app = FastAPI()

# 🔹 Initialize API Client
API_KEY = "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"
client = Groq(api_key=API_KEY)

# 🔹 Directories for storing files
UPLOAD_DIR = "app/uploads"
RESPONSE_DIR = "app/ai_responses"
OUTPUT_AUDIO_DIR = "app/output"

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESPONSE_DIR, exist_ok=True)
os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

RESPONSE_FILE = os.path.join(RESPONSE_DIR, "response.txt")
OUTPUT_AUDIO_FILE = os.path.join(OUTPUT_AUDIO_DIR, "answer.mp3")

# 🔹 Allowed audio extensions
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}

def is_audio_file(filename: str) -> bool:
    """Check if file is an audio type based on extension and MIME type."""
    ext = os.path.splitext(filename)[1].lower()
    mime_type, _ = mimetypes.guess_type(filename)
    return ext in ALLOWED_EXTENSIONS and (mime_type and mime_type.startswith("audio"))

@app.get("/")
def home():
    return {"message": "Welcome to AI Audio Processing API"}

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Upload an audio file and automatically process it."""
    try:
        if not is_audio_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

        file_path = os.path.join(UPLOAD_DIR, "uploaded_audio.wav")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer, length=1024 * 1024)

        return await transcribe_and_generate_ai_response(file_path)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/transcribe-audio/")
async def transcribe_and_generate_ai_response(file_path: str):
    """Transcribe audio, generate AI response, and convert to speech."""
    if not os.path.exists(file_path):
        return {"message": "No uploaded file found. Please upload an audio file first."}

    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3-turbo", 

                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language="en",
                temperature=0.0
            )
        
        transcribed_text = transcription.text
        print(f"🔹 Transcribed Text: {transcribed_text}")
        
        if not transcribed_text.strip():
            raise HTTPException(status_code=400, detail="Transcription failed or returned empty text.")
        
        ai_response = generate_ai_response(transcribed_text)
        print(f"🔹 AI Response: {ai_response}")
        
        if not ai_response.strip():
            raise HTTPException(status_code=400, detail="AI response generation failed.")
        
        convert_text_to_speech(ai_response)

        return {
            "transcription": transcribed_text,
            "ai_response": ai_response,
            "response_text_file": RESPONSE_FILE,
            "audio_path": OUTPUT_AUDIO_FILE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription Error: {str(e)}")

def generate_ai_response(text: str):
    """Use Llama3 to generate AI response and save it."""
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": text}],
            temperature=1,
            max_tokens=1024,
            top_p=1
        )

        ai_text = response.choices[0].message.content
        with open(RESPONSE_FILE, "w", encoding="utf-8") as file:
            file.write(ai_text)

        return ai_text
    except Exception as e:
        return f"AI Response Error: {str(e)}"

def convert_text_to_speech(text: str):
    """Convert AI-generated text to speech using gTTS and save as an MP3 file."""
    try:
        print(f"🔹 Converting text to speech: {text}")
        tts = gTTS(text, lang='en')
        tts.save(OUTPUT_AUDIO_FILE)
        print(f"✅ Speech saved to: {OUTPUT_AUDIO_FILE}")
    except Exception as e:
        print(f"❌ TTS Error: {str(e)}")
