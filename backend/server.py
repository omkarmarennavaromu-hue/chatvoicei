import os
import tempfile
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Voice mapping for TTS
VOICE_MAP = {
    "Male": "onyx",
    "Female": "nova",
    "Deep Male": "echo",
    "Clear Female": "shimmer"
}

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Get audio file and selected voice from request
        audio_file = request.files['audio']
        selected_voice = request.form.get('voice', 'Male')
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
            audio_file.save(tmp_audio.name)
            tmp_audio_path = tmp_audio.name
        
        # Step 1: Transcribe audio using Whisper
        with open(tmp_audio_path, 'rb') as audio:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                language=None  # Auto-detect language
            )
        
        user_text = transcription.text
        logging.info(f"User said: {user_text}")
        
        # Step 2: Get AI response from GPT
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Respond naturally in the same language as the user."},
                {"role": "user", "content": user_text}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        ai_text = completion.choices[0].message.content
        logging.info(f"AI response: {ai_text}")
        
        # Step 3: Convert AI response to speech
        voice = VOICE_MAP.get(selected_voice, "onyx")
        
        speech_response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=ai_text
        )
        
        # Save speech to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_speech:
            speech_response.stream_to_file(tmp_speech.name)
            tmp_speech_path = tmp_speech.name
        
        # Clean up audio file
        os.unlink(tmp_audio_path)
        
        # Return both the AI text and audio
        return jsonify({
            'success': True,
            'text': ai_text,
            'audio_url': f'/get_audio/{os.path.basename(tmp_speech_path)}'
        })
        
    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_audio/<filename>')
def get_audio(filename):
    """Serve audio files"""
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, filename)
    return send_file(audio_path, mimetype='audio/mpeg', as_attachment=False)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
