from flask import Flask, request, render_template, send_from_directory, jsonify
from google.cloud import texttospeech
import os
import whisper
from dotenv import load_dotenv
import uuid
import tempfile

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure folders for audio files
AUDIO_FOLDER = os.path.join('static', 'audio')
UPLOAD_FOLDER = os.path.join('static', 'uploads')

for folder in [AUDIO_FOLDER, UPLOAD_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Load Whisper model
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("tiny.en")
    return whisper_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize_text():
    """Synthesizes speech from the input text."""
    text = request.form.get('text', '')
    
    if not text:
        return {"error": "No text provided"}, 400
    
    # Instantiate a client
    client = texttospeech.TextToSpeechClient()
    
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code=request.form.get('language', 'en-US'),
        name=request.form.get('voice', 'en-US-Standard-B'),
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    
    # Select the type of audio file you want
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=float(request.form.get('rate', 1.0)),
        pitch=float(request.form.get('pitch', 0.0))
    )
    
    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    # Generate a unique filename
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(app.config['AUDIO_FOLDER'], filename)
    
    # Save the audio content to a file
    with open(filepath, 'wb') as out:
        out.write(response.audio_content)
    
    # Return the path to the audio file
    return {"audio_path": os.path.join('static', 'audio', filename)}

@app.route('/transcribe', methods=['POST'])
def transcribe_speech():
    """Transcribes speech from an audio file using Whisper."""
    # Check if the post request has the file part
    if 'audio' not in request.files:
        return {"error": "No audio file provided"}, 400
    
    file = request.files['audio']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return {"error": "No selected file"}, 400
    
    # Save the uploaded file
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Get the Whisper model
        model = get_whisper_model()
        
        # Transcribe the audio file
        result = model.transcribe(filepath)
        
        # Return the transcription
        return {
            "text": result["text"],
            "audio_path": os.path.join('static', 'uploads', filename)
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)