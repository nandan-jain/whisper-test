from flask import Flask, request, render_template, send_from_directory
from google.cloud import texttospeech
import os
import whisper
from dotenv import load_dotenv
import uuid
import logging

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure folders for audio files
AUDIO_FOLDER = os.path.join('static', 'audio')
UPLOAD_FOLDER = os.path.join('static', 'uploads')

for folder in [AUDIO_FOLDER, UPLOAD_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['AUDIO_FOLDER'] = AUDIO_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Path to service account file (in the same directory as app.py)
SERVICE_ACCOUNT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "career-engine-453114-89f27bc7663b.json")

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
    
    try:
        # Instantiate a client with explicit credentials
        client = texttospeech.TextToSpeechClient.from_service_account_json(SERVICE_ACCOUNT_FILE)
        
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
    
    except FileNotFoundError:
        logger.error(f"Service account file not found at: {SERVICE_ACCOUNT_FILE}")
        return {"error": "Google Cloud service account file not found. Please check the file path."}, 500
    except Exception as e:
        logger.error(f"Text-to-speech error: {str(e)}")
        return {"error": f"Text-to-speech error: {str(e)}"}, 500

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
        logger.error(f"Transcription error: {str(e)}")
        return {"error": str(e)}, 500


@app.route('/transcribe-audio', methods=['GET'])
def transcribe_audio():
    """
    Transcribes a specific audio file located next to app.py
    and returns the text as HTML.
    """
    # Initialize context dictionary for template
    context = {
        'has_audio': False,
        'error': None,
        'text': None,
        'filename': None,
        'audio_path': None
    }
    
    audio_filename = "recording.m4a"  
    
    try:
        # Get the directory where app.py is located
        app_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Full path to the audio file
        filepath = os.path.join(app_dir, audio_filename)
        
        if not os.path.exists(filepath):
            context['error'] = f"Audio file '{audio_filename}' not found. Please place it next to app.py."
            return render_template('audio_transcribe.html', **context)
        
        # Get the Whisper model
        model = get_whisper_model()
        
        # Transcribe the audio file
        result = model.transcribe(filepath)
        
        # Update context with success data
        context['has_audio'] = True
        context['text'] = result["text"]
        context['filename'] = audio_filename
        
        # Reference the file directly from the app directory
        context['audio_path'] = f"/app-files/{audio_filename}"
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        context['error'] = str(e)
    
    # Return the single template with appropriate context
    return render_template('audio_transcribe.html', **context)


@app.route('/static/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(app.config['AUDIO_FOLDER'], filename)

@app.route('/static/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Check if service account file exists
    if not os.path.isfile(SERVICE_ACCOUNT_FILE):
        logger.warning(f"Google Cloud service account file not found at: {SERVICE_ACCOUNT_FILE}")
        logger.warning("Please make sure to place your service account JSON file in the same directory as app.py")
        logger.warning("Text-to-Speech functionality will not work without valid credentials")
    else:
        logger.info(f"Found Google Cloud service account file at: {SERVICE_ACCOUNT_FILE}")
    
    app.run(debug=True)