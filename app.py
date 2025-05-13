from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import os
import cv2
import numpy as np
import tempfile
from werkzeug.utils import secure_filename
import time
import threading
import base64
from utils.sign_language import SignLanguageTranslator
from utils.speech import SpeechProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Initialize translators
sign_translator = SignLanguageTranslator()
speech_processor = SpeechProcessor()

# Global variables for live capture
live_capture_active = False
captured_frames = []
current_word = []
completed_sentence = ""

@app.route('/')
def index():
    available_languages = speech_processor.get_available_languages()
    return render_template('index.html', available_languages=available_languages)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded video
            result = sign_translator.process_video(filepath)
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return jsonify({'result': result})
    
    available_languages = speech_processor.get_available_languages()
    return render_template('upload.html', available_languages=available_languages)

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global live_capture_active, captured_frames, current_word, completed_sentence
    live_capture_active = True
    captured_frames = []
    current_word = []
    completed_sentence = ""
    
    return jsonify({'status': 'Capture started'})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global live_capture_active
    live_capture_active = False
    
    return jsonify({'status': 'Capture stopped'})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global current_word, completed_sentence
    
    data = request.get_json()
    frame_data = data['frame'].split(',')[1]  # Remove the data:image/jpeg;base64, part
    frame_bytes = base64.b64decode(frame_data)
    
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Process the frame for sign detection
    detected_sign = sign_translator.process_frame(frame)
    
    # Check for the special thumb sign (space)
    if sign_translator.is_thumb_sign(frame):
        if current_word:
            word = ''.join(current_word)
            completed_sentence += word + " "
            current_word = []
            return jsonify({'status': 'Word completed', 'current_sentence': completed_sentence})
    
    # Check for sentence completion gesture
    elif sign_translator.is_sentence_complete_sign(frame):
        if current_word:
            word = ''.join(current_word)
            completed_sentence += word
            current_word = []
        
        result = {
            'status': 'Sentence completed', 
            'final_sentence': completed_sentence,
            'audio_url': None
        }
        
        # Generate speech for the completed sentence
        if completed_sentence:
            audio_path = speech_processor.text_to_speech(completed_sentence)
            result['audio_url'] = f'/audio/{os.path.basename(audio_path)}'
        
        completed_sentence = ""
        return jsonify(result)
    
    # Normal sign processing
    elif detected_sign:
        current_word.append(detected_sign)
        return jsonify({
            'status': 'Sign detected', 
            'sign': detected_sign, 
            'current_word': ''.join(current_word),
            'current_sentence': completed_sentence
        })
    
    return jsonify({'status': 'No sign detected'})

@app.route('/translate_speech', methods=['POST'])
def translate_speech():
    data = request.get_json()
    audio_data = data.get('audio')
    target_language = data.get('target_language', 'en')
    text = data.get('text')
    
    if audio_data:
        # Save the audio data to a temporary file
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        temp_audio_file = os.path.join(app.config['UPLOAD_FOLDER'], f"speech_{int(time.time())}.webm")
        
        with open(temp_audio_file, 'wb') as f:
            f.write(audio_bytes)
        
        # Process the speech
        text = speech_processor.speech_to_text(temp_audio_file)
        
        # Clean up
        os.remove(temp_audio_file)
    
    # Translate if needed
    if target_language != 'en':
        translated_text = speech_processor.translate_text(text, target_language)
    else:
        translated_text = text
    
    # Convert to speech
    audio_path = speech_processor.text_to_speech(translated_text, target_language)
    
    return jsonify({
        'original_text': text,
        'translated_text': translated_text,
        'audio_url': f'/audio/{os.path.basename(audio_path)}'
    })

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_from_directory(speech_processor.get_audio_dir(), filename)

if __name__ == '__main__':
    os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'audio'), exist_ok=True)
    app.run(debug=True)