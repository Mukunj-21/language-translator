import os
import tempfile
import numpy as np
import whisper
from googletrans import Translator
from gtts import gTTS
import time

class SpeechProcessor:
    def __init__(self, models_dir="models/speech_model"):
        """
        Initialize the speech processor with necessary models.
        
        Args:
            models_dir: Optional directory where pre-downloaded whisper models are stored
        """
        # Set models directory
        self.models_dir = models_dir
        
        # Initialize whisper model for speech recognition
        try:
            if self.models_dir and os.path.exists(self.models_dir):
                print(f"Loading Whisper model from: {self.models_dir}")
                # When using a local model, we can specify the path
                self.whisper_model = whisper.load_model("medium", download_root=self.models_dir)
            else:
                print("Loading Whisper model from default location")
                self.whisper_model = whisper.load_model("base")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise
        
        # Initialize translator
        self.translator = Translator()
        
        # Directory to store audio files
        self.audio_dir = os.path.join(tempfile.gettempdir(), 'audio')
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Map of language codes to full names for UI display
        self.language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'pl': 'Polish',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh-CN': 'Chinese (Simplified)',
            'zh-TW': 'Chinese (Traditional)',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'tr': 'Turkish',
            'sv': 'Swedish',
            'fi': 'Finnish',
            'da': 'Danish',
            'no': 'Norwegian',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'th': 'Thai',
            'vi': 'Vietnamese',
            'id': 'Indonesian',
            'ms': 'Malay',
            'he': 'Hebrew',
            'el': 'Greek',
            'ro': 'Romanian'
        }
        
        # Map language codes to Whisper language codes
        self.whisper_language_map = {
            'en': 'en',
            'es': 'es',
            'fr': 'fr',
            'de': 'de',
            'it': 'it',
            'pt': 'pt',
            'nl': 'nl',
            'pl': 'pl',
            'ru': 'ru',
            'ja': 'ja',
            'ko': 'ko',
            'zh-CN': 'zh',
            'zh-TW': 'zh',
            'ar': 'ar',
            'hi': 'hi',
            'tr': 'tr',
            'sv': 'sv',
            'fi': 'fi',
            'da': 'da',
            'no': 'no',
            'cs': 'cs',
            'hu': 'hu',
            'th': 'th',
            'vi': 'vi',
            'id': 'id',
            'ms': 'ms',
            'he': 'he',
            'el': 'el',
            'ro': 'ro'
        }
    
    def get_available_languages(self):
        """Return available languages for translation."""
        return self.language_map
    
    def get_audio_dir(self):
        """Return the directory where audio files are stored."""
        return self.audio_dir
    
    def speech_to_text(self, audio_file, source_lang=None):
        """
        Convert speech audio to text using Whisper.
        
        Args:
            audio_file: Path to the audio file
            source_lang: Source language code (optional)
            
        Returns:
            Transcribed text
        """
        try:
            # Prepare options for transcription
            options = {}
            
            # If source language is specified, set it in the options
            if source_lang and source_lang in self.whisper_language_map:
                whisper_lang = self.whisper_language_map[source_lang]
                options["language"] = whisper_lang
                print(f"Using specified language for transcription: {whisper_lang}")
            
            # Load audio and run inference
            result = self.whisper_model.transcribe(audio_file, **options)
            return result["text"]
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return ""
    
    def translate_text(self, text, source_lang=None, target_lang='en'):
        """
        Translate text to the target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (can be None for auto-detection)
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        try:
            # If source and target languages are the same, no need to translate
            if source_lang and source_lang == target_lang:
                return text
                
            translation = self.translator.translate(
                text, 
                src=source_lang if source_lang else 'auto', 
                dest=target_lang
            )
            return translation.text
        except Exception as e:
            print(f"Error in translation: {e}")
            return text  # Return original text if translation fails
    
    def text_to_speech(self, text, lang='en'):
        """
        Convert text to speech using gTTS.
        
        Args:
            text: Text to convert to speech
            lang: Language code
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Generate a unique filename
            filename = f"speech_{int(time.time())}.mp3"
            filepath = os.path.join(self.audio_dir, filename)
            
            # Handle language code conversion for gTTS if needed
            if lang == 'zh-CN':
                gtts_lang = 'zh-CN'
            elif lang == 'zh-TW':
                gtts_lang = 'zh-TW'
            else:
                # Extract base language code without country variant
                gtts_lang = lang.split('-')[0]
            
            # Generate speech
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            tts.save(filepath)
            
            return filepath
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return None