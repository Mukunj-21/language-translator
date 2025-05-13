import os
import tempfile
import whisper
from googletrans import Translator
from gtts import gTTS
import time

class SpeechProcessor:
    def __init__(self):
        """Initialize the speech processor with necessary models."""
        # Initialize whisper model for speech recognition
        self.whisper_model = whisper.load_model("base")
        
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
            'zh-cn': 'Chinese (Simplified)',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'tr': 'Turkish',
            'sv': 'Swedish',
            'fi': 'Finnish',
            'da': 'Danish',
            'no': 'Norwegian',
            'cs': 'Czech',
        }
    
    def get_available_languages(self):
        """Return available languages for translation."""
        return self.language_map
    
    def get_audio_dir(self):
        """Return the directory where audio files are stored."""
        return self.audio_dir
    
    def speech_to_text(self, audio_file):
        """
        Convert speech audio to text using Whisper.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Load audio and run inference
            result = self.whisper_model.transcribe(audio_file)
            return result["text"]
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            return ""
    
    def translate_text(self, text, target_lang='en'):
        """
        Translate text to the target language.
        
        Args:
            text: Text to translate
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        try:
            translation = self.translator.translate(text, dest=target_lang)
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
            
            # Generate speech
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(filepath)
            
            return filepath
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return None
