# Universal Language Translator

A comprehensive translation platform that bridges communication gaps through both sign language translation and multi-language speech translation.

## Features

### Sign Language Translation
- **Real-time Sign Language Detection**: Converts American Sign Language (ASL) alphabet signs to text
- **Word and Sentence Formation**: Special gestures for spaces between words and sentence completion
- **Text-to-Speech**: Converts translated sign language to spoken audio
- **High-accuracy Hand Detection**: Uses MediaPipe for precise hand tracking and gesture recognition

### Speech Translation
- **Speech Recognition**: Powered by OpenAI's Whisper model for high-accuracy speech-to-text
- **Multi-language Support**: Translate between 30+ languages
- **Text-to-Speech Synthesis**: Converts translated text back to spoken language
- **Real-time Processing**: Fast and efficient translation pipeline

## Architecture

The application consists of three main components:

1. **Sign Language Processor**: Uses computer vision and machine learning to detect and interpret sign language gestures.
2. **Speech Processor**: Handles speech recognition, language translation, and speech synthesis.
3. **Web Interface**: A Flask-based application that provides an intuitive UI for both translation modes.

## Prerequisites

- Python 3.8+
- Flask
- OpenCV
- PyTorch
- Whisper (OpenAI)
- googletrans
- gTTS (Google Text-to-Speech)
- MediaPipe
- Transformers (Hugging Face)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Mukunj-21/language-translator.git
   cd language-translator
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the sign language model:
   ```
   mkdir -p models/sign_language_model
   # Download the sign language model from Hugging Face or use your own trained model
   # Place it in the models/sign_language_model directory
   ```

5. Download the Whisper model:
   ```
   mkdir -p models/speech_model
   # Whisper will download automatically on first use, but you can pre-download it
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Use the web interface to:
   - Translate sign language to text and speech in real-time
   - Upload sign language videos for translation
   - Translate spoken language to text and other languages

## Sign Language Gestures

- **Letter Signs**: ASL alphabet (A-Z)
- **Space Between Words**: Thumb up gesture
- **End of Sentence**: Open palm with all fingers extended

## Supported Languages

The application supports translation between 30+ languages including:
- English
- Spanish
- French
- German
- Chinese (Simplified and Traditional)
- Japanese
- Korean
- Arabic
- Russian
- Portuguese
- Hindi
- And many more...

## Project Structure

```
language-translator/
├── app.py                  # Main Flask application
├── utils/
│   ├── speech.py           # Speech processing module
│   └── sign_language.py    # Sign language processing module
├── models/
│   ├── speech_model/       # Directory for Whisper models
│   └── sign_language_model/# Directory for sign language models
├── templates/              # HTML templates for the web interface
├── static/                 # CSS, JavaScript, and other static files
└── requirements.txt        # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [MediaPipe](https://mediapipe.dev/) for hand tracking
- [Hugging Face Transformers](https://huggingface.co/transformers/) for sign language models
- [Google Translate](https://cloud.google.com/translate) for text translation
- [gTTS](https://github.com/pndurette/gTTS) for text-to-speech synthesis
