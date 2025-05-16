# Language Translator

A comprehensive web-based language translation application that supports speech-to-text, text-to-text translation, and text-to-speech conversion across multiple languages.

## Features

- **Multi-language Support**: Translate between 10+ languages including English, Spanish, French, German, Hindi, and more
- **Speech Recognition**: Convert spoken language to text using OpenAI's Whisper model
- **Text Translation**: Accurate text translation powered by Google Translate API
- **Text-to-Speech**: Convert translated text back to speech using Google Text-to-Speech (gTTS)
- **User-friendly Interface**: Clean, responsive web design for seamless translation experience

## Tech Stack

- **Backend**: Python, Flask
- **Speech Recognition**: OpenAI Whisper
- **Translation Engine**: Google Translate API
- **Text-to-Speech**: Google Text-to-Speech (gTTS)
- **Frontend**: HTML, CSS, JavaScript

## Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)
- Google Cloud account (for Google Translate API)

### Setup

1. Clone the repository
```bash
git clone https://github.com/Mukunj-21/language-translator.git
cd language-translator
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up Google Cloud credentials and API key
   - Create a Google Cloud project
   - Enable Google Translate API
   - Create and download credentials
   - Set the environment variable: `export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"`

5. Run the application
```bash
python app.py
```

6. Access the application at `http://localhost:5000`

## Usage

1. Select source and target languages from the dropdown menus
2. Choose input method:
   - **Text**: Type text directly into the input field
   - **Speech**: Click the microphone button and speak
3. Click "Translate" to process the translation
4. For text-to-speech output, click the speaker icon to hear the translated text

## How It Works

1. **Speech-to-Text**: When speech input is selected, the application uses OpenAI's Whisper model to transcribe the spoken words into text
2. **Text Translation**: The transcribed text (or directly entered text) is sent to Google Translate API for translation
3. **Text-to-Speech**: The translated text is converted back to speech using Google's Text-to-Speech (gTTS) service

## Future Improvements

- Add user accounts for saving translation history
- Support for document translation
- Offline translation mode
- Mobile application
- Batch processing for multiple translations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Google Translate API](https://cloud.google.com/translate) for text translation
- [Google Text-to-Speech (gTTS)](https://gtts.readthedocs.io/) for speech synthesis
- [Flask](https://flask.palletsprojects.com/) for the web framework
- All contributors and supporters of the project
