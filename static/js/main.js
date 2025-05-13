// DOM Elements
const videoElement = document.getElementById('videoElement');
const canvasElement = document.getElementById('canvasElement');
const startCaptureBtn = document.getElementById('startCaptureBtn');
const stopCaptureBtn = document.getElementById('stopCaptureBtn');
const uploadVideoBtn = document.getElementById('uploadVideoBtn');
const videoUpload = document.getElementById('videoUpload');
const currentSign = document.getElementById('currentSign');
const currentWord = document.getElementById('currentWord');
const translatedSentence = document.getElementById('translatedSentence');
const translationOptions = document.querySelector('.translation-options');
const targetLanguage = document.getElementById('targetLanguage');
const translateBtn = document.getElementById('translateBtn');
const translationResult = document.getElementById('translationResult');
const audioPlayer = document.getElementById('audioPlayer');
const audioElement = document.getElementById('audioElement');

// Speech Elements
const startRecordingBtn = document.getElementById('startRecordingBtn');
const stopRecordingBtn = document.getElementById('stopRecordingBtn');
const speechTargetLanguage = document.getElementById('speechTargetLanguage');
const recordingStatus = document.getElementById('recordingStatus');
const originalText = document.getElementById('originalText');
const translatedText = document.getElementById('translatedText');
const speechAudioElement = document.getElementById('speechAudioElement');

// Mode Switch Elements
const signModeBtn = document.getElementById('signModeBtn');
const speechModeBtn = document.getElementById('speechModeBtn');
const signLanguageSection = document.getElementById('signLanguageSection');
const speechSection = document.getElementById('speechSection');

// Variables
let stream = null;
let mediaRecorder = null;
let audioChunks = [];
let isCapturing = false;
let isRecording = false;
let captureInterval = null;
let lastCompletedSentence = "";

// Switch between sign language and speech modes
signModeBtn.addEventListener('click', () => {
    signModeBtn.classList.add('active');
    speechModeBtn.classList.remove('active');
    signLanguageSection.style.display = 'block';
    speechSection.style.display = 'none';
});

speechModeBtn.addEventListener('click', () => {
    signModeBtn.classList.remove('active');
    speechModeBtn.classList.add('active');
    signLanguageSection.style.display = 'none';
    speechSection.style.display = 'block';
});

// Initialize webcam
async function initializeWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: true,
            audio: false
        });
        videoElement.srcObject = stream;
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Error accessing webcam. Please make sure your camera is connected and permissions are granted.");
    }
}

// Start sign language capture
startCaptureBtn.addEventListener('click', () => {
    if (!stream) {
        initializeWebcam().then(() => {
            startCapturing();
        });
    } else {
        startCapturing();
    }
});

function startCapturing() {
    isCapturing = true;
    startCaptureBtn.disabled = true;
    stopCaptureBtn.disabled = false;
    
    // Reset displays
    currentSign.textContent = '';
    currentWord.textContent = '';
    
    // Start capturing frames at intervals
    captureInterval = setInterval(() => {
        captureFrame();
    }, 500); // Capture every 500ms
}

// Stop sign language capture
stopCaptureBtn.addEventListener('click', () => {
    stopCapturing();
});

function stopCapturing() {
    isCapturing = false;
    clearInterval(captureInterval);
    startCaptureBtn.disabled = false;
    stopCaptureBtn.disabled = true;
}

// Capture frame and send to backend
function captureFrame() {
    if (!isCapturing) return;
    
    const context = canvasElement.getContext('2d');
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
    context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    
    // Convert canvas to data URL
    const frameData = canvasElement.toDataURL('image/jpeg');
    
    // Send frame to backend
    fetch('/process_frame', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ frame: frameData }),
    })
    .then(response => response.json())
    .then(data => {
        handleFrameResponse(data);
    })
    .catch(error => {
        console.error('Error processing frame:', error);
    });
}

// Handle frame processing response
function handleFrameResponse(data) {
    switch (data.status) {
        case 'Sign detected':
            currentSign.textContent = data.sign;
            currentWord.textContent = data.current_word;
            translatedSentence.textContent = data.current_sentence;
            break;
            
        case 'Word completed':
            currentSign.textContent = '';
            currentWord.textContent = '';
            translatedSentence.textContent = data.current_sentence;
            break;
            
        case 'Sentence completed':
            currentSign.textContent = '';
            currentWord.textContent = '';
            translatedSentence.textContent = data.final_sentence;
            lastCompletedSentence = data.final_sentence;
            
            // Stop capturing
            stopCapturing();
            
            // Show translation options
            translationOptions.style.display = 'block';
            
            // If audio URL is provided, play it
            if (data.audio_url) {
                audioElement.src = data.audio_url;
                audioPlayer.style.display = 'block';
                audioElement.play();
            }
            break;
            
        case 'No sign detected':
            // Do nothing
            break;
    }
}

// Handle video upload
uploadVideoBtn.addEventListener('click', () => {
    videoUpload.click();
});

videoUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Reset displays
    currentSign.textContent = '';
    currentWord.textContent = '';
    translatedSentence.textContent = 'Processing video...';
    
    fetch('/upload', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            translatedSentence.textContent = `Error: ${data.error}`;
        } else {
            translatedSentence.textContent = data.result.detected_text;
            lastCompletedSentence = data.result.detected_text;
            
            // Show translation options
            translationOptions.style.display = 'block';
        }
    })
    .catch(error => {
        console.error('Error uploading video:', error);
        translatedSentence.textContent = 'Error processing video';
    });
});

// Translate sentence to another language
translateBtn.addEventListener('click', () => {
    if (!lastCompletedSentence) {
        alert('No sentence to translate!');
        return;
    }
    
    const selectedLanguage = targetLanguage.value;
    
    translationResult.textContent = `Translating to ${targetLanguage.options[targetLanguage.selectedIndex].text}...`;
    
    // Use Google Translate API via our backend
    fetch('/translate_speech', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            audio: null,  // No audio, using text
            text: lastCompletedSentence,
            target_language: selectedLanguage
        }),
    })
    .then(response => response.json())
    .then(data => {
        translationResult.textContent = data.translated_text;
        
        // Play audio if available
        if (data.audio_url) {
            audioElement.src = data.audio_url;
            audioPlayer.style.display = 'block';
            audioElement.play();
        }
    })
    .catch(error => {
        console.error('Error translating text:', error);
        translationResult.textContent = 'Error translating text';
    });
});

// Speech recording functionality
startRecordingBtn.addEventListener('click', async () => {
    try {
        const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(audioStream);
        audioChunks = [];
        
        mediaRecorder.addEventListener('dataavailable', event => {
            audioChunks.push(event.data);
        });
        
        mediaRecorder.addEventListener('stop', () => {
            processAudioRecording();
        });
        
        // Start recording
        mediaRecorder.start();
        isRecording = true;
        recordingStatus.textContent = 'Recording...';
        recordingStatus.classList.add('recording');
        startRecordingBtn.disabled = true;
        stopRecordingBtn.disabled = false;
        
        // Reset displays
        originalText.textContent = '';
        translatedText.textContent = '';
        speechAudioElement.src = '';
        
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Error accessing microphone. Please ensure microphone permissions are granted.');
    }
});

stopRecordingBtn.addEventListener('click', () => {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        recordingStatus.textContent = 'Processing...';
        recordingStatus.classList.remove('recording');
        startRecordingBtn.disabled = false;
        stopRecordingBtn.disabled = true;
    }
});

function processAudioRecording() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
    const reader = new FileReader();
    
    reader.onload = function() {
        const base64Audio = reader.result;
        const selectedLanguage = speechTargetLanguage.value;
        
        // Send audio to backend for processing
        fetch('/translate_speech', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                audio: base64Audio,
                target_language: selectedLanguage
            }),
        })
        .then(response => response.json())
        .then(data => {
            originalText.textContent = data.original_text;
            translatedText.textContent = data.translated_text;
            
            if (data.audio_url) {
                speechAudioElement.src = data.audio_url;
                speechAudioElement.play();
            }
            
            recordingStatus.textContent = 'Not recording';
        })
        .catch(error => {
            console.error('Error processing speech:', error);
            recordingStatus.textContent = 'Error processing speech';
        });
    };
    
    reader.readAsDataURL(audioBlob);
}

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
    // Pre-initialize webcam if possible
    initializeWebcam().catch(err => {
        console.log('Webcam will be initialized when needed');
    });
});
