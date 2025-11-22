## SPEECH-TO-SPEECH AI BOT

A real-time conversational AI system that listens to user speech, converts it into text, understands the meaning using an LLM, detects emotion, and replies back using expressive emotional Text-to-Speech (TTS). Designed for natural, human-like interaction.

----------------------------------------------------

## FEATURES

- Real-time voice input via microphone
- Whisper-based Speech-to-Text (STT)
- LLM-powered response generation (OpenAI / Gemini / Llama supported)
- Automatic emotion detection (happy, sad, excited, calm, neutral)
- Emotion-based TTS output using expressive voices
- Multi-model support for STT and TTS
- .env support for API key management
- Modular, clean, maintainable code structure
- Compatible with both local environment and Docker

----------------------------------------------------

## EMOTION DETECTION WORKFLOW

The bot identifies speaker emotion using:
- Sentiment keyword analysis
- Voice intensity (optional)
- Speaking speed (optional)
- LLM-based emotional classification

Supported emotions:
- happy
- sad
- excited
- calm
- neutral

The detected emotion is mapped to a matching TTS emotion profile.

----------------------------------------------------

## INSTALLATION

### 1. Create virtual environment
python3 -m venv env

### 2. Activate environment
Linux/macOS:
    source env/bin/activate
Windows:
    env\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Configure .env file
Create a .env file and add:
API_KEY=your_api_key
LLM_MODEL="meta-llama/llama-4-scout-17b-16e-instruct"

----------------------------------------------------

## GIT WORKFLOW

git init  
git add .  
git commit -m "initial commit"  
git status  

----------------------------------------------------

## RUNNING THE BOT

### Run locally:
python3 main.py

### Run using Docker:
docker build -t speech-bot .
docker run --rm -it speech-bot

----------------------------------------------------

## HOW IT WORKS (PIPELINE)

1. Listens to your voice through the microphone  
2. Converts speech to text using Whisper  
3. Detects your emotion  
4. Generates a smart response using LLM  
5. Converts the response into emotional speech  
6. Plays the audio back instantly  

----------------------------------------------------

## TECHNOLOGIES USED

- Python 3.10+
- Whisper STT
- Coqui TTS (Emotional voices)
- Llama LLMs
- NLP-based emotion detection
- SoundDevice (microphone input)
- Pydub (audio playback)

----------------------------------------------------

## PROJECT STRUCTURE
```
speech_bot/
├── Dockerfile
├── main.py
├── README.txt
├── requirements.txt
├── src/
│   └── settings.py
```
----------------------------------------------------

## USAGE
To start:
```
python3 main.py
or
docker build -t speech-bot .
