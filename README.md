## SPEECH TO SPEECH AI BOT

This project is a real-time Speech-to-Speech conversational bot.
It listens to the user's voice, converts it into text, understands it using an LLM,
detects emotion, and replies back using emotional text-to-speech audio.

-----------------------------------------
FEATURES
-----------------------------------------

1. Real-time voice input using microphone
2. Whisper-based STT (Speech to Text)
3. LLM for understanding and generating response
4. Automatic emotion detection from text
5. Emotional TTS voice response (happy, sad, excited, calm, neutral)
6. Multi-model support for Whisper and TTS
7. Uses .env file to store API keys
8. Clean modular code structure
9. Works in local environment and inside Docker

-----------------------------------------
ENVIRONMENT VARIABLES (.env)
-----------------------------------------

Create a .env file in project root:

API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini   (or any other model)

-----------------------------------------
EMOTION DETECTION LOGIC
-----------------------------------------

The bot identifies user emotion from transcribed text using:
- sentiment keywords
- voice intensity
- speaking speed (optional)
- LLM-based emotional classification

It classifies into:
- happy
- sad
- excited
- calm
- neutral

This detected emotion is mapped to TTS_EMOTIONS for emotional speech output.

-----------------------------------------
INSTALLATION STEPS
-----------------------------------------

1. Create virtual environment
   python3 -m venv env

2. Activate venv:
   Linux: source env/bin/activate
   Windows: env\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Create .env file:
   API_KEY=your_api_key
   LLM_MODEL="meta-llama/llama-4-scout-17b-16e-instruct

-----------------------------------------
GIT WORKFLOW
-----------------------------------------

Initialize git:
   git init

Add files:
   git add filename

Commit:
   git commit -m "initial commit"

Check status:
   git status

-----------------------------------------
RUNNING THE BOT
-----------------------------------------

Start the application:
   python3 main.py

The bot will:
1. Listen from microphone
2. Convert speech to text (Whisper)
3. Detect emotion from text
4. Get response from LLM
5. Convert response to emotional speech (TTS)
6. Play the audio output

-----------------------------------------
SUPPORTED TECHNOLOGIES
-----------------------------------------

- Python 3.10+
- Whisper STT
- Coqui TTS
- OpenAI / Gemini / Llama LLMs
- Emotion detection using NLP
- SoundDevice for microphone input
- Pydub for audio playback

-----------------------------------------
FOLDER STRUCTURE
-----------------------------------------

```
speech_bot/
├── Dockerfile
├── main.py
├── README.md
├── requirements.txt
├── src/
│   └── seetings.py
```
# Run the code
```
docker build -t speech-bot .
or 
python3 main.py
```



