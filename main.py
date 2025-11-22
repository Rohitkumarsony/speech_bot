import numpy as np
import sounddevice as sd
import threading
import time
import os
import torch
import requests
import tempfile
import sys
import scipy.signal as signal
from src.seetings import API_KEY, LLM_MODEL,TTS_MODELS, TTS_EMOTIONS, WHISPER_MODELS,tts_quality, whisper_size, language

# Try importing required packages - install if needed
try:
    import whisper
    from TTS.api import TTS
    from pydub import AudioSegment
    from pydub.playback import play
    import noisereduce as nr
except ImportError:
    import pip
    print("Installing required packages...")
    pip.main(['install', 'openai-whisper', 'TTS', 'sounddevice', 'numpy', 'scipy', 'requests', 'pydub', 'PyAudio', 'librosa', 'noisereduce'])
    import whisper
    from TTS.api import TTS
    from pydub import AudioSegment
    from pydub.playback import play
    import librosa
    import noisereduce as nr

class SequentialSpeechSystem:
    def __init__(self, tts_quality="natural", whisper_size="small", language="en"):
        # Basic configuration
        self.sample_rate = 16000
        self.language = language
        self.recording_duration = 7.0  # Default recording duration in seconds
        
        # Device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Processing flags
        self.running = False
        self.listening = False
        self.processing = False
        self.speaking = False
        
        # Initialize locks and events
        self.state_lock = threading.Lock()
        self.speech_detected_event = threading.Event()
        self.processing_complete_event = threading.Event()
        
        # Initialize whisper model
        whisper_choice = WHISPER_MODELS.get(whisper_size, WHISPER_MODELS["base"])
        print(f"Loading Whisper model '{whisper_choice['model']}' on {self.device}...")
        self.whisper = whisper.load_model(whisper_choice["model"], device=self.device)
        
        # Initialize TTS model
        self.tts_model = TTS_MODELS.get(tts_quality, TTS_MODELS["natural"])  # Default to natural voice
        print(f"Loading TTS model: {self.tts_model}...")
        self.tts = TTS(self.tts_model, progress_bar=False, gpu=(self.device == "cuda"))
        
        # Get available speakers if the model supports them
        self.available_speakers = []
        self.speaker_manager = None
        try:
            if hasattr(self.tts, "speaker_manager") and self.tts.speaker_manager is not None:
                self.speaker_manager = self.tts.speaker_manager
                self.available_speakers = self.tts.speaker_manager.speaker_names
                print(f"Available speakers: {self.available_speakers}")
                if self.available_speakers:
                    self.current_speaker = self.available_speakers[0]
                    print(f"Using speaker: {self.current_speaker}")
        except Exception as e:
            print(f"Error getting speakers: {e}")
            self.speaker_manager = None
        
        # Preload TTS to reduce initial latency
        _ = self.tts.tts("System initializing", speed=1.0)
        
        # Caches for LLM responses and TTS
        self.llm_cache = {}
        self.tts_cache = {}
        
        # Temporary directory for audio files
        self.temp_dir = tempfile.mkdtemp()
        print(f"Using temporary directory: {self.temp_dir}")
        
        # History for context
        self.conversation_history = []
        self.max_history_items = 5
        
        # Audio detection parameters
        self.vad_threshold = 0.03  # Initial threshold for voice detection
        self.silence_duration = 1.0  # Seconds of silence to consider end of speech
        self.max_listen_time = 21.0  # Maximum time to listen for input
        
        # Current audio data
        self.current_audio = []
        
        # Emotion detection and processing
        self.current_emotion = "neutral"
        self.emotion_keywords = {
            "happy": ["happy", "glad", "joy", "exciting", "wonderful", "fantastic"],
            "sad": ["sad", "sorry", "unfortunate", "regret", "disappointing"],
            "excited": ["wow", "amazing", "awesome", "incredible", "excellent"],
            "calm": ["calm", "relax", "gentle", "soothing", "peaceful"]
        }
        
        print("System initialized!")
    
    def _detect_emotion_from_text(self, text):
        """Detect emotion from text content"""
        text_lower = text.lower()
        
        # Count emotion keywords
        emotion_scores = {emotion: 0 for emotion in self.emotion_keywords}
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                emotion_scores[emotion] += text_lower.count(keyword)
        
        # Check for question marks (curious tone)
        if "?" in text:
            emotion_scores["excited"] += 1
        
        # Check for exclamation marks (excited tone)
        if "!" in text:
            emotion_scores["excited"] += 2
        
        # Get max score emotion or default to neutral
        max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        if max_emotion[1] > 0:
            return max_emotion[0]
        return "neutral"
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio recording"""
        if status:
            print(f"Audio status: {status}")
        
        # Get audio data as mono and float32
        audio_chunk = np.squeeze(indata).astype(np.float32)
        
        # Add the new audio data to the current recording
        self.current_audio.append(audio_chunk.copy())
        
        # Detect if this is speech
        energy = np.sqrt(np.mean(audio_chunk**2))
        if energy > self.vad_threshold:
            self.last_speech_time = time.time()
            self.speech_detected_event.set()
    
    def _listen_for_speech(self):
        """Record audio until silence is detected or max time reached"""
        print("🎤 Listening for speech...")
        self.current_audio = []
        self.last_speech_time = time.time()
        self.speech_detected_event.clear()
        
        start_time = time.time()
        with self.state_lock:
            self.listening = True
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=self._audio_callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            ):
                # Wait for initial speech
                speech_wait_timeout = 15.0  # Wait up to 5 seconds for speech to start
                if not self.speech_detected_event.wait(timeout=speech_wait_timeout):
                    print("No speech detected, listening again...")
                    with self.state_lock:
                        self.listening = False
                    return None
                
                # Continue recording until silence is detected or max time is reached
                while self.running:
                    time_elapsed = time.time() - start_time
                    time_since_last_speech = time.time() - self.last_speech_time
                    
                    # Stop if silence for too long or max time reached
                    if time_since_last_speech > self.silence_duration or time_elapsed > self.max_listen_time:
                        break
                    
                    time.sleep(0.1)
                
                # Process the recorded audio if we have enough
                if len(self.current_audio) > 5:  # Ensure we have enough audio chunks
                    # Combine audio chunks
                    combined_audio = np.concatenate(self.current_audio).astype(np.float32)
                    print(f"Recorded {len(combined_audio)/self.sample_rate:.1f} seconds of audio")
                    
                    with self.state_lock:
                        self.listening = False
                    
                    return combined_audio
                else:
                    print("Not enough audio recorded")
                    with self.state_lock:
                        self.listening = False
                    return None
                
        except Exception as e:
            print(f"Recording error: {e}")
            with self.state_lock:
                self.listening = False
            return None
    
    def _normalize_audio(self, audio):
        """Normalize audio for whisper"""
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio
    
    def _enhance_audio(self, audio):
        """Apply audio enhancements to improve speech quality"""
        try:
            # Ensure audio is float32
            audio = audio.astype(np.float32)
            
            # Check if audio is too quiet
            if np.max(np.abs(audio)) < 0.1:
                # Apply gain
                audio = audio * (0.5 / (np.max(np.abs(audio)) + 1e-10))
            
            # Apply a bandpass filter to focus on speech frequencies (300Hz-3400Hz)
            sos = signal.butter(4, [300, 3400], 'bandpass', fs=self.sample_rate, output='sos')
            audio = signal.sosfilt(sos, audio).astype(np.float32)
            
            # Apply noise reduction with more aggressive settings
            try:
                # Get noise profile from the first 0.3 seconds
                noise_part = audio[:int(self.sample_rate * 0.3)]
                enhanced = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    prop_decrease=0.8,
                    stationary=True,
                    n_std_thresh_stationary=1.5
                )
                if not np.isnan(enhanced).any():
                    audio = enhanced
            except Exception as nr_error:
                print(f"Noise reduction error (non-critical): {nr_error}")
            
            # Apply a slight compression to level out volume
            threshold = 0.5
            ratio = 4.0  # 4:1 compression
            audio_compressed = np.copy(audio)
            mask = np.abs(audio) > threshold
            audio_compressed[mask] = np.sign(audio[mask]) * (threshold + (np.abs(audio[mask]) - threshold) / ratio)
            
            return audio_compressed.astype(np.float32)
            
        except Exception as e:
            print(f"Audio enhancement error: {e}")
            # Return original audio if enhancement fails
            return audio.astype(np.float32)
    
    def _process_audio(self, audio):
        """Process audio to text using Whisper"""
        try:
            with self.state_lock:
                self.processing = True
            
            print("Processing speech...")
            
            # Enhance and normalize the audio
            enhanced_audio = self._enhance_audio(audio)
            normalized_audio = self._normalize_audio(enhanced_audio)
            
            # Debug: Save audio to file if needed
            # debug_path = os.path.join(self.temp_dir, f"speech_{time.time()}.wav")
            # wavfile.write(debug_path, self.sample_rate, (normalized_audio * 32767).astype(np.int16))
            
            # Transcribe with whisper
            result = self.whisper.transcribe(
                normalized_audio,
                language=self.language,
                initial_prompt="The following is a transcription of spoken words.",
                fp16=(self.device == "cuda"),
                word_timestamps=False,
                temperature=0.0  # Use greedy decoding for more predictable results
            )
            
            text = result.get('text', '').strip()
            
            # Only process meaningful text
            if len(text) > 2 and len(text.split()) > 1:
                print(f"📝 Transcription: {text}")
                return text
            else:
                print("Empty or too short transcription")
                return None
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
        finally:
            with self.state_lock:
                self.processing = False
    
    def _process_text(self, text):
        """Process text with LLM"""
        try:
            if not text:
                return None
                
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Limit history size
            if len(self.conversation_history) > self.max_history_items * 2:
                self.conversation_history = self.conversation_history[-self.max_history_items*2:]
            
            # Use cache if possible for exact matches
            cache_key = text.lower().strip()
            if cache_key in self.llm_cache:
                response = self.llm_cache[cache_key]
                print(f"💬 [Cached] Response: {response}")
            else:
                # Query LLM API
                print("Generating response...")
                response = self._query_llm(text)
                
                # Cache the response
                if len(self.llm_cache) > 50:  # Limit cache size
                    self.llm_cache.clear()
                self.llm_cache[cache_key] = response
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Detect emotion from response for TTS
            self.current_emotion = self._detect_emotion_from_text(response)
            print(f"Detected emotion for TTS: {self.current_emotion}")
            
            return response
            
        except Exception as e:
            print(f"LLM processing error: {e}")
            return "Sorry, I couldn't process your request."
    
    def _query_llm(self, text):
        """Query LLM API with text and conversation history"""
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            }
            
            # Build messages with history
            messages = [
                {
                    "role": "system", 
                    "content": "You are a professional, conversational voice assistant. Your responses must be natural, concise, and clear. Avoid exaggerated expressions like 'wow' or 'aww'. Keep answers short (1–3 sentences or about 15–45 words) but maintain a friendly and engaging tone."
                }
            ]
            
            # Add relevant conversation history
            if len(self.conversation_history) > 0:
                # Add up to max_history_items of history
                for item in self.conversation_history[-self.max_history_items*2:]:
                    messages.append(item)
            
            # Add current message if not already in history
            if not any(item["content"] == text and item["role"] == "user" for item in messages):
                messages.append({"role": "user", "content": text})
            
            data = {
                "model": LLM_MODEL,
                "messages": messages,
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                print(f"💬 Response: {response_text}")
                return response_text
            else:
                print(f"API error: {response.status_code}")
                return "I'm having trouble connecting right now."
        except Exception as e:
            print(f"API error: {e}")
            return "Sorry, I couldn't generate a response."
    
    def _speak_response(self, text):
        """Convert text to speech and play it"""
        try:
            if not text:
                return
                
            with self.state_lock:
                self.speaking = True
                
            print("Converting to speech...")
            
            # Use cache if possible (with emotion consideration)
            cache_key = f"{self.current_emotion}_{text[:50]}"  # Use emotion + first 50 chars as key
            if cache_key in self.tts_cache and os.path.exists(self.tts_cache[cache_key]):
                audio_path = self.tts_cache[cache_key]
                print("Using cached audio")
            else:
                # Convert to speech with emotional parameters
                audio_path = self._text_to_speech(text)
                
                # Cache the result
                if audio_path:
                    if len(self.tts_cache) > 20:  # Limit cache size
                        for old_path in list(self.tts_cache.values()):
                            if os.path.exists(old_path) and old_path != audio_path:
                                try:
                                    os.unlink(old_path)
                                except:
                                    pass
                        self.tts_cache.clear()
                    
                    self.tts_cache[cache_key] = audio_path
            
            # Play the audio
            if audio_path and os.path.exists(audio_path):
                self._play_audio(audio_path)
            
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            with self.state_lock:
                self.speaking = False
    
    def _text_to_speech(self, text):
        """Convert text to speech using Coqui TTS with emotional parameters"""
        try:
            # Create temp file
            temp_file = os.path.join(self.temp_dir, f"response_{time.time()}.wav")
            
            # Get emotion parameters
            emotion_params = TTS_EMOTIONS.get(self.current_emotion, TTS_EMOTIONS["neutral"])
            
            # Apply additional text processing for better prosody
            processed_text = self._preprocess_text_for_tts(text)
            
            # Generate speech with emotional parameters
            try:
                # Use Coqui TTS with speaker if available
                if self.speaker_manager and self.available_speakers:
                    # Random speaker variation to add diversity
                    speaker = self.available_speakers[hash(processed_text) % len(self.available_speakers)]
                    
                    # Generate with speaker and emotion
                    self.tts.tts_to_file(
                        text=processed_text,
                        file_path=temp_file,
                        speaker=speaker,
                        speed=emotion_params["speed"],
                    )
                else:
                    # Generate with emotion only
                    self.tts.tts_to_file(
                        text=processed_text, 
                        file_path=temp_file,
                        speed=emotion_params["speed"]
                    )
                
                # Process the generated audio file to apply emotional effects
                if os.path.exists(temp_file):
                    self._apply_audio_effects(temp_file, emotion_params)
                    return temp_file
                else:
                    print("TTS failed to create audio file")
                    return None
                    
            except Exception as e:
                print(f"Coqui TTS error: {e}")
                
                # Fall back to gTTS if Coqui fails
                try:
                    from gtts import gTTS
                    temp_file = os.path.join(self.temp_dir, f"response_{time.time()}.mp3")
                    tts = gTTS(text=processed_text, lang='en', slow=False)
                    tts.save(temp_file)
                    return temp_file
                except Exception as gtts_error:
                    print(f"gTTS error: {gtts_error}")
                    return None
                
        except Exception as e:
            print(f"TTS error: {e}")
            print(f"🔊 SPOKEN RESPONSE: {text}")
            return None
    
    def _preprocess_text_for_tts(self, text):
        """Preprocess text to improve TTS prosody and emotional expression"""
        # Replace common abbreviations
        replacements = {
            "I'm": "I am",
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "I'll": "I will",
            "you'll": "you will",
            "they'll": "they will",
            "we'll": "we will",
            "he'll": "he will",
            "she'll": "she will",
            "it'll": "it will",
            "you're": "you are",
            "they're": "they are",
            "we're": "we are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is"
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Add prosody markers based on emotion
        if self.current_emotion == "excited":
            # Add emphasis on key words
            words = text.split()
            for i in range(len(words)):
                if len(words[i]) > 4 and i % 4 == 0:  # Emphasize longer words periodically
                    words[i] = words[i].upper()  # TTS often emphasizes uppercase words
            text = " ".join(words)
            
            # Add occasional exclamations for excited emotion
            if not text.endswith("!") and not text.endswith("."):
                text += "!"
                
        elif self.current_emotion == "sad":
            # Add pauses for sad emotion
            sentences = text.split('.')
            processed_sentences = []
            for sentence in sentences:
                if sentence.strip():
                    processed_sentences.append(sentence.strip() + "...")  # Add pause
            text = ". ".join(processed_sentences)
            
        elif self.current_emotion == "happy":
            # Make happy text more expressive
            if not text.endswith("!") and random.random() > 0.7:
                text = text.rstrip('.') + "!"
                
        # Add SSML-like markers for breath between sentences (works with some TTS systems)
        text = text.replace(". ", ". <break time='300ms'> ")
        
        return text
    
    def _apply_audio_effects(self, audio_path, emotion_params):
        """Apply audio effects based on emotion parameters"""
        try:
            # Load audio
            sound = AudioSegment.from_file(audio_path)
            
            # Apply pitch shift if needed (some emotions have higher/lower pitch)
            if emotion_params["pitch"] != 1.0:
                # For pydub, we can adjust pitch by changing the frame rate
                # This is a simple approach - a more sophisticated one would use librosa
                new_frame_rate = int(sound.frame_rate * emotion_params["pitch"])
                sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_frame_rate})
                sound = sound.set_frame_rate(44100)  # Restore standard frame rate
            
            # Apply energy/volume adjustment
            if emotion_params["energy"] != 1.0:
                gain_db = 20 * np.log10(emotion_params["energy"])  # Convert to dB
                sound = sound + gain_db
            
            # Add slight reverb for some emotions (sad, calm)
            if self.current_emotion in ["sad", "calm"]:
                # Simple reverb effect by mixing with delayed version
                reverb = sound.fade_out(300)
                reverb = AudioSegment.silent(duration=100) + reverb
                sound = sound.overlay(reverb, gain_matrix=[(0.2, 0)])
            
            # Export modified audio
            sound.export(audio_path, format="wav")
            
            return True
        except Exception as e:
            print(f"Audio effects error (non-critical): {e}")
            return False
    
    def _play_audio(self, file_path):
        """Play audio file"""
        if not file_path or not os.path.exists(file_path):
            return
        
        try:
            # Try using pydub
            sound = AudioSegment.from_file(file_path)
            play(sound)
        except Exception as e:
            print(f"Pydub playback error: {e}")
            try:
                # Fallback to system commands
                if os.name == 'nt':  # Windows
                    os.system(f'start {file_path}')
                elif os.name == 'posix':  # macOS or Linux
                    if os.path.exists('/usr/bin/afplay'):  # macOS
                        os.system(f'afplay {file_path}')
                    else:  # Linux
                        os.system(f'mpg123 {file_path}')
                # Wait approximately for the audio length
                time.sleep(len(sound) / 1000.0 if 'sound' in locals() else 3.0)
            except Exception as e2:
                print(f"Audio playback fallback error: {e2}")
    
    def _main_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Step 1: Listen for speech
                audio = self._listen_for_speech()
                
                if audio is not None:
                    # Step 2: Process audio to text
                    text = self._process_audio(audio)
                    
                    if text:
                        # Step 3: Process text with LLM
                        response = self._process_text(text)
                        
                        # Step 4: Speak the response
                        self._speak_response(response)
                
                # Short pause before listening again
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Main loop error: {e}")
                time.sleep(1)
    
    def calibrate_microphone(self):
        """Calibrate microphone for speech detection"""
        print("Calibrating microphone (2 seconds of background noise)...")
        try:
            audio_chunks = []
            
            def callback(indata, frames, time_info, status):
                if status:
                    print(f"Calibration status: {status}")
                audio_chunks.append(np.squeeze(indata).astype(np.float32))
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=callback,
                blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
            ):
                # Record 2 seconds of background noise
                time.sleep(1.0)
            
            # Calculate background noise level
            if audio_chunks:
                all_audio = np.concatenate(audio_chunks)
                noise_level = np.sqrt(np.mean(all_audio**2))
                # Set threshold to 2.5x the background noise
                self.vad_threshold = max(0.01, noise_level * 2.5)
                print(f"Calibration complete. Background noise level: {noise_level:.5f}")
                print(f"Voice activity detection threshold set to: {self.vad_threshold:.5f}")
            else:
                print("Calibration failed - no audio received")
                
        except Exception as e:
            print(f"Calibration error: {e}")
            self.vad_threshold = 0.03  # Use default
            print(f"Using default threshold: {self.vad_threshold}")
    
    def start(self):
        """Start the system"""
        if self.running:
            print("System already running")
            return
        
        self.running = True
        
        # Calibrate microphone
        self.calibrate_microphone()
        
        # Start main loop in a separate thread
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        
        print("System started! Speak into the microphone. Press Ctrl+C to stop.")
    
    def stop(self):
        """Stop the system"""
        print("Stopping system...")
        self.running = False
        
        # Wait for thread to finish
        if hasattr(self, 'main_thread'):
            self.main_thread.join(timeout=1.0)
        
        # Clean up temp directory
        try:
            for file in os.listdir(self.temp_dir):
                os.unlink(os.path.join(self.temp_dir, file))
            os.rmdir(self.temp_dir)
        except:
            pass
        
        print("System stopped")

# Import random for emotion effects
import random

# Main execution
if __name__ == "__main__":
    print("=== Enhanced Emotional Speech Processing System ===")
    
    print("\nAvailable TTS quality options:")
    for quality, model in TTS_MODELS.items():
        print(f"  - {quality}: {model}")
    
    print("\nAvailable Whisper model options:")
    for size, info in WHISPER_MODELS.items():
        print(f"  - {size}: {info['description']}")
    
   
    # Allow command line overrides
        # Allow command line overrides
    if len(sys.argv) > 1:
        tts_quality = sys.argv[1] if sys.argv[1] in TTS_MODELS else tts_quality
        whisper_size = sys.argv[2] if sys.argv[2] in WHISPER_MODELS else whisper_size
        language = sys.argv[3] if len(sys.argv) > 3 else language

    # Initialize the speech system with selected parameters
    system = SequentialSpeechSystem(tts_quality=tts_quality, whisper_size=whisper_size, language=language)
    
    # Start the system
    system.start()

    try:
        # Run the system indefinitely (or until the user interrupts)
        while system.running:
            time.sleep(1)

    except KeyboardInterrupt:
        # Allow the user to gracefully stop the system with Ctrl+C
        system.stop()
        print("\nSystem stopped by user.")