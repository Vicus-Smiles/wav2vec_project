"""
Wav2Vec 2.0 Self-Supervised Learning Implementation
Group 6 - Machine Learning & Data Science Project
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import librosa
import soundfile as sf
import os
import sys
from datetime import datetime

class Wav2Vec2SpeechRecognition:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        print(f"🚀 Loading model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        print("✅ Model loaded successfully")
        
    def load_audio(self, audio_path, target_sr=16000):
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr)
            duration = len(audio) / sr
            print(f"📊 Audio loaded: {duration:.2f}s, {sr}Hz")
            return audio, sr
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return None, None
    
    def transcribe(self, audio, sr=16000):
        try:
            input_values = self.processor(
                audio, 
                sampling_rate=sr, 
                return_tensors="pt"
            ).input_values
            
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.decode(predicted_ids[0])
            return transcription
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def transcribe_file(self, audio_path):
        print(f"\n🎵 Processing: {os.path.basename(audio_path)}")
        audio, sr = self.load_audio(audio_path)
        if audio is not None:
            result = self.transcribe(audio, sr)
            print(f"📝 Transcription: {result}")
            return result
        return None
    
    def batch_transcribe(self, audio_folder):
        results = {}
        audio_files = [f for f in os.listdir(audio_folder) 
                      if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        for audio_file in audio_files:
            audio_path = os.path.join(audio_folder, audio_file)
            transcription = self.transcribe_file(audio_path)
            results[audio_file] = transcription
        
        return results

def create_sample_dataset():
    """Create sample audio files for demonstration"""
    try:
        from gtts import gTTS
        
        samples = [
            ("Hello world, this is wav2vec 2.0", "sample1.wav"),
            ("Self-supervised learning is powerful", "sample2.wav"),
            ("Speech recognition without labeled data", "sample3.wav"),
            ("Machine learning project by Group Six", "sample4.wav")
        ]
        
        created_files = []
        for text, filename in samples:
            if not os.path.exists(filename):
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(filename)
                print(f"✅ Created: {filename}")
                created_files.append(filename)
        
        return created_files
    except ImportError:
        print("⚠️  Install gTTS: pip install gTTS")
        return []

def demonstrate_ssl_concept():
    """Explain self-supervised learning principles"""
    print("\n" + "═"*70)
    print("SELF-SUPERVISED LEARNING: WAV2VEC 2.0 ARCHITECTURE")
    print("═"*70)
    
    stages = {
        "1. UNLABELED DATA COLLECTION": "1000s hours of raw audio",
        "2. CONTRASTIVE PRE-TRAINING": "Mask & predict task",
        "3. QUANTIZATION": "Continuous → discrete representations",
        "4. TRANSFORMER ENCODER": "Contextual understanding",
        "5. FINE-TUNING (Optional)": "Add CTC head for ASR"
    }
    
    for stage, description in stages.items():
        print(f"\n{stage}")
        print(f"   ↪ {description}")
    
    print("\n" + "═"*70)

def check_environment():
    """Verify all required packages are installed"""
    required_packages = [
        ("torch", lambda: torch.__version__),
        ("transformers", lambda: __import__('transformers').__version__),
        ("librosa", lambda: librosa.__version__),
        ("soundfile", lambda: sf.__version__)
    ]
    
    print("🔍 Environment Check")
    print("-"*40)
    
    for package, version_func in required_packages:
        try:
            version = version_func()
            print(f"✅ {package}: {version}")
        except:
            print(f"❌ {package}: NOT INSTALLED")
    
    print("-"*40)

def save_results(transcriptions, filename="results.txt"):
    """Save transcription results to file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("WAV2VEC 2.0 TRANSCRIPTION RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for audio_file, transcription in transcriptions.items():
            f.write(f"Audio: {audio_file}\n")
            f.write(f"Transcription: {transcription}\n")
            f.write("-"*40 + "\n")
    
    print(f"💾 Results saved to: {filename}")

def main():
    print("\n" + "✨"*25)
    print("WAV2VEC 2.0 - SELF-SUPERVISED SPEECH RECOGNITION")
    print("Group 6 | Machine Learning & Data Science")
    print("✨"*25)
    
    # Environment check
    check_environment()
    
    # Explain SSL concept
    demonstrate_ssl_concept()
    
    # Initialize model
    print("\n🔄 Initializing wav2vec 2.0 model...")
    recognizer = Wav2Vec2SpeechRecognition()
    
    # Create sample dataset if needed
    current_dir = os.getcwd()
    audio_files = [f for f in os.listdir(current_dir) 
                  if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
    
    if not audio_files:
        print("\n📁 No audio files found. Creating sample dataset...")
        audio_files = create_sample_dataset()
    
    # Process audio files
    print(f"\n🎯 Found {len(audio_files)} audio file(s) for processing")
    print("-"*40)
    
    results = {}
    for audio_file in audio_files:
        result = recognizer.transcribe_file(audio_file)
        if result:
            results[audio_file] = result
    
    # Save results
    if results:
        save_results(results)
    
    # Performance summary
    print("\n" + "📊"*20)
    print("PERFORMANCE SUMMARY")
    print("📊"*20)
    print(f"Total files processed: {len(results)}")
    print(f"Successful transcriptions: {len(results)}")
    print("\n✅ Implementation demonstrates:")
    print("   • Self-supervised learning concept")
    print("   • Pre-trained wav2vec 2.0 usage")
    print("   • Speech recognition pipeline")
    print("   • Practical application with audio")
    
    print("\n" + "🎉"*25)
    print("PROJECT IMPLEMENTATION COMPLETE")
    print("🎉"*25)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)