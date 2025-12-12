import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import librosa
import soundfile as sf
import os

class Wav2Vec2SpeechRecognition:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        print(f"Loading model: {model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        print("Model loaded successfully!\n")
        
    def load_audio(self, audio_path, target_sr=16000):
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio
    
    def transcribe(self, audio):
        input_values = self.processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_values
        
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        return transcription
    
    def transcribe_file(self, audio_path):
        print(f"Processing: {audio_path}")
        audio = self.load_audio(audio_path)
        result = self.transcribe(audio)
        print(f"Transcription: {result}\n")
        return result

def create_sample_audio(text="Machine learning is fascinating", filename="sample_audio.wav"):
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        print(f"Created sample audio: {filename}\n")
        return filename
    except ImportError:
        print("Warning: gTTS not installed. Using existing audio files only.\n")
        return None

def main():
    print("="*70)
    print("WAV2VEC 2.0 - SELF-SUPERVISED SPEECH RECOGNITION")
    print("Group 6 - Machine Learning and Data Science")
    print("="*70 + "\n")
    
    # Initialize model
    recognizer = Wav2Vec2SpeechRecognition()
    
    # Display self-supervised learning concept
    print("="*70)
    print("SELF-SUPERVISED LEARNING EXPLANATION")
    print("="*70)
    print("\nWav2Vec 2.0 Two-Stage Approach:")
    print("\n1. CONTRASTIVE LEARNING (Self-Supervised):")
    print("   - Input: Raw audio waveform")
    print("   - Process: Mask random portions of audio")
    print("   - Task: Predict masked representations")
    print("   - Key: NO labels required!")
    print("\n2. FINE-TUNING (Supervised - Optional):")
    print("   - Add CTC layer for classification")
    print("   - Train with small labeled dataset")
    print("   - Achieves high accuracy with minimal data")
    print("\nInnovation: Pre-training on unlabeled audio creates robust")
    print("representations that transfer to downstream tasks efficiently.")
    print("="*70 + "\n")
    
    # Check for existing audio files
    audio_files = [f for f in os.listdir('.') if f.endswith(('.wav', '.mp3', '.flac'))]
    
    if audio_files:
        print(f"Found {len(audio_files)} audio file(s) in directory:")
        for audio_file in audio_files:
            recognizer.transcribe_file(audio_file)
    else:
        print("No audio files found. Creating sample...")
        sample_file = create_sample_audio("Hello world, this is a demonstration of wav2vec")
        if sample_file and os.path.exists(sample_file):
            recognizer.transcribe_file(sample_file)
    
    print("="*70)
    print("IMPLEMENTATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()