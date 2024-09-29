import sys
import numpy as np
import librosa

def detect_pitches(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file)
    
    # Define the pitch range (adjust as needed)
    fmin = librosa.note_to_hz('C2')  # Minimum frequency (e.g., C2)
    fmax = librosa.note_to_hz('C7')  # Maximum frequency (e.g., C7)
    
    # Use librosa's pitch detection
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=fmin, fmax=fmax)
    
    # Extract the pitches from the result
    pitch_times = []
    pitch_values = []
    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            time = librosa.frames_to_time(i, sr=sr)
            pitch_times.append(time)
            pitch_values.append(pitch)
    
    # Display the detected pitches
    print("Detected pitches:")
    for time, pitch in zip(pitch_times, pitch_values):
        print(f"Time {time:.2f}s: Pitch {pitch:.2f} Hz")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pitch_detection.py <audio_file>")
        sys.exit(1)
    audio_file = sys.argv[1]
    detect_pitches(audio_file)
