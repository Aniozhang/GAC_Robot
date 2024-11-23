import openai
import pyaudio
import wave
import numpy as np

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from robot import RobotArm
from recording import Recording

robot_name = "WX250"  # Replace with the actual robot name in your configuration
robot = RobotArm(robot_name, port="COM3")  # Adjust port as needed


rec_1 = Recording.load("./recording_1.rec")
rec_2 = Recording.load("./recording_2.rec")


# Set up OpenAI API key
openai.api_key = ""

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5
OUTPUT_FILENAME = "output.wav"

def record_audio():
    """Record audio from the microphone and save it to a file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the audio to a file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def transcribe_audio(filename):
    """Transcribe audio to text using OpenAI's Whisper."""
    with open(filename, "rb") as audio_file:
        response = openai.Audio.transcribe("whisper-1", audio_file)
    return response["text"]

def identify_box(transcription, n_boxes):
    """Identify the box mentioned in the transcription and generate probabilities."""
    import re

    # Define mappings for numbers in words (e.g., "one" -> 1, "first" -> 1, etc.)
    number_words = {
        "one": 1, "first": 1, "two": 2, "second": 2, "three": 3, "third": 3,
        "four": 4, "fourth": 4, "five": 5, "fifth": 5,
        "six": 6, "sixth": 6, "seven": 7, "seventh": 7,
        "eight": 8, "eighth": 8, "nine": 9, "ninth": 9, "ten": 10, "tenth": 10,
    }

    probabilities = np.zeros(n_boxes)

    # Normalize transcription to lowercase
    transcription = transcription.lower()

    # Check for explicit box mentions
    for i in range(1, n_boxes + 1):
        if f"box {i}" in transcription:
            probabilities[i - 1] = 1.0

    # Match word-based numbers (e.g., "first", "second", "third")
    for word, number in number_words.items():
        if word in transcription and 1 <= number <= n_boxes:
            probabilities[number - 1] = 1.0

    # Fallback: If no clear match, default to uniform probabilities
    if probabilities.sum() == 0:
        probabilities[:] = 1 / n_boxes  # Default to uniform distribution
    else:
        probabilities /= probabilities.sum()

    return probabilities


def main():
    n_boxes = int(input("Enter the number of boxes: "))
    
    
    while True:
        record_audio()

        transcription = transcribe_audio(OUTPUT_FILENAME)
        print(f"Transcription: {transcription}")

        probabilities = identify_box(transcription, n_boxes)
        print("Probability Distribution:", probabilities)

        boxSelected = np.argmax(probabilities)

        if boxSelected == 0:
            robot.playBack(rec_1)
        if boxSelected == 1:
            robot.playBack(rec_2)

main()