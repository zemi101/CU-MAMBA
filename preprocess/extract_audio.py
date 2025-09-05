# preprocess/extract_audio.py

import os
import subprocess
import librosa
import soundfile as sf
from tqdm import tqdm

# === CONFIGURATION ===
VIDEO_DIR = r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\videos'
OUTPUT_DIR = r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\audio'
TARGET_SR = 16000
DURATION = 3.0  # seconds


os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_audio_segment(video_path, output_path):
    temp_audio = "temp_audio.wav"

    # Extract raw audio using ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(TARGET_SR), "-t", str(DURATION),
        temp_audio
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        audio, sr = librosa.load(temp_audio, sr=TARGET_SR)
        audio = pad_or_truncate(audio, sr)
        sf.write(output_path, audio, sr)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


def pad_or_truncate(audio, sr):
    expected_len = int(sr * DURATION)
    if len(audio) < expected_len:
        return librosa.util.fix_length(audio, expected_len)
    else:
        return audio[:expected_len]


def process_all_videos():
    for filename in tqdm(os.listdir(VIDEO_DIR)):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        video_path = os.path.join(VIDEO_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{video_id}.wav")

        if os.path.exists(output_path):
            continue

        extract_audio_segment(video_path, output_path)


if __name__ == "__main__":
    process_all_videos()
