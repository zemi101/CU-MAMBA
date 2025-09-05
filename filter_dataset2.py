import os
import json
import cv2
import shutil
import random
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import face_alignment

# === PATHS ===
SOURCE_VIDEO_DIR = r"E:\SSD DATA\dfdc_train_all2\all_in_folders"
METADATA_FILE = os.path.join(SOURCE_VIDEO_DIR, "metadata_ALL.json")
OUTPUT_DIR = r"E:\SSD DATA\dfdc_train_all2\filtered_videos"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "metadata_filtered.json")

# === INIT ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🟢 CUDA Available: {torch.cuda.get_device_name(0) if device == 'cuda' else '❌ No GPU - Using CPU'}")

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)

# === Helper Functions ===
def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 0

def has_audio_stream(video_path):
    try:
        import subprocess
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_type", "-of", "json", video_path
        ], capture_output=True, text=True)
        return '"codec_type": "audio"' in result.stdout
    except:
        return False

def has_required_face_features_all_frames(video_path, required_frames=60):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < required_frames:
        cap.release()
        return False

    frames_checked = 0
    valid = True
    while frames_checked < required_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames_checked * total_frames / required_frames))
        ret, frame = cap.read()
        if not ret:
            valid = False
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = fa.get_landmarks(rgb_frame)
        if not landmarks:
            valid = False
            break
        lm = landmarks[0]
        if not (np.all(lm[36:48]) and np.all(lm[27:36]) and np.all(lm[48:68]) and np.all(lm[17:27])):
            valid = False
            break

        frames_checked += 1

    cap.release()
    return valid

# === Load Metadata ===
with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

real_videos = [vid for vid, info in metadata.items() if info['label'] == 'REAL']
real_to_fakes = defaultdict(list)
for vid, info in metadata.items():
    if info['label'] == 'FAKE':
        real_to_fakes[info['original']].append(vid)

random.shuffle(real_videos)
filtered_metadata = {}
real_count = 0
fake_count = 0

print("🔍 Filtering videos with valid FPS, 60-frame features, and audio...\n")

for real_vid in tqdm(real_videos, desc="Filtering"):
    if real_count >= 5000 or fake_count >= 5000:
        break

    real_path = os.path.join(SOURCE_VIDEO_DIR, real_vid)
    if not os.path.exists(real_path):
        continue

    if not (29.0 < get_fps(real_path) < 31.0):
        continue
    if not has_audio_stream(real_path):
        continue
    if not has_required_face_features_all_frames(real_path):
        continue
    if real_vid not in real_to_fakes:
        continue

    # Check and select a matching fake
    fakes = real_to_fakes[real_vid]
    random.shuffle(fakes)
    selected_fake = None

    for fake_vid in fakes:
        fake_path = os.path.join(SOURCE_VIDEO_DIR, fake_vid)
        if not os.path.exists(fake_path):
            continue
        if not (29.0 < get_fps(fake_path) < 31.0):
            continue
        if not has_audio_stream(fake_path):
            continue
        if not has_required_face_features_all_frames(fake_path):
            continue
        selected_fake = fake_vid
        break

    if not selected_fake:
        continue

    # ✅ Copy both videos
    shutil.copy(real_path, os.path.join(OUTPUT_DIR, real_vid))
    filtered_metadata[real_vid] = metadata[real_vid]
    real_count += 1

    shutil.copy(os.path.join(SOURCE_VIDEO_DIR, selected_fake), os.path.join(OUTPUT_DIR, selected_fake))
    filtered_metadata[selected_fake] = metadata[selected_fake]
    fake_count += 1

# === Save Metadata ===
with open(OUTPUT_JSON, 'w') as f_out:
    json.dump(filtered_metadata, f_out, indent=4)

# === Summary ===
print("\n✅ Done!")
print(f"📁 Output: {OUTPUT_DIR}")
print(f"🎬 REAL: {real_count} | 🎭 FAKE: {fake_count}")
print(f"📝 Metadata saved to: {OUTPUT_JSON}")
