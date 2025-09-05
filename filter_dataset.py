import os
import json
import cv2
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

# === PATHS ===
SOURCE_VIDEO_DIR = r"E:\SSD DATA\dfdc_train_all2\all_in_folders"
METADATA_FILE = r"E:\SSD DATA\dfdc_train_all2\all_in_folders\metadata_ALL.json"
OUTPUT_DIR = r"E:\SSD DATA\dfdc_train_all2\filtered_videos"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "metadata_filtered.json")

# === Create Output Directory ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_fps(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    except:
        return 0

# === Load Metadata ===
with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

# === Build REAL and FAKE references ===
real_videos = [vid for vid, info in metadata.items() if info['label'] == 'REAL']
real_to_fakes = defaultdict(list)
for vid, info in metadata.items():
    if info['label'] == 'FAKE':
        real_to_fakes[info['original']].append(vid)

# === Shuffle REALs randomly ===
random.shuffle(real_videos)

filtered_metadata = {}
real_count = 0
fake_count = 0

print("🔄 Selecting videos randomly and filtering by FPS (1 FAKE per REAL)...\n")

for real_vid in tqdm(real_videos, desc="Selecting"):
    if real_count >= 5000 or fake_count >= 5000:
        break

    real_path = os.path.join(SOURCE_VIDEO_DIR, real_vid)
    if not os.path.exists(real_path):
        continue

    real_fps = get_fps(real_path)
    if not (29.00 < real_fps < 31.00):
        continue

    if real_vid not in real_to_fakes:
        continue

    # Shuffle FAKEs to randomize selection
    fakes = real_to_fakes[real_vid]
    random.shuffle(fakes)

    selected_fake = None
    for fake_vid in fakes:
        fake_path = os.path.join(SOURCE_VIDEO_DIR, fake_vid)
        if not os.path.exists(fake_path):
            continue
        fake_fps = get_fps(fake_path)
        if 29.00 < fake_fps < 31.00:
            selected_fake = fake_vid
            break

    if not selected_fake:
        continue

    # ✅ Copy REAL
    shutil.copy(real_path, os.path.join(OUTPUT_DIR, real_vid))
    filtered_metadata[real_vid] = metadata[real_vid]
    real_count += 1

    # ✅ Copy FAKE
    shutil.copy(os.path.join(SOURCE_VIDEO_DIR, selected_fake), os.path.join(OUTPUT_DIR, selected_fake))
    filtered_metadata[selected_fake] = metadata[selected_fake]
    fake_count += 1

# === Save Metadata ===
with open(OUTPUT_JSON, 'w') as f_out:
    json.dump(filtered_metadata, f_out, indent=4)

# === Summary ===
print("\n✅ Done!")
print(f"📁 Saved in: {OUTPUT_DIR}")
print(f"🎬 REAL videos: {real_count}")
print(f"🎭 FAKE videos: {fake_count}")
print(f"📝 Metadata saved at: {OUTPUT_JSON}")
