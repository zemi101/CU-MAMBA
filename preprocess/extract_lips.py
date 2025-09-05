# preprocess/extract_lips.py

import os
import cv2
import numpy as np
import torch
import face_alignment
from tqdm import tqdm

# Paths
FACES_DIR =  r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\faces'
OUTPUT_DIR =  r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\lips'
IMG_SIZE = 112

# === GPU CHECK ===
if torch.cuda.is_available():
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🟢 CUDA is available. Using GPU: {gpu_name}")
else:
    device = 'cpu'
    print("🔴 CUDA not available. Using CPU.")

# === Initialize Face Aligner Once ===
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    flip_input=False,
    device=device
)

def extract_lips_from_frame(img):
    try:
        landmarks = fa.get_landmarks(img)
        if not landmarks:
            return None
        landmarks = landmarks[0]

        # Lip landmarks 48–67
        lip_points = landmarks[48:68]
        x_min = int(np.min(lip_points[:, 0])) - 10
        y_min = int(np.min(lip_points[:, 1])) - 10
        x_max = int(np.max(lip_points[:, 0])) + 10
        y_max = int(np.max(lip_points[:, 1])) + 10

        # Crop lip region
        lip_crop = img[y_min:y_max, x_min:x_max]
        if lip_crop.size == 0:
            return None

        lip_crop = cv2.resize(lip_crop, (IMG_SIZE, IMG_SIZE))
        return lip_crop

    except Exception as e:
        print(f"[Error] Lip extraction failed: {e}")
        return None

def process_video_lips(video_id):
    face_dir = os.path.join(FACES_DIR, video_id)
    output_dir = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(60):
        face_path = os.path.join(face_dir, f"{i:03d}.jpg")
        if not os.path.exists(face_path):
            continue
        frame = cv2.imread(face_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        lip = extract_lips_from_frame(frame_rgb)
        if lip is not None:
            save_path = os.path.join(output_dir, f"{i:03d}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(lip, cv2.COLOR_RGB2BGR))

def process_all_videos():
    for video_id in tqdm(os.listdir(FACES_DIR)):
        if not os.path.isdir(os.path.join(FACES_DIR, video_id)):
            continue
        process_video_lips(video_id)

if __name__ == "__main__":
    process_all_videos()
