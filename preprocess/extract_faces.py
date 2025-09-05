# extract_faces_fast.py

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import face_alignment

# === CONFIGURATION ===
VIDEO_DIR = r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\videos'
OUTPUT_DIR = r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\faces'
FRAME_COUNT = 60
IMG_SIZE = 224

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

# === Extract Face Frames from One Video ===
def extract_faces_from_video(video_path, output_path):
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // FRAME_COUNT)

        frames = []
        count = 0
        while cap.isOpened() and len(frames) < FRAME_COUNT:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preds = fa.get_landmarks(frame_rgb)
                if preds:
                    landmarks = preds[0]
                    x_min = max(0, int(np.min(landmarks[:, 0])) - 40)
                    x_max = int(np.max(landmarks[:, 0])) + 40
                    y_min = max(0, int(np.min(landmarks[:, 1])) - 40)
                    y_max = int(np.max(landmarks[:, 1])) + 40

                    face_crop = frame_rgb[y_min:y_max, x_min:x_max]
                    if face_crop.size > 0:
                        face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                        frames.append(face_crop)
            count += 1
        cap.release()

        if frames:
            os.makedirs(output_path, exist_ok=True)
            for i, frame in enumerate(frames):
                save_path = os.path.join(output_path, f"{i:03d}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"⚠️ Error processing {video_path}: {e}")

# === Process All Videos Sequentially ===
def process_all_videos():
    video_files = [
        f for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(('.mp4', '.avi', '.mov'))
    ]

    print(f"📦 Total videos to process: {len(video_files)}")

    for filename in tqdm(video_files):
        video_path = os.path.join(VIDEO_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, video_id)
        if os.path.exists(output_path) and len(os.listdir(output_path)) >= FRAME_COUNT:
            continue
        extract_faces_from_video(video_path, output_path)

if __name__ == "__main__":
    torch.set_num_threads(1)  # 🧠 Avoid CPU thread spikes
    process_all_videos()
