# preprocess/extract_headpose.py

import os
import cv2
import json
import numpy as np
import face_alignment
from tqdm import tqdm
import torch

# Configuration
FACES_DIR = r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\faces'
OUTPUT_DIR = r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\headpose'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# 3D model points (reference points on a generic face model)
MODEL_POINTS_68 = np.array([
    [0.0, 0.0, 0.0],          # Nose tip (30)
    [0.0, -330.0, -65.0],     # Chin (8)
    [-225.0, 170.0, -135.0],  # Left eye left corner (36)
    [225.0, 170.0, -135.0],   # Right eye right corner (45)
    [-150.0, -150.0, -125.0], # Left mouth corner (48)
    [150.0, -150.0, -125.0]   # Right mouth corner (54)
], dtype=np.float64)

def get_head_pose(frame_rgb):
    landmarks = fa.get_landmarks(frame_rgb)
    if not landmarks:
        return None
    landmarks = landmarks[0]

    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left mouth corner
        landmarks[54]   # Right mouth corner
    ], dtype=np.float64)

    height, width, _ = frame_rgb.shape
    focal_length = width
    center = (width / 2, height / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS_68, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rmat, np.zeros((3, 1))))
    euler_angles, _, _, _, _, _ = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = float(euler_angles[0])
    yaw = float(euler_angles[1])
    roll = float(euler_angles[2])

    return [yaw, pitch, roll]

def process_video_headpose(video_id):
    video_path = os.path.join(FACES_DIR, video_id)
    angles = []

    for i in range(60):
        frame_path = os.path.join(video_path, f"{i:03d}.jpg")
        if not os.path.exists(frame_path):
            continue
        frame = cv2.imread(frame_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose = get_head_pose(frame_rgb)
        if pose:
            angles.append(pose)
        else:
            angles.append([0.0, 0.0, 0.0])  # Default if estimation fails

    # Save head pose as JSON
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}.json")
    with open(output_path, "w") as f:
        json.dump(angles, f)

def process_all_videos():
    for video_id in tqdm(os.listdir(FACES_DIR)):
        if not os.path.isdir(os.path.join(FACES_DIR, video_id)):
            continue
        process_video_headpose(video_id)

if __name__ == "__main__":
    process_all_videos()
