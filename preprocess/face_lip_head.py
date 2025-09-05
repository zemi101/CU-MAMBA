import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
import face_alignment

# === CONFIGURATION ===
VIDEO_DIR = r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\videos'
FACE_OUTPUT_DIR = r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\faces'
LIP_OUTPUT_DIR = r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\lips'
HEADPOSE_OUTPUT_DIR = r'D:\WORK\PycharmProjects\DYNAMAMBAU++\data\headpose'
FRAME_COUNT = 60
FACE_SIZE = 224
LIP_SIZE = 112

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

# 3D model points for head pose
MODEL_POINTS_68 = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -330.0, -65.0],
    [-225.0, 170.0, -135.0],
    [225.0, 170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0, -150.0, -125.0]
], dtype=np.float64)

def extract_faces_and_lips_and_pose(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // FRAME_COUNT)

    face_dir = os.path.join(FACE_OUTPUT_DIR, video_id)
    lip_dir = os.path.join(LIP_OUTPUT_DIR, video_id)
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(lip_dir, exist_ok=True)

    head_pose_data = []

    count = 0
    saved = 0

    while cap.isOpened() and saved < FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = fa.get_landmarks(frame_rgb)
            if preds:
                landmarks = preds[0]

                # === Face ===
                x_min = max(0, int(np.min(landmarks[:, 0])) - 40)
                x_max = int(np.max(landmarks[:, 0])) + 40
                y_min = max(0, int(np.min(landmarks[:, 1])) - 40)
                y_max = int(np.max(landmarks[:, 1])) + 40
                face_crop = frame_rgb[y_min:y_max, x_min:x_max]
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))
                    face_save_path = os.path.join(face_dir, f"{saved:03d}.jpg")
                    cv2.imwrite(face_save_path, cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))

                # === Lip ===
                lip_points = landmarks[48:68]
                lx_min = int(np.min(lip_points[:, 0])) - 10
                ly_min = int(np.min(lip_points[:, 1])) - 10
                lx_max = int(np.max(lip_points[:, 0])) + 10
                ly_max = int(np.max(lip_points[:, 1])) + 10
                lip_crop = frame_rgb[ly_min:ly_max, lx_min:lx_max]
                if lip_crop.size > 0:
                    lip_crop = cv2.resize(lip_crop, (LIP_SIZE, LIP_SIZE))
                    lip_save_path = os.path.join(lip_dir, f"{saved:03d}.jpg")
                    cv2.imwrite(lip_save_path, cv2.cvtColor(lip_crop, cv2.COLOR_RGB2BGR))

                # === Head Pose ===
                image_points = np.array([
                    landmarks[30], landmarks[8], landmarks[36],
                    landmarks[45], landmarks[48], landmarks[54]
                ], dtype=np.float64)

                height, width, _ = frame_rgb.shape
                focal_length = width
                center = (width / 2, height / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)

                dist_coeffs = np.zeros((4, 1))
                success, rotation_vector, _ = cv2.solvePnP(
                    MODEL_POINTS_68, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    rmat, _ = cv2.Rodrigues(rotation_vector)
                    proj_matrix = np.hstack((rmat, np.zeros((3, 1))))
                    euler_angles, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(proj_matrix)
                    pitch = float(euler_angles[0][0])
                    yaw = float(euler_angles[1][0])
                    roll = float(euler_angles[2][0])
                    head_pose_data.append([yaw, pitch, roll])
                else:
                    head_pose_data.append([0.0, 0.0, 0.0])

                saved += 1
        count += 1
    cap.release()

    # Save head pose
    os.makedirs(HEADPOSE_OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(HEADPOSE_OUTPUT_DIR, f"{video_id}.json"), "w") as f:
        json.dump(head_pose_data, f)

def process_all_videos():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"📦 Total videos to process: {len(video_files)}")

    for filename in tqdm(video_files):
        video_path = os.path.join(VIDEO_DIR, filename)
        video_id = os.path.splitext(filename)[0]
        if os.path.exists(os.path.join(FACE_OUTPUT_DIR, video_id)) and \
           len(os.listdir(os.path.join(FACE_OUTPUT_DIR, video_id))) >= FRAME_COUNT:
            continue
        extract_faces_and_lips_and_pose(video_path, video_id)

if __name__ == "__main__":
    torch.set_num_threads(1)
    process_all_videos()