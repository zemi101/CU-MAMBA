# utils/dataset.py

import os
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import librosa

class MultimodalDFDataset(Dataset):
    def __init__(self, video_list, labels, base_dir='data', augment=False):
        self.video_list = video_list
        self.labels = labels
        self.base_dir = base_dir
        self.augment = augment

        self.face_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.lip_tf = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_id = self.video_list[idx]
        label = self.labels[idx]

        # --- Face stream ---
        face_dir = os.path.join(self.base_dir, 'faces', video_id)
        face_frames = self._load_frames(face_dir, self.face_tf, 60)

        # --- Audio stream ---
        audio_path = os.path.join(self.base_dir, 'audio', f"{video_id}.wav")
        mel = self._load_mel(audio_path)  # [1, 64, 96]

        # --- Lip sync ---
        lip_dir = os.path.join(self.base_dir, 'lips', video_id)
        lips = self._load_frames(lip_dir, self.lip_tf, 60)

        # --- Head pose ---
        hp_path = os.path.join(self.base_dir, 'headpose', f"{video_id}.json")
        headpose = self._load_headpose(hp_path)  # [60, 3]

        return {
            'face': face_frames,         # [60, 3, 224, 224]
            'audio': mel,                # [1, 64, 96]
            'lips': lips,                # [60, 3, 112, 112]
            'audio_lip': mel.clone(),    # reused audio for sync
            'headpose': headpose,        # [60, 3]
            'label': torch.tensor(label, dtype=torch.long)
        }

    def _load_frames(self, folder, tf, max_len):
        frames = sorted(os.listdir(folder))[:max_len]
        imgs = []
        for f in frames:
            path = os.path.join(folder, f)
            img = tf(Image.open(path).convert('RGB'))
            imgs.append(img)
        while len(imgs) < max_len:
            imgs.append(imgs[-1])  # pad
        return torch.stack(imgs)  # [T, C, H, W]

    def _load_headpose(self, json_path):
        with open(json_path, 'r') as f:
            hp = json.load(f)
        hp = torch.tensor(hp, dtype=torch.float32)
        if len(hp) < 60:
            pad = hp[-1].repeat(60 - len(hp), 1)
            hp = torch.cat([hp, pad], dim=0)
        return hp

    def _load_mel(self, path):
        y, sr = librosa.load(path, sr=16000)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=160, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db[:, :96]
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)
        return mel_tensor
