# test.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import MultimodalDFDataset
from models.assemble_model import MultiModalDeepfakeModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import pandas as pd

# Dummy test split
test_ids = [f'real_{i}' for i in range(800, 1000)] + [f'fake_{i}' for i in range(800, 1000)]
test_labels = [0]*200 + [1]*200

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = MultiModalDeepfakeModel(embed_dim=256).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Dataset & loader
test_dataset = MultimodalDFDataset(test_ids, test_labels)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

# Metrics
all_preds = []
all_trues = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)

        output = model(**inputs)
        preds = torch.argmax(output, dim=1)

        all_preds += preds.cpu().tolist()
        all_trues += labels.cpu().tolist()

# Compute metrics
acc = accuracy_score(all_trues, all_preds)
prec = precision_score(all_trues, all_preds)
rec = recall_score(all_trues, all_preds)
f1 = f1_score(all_trues, all_preds)
cm = confusion_matrix(all_trues, all_preds)

print(f"\n🧪 Test Results:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"Confusion Matrix:\n{cm}")

# Optional: save predictions
df = pd.DataFrame({
    'video_id': test_ids,
    'true_label': test_labels,
    'predicted': all_preds
})
df.to_csv("predictions.csv", index=False)
print("📄 Predictions saved to predictions.csv")
