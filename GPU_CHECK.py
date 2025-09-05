import torch
import os
import multiprocessing

# ✅ GPU Info
if torch.cuda.is_available():
    print("🟢 CUDA Available")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Memory: {torch.cuda.get_device_properties(i).total_memory // 1024**2} MB")
else:
    print("🔴 CUDA not available")

# ✅ CPU Info
print(f"\n🧠 Logical CPU cores: {os.cpu_count()}")
print(f"🧠 Physical CPU cores: {multiprocessing.cpu_count()}")
