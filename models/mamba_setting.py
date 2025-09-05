# models/mamba_settings.py

# Toggle this if you want to load from your local folder (recommended)
USE_LOCAL_PRETRAINED = True

# Folder that contains config.json + model.safetensors (or pytorch_model.bin)
LOCAL_MAMBA = r"D:\WORK\PycharmProjects\DYNAMAMBAU++\models\mamba-130m"

# Freeze Mamba weights (set True if you only want to fine-tune heads)
FREEZE_BACKBONE = False

# If you set USE_LOCAL_PRETRAINED = False, a small random-initialized Mamba will be used.
# You can optionally control its hidden size here (if None, it defaults to max(256, input_dim))
CUSTOM_HIDDEN_SIZE = None
