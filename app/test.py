from huggingface_hub import snapshot_download, HfFileSystem
import os

MODEL_NAME = "tuman/vit-rugpt2-image-captioning"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_snapshot")

# Delete empty directory if it exists
if os.path.exists(LOCAL_MODEL_DIR) and not os.listdir(LOCAL_MODEL_DIR):
    os.rmdir(LOCAL_MODEL_DIR)

try:
    # Download with verbose logging
    path = snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,
        force_download=True,
        resume_download=False
    )
    print(f"Downloaded to: {path}")
except Exception as e:
    print(f"Download failed: {str(e)}")