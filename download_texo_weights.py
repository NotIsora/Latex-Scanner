from huggingface_hub import snapshot_download
import os

repo_id = "alephpi/FormulaNet"
local_dir = "weight"

print(f"Downloading model from {repo_id} to {local_dir} (excluding checkpoints)...")
# Optimize download by excluding large training checkpoints
snapshot_download(repo_id=repo_id, local_dir=local_dir, ignore_patterns=["checkpoints/**"])
print("Download complete.")
