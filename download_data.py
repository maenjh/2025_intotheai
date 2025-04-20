import kagglehub
import os

# Ensure the target directory exists
target_path = "c:/Users/jjmjh/Desktop/2025_intotheai/dataset"
os.makedirs(target_path, exist_ok=True)

# Download latest version to the specified folder
path = kagglehub.dataset_download(
    "albertobircoci/ai-generated-dogs-jpg-vs-real-dogs-jpg",
    path=target_path
)

print("Path to dataset files:", path)