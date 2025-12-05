from huggingface_hub import HfApi
import os

# Initialize HF API using your token (stored in environment variable)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload the entire deployment folder to HuggingFace Space
api.upload_folder(
    folder_path="Tourism_Package_Prediction/deployment",   # Local folder to upload
    repo_id="Amitgupta2982/Tourism-Package-Prediction",    # Your HuggingFace Space ID
    repo_type="space",                                      # Must be 'space' for deployment
    path_in_repo=""                                         # Upload to root of the Space
)

print("Hosting upload completed successfully!")
