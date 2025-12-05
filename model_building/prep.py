
# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Authenticate HuggingFace
login(token=os.getenv("HF_TOKEN"))

api = HfApi()

# ---------------------------------------------------
#  Load Dataset from HuggingFace
# ---------------------------------------------------
DATASET_PATH = "hf://datasets/Amitgupta2982/Tourism-Package-Prediction/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully:", df.shape)

# ---------------------------------------------------
#  Data Cleaning Specific to Tourism Dataset
# ---------------------------------------------------

# Remove unnecessary identifier columns
cols_to_drop = ["Unnamed: 0", "CustomerID"]
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors="ignore")

# Deduplicate data
df = df.drop_duplicates()

# Standardize categorical columns
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()

print("Dataset cleaned:", df.shape)

# ---------------------------------------------------
#  Split Into Train and Test
# ---------------------------------------------------

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create folder to store split datasets
os.makedirs("Tourism_Package_Prediction/data_split", exist_ok=True)

train_path = "Tourism_Package_Prediction/data_split/train.csv"
test_path = "Tourism_Package_Prediction/data_split/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Train/Test created:")
print("Train:", train_df.shape)
print("Test :", test_df.shape)

# ---------------------------------------------------
# Upload Split Files Back to HuggingFace Dataset Repo
# ---------------------------------------------------

api.upload_folder(
    folder_path="Tourism_Package_Prediction/data_split",
    repo_id="Amitgupta2982/Tourism-Package-Prediction",
    repo_type="dataset",
)

print("Train/Test files uploaded successfully to HuggingFace!")
