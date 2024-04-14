import wandb
from datasets import Dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import pandas as pd
import sys
import tarfile
import zipfile
import os

# Initialize wandb
wandb.init(project="train-mamba-2", entity="algaeai")

# Ensure there is a command-line argument provided
if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_archive_file>")
    sys.exit(1)

archive_file_path = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf")

# Function to extract files and read all text and CSV files
def load_data_from_archive(archive_path):
    text_data = []
    if archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall("temp_data")
    elif archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall("temp_data")
    else:
        raise ValueError("Unsupported file format")
        
    for root, dirs, files in os.walk("temp_data"):
        for file in files:
            if file.endswith(".txt") or file.endswith(".csv"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file.endswith(".txt"):
                        text_data.extend(f.readlines())
                    elif file.endswith(".csv"):
                        df = pd.read_csv(file_path)
                        text_data.extend(df.iloc[:, 0].astype(str).tolist())  # Assuming the first column contains the text

    return {"text": [line.strip() for line in text_data if line.strip()]}

# Load and preprocess the data
data = load_data_from_archive(archive_file_path)
dataframe = pd.DataFrame(data)
dataset = Dataset.from_pandas(dataframe)

training_args = TrainingArguments(
    output_dir="./results-Apr2nd2024-200e",
    num_train_epochs=200,
    per_device_train_batch_size=4,
    logging_dir='./logs-Apr2nd2024b-200e',
    logging_steps=10,
    learning_rate=2e-3,
    report_to="wandb"  # Integrate Weights & Biases for tracking
)

lora_config = LoraConfig(
    r=8,
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="text",  # Ensure this matches your data's text field
)

trainer.train()

# Clean up the extracted files
os.system("rm -rf temp_data")

# Finish the wandb run
wandb.finish()
