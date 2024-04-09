import os
import sys
import zipfile
import pandas as pd
import datetime
import wandb
from datasets import Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if len(sys.argv) < 2:
    print("Usage: python script.py <path_to_zip_file>")
    sys.exit(1)

zip_file_path = sys.argv[1]
base_name = os.path.splitext(os.path.basename(zip_file_path))[0]  # Extract the base name without extension

# Get the current system date and time for unique file naming
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Initialize wandb with a dynamic project name based on the zip file and current time
wandb.init(project=f"{base_name}-{current_time}-train-project", entity="algaeai")

rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
setup(rank, world_size)

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf")
model.cuda(rank)
model = DDP(model, device_ids=[rank])

def load_text_from_zip(zip_path):
    text_data = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("temp_data")
        for root, dirs, files in os.walk("temp_data"):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_data.extend(f.readlines())
    return {"text": [line.strip() for line in text_data if line.strip()]}

data = load_text_from_zip(zip_file_path)
dataframe = pd.DataFrame(data)
dataset = Dataset.from_pandas(dataframe)

sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler)

# Include the current time in the output and log directories
output_dir = f"{base_name}_{current_time}_results.output"
logging_dir = f"{base_name}_{current_time}_logs.log"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=40,
    per_device_train_batch_size=2,
    logging_dir=logging_dir,
    logging_steps=10,
    learning_rate=2e-3,
    report_to="wandb"
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
    train_dataset=dataset,  # Assume data_loader should be used here
)

if rank == 0:
    wandb.watch(model)

trainer.train()

cleanup()
