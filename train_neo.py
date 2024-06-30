import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_from_disk
import wandb
import torch
from transformers import get_linear_schedule_with_warmup

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Specify the model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained("lhy/character-level-tokenizer")
model = AutoModelForCausalLM.from_pretrained(model_name, is_decoder=True, torch_dtype=torch.float32).to(device)

# Load the tokenized dataset
tokenized_dataset = load_from_disk("/scratch/drn2/PROJECTS/AI/tokenized_dataset-duality-K")

# Split the training data into train and validation sets
train_val_split = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Create the data collator with the tokenizer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training Arguments with Optimizations, Decay, and Evaluation
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=16,
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    save_steps=10_000,
    save_total_limit=50,
    prediction_loss_only=True,
    report_to="wandb",
    weight_decay=0.01,
    learning_rate=2e-5,
    warmup_steps=500,
    lr_scheduler_type="linear",
    evaluation_strategy="steps",
    eval_steps=5000,
    logging_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Initialize wandb
wandb.init(project="Pretrain-Mamba130m-decay-experiment", entity="algaeai", config=training_args)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Start training
trainer.train()

# Save the trained model
trainer.save_model("./pretrained_model-gptneo125-singlechar")

# Finish wandb logging
wandb.finish()

print("Training completed. Model saved to './pretrained_model' directory.")
