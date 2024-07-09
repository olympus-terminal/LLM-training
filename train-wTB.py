import sys
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_from_disk
import wandb
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#  existing code for device, tokenizer, model configuration, and dataset loading remains unchanged]
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Specify the tokenizer
tokenizer = AutoTokenizer.from_pretrained("hmbyt5/byt5-small-english", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Test the tokenizer
test_string = "Hello, world! This is a test."
encoded = tokenizer.encode(test_string)
decoded = tokenizer.decode(encoded)
print(f"Original: {test_string}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")

# Create a new model configuration
config = AutoConfig.from_pretrained(
    "gpt2",  # Using GPT-2 as a base for causal LM
    vocab_size=len(tokenizer),
    n_positions=1024,  # Adjust as needed
    n_ctx=1024,  # Adjust as needed
    n_embd=768,  # Adjust as needed
    n_layer=12,  # Adjust as needed
    n_head=12,  # Adjust as needed
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Initialize a new model from scratch
model = AutoModelForCausalLM.from_config(config).to(device)

# Ensure the model's embedding size matches the tokenizer's vocabulary size
model.resize_token_embeddings(len(tokenizer))

# Load the tokenized dataset
tokenized_dataset = load_from_disk("/scratch/drn2/PROJECTS/AI/Byte5/byte5-tokenized_dataset")

# Split the training data into train and validation sets
train_val_split = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Create the data collator with the tokenizer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize TensorBoard writer
writer = SummaryWriter('runs/byt5_causal_lm_experiment')

# Training Arguments with Optimizations, Decay, and Evaluation
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,
    fp16=True,
    gradient_checkpointing=True,
    save_steps=10000,
    save_total_limit=100,
    prediction_loss_only=True,
    report_to="wandb",
    weight_decay=0.01,
    learning_rate=1e-4,
    warmup_steps=500,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Initialize wandb
wandb.init(project="ByT5-Causal-LM-Training", entity="algaeai", config=training_args)

# Custom optimizer
#optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

# Custom learning rate scheduler
num_training_steps = (len(train_dataset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps
)

# Modify the Trainer to include TensorBoard logging
class TensorBoardTrainer(Trainer):
    def __init__(self, *args, writer=None, **kwargs):
        self.writer = writer
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self.writer:
            self.writer.add_scalar('Loss/train', loss.item(), self.state.global_step)
        return loss

    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        if self.writer and 'eval_loss' in output.metrics:
            self.writer.add_scalar('Loss/eval', output.metrics['eval_loss'], self.state.global_step)
        return output

    def log(self, logs):
        super().log(logs)
        if self.writer:
            for key, value in logs.items():
                self.writer.add_scalar(key, value, self.state.global_step)

# Create the trainer with custom optimizer, scheduler, and TensorBoard logging
trainer = TensorBoardTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, scheduler),
    writer=writer  # This will now be handled correctly
)
# Start training
trainer.train()

# Log histograms of model parameters after training
for name, param in model.named_parameters():
    writer.add_histogram(name, param, 0)

# Save the trained model
trainer.save_model("./pretrained_byt5_causal_lm")

# Finish wandb logging
wandb.finish()

# Close TensorBoard writer
writer.close()

print("Training completed. Model saved to './pretrained_byt5_causal_lm' directory.")
