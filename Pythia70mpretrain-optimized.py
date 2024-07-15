import sys
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_from_disk
import wandb
import torch
from transformers import AdamW, get_cosine_schedule_with_warmup

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



from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

# Initialize the configuration
#config = GPTNeoXConfig.from_pretrained("EleutherAI/pythia-70m")

# Create the model with random weights
#model = GPTNeoXForCausalLM(config)


# Create a new model configuration
config = GPTNeoXConfig.from_pretrained("EleutherAI/pythia-70m",  # Using GPT-2 as a base for causal LM
    vocab_size=len(tokenizer),
    n_positions=1024,  # Adjust as needed
    n_ctx=1024,  # Adjust as needed
    n_embd=768,  # Adjust as needed
    n_layer=12,  # Adjust as needed
    n_head=12,  # Adjust as needed
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)


model = GPTNeoXForCausalLM(config)

# Initialize a new model from scratch
#model = AutoModelForCausalLM.from_config(config).to(device)

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

# Training Arguments with Optimizations, Decay, and Evaluation
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Increased number of epochs
    per_device_train_batch_size=16,  # Adjusted batch size
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,
    fp16=True,  # Enable mixed precision training
    gradient_checkpointing=True,
    save_steps=5000,
    save_total_limit=3,
    prediction_loss_only=True,
    report_to="wandb",
    weight_decay=0.01,
    learning_rate=1e-4,  # Adjusted learning rate
    warmup_steps=2000,
    lr_scheduler_type="cosine",
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# Initialize wandb
wandb.init(project="ByT5-Causal-LM-Training", entity="algaeai", config=training_args)

# Custom optimizer
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

# Custom learning rate scheduler
num_training_steps = (len(train_dataset) // training_args.per_device_train_batch_size) * training_args.num_train_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps
)

# Create the trainer with custom optimizer and scheduler
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, scheduler)
)

# Start training
trainer.train()

# Save the trained model
trainer.save_model("./pretrained_byt5_causal_lm")

# Finish wandb logging
wandb.finish()

print("Training completed. Model saved to './pretrained_byt5_causal_lm' directory.")
