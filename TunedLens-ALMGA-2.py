import torch
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoConfig, GPTNeoXForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm

# Set up argument parser
parser = argparse.ArgumentParser(description="Analyze amino acid sequences using a GPTNeoX model.")
parser.add_argument('file_paths', nargs='+', help='Paths to files containing amino acid sequences (one per line)')
args = parser.parse_args()

# Define the HuggingFace model name
model_name = "ChlorophyllChampion/duality100s-ckpt-30000_pythia70m-arc"

# Load the configuration
config = AutoConfig.from_pretrained(model_name)

# Load the ByteT5 tokenizer
tokenizer = AutoTokenizer.from_pretrained("hmbyt5/byt5-small-english")

# Explicitly download the safetensors file
safetensors_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")

# Manually create the model
model = GPTNeoXForCausalLM(config)

# Load the state dict using safetensors
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict, strict=False)

# Move model to GPU if available and use half precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).half()

def cleanup_model(model):
    try:
        if hasattr(model, 'base_model_prefix') and len(model.base_model_prefix) > 0:
            bm = getattr(model, model.base_model_prefix)
            del bm
    except:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()

def read_sequences_from_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def aa_to_input_ids(sequence):
    toks = tokenizer.encode(sequence, add_special_tokens=False)
    return torch.as_tensor(toks).view(1, -1).to(model.device)

def analyze_sequence(model, tokenizer, input_ids, num_tokens=4):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        
    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states)
    sequence_length = input_ids.size(1)
    vocab_size = model.config.vocab_size
    
    probabilities = torch.zeros(num_layers, num_tokens, vocab_size, device=model.device)
    
    for layer in range(num_layers):
        layer_logits = model.gpt_neox.final_layer_norm(hidden_states[layer])
        layer_logits = model.embed_out(layer_logits)
        layer_probs = torch.softmax(layer_logits, dim=-1)
        probabilities[layer, :, :] = layer_probs[0, sequence_length-num_tokens:sequence_length, :]
    
    return probabilities.detach().cpu().numpy()

def plot_summary_heatmap(avg_probs_list, file_name):
    avg_probs_array = np.array(avg_probs_list)
    overall_avg_probs = np.mean(avg_probs_array, axis=0)
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(overall_avg_probs, cmap="YlOrRd")
    plt.title(f"Average Token Probabilities - {file_name}")
    plt.xlabel("Token")
    plt.ylabel("Layer")
    plt.savefig(f"heatmap_summary_{file_name}.svg", format="svg")
    plt.close()

def plot_summary_token_probabilities(probs_list, token_ids_list, file_name):
    num_layers, num_tokens = probs_list[0].shape[:2]
    avg_probs = np.mean([probs[:, range(num_tokens), tokens] for probs, tokens in zip(probs_list, token_ids_list)], axis=0)
    
    plt.figure(figsize=(15, 10))
    for i in range(num_tokens):
        plt.plot(avg_probs[:, i], label=f'Token position: {i+1}')
    
    plt.xlabel('Layer')
    plt.ylabel('Average Probability')
    plt.title(f"Average Probabilities of next {num_tokens} tokens across layers - {file_name}")
    plt.legend()
    plt.savefig(f"token_probs_summary_{file_name}.svg", format="svg")
    plt.close()

# Main execution
for file_path in args.file_paths:
    sequences = read_sequences_from_file(file_path)
    file_name = file_path.split('/')[-1].split('.')[0]  # Extract file name without extension
    
    avg_probs_list = []
    probs_list = []
    token_ids_list = []
    
    for sequence in tqdm(sequences, desc=f"Processing sequences in {file_name}"):
        input_ids = aa_to_input_ids(sequence)
        probabilities = analyze_sequence(model, tokenizer, input_ids)
        
        avg_probs = probabilities.mean(axis=1)  # Average over the 4 tokens
        avg_probs_list.append(avg_probs)
        
        next_tokens = input_ids[0, -4:].detach().cpu().numpy()
        token_ids_list.append(next_tokens)
        probs_list.append(probabilities)
    
    # Generate summary visualizations for the file
    plot_summary_heatmap(avg_probs_list, file_name)
    plot_summary_token_probabilities(probs_list, token_ids_list, file_name)
