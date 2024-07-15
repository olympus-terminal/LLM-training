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
from sklearn.decomposition import PCA

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

def analyze_sequence(model, tokenizer, input_ids):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    return [h.squeeze(0).cpu().numpy() for h in hidden_states]

def plot_sequence_representation(hidden_states, file_name):
    num_layers = len(hidden_states)
    sequence_length = hidden_states[0].shape[0]
    
    # Use PCA to reduce dimensionality for visualization
    pca = PCA(n_components=2)
    reduced_states = [pca.fit_transform(layer) for layer in hidden_states]
    
    plt.figure(figsize=(20, 5 * ((num_layers + 3) // 4)))
    for i, state in enumerate(reduced_states):
        plt.subplot(((num_layers + 3) // 4), 4, i+1)
        plt.scatter(state[:, 0], state[:, 1], c=range(sequence_length), cmap='viridis')
        plt.title(f'Layer {i}')
        plt.colorbar(label='Sequence Position')
    
    plt.tight_layout()
    plt.savefig(f"sequence_representation_{file_name}.svg", format="svg", bbox_inches='tight')
    plt.close()

def plot_layer_similarity(hidden_states, file_name):
    num_layers = len(hidden_states)
    similarity_matrix = np.zeros((num_layers, num_layers))
    
    for i in range(num_layers):
        for j in range(num_layers):
            similarity_matrix[i, j] = np.corrcoef(hidden_states[i].flatten(), hidden_states[j].flatten())[0, 1]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Layer Similarity - {file_name}')
    plt.xlabel('Layer')
    plt.ylabel('Layer')
    plt.savefig(f"layer_similarity_{file_name}.svg", format="svg", bbox_inches='tight')
    plt.close()

# Main execution
for file_path in args.file_paths:
    sequences = read_sequences_from_file(file_path)
    file_name = file_path.split('/')[-1].split('.')[0]  # Extract file name without extension
    
    for idx, sequence in enumerate(tqdm(sequences, desc=f"Processing sequences in {file_name}")):
        input_ids = aa_to_input_ids(sequence)
        hidden_states = analyze_sequence(model, tokenizer, input_ids)
        
        # Plot sequence representation for each layer
        plot_sequence_representation(hidden_states, f"{file_name}_seq_{idx}")
        
        # Plot layer similarity
        plot_layer_similarity(hidden_states, f"{file_name}_seq_{idx}")

# Cleanup
