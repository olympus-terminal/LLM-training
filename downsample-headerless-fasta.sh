#!/bin/bash

# Check if correct number of arguments was passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_file> <output_file> <original_size_gb> <desired_size_gb>"
    exit 1
fi

# Assign arguments to variables
input_file="$1"
output_file="$2"
original_size_gb="$3"
desired_size_gb="$4"

# Calculate sampling rate
sampling_rate=$(echo "$desired_size_gb / $original_size_gb" | bc -l)

# Use awk to select lines randomly based on the sampling rate
awk -v rate="$sampling_rate" 'BEGIN { srand() } rand() < rate' "$input_file" > "$output_file"

echo "Downsampling complete. Output saved to $output_file"
