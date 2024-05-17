import os
import shutil
import sys

# Check if enough arguments are provided
if len(sys.argv) < 3:
    print("Usage: python organize_files.py <species_division_file> <directory>")
    sys.exit(1)

# Get arguments from command line
file_path = sys.argv[1]
directory = sys.argv[2]

# Dictionary to store species and their corresponding divisions
species_divisions = {}

# Read the file and populate the dictionary
with open(file_path, "r") as f:
    next(f)  # Skip the header line
    for line in f:
        species, division = line.strip().split("\t")
        species_divisions[species] = division

# Create directories for each division inside the specified directory
for division in set(species_divisions.values()):
    os.makedirs(os.path.join(directory, division), exist_ok=True)

# Copy files to their respective directories
for species, division in species_divisions.items():
    for filename in os.listdir(directory):
        if filename.startswith(species):
            shutil.copy(filename, os.path.join(directory, division, filename))
