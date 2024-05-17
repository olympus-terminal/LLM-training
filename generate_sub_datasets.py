import os
import sys

# Check if a directory path is provided as an argument
if len(sys.argv) < 2:
    print("Usage: python concatenate_files.py <directory>")
    sys.exit(1)

# Get the directory path from the command line argument
directory = sys.argv[1]

# Iterate through each directory
for dir_name in os.listdir(directory):
    dir_path = os.path.join(directory, dir_name)
    if os.path.isdir(dir_path):

        # Create a master file for the directory
        master_file_path = os.path.join(directory, f"{dir_name}_master.txt")
        with open(master_file_path, "w") as master_file:

            # Iterate through each file in the directory
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    with open(file_path, "r") as f:
                        # Read lines, filtering out those containing '>' and removing newlines
                        for line in f:
                            if ">" not in line:
                                master_file.write(line.strip())
