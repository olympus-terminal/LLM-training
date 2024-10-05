#usage python script.py $file

import sys
import re

def format_amino_acid_sequence(input_file, output_file):
    # Step 1: Read the file and remove all newlines
    with open(input_file, 'r') as file:
        content = file.read().replace('\n', '')

    # Step 2: Remove non-amino acid characters
    # Amino acids are typically represented by 20 standard letters: ACDEFGHIKLMNPQRSTVWY
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    content = ''.join(char for char in content.upper() if char in amino_acids)

    # Step 3: Break the sequence into chunks of 100 characters
    chunks = [content[i:i+100] for i in range(0, len(content), 100)]

    # Step 4: Write the result to a new file
    with open(output_file, 'w') as file:
        for chunk in chunks:
            file.write(chunk + '\n')

    print(f"Processed sequence written to {output_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        format_amino_acid_sequence(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied when trying to read '{input_file}' or write to '{output_file}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
