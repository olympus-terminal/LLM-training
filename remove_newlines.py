import argparse

def remove_newlines(input_file, output_file):
    # Open the input file in read mode
    with open(input_file, 'r') as file:
        # Read the entire content of the file
        content = file.read()

    # Replace all newline characters with an empty string
    content_without_newlines = content.replace('\n', '')

    # Open the output file in write mode
    with open(output_file, 'w') as file:
        # Write the content without newlines to the output file
        file.write(content_without_newlines)

    print(f"Newlines have been removed from {input_file} and written to {output_file}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove newlines from a text file')
    parser.add_argument('input_file', type=str, help='Path to the input file')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    args = parser.parse_args()

    remove_newlines(args.input_file, args.output_file)
