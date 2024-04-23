import sys
import random

def split_file(input_file, output_file1, output_file2, percentage=50):
    try:
        # Read lines from the input file
        with open(input_file, 'r') as infile:
            lines = infile.readlines()

        # Calculate the number of lines for each part
        num_lines = len(lines)
        num_lines_part1 = int(num_lines * (percentage / 100))
        num_lines_part2 = num_lines - num_lines_part1

        # Randomly shuffle the lines
        random.shuffle(lines)

        # Split the lines into two parts
        part1_lines = lines[:num_lines_part1]
        part2_lines = lines[num_lines_part1:]

        # Write the two parts to output files
        with open(output_file1, 'w') as outfile1:
            outfile1.writelines(part1_lines)

        with open(output_file2, 'w') as outfile2:
            outfile2.writelines(part2_lines)

        print(f"File split into two parts: {num_lines_part1} lines in {output_file1} and {num_lines_part2} lines in {output_file2}.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <input_file> <output_file1> <output_file2> <percentage>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file1 = sys.argv[2]
    output_file2 = sys.argv[3]
    percentage = int(sys.argv[4])

    split_file(input_file, output_file1, output_file2, percentage)

