import argparse
import re
from pprint import pformat



def process_content(content):
    # Example regex pattern (modify this based on your needs)
    # This pattern is just an example, it matches email addresses
    pattern = r'Performance metrics for section 2([^.\n]+(?:\n(?!\n).+)*)'
    # Find all matches and return them
    text = re.findall(pattern, content)[0]

    pattern = r'(\w+(?:_\w+)*)\s+((0x[0-9A-Fa-f]+)|(\d+\.\d+)|(\d+))'

    matches = re.findall(pattern, text)

    data = {}

    for match in matches:
        key = match[0]
        # Determine the type of value and convert accordingly
        if match[2]:  # Hexadecimal
            value = int(match[2], 16)
        elif match[3]:  # Floating-point
            value = float(match[3])
        else:  # Integer
            value = int(match[4])
        
        # Add to dictionary
        data[key] = value
    
    return data

def main(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Process the content with regex
    processed_content = process_content(content)

    # Write the processed content to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(pformat(processed_content))

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Process log files.')
    parser.add_argument('-i', '--input', required=True, help='Input file path')
    parser.add_argument('-o', '--output', required=True, help='Output file path')

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with the input and output file paths
    main(args.input, args.output)
