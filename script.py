#!/usr/bin/env python3
import sys
import re

def convert_indent(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    converted_lines = []
    for line in lines:
        # Replace only leading 2-space indentation, not spaces inside the line
        leading_spaces = re.match(r'^( *)', line).group(1)
        if len(leading_spaces) % 2 != 0:
            print(f"Warning: Odd number of leading spaces in line: {line.strip()}")
        new_leading = leading_spaces.replace('  ', '    ')
        converted_line = new_leading + line[len(leading_spaces):]
        converted_lines.append(converted_line)

    with open(filename, 'w') as f:
        f.writelines(converted_lines)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./script.py file")
        sys.exit(1)

    convert_indent(sys.argv[1])

