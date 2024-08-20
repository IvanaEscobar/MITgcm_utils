###
# conda activate myenv
# copy this file into you code or build directory
# Run: 
# python check_ifdef_balance.py

import os
import re

def check_ifdef_endif_balance(file_path):
    with open(file_path, 'r') as file:
        stack = []
        for line_number, line in enumerate(file, start=1):
            # Match #ifdef, #ifndef, and #if (defined with up to three spaces after #
            if re.search(r'#\s{0,3}ifdef', line) or re.search(r'#\s{0,3}ifndef', line) or re.search(r'#\s{0,3}if\s*\(\s*defined', line):
                stack.append((line_number, line.strip()))
            # Match #endif with up to three spaces after #
            elif re.search(r'#\s{0,3}endif', line):
                if stack:
                    stack.pop()
                else:
                    print(f"Unmatched #endif at line {line_number} in {file_path}")

        if stack:
            for unmatched_line, directive in stack:
                print(f"Unmatched {directive} at line {unmatched_line} in {file_path}")

def check_all_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.F') or file.endswith('.F90'):
                check_ifdef_endif_balance(os.path.join(root, file))

if __name__ == "__main__":
    directory = "."  # specify your directory here
    check_all_files(directory)
