import os
import sys

if len(sys.argv) != 2:
    print("Usage: python script.py <directory_path>")
    sys.exit(1)

directory = sys.argv[1]

if not os.path.isdir(directory):
    print("Error: Directory does not exist.")
    sys.exit(1)

# OUTPUT PATH
output_bleu_file = os.path.join(directory, "a_bleu_results.txt")
output_comet_file = os.path.join(directory, "a_comet_results.txt")
output_cometkiwi_file = os.path.join(directory, "a_cometkiwi_results.txt")

# BLEU
with open(output_bleu_file, "w", encoding="utf-8") as bleu_output:
    for filename in os.listdir(directory):
        if filename.endswith(".bleu"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    lines = file.readlines()
                    for line in lines:
                        if "version:2.4.1 =" in line:
                            parts = line.split("version:2.4.1 = ")
                            filename_parts = filename.split('.')
                            filename_without_ext = filename_parts[0].strip()
                            score = parts[1].split()[0].strip()
                            bleu_output.write(f"{filename_without_ext}: {score}\n")
                            break  
                except:
                    pass

# COMET
with open(output_comet_file, "w", encoding="utf-8") as comet_output:
    for filename in os.listdir(directory):
        if filename.endswith(".comet"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    lines = file.readlines()
                    filename_parts = filename.split('.')
                    filename_without_ext = '.'.join(filename_parts[:-1])
                    last_line_parts = lines[-1].split()
                    score = last_line_parts[-1]
                    comet_output.write(f"{filename_without_ext}: {score}\n")
                except:
                    pass

# COMETKIWI
with open(output_cometkiwi_file, "w", encoding="utf-8") as cometkiwi_output:
    for filename in os.listdir(directory):
        if filename.endswith(".cometkiwi"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    lines = file.readlines()
                    filename_parts = filename.split('.')
                    filename_without_ext = '.'.join(filename_parts[:-1])
                    last_line_parts = lines[-1].split()
                    score = last_line_parts[-1]
                    cometkiwi_output.write(f"{filename_without_ext}: {score}\n")
                except:
                    pass
