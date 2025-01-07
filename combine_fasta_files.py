# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:13:02 2024

@author: lorenzo
"""

############# COMBINE FASTA FILES FAST #############

import os
import concurrent.futures



def process_file(file_path, seen_labels):
    combined_data = []
    with open(file_path, 'r') as infile:
        write_data = False
        for line in infile:
            if line.startswith(">"):
                label = line.strip()
                if label not in seen_labels:
                    seen_labels.add(label)
                    combined_data.append(line)
                    write_data = True
                else:
                    write_data = False
            elif write_data:
                combined_data.append(line)
    return combined_data

def combine_fasta_files(input_directory, output_file):
    seen_labels = set()
    combined_data = []

    # Use ThreadPoolExecutor to process files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for root, dirs, files in os.walk(input_directory):
            for file in files:
                if file.endswith(".fasta") or file.endswith(".fa"):
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(process_file, file_path, seen_labels))
        
        for future in concurrent.futures.as_completed(futures):
            combined_data.extend(future.result())
    
    with open(output_file, 'w') as outfile:
        outfile.writelines(combined_data)

def delete_fasta_files(input_directory, exclude_file):
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if not "combined" in file and (file.endswith(".fasta") or file.endswith(".fa")):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                #print(f"Deleting: {file}")

def process_subfolders_recursively_fast(base_directory):
    for root, dirs, files in os.walk(base_directory):
        # Only process subfolders, skip the root folder
        for subdir in dirs:
            subfolder_path = os.path.join(root, subdir)
            output_file = os.path.join(subfolder_path, f"{subdir}_combined.fasta")
            combine_fasta_files(subfolder_path, output_file)
            delete_fasta_files(subfolder_path, output_file)



#input_directory = "C:/Users/lorenzo/Desktop/phagescope/FASTA/protein_fasta/RefSeq"
#input_directory = "C:/Users/lorenzo/Desktop/aaa"
#input_directory = "C:/Users/lorenzo/Desktop/phagescope/FASTA/phage_fasta/RefSeq"

# DO NOT RUN IF A COPY HASN'T BEEN MADE!!!!!
input_directory ="C:/Users/lorenzo/Desktop/phagescope/FASTA/phage_fasta"
#input_directory = "C:/Users/lorenzo/Desktop/TMP - Copia"
import time 
start = time.time()

process_subfolders_recursively_fast(input_directory)

print("elapsed fast: " +  str(time.time() - start))


