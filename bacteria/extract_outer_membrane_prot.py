import os
from Bio import SeqIO
import subprocess


def parse_psortb_results(psortb_output_file):
    """Parse psortb results to extract SeqIDs with 'OuterMembrane' as Final Prediction."""
    outer_membrane_ids = []
    try:
        with open(psortb_output_file, 'r') as file:
            seq_id = None
            is_outer_membrane = False
            for line in file:
                if line.startswith("SeqID:"):
                    if seq_id and is_outer_membrane:
                        outer_membrane_ids.append(seq_id)
                    seq_id = line.split(" ", 1)[1].strip()
                    is_outer_membrane = False  # Reset for the new sequence
                elif line.startswith("  Final Prediction:"):
                    if "OuterMembrane" in line:
                        is_outer_membrane = True
            # Append the last seq_id if applicable
            if seq_id and is_outer_membrane:
                outer_membrane_ids.append(seq_id)
    except FileNotFoundError:
        print(f"Error: PSORTb result file not found: {psortb_output_file}")
    return outer_membrane_ids


def filter_fasta_by_ids(fasta_file, output_file, ids_to_keep):
    """Filter a FASTA file to include only sequences with IDs in ids_to_keep."""
    with open(output_file, 'w') as out_f:
        for record in SeqIO.parse(fasta_file, "fasta"):
            if record.id in ids_to_keep:
                SeqIO.write(record, out_f, "fasta")


import os

def get_last_modified_file(folder):
    """Get the most recently modified file in the given folder."""
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    except Exception as e:
        print(f"Error finding last modified file: {e}")
        return None
    
def run_psortb(psortb_executable, fasta_file, psortb_results_folder):
    """Run PSORTb for a single FASTA file and return the path to the result file."""
    command = [
        psortb_executable,
        "-i", fasta_file,
        "-r", psortb_results_folder,
        "-n",  # Adjust for Gram-negative
        "--verbose"
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running PSORTb on {fasta_file}: {e.stderr}")
        return None
    
    # Find the last modified result file
    result_file = get_last_modified_file(psortb_results_folder)
    if not result_file:
        print(f"Error: No result file found in {psortb_results_folder}after running PSORTb.")
              
    return result_file


def main(psortb_executable, fasta_folder, psortb_results_folder, output_folder):
    """Process each FASTA file, run PSORTb, and create filtered output."""
    os.makedirs(psortb_results_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    for fasta_file in os.listdir(fasta_folder):
        if not fasta_file.endswith(".fasta"):
            continue

        fasta_path = os.path.join(fasta_folder, fasta_file)
        print(f"Processing {fasta_path}...")

        # 1) Run PSORTb
        psortb_result_file = run_psortb(psortb_executable, fasta_path, psortb_results_folder)
        if not psortb_result_file:
            print(f"Skipping {fasta_path} due to PSORTb error.")
            continue

        # 2) Parse PSORTb Results
        outer_membrane_ids = parse_psortb_results(psortb_result_file)

        # 3) Create a new FASTA file containing only OuterMembrane proteins
        output_fasta_file = os.path.join(output_folder, f"{os.path.splitext(fasta_file)[0]}_only_OuterMembrane.fasta")
        if not outer_membrane_ids:
            print(f"No OuterMembrane proteins found in {fasta_file}.")
            continue
        filter_fasta_by_ids(fasta_path, output_fasta_file, outer_membrane_ids)
        print(f"Created {output_fasta_file} with {len(outer_membrane_ids)} OuterMembrane proteins.")


if __name__ == "__main__":
    # Paths
    PSORTB_EXECUTABLE = r"./psortb_app"  # Update to your PSORTb executable
    FASTA_FOLDER = r"/mnt/c/Users/lorenzo/Desktop/datasets/bacteria_proteomes"
    PSORTB_RESULTS_FOLDER = r"/mnt/c/Users/lorenzo/Desktop/results"
    OUTPUT_FOLDER = r"/mnt/c/Users/lorenzo/Desktop/filtered_fasta"

    main(PSORTB_EXECUTABLE, FASTA_FOLDER, PSORTB_RESULTS_FOLDER, OUTPUT_FOLDER)
