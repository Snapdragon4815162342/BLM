#import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.manifold import TSNE
import time
import glob
from joblib import Parallel, delayed
import argparse
import sys

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import HashingVectorizer
import psutil



def print_umap_tsne(k, fasta_folder):
    print(f"Processing with k={k} on folder={fasta_folder}")
    
    # Parse Sequences and Create k-mer Frequency Profiles
    #k = 6  # k-mer length
    def get_kmers(sequence, k):
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    

    
    # Find all FASTA files in the folder
    fasta_files = glob.glob(fasta_folder + "**/*.fasta", recursive=True)
    
    # Load sequences
    sequences = []
    prev = 0
    for file in fasta_files:
        sequences.extend([str(record.seq) for record in SeqIO.parse(file, "fasta")])
        print(f"Loaded {len(sequences) - prev} sequences from {file}")
        prev = len(sequences)
    
    print(f"Loaded {len(sequences)} sequences")
    
    
    
 
# =============================================================================
#     
#     # 2ND OPTION TO VECTORIZE IN PARALLEL
    
    # Step 1: Parallel k-mer extraction and conversion to strings
    # def extract_kmers_as_string(sequence, k):
    #     """Convert k-mers to a single string for CountVectorizer."""
    #     return " ".join(get_kmers(sequence, k))
    
    # start_preprocess = time.time()
    
    # # Parallel preprocessing
    # preprocessed_sequences = Parallel(n_jobs=-1, prefer="threads")(
    #     delayed(extract_kmers_as_string)(seq, k) for seq in sequences
    # )
    # print("Elapsed preprocessing time:", time.time() - start_preprocess)
    
    # # Step 2: Fit and transform with CountVectorizer
    # start_vectorize = time.time()
    # vectorizer = CountVectorizer(lowercase=False)
    # X_kmers = vectorizer.fit_transform(preprocessed_sequences)
    # print("Elapsed vectorizing time:", time.time() - start_vectorize)
    # print(f" sparse matrix [{len(sequences)}, {vectorizer.get_feature_names_out().size}]of {X_kmers.data.nbytes} bytes allocated")
    
    
# 
# =============================================================================
    
    #%%
    
# =============================================================================
#     
    # 3rd option with HashVectorizer
    # 
    # Step 2: Convert sequences to k-mer frequency profiles using HashingVectorizer
    

    vectorizer = HashingVectorizer(analyzer=lambda seq: get_kmers(seq, k), lowercase=False, n_features=2**20)  # You can adjust n_features as needed

    start2 = time.time()
    X_kmers = vectorizer.fit_transform(sequences)

    print("elapsed Vectorizer.fit_transform: " + str(time.time() - start2))
    #print(f" sparse matrix [{len(sequences)}, {vectorizer.get_feature_names_out().size}]of {X_kmers.data.nbytes} bytes allocated")

# 
# =============================================================================
    
    #%%

    # Now that i have the k-mer frequency matrix i can delete sequences from ram
    print (f"deleting sequences and freeing {str(sys.getsizeof(sequences))}" )
    del sequences
    
    #%%

    # =============================================================================
    # To see that FIRST 50 PCA COMPONENTS COVER 99% OF VARIANCE 
    #
    # pca = pca_model.fit(X_kmers.toarray())
    # 
    # # Print explained variance ratio for each component
    # explained_variance = pca.explained_variance_ratio_
    # print("Variance explained by each of the first 50 PCA components:")
    # print(sum(explained_variance))
    # 
    # =============================================================================
    
    #X_pca = pca_model.fit_transform(X_kmers.toarray())
    
    # =============================================================================
    # INCREMENTAL PCA

    # Step 1: Normalize the sparse matrix incrementally
    scaler = MaxAbsScaler()
    X_kmers = scaler.fit_transform(X_kmers)  # MaxAbsScaler works on sparse matrices directly

    # Step 2: Incremental PCA on the sparse matrix
    #chunk_size = 1000  # Adjust based on available memory
    
    # Calculate optimal chunk size at runtime
    # Estimate the memory size of one dense row
    sample_row_dense = X_kmers[0].toarray()
    row_size_in_bytes = sample_row_dense.nbytes

    # Get available memory in bytes
    available_memory = psutil.virtual_memory().available

    # Leave 20% of available memory as headroom
    memory_headroom = 0.2
    usable_memory = available_memory * (1 - memory_headroom)

    # Calculate optimal chunk size
    chunk_size = int(usable_memory // row_size_in_bytes)

    # Set a minimum and maximum limit for chunk size
    #chunk_size = max(100, min(chunk_size, 10000))  # Between 100 and 10,000 rows
    print(f"Using pca chunk size: {chunk_size}")    

    
    ipca_model = IncrementalPCA(n_components=50)

    # Fit IncrementalPCA in batches (convert chunks to dense temporarily)
    print("Fitting Incremental PCA:")
    num_batches = (X_kmers.shape[0] + chunk_size - 1) // chunk_size
    for start in tqdm(range(0, X_kmers.shape[0], chunk_size), total=num_batches, desc="Fitting PCA"):
        end = min(start + chunk_size, X_kmers.shape[0])
        chunk = X_kmers[start:end].toarray()  # Convert chunk to dense
        ipca_model.partial_fit(chunk)

    # Transform the data in batches
    X_pca = []
    print("Transforming data with Incremental PCA:")
    for start in tqdm(range(0, X_kmers.shape[0], chunk_size), total=num_batches, desc="Transforming Data"):
        end = min(start + chunk_size, X_kmers.shape[0])
        chunk = X_kmers[start:end].toarray()  # Convert chunk to dense
        X_pca.append(ipca_model.transform(chunk))

    # Concatenate results into a single dense array
    X_pca = np.vstack(X_pca)

    # Save pca to disk
    np.save("X_pca.npy", X_pca)

    print(f"size X_kmers: {X_kmers.data.nbytes} bytes")
    print(f"size X_pca: {X_pca.nbytes} bytes")

    #%%
    
    # Now that i have the 50-component PCA i can delete frequency matrix from ram
    print (f"deleting freq matrix and freeing {X_kmers.data.nbytes} bytes" )
    del X_kmers
    
    
    #%%
    # Step 2: UMAP on the PCA-transformed data
    umap_model = UMAP(n_neighbors=5, min_dist=0.1, metric='cosine', low_memory=True)
    X_umap = umap_model.fit_transform(X_pca)
    
    np.save("X_umap.npy", X_umap)

    
    # Step 4: t-SNE on the PCA-transformed data
    tsne_model = TSNE(n_components=2, perplexity=30, metric='cosine')
    X_tsne = tsne_model.fit_transform(X_pca)

    np.save("X_tsne.npy", X_tsne)




if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process a folder of FASTA files with a specified k-mer size.")

    # Add arguments
    parser.add_argument("-k", type=int, required=True, help="The k-mer size (integer).")
    parser.add_argument("-f", "--fasta_folder", type=str, required=True, help="Path to the folder containing FASTA files.")




    # Parse arguments
    args = parser.parse_args()

    # Use the parsed arguments
    print_umap_tsne(args.k, args.fasta_folder)


