# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 22:55:19 2024

@author: lorenzo
"""
# =============================================================================
# pip install pandas
# pip install -U scikit-learn
# pip install umap-learn
# pip install matplotlib
# pip install biopython
# pip install joblib
# pip install scipy
# pip install numpy
# pip install tqdm
# pip install psutil
# =============================================================================



#%%
#import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from umap.umap_ import UMAP
#from umap import UMAP
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



import holoviews as hv
from holoviews.operation.datashader import datashade, shade, dynspread
import datashader as ds
import pandas as pd
#import numpy as np

import panel as pn




#%%
def print_umap_tsne(k, fasta_folder):
    print(f"Processing with k={k} on folder={fasta_folder}")
    #%%
    
    # Parse Sequences and Create k-mer Frequency Profiles
    #k = 6  # k-mer length
    def get_kmers(sequence, k):
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    
    # toy example sequences
    #toy_sequences = ["AGCTAGC", "TTCGAAG", "AGCTTGC", "AGGCTTGA", "TTAAAAGCACCC", "GCCACCGAACT"]
    
    
    #%%
    
    # Specify the folder containing the FASTA files
    #fasta_folder = "C:/Users/lorenzo/Desktop/tmp" # on my pc
    #fasta_folder = "/fasta_phages" # on cluster
    
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
    
    
    
    
    
    #%%
    # =============================================================================
    # # Section to load a single fasta file instead of an entire folder
    # 
    # 
    # 
    # fasta_files = [
    #     "C:/Users/lorenzo/Desktop/phagescope/FASTA/phage_fasta/STV.fasta",
    #     #"C:/Users/lorenzo/Desktop/phagescope/FASTA/phage_fasta/IGVD.fasta",
    #     #"C:/Users/lorenzo/Desktop/phagescope/FASTA/phage_fasta/RefSeq.fasta"
    # ]
    # 
    # # Load sequences
    # sequences = []
    # for file in fasta_files:
    #     sequences.extend([str(record.seq) for record in SeqIO.parse(file, "fasta")])
    # 
    # print(f"Loaded {len(sequences)} sequences")
    # 
    # =============================================================================
    #%%
    
    # =============================================================================
    # # 1ST OPTION TO VECTORIZE SEQUENTIALLY
    # 
    # # Step 2: Convert sequences to k-mer frequency profiles
    # start1 = time.time()
    # 
    # vectorizer = CountVectorizer(analyzer=lambda seq: get_kmers(seq, k), lowercase=False)
    # print("elapsed countVectorizer: " +  str(time.time() - start1))
    # start2 = time.time()
    # X_kmers = vectorizer.fit_transform(sequences)
    # 
    # print("elapsed Vectorizer.fit: " +  str(time.time() - start2))
    # 
    # 
    # =============================================================================
    #%%
    
# =============================================================================
#     
#     # 2ND OPTION TO VECTORIZE IN PARALLEL
    
    # Step 1: Parallel k-mer extraction and conversion to strings
    def extract_kmers_as_string(sequence, k):
        """Convert k-mers to a single string for CountVectorizer."""
        return " ".join(get_kmers(sequence, k))
    
    start_preprocess = time.time()
    
    # Parallel preprocessing
    preprocessed_sequences = Parallel(n_jobs=-1, prefer="threads")(
        delayed(extract_kmers_as_string)(seq, k) for seq in sequences
    )
    print("Elapsed preprocessing time:", time.time() - start_preprocess)
    
    # Step 2: Fit and transform with CountVectorizer
    start_vectorize = time.time()
    vectorizer = CountVectorizer(lowercase=False)
    X_kmers = vectorizer.fit_transform(preprocessed_sequences)
    print("Elapsed vectorizing time:", time.time() - start_vectorize)
    print(f" sparse matrix [{len(sequences)}, {vectorizer.get_feature_names_out().size}]of {X_kmers.data.nbytes} bytes allocated")
    
    
# 
# =============================================================================
    
    #%%
    
# =============================================================================
#     
#     # 3rd option with HashVectorizer
#     # 
#     # Step 2: Convert sequences to k-mer frequency profiles using HashingVectorizer
#     start1 = time.time()
# 
#     vectorizer = HashingVectorizer(analyzer=lambda seq: get_kmers(seq, k), lowercase=False, n_features=2**20)  # You can adjust n_features as needed
#     print("elapsed HashingVectorizer initialization: " + str(time.time() - start1))
# 
#     start2 = time.time()
#     X_kmers = vectorizer.fit_transform(sequences)
# 
#     print("elapsed Vectorizer.fit_transform: " + str(time.time() - start2))
#     #print(f" sparse matrix [{len(sequences)}, {vectorizer.get_feature_names_out().size}]of {X_kmers.data.nbytes} bytes allocated")
# 
# 
# =============================================================================
    
    #%%

    # Now that i have the k-mer frequency matrix i can delete sequences from ram
    print (f"deleting sequences and freeing {str(sys.getsizeof(sequences))}" )
    del sequences
    
    #%%
    # Display k-mer frequency profiles
    #print("k-mer Frequency Matrix:")
    #print(pd.DataFrame(X_kmers.toarray(), columns=vectorizer.get_feature_names_out()))
    
    # =============================================================================
    # # UAMP AND TSNE PLOTS WITHOUT PCA
    # #%%
    # # Step 3: Apply UMAP
    # umap_model = UMAP(n_neighbors=20, min_dist=0.15, metric='correlation')
    # X_umap = umap_model.fit_transform(X_kmers.toarray())
    # 
    # #%%
    # # Step 4: Plot
    # plt.figure(figsize=(6, 4))
    # plt.scatter(X_umap[:, 0], X_umap[:, 1], s=5, alpha= 0.2)
    # plt.title("UMAP projection of k-mer frequencies")
    # plt.xlabel("UMAP1")
    # plt.ylabel("UMAP2")
    # plt.show()
    # 
    # 
    # #%%
    # 
    # 
    # 
    # # Step 1: Set up the t-SNE model
    # # Here, n_components=2 projects down to 2D, and the metric is set to cosine for similarity based on k-mer frequency profiles.
    # tsne_model = TSNE(n_components=2, perplexity=50, metric='cosine', random_state=42)
    # 
    # # Step 2: Fit and transform the k-mer frequency data
    # # Using .toarray() to convert sparse matrix to dense
    # X_tsne = tsne_model.fit_transform(X_kmers.toarray())
    # 
    # # Step 3: Plot the t-SNE result
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, color='green', alpha= 0.2)
    # plt.title("t-SNE projection of k-mer frequencies")
    # plt.xlabel("t-SNE Component 1")
    # plt.ylabel("t-SNE Component 2")
    # plt.show()
    # 
    # =============================================================================
    #%%
    # First PCA and then umap
    
    #from sklearn.decomposition import PCA
    
    # Step 1: PCA Reduction
    #pca_model = PCA(n_components=50, random_state=42)
    
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

    # Step 3: Plot the PCA results




    hv.extension('bokeh')  # Enables interactive plots with the Bokeh backend

    # Assuming X_pca and k are already defined
    data = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1]})

    # Convert the data to a HoloViews Points object
    points = hv.Points(data, kdims=['PCA1', 'PCA2'])

    # Use datashade to dynamically aggregate and render the points
    interactive_plot = datashade(
        points,
        aggregator=ds.count(),
        cmap=['lightblue', 'red']
    )

    # Optionally spread out sparse points for better visibility
    interactive_plot = dynspread(interactive_plot, threshold=0.5, max_px=4)

    # Customize plot options
    interactive_plot = interactive_plot.opts(
        title=f"PCA k = {k}",
        xlabel="PCA Component 1",
        ylabel="PCA Component 2",
        width=800,
        height=600,
        tools=['box_zoom', 'pan', 'wheel_zoom', 'reset'],
        active_tools=['wheel_zoom']
    )

    # Show the plot in a notebook or save as an HTML file
    hv.save(interactive_plot, f'PCA_plot_k{k}.html', backend='bokeh')  # Save to HTML
    interactive_plot

    pn.extension()

    # Serve the interactive plot
    pn.panel(interactive_plot).show(port=5006)










    #%%
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=5, color='red', alpha=0.2)
    plt.title(f"PCA k = {k}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    #plt.show()
    plt.savefig(f"PCA_plot_k{k}.tiff", format='tiff', dpi=300)
    plt.close()
    
    #%%
    
    # Now that i have the 50-component PCA i can delete frequency matrix from ram
    # print (f"deleting freq matrix and freeing {str(sys.getsizeof(X_kmers))}" )
    # del X_kmers
    
    
    #%%
    # Step 2: UMAP on the PCA-transformed data
    umap_model = UMAP(n_neighbors=5, min_dist=0.1, metric='cosine', low_memory=True)
    X_umap = umap_model.fit_transform(X_pca)
    
    #%
    # Step 3: Plot UMAP result
    plt.figure(figsize=(8, 6))
    plt.scatter(X_umap[:, 0], X_umap[:, 1], s=5, color='blue', alpha=0.2)
    plt.title(f"UMAP projection after PCA reduction k = {k}")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    #plt.show()
    plt.savefig(f"UMAP_plot_k{k}.tiff", format='tiff', dpi=300)
    plt.close()
    #%%
    
    # Step 4: t-SNE on the PCA-transformed data
    tsne_model = TSNE(n_components=2, perplexity=30, metric='cosine')
    X_tsne = tsne_model.fit_transform(X_pca)
    
    
    # Step 5: Plot t-SNE result
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=5, color='green', alpha=0.2)
    plt.title(f"t-SNE projection after PCA reduction k = {k}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    #plt.show()
    plt.savefig(f"tSNE_plot_k{k}.tiff", format='tiff', dpi=300)
    plt.close()
    




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






