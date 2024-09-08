import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import textwrap
import time
import faiss  # Import FAISS for similarity search

# Load the embedding model
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", trust_remote_code=True, device="cuda")

all_embeddings = []

def load_embeddings(embeddings_df_save_paths: list[str]) -> list[dict]:
    """
    Load multiple CSVs with embeddings and return a list of dictionaries containing embeddings and metadata.

    Parameters:
        embeddings_df_save_paths (list[str]): List of paths to CSV files containing embeddings.

    Returns:
        list[dict]: A list of dictionaries where each dictionary contains the embeddings, page chunks, and source (book).
    """
    global all_embeddings

    for csv_path in embeddings_df_save_paths:
        df = pd.read_csv(csv_path)

        # Convert the stringified numpy arrays in the 'embeddings' column back to actual arrays
        df["embeddings"] = df["embeddings"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        embeddings = np.array(df["embeddings"].tolist()).astype('float32')

        # Create a record for each chunk with its metadata and the source (book) name
        pages_and_chunks = df.to_dict(orient="records")
        source_name = csv_path.split("/")[-1]  # Extract the source name (e.g., the file name)
        for item in pages_and_chunks:
            item["source"] = source_name  # Add source information to each record

        # Store embeddings and metadata in a dictionary
        all_embeddings.append({
            "embeddings": embeddings,
            "pages_and_chunks": pages_and_chunks
        })

    return all_embeddings

def build_faiss_index(all_embeddings: list[dict]) -> list[faiss.IndexFlatIP]:
    """
    Build FAISS indexes for all embeddings.

    Parameters:
        all_embeddings (list[dict]): List of embeddings from multiple CSV files.

    Returns:
        list[faiss.IndexFlatIP]: List of FAISS indexes built from the embeddings.
    """
    faiss_indices = []
    
    for dataset in all_embeddings:
        embeddings = dataset["embeddings"]

        # Create a FAISS index for inner product (cosine similarity equivalent)
        index = faiss.IndexFlatIP(embeddings.shape[1])  # dimension is the size of the embedding vector

        # Normalize the embeddings to ensure cosine similarity
        faiss.normalize_L2(embeddings)

        # Add embeddings to the FAISS index
        index.add(embeddings)

        faiss_indices.append({
            "index": index,
            "pages_and_chunks": dataset["pages_and_chunks"]
        })

    return faiss_indices

def retrieve_relevant_resources(query: str, faiss_indices: list[dict], model: SentenceTransformer, top_k: int = 5):
    """
    Retrieve the top-k most relevant passages across all FAISS indices.

    Parameters:
        query (str): The user's query.
        faiss_indices (list[dict]): List of FAISS indices and their associated metadata.
        model (SentenceTransformer): The sentence transformer model to embed the query.
        top_k (int): Number of top results to return.

    Returns:
        list[dict]: A list of top results sorted by similarity scores.
    """
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=False, device="cuda").astype('float32')
    query_embedding = query_embedding.reshape(1, -1)

    # Normalize query embedding for cosine similarity search
    faiss.normalize_L2(query_embedding)

    all_results = []
    start_time = time.time()

    # Search in each FAISS index
    for dataset in faiss_indices:
        index = dataset["index"]
        pages_and_chunks = dataset["pages_and_chunks"]

        # Perform FAISS search (using inner product)
        distances, indices = index.search(query_embedding.reshape(1, -1), top_k)

        # Collect top results with metadata
        for score, idx in zip(distances[0], indices[0]):
            all_results.append({
                "score": score,
                "page_number": pages_and_chunks[idx]["page_number"],
                "sentence_chunk": pages_and_chunks[idx]["sentence_chunk"],
                "source": pages_and_chunks[idx]["source"]
            })
    
    end_time = time.time()
    
    print(f"[INFO] Time taken to search across all FAISS indices: {end_time-start_time:.5f} seconds.")

    # Sort all results by score in descending order
    sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

    return sorted_results

def print_top_results(query: str, top_results: list[dict], top_k: int = 5):
    """
    Print the top-k relevant passages with their scores, page numbers, and sources.

    Parameters:
        query (str): The user's query.
        top_results (list[dict]): List of the top relevant passages.
        top_k (int): Number of top results to print.
    """
    print(f"Query: {query}\n")
    print("Top Results:\n")

    for i, result in enumerate(top_results[:top_k]):
        print(f"Result {i+1}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page Number: {result['page_number']}")
        print("Passage:")
        print_wrapped(result["sentence_chunk"])
        print("\n")

def print_wrapped(text, wrap_length=80):
    """
    Helper function to wrap text for better readability in the console.
    
    Parameters:
        text (str): The text to wrap.
        wrap_length (int): The maximum line length.
    """
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

# Example usage

# List of CSV files containing embeddings from different textbooks
embedding_csvs = [
    "embeddings/ostep_text_chunks_and_embeddings.csv",
    "embeddings/neural_network_text_chunks_and_embeddings.csv",
    "embeddings/algorithm_design_manual_text_chunks_and_embeddings.csv",
]

# Load embeddings from multiple sources
all_embeddings = load_embeddings(embedding_csvs)

# Build FAISS indices for all embeddings
faiss_indices = build_faiss_index(all_embeddings)

with open("queries.txt", "r") as file:
    queries = file.readlines()

for query in queries:
    # Retrieve top 5 passages from all textbooks using FAISS
    top_results = retrieve_relevant_resources(query, faiss_indices, embedding_model, top_k=1)

    # Print the results
    print_top_results(query, top_results, top_k=5)
