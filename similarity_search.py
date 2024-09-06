import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import textwrap
import time

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
#    all_embeddings = []

    for csv_path in embeddings_df_save_paths:
        df = pd.read_csv(csv_path)

        # Convert the stringified numpy arrays in the 'embeddings' column back to actual arrays
        df["embeddings"] = df["embeddings"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        embeddings = torch.tensor(np.array(df["embeddings"].tolist()), dtype=torch.float32).to("cuda")

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

def retrieve_relevant_resources(query: str, all_embeddings: list[dict], model: SentenceTransformer, top_k: int=5):
    """
    Retrieve the top-k most relevant passages across all embedding databases.

    Parameters:
        query (str): The user's query.
        all_embeddings (list[dict]): List of embeddings and their associated metadata.
        model (SentenceTransformer): The sentence transformer model to embed the query.
        top_k (int): Number of top results to return.

    Returns:
        list[dict]: A list of top results sorted by similarity scores.
    """
    # Embed the query
    query_embedding = model.encode(query, convert_to_tensor=True, device="cuda")

    all_results = []

    start_time = time.time()

    # Calculate cosine similarity for each embedding database
    for dataset in all_embeddings:
        embeddings = dataset["embeddings"]
        pages_and_chunks = dataset["pages_and_chunks"]

        # Compute cosine similarity between query embedding and text embeddings
        cosine_similarities = torch.nn.functional.cosine_similarity(query_embedding, embeddings)

        # Get the top-k results for this dataset
        top_scores, top_indices = torch.topk(cosine_similarities, k=top_k)

        # Collect top results with metadata
        for score, idx in zip(top_scores, top_indices):
            all_results.append({
                "score": score,
                "page_number": pages_and_chunks[idx]["page_number"],
                "sentence_chunk": pages_and_chunks[idx]["sentence_chunk"],
                "source": pages_and_chunks[idx]["source"]
            })
    
    end_time = time.time()
    
    print(f"[INFO] Time taken to get scores on all embeddings: {end_time-start_time:.5f} seconds.")

    # Sort all results by score in descending order
    sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)

    return sorted_results[:top_k]

def print_top_results(query: str, top_results: list[dict], top_k: int=5):
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

# User query
# query = "How do operating systems use virtualization to manage memory?"
# query = "Explain the Perceptron Algorithm."
query = "Create the smallest deterministic finite automaton M' such that M' behaves identically to M."

# Retrieve top 5 passages from all textbooks
top_results = retrieve_relevant_resources(query, all_embeddings, embedding_model, top_k=5)

# Print the results
print_top_results(query, top_results, top_k=5)
