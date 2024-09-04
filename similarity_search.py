import random
from sentence_transformers import util, SentenceTransformer
import textwrap
import torch
import numpy as np
import pandas as pd
import time

# Load embedding model
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", 
                                    trust_remote_code=True,
                                    device="cuda")

def load_embeddings(embeddings_df_save_path: str) -> tuple:
    # Import text embeddings
    text_chunks_and_embedding_df = pd.read_csv(embeddings_df_save_path)
    device = "cuda"
    # Convert embedding column back to np.array (it got converted to string when it got saved to CSV)
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    # Convert texts and embedding df to list of dicts
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")
    # Convert embeddings to torch tensor and send to device (note: NumPy arrays are float64, torch tensors are float32 by default)
    embeddings = torch.tensor(np.array(text_chunks_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)
    
    return embeddings, pages_and_chunks

embeddings_df_save_path = "ostep_text_chunks_and_embeddings_df.csv"
embeddings, pages_and_chunks = load_embeddings(embeddings_df_save_path)

def retrieve_relevant_resources(query: str,
                                embeddings: torch.tensor,
                                model: SentenceTransformer=embedding_model,
                                n_resources_to_return: int=5,
                                print_time: bool=True):
    """
    Embeds a query with model and returns top k scores and indices from embeddings.
    """

    # Embed the query
    query_prompt_name = "s2p_query"
    query_embedding = model.encode(query,
                                   prompt_name=query_prompt_name,
                                   convert_to_tensor=True,
                                   device="cuda")

    # Get dot product or cosine_similarity scores on embeddings
    start_time = time.time()

    # cosine_similarity_scores = cosine_similarity(query_embedding, embeddings)
    dot_scores = util.dot_score(query_embedding, embeddings)[0]
    end_time = time.time()
    
    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time-start_time:.5f} seconds.")
    
    scores, indices = torch.topk(input=dot_scores,
                                 k=n_resources_to_return)
    
    return scores, indices

def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

def print_top_results_and_scores(query: str,
                                 embeddings: torch.tensor,
                                 pages_and_chunks: list[dict]=pages_and_chunks,
                                 n_resources_to_return: int=5):
    """
    Takes a query, retrieves most relevant resources and prints them out in descending order.
    Note: Requires pages_and_chunks to be formatted in a specific way (see above for reference).
    """
    scores, indices = retrieve_relevant_resources(query=query,
                                                  embeddings=embeddings,
                                                  n_resources_to_return=n_resources_to_return)
    
    print(f"Query: {query}\n")
    print("Results:")

    # Loop through zipped together scores and indices
    for score, index in zip(scores, indices):
        print(f"Score: {score:.4f}")
        # Print relevant sentence chunk (since the scores are in descending order, the most relevant chunk will be first)
        print_wrapped(pages_and_chunks[index]["sentence_chunk"])
        print(f"Page number: {pages_and_chunks[index]['page_number']}")
        print("\n")

query = "How do operating systems use virtualization to manage memory"

# Get just the scores and indices of top related results
scores, indices = retrieve_relevant_resources(query=query,
                                              embeddings=embeddings)
print_top_results_and_scores(query=query,
                             embeddings=embeddings)
