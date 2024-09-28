import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import textwrap
import time

from unixcoder import UniXcoder

from pdf_processing_and_embedding import get_single_code_embedding

# Load the embedding model
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", trust_remote_code=True, device="cuda")

# Set up UniXcoder for code embeddings
device = torch.device("cuda")
code_embedding_model = UniXcoder("microsoft/unixcoder-base")
code_embedding_model.to(device)

all_embeddings = []

def load_embeddings(embeddings_df_save_paths: list[str], is_text: bool=True) -> list[dict]:
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
        if is_text:
            df["embeddings"] = df["embeddings"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
            embeddings = torch.tensor(np.array(df["embeddings"].tolist()), dtype=torch.float32).to("cuda")

        else:
            df["code_embeddings"] = df["code_embeddings"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
            embeddings = torch.tensor(np.array(df["code_embeddings"].tolist()), dtype=torch.float32).to("cuda")

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

def retrieve_top_code_embeddings(query: str, all_embeddings: list[dict], top_k: int=5) -> float: 
    """
    Retrieve the top-k most relevant code snippets across all embedding databases.
    Parameters:
        query (str): The user's query.
        all_embeddings (list[dict]): List of embeddings and their associated metadata.
        top_k (int): Number of top results to return.

    Returns:
        list[dict]: A list of top results sorted by similarity scores.
    """
    nlq_emb = torch.from_numpy(get_single_code_embedding(query)).cuda()
    for dataset in all_embeddings:
        embeddings = dataset["embeddings"]
        pages_and_chunks = dataset["pages_and_chunks"]
        cos_scores = torch.nn.functional.cosine_similarity(nlq_emb, embeddings)
        top_results = torch.topk(cos_scores, k=top_k)
        top_values, top_indices = top_results.values, top_results.indices
    all_results = []
    # Collect top results with metadata
    for score, idx in zip(top_values, top_indices):
        all_results.append({
            "score": score,
            "page_number": pages_and_chunks[idx]["page_number"],
            "code_snippet": pages_and_chunks[idx]["code"],
            "source": pages_and_chunks[idx]["source"]
        })
    sorted_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
    return sorted_results

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

    return sorted_results #[:top_k]

def format_code_snippet(code_snippet: str) -> str:
    """
    Format the code snippet by breaking lines at certain symbols and adding indentation to improve readability.
    Parameters:
        code_snippet (str): The raw code snippet as a single line.

    Returns:
        str: Formatted code snippet with proper line breaks and indentation.
    """
    # Define symbols where we break the line
    break_symbols = ['{', '}', ';']

    # Initialize variables for formatted code and indentation level
    formatted_code = ""
    indent_level = 0
    indent_spaces = 4  # Number of spaces for each indent level

    # Split the code snippet into tokens based on break symbols
    tokens = []
    current_token = ""

    for char in code_snippet:
        current_token += char
        if char in break_symbols:
            tokens.append(current_token.strip())
            current_token = ""

    # Append any remaining characters as the final token
    if current_token.strip():
        tokens.append(current_token.strip())

    # Process each token and apply indentation
    for token in tokens:
        stripped_token = token.strip()

        # Dedent if the token starts with '}', since this ends a block
        if stripped_token.startswith("}"):
            indent_level -= 1

        # Add the token with proper indentation
        formatted_code += " " * (indent_level * indent_spaces) + stripped_token + "\n"

        # Indent if the token ends with '{', since this starts a block
        if stripped_token.endswith("{"):
            indent_level += 1

    return formatted_code.strip()

def print_top_results(query: str, top_results: list[dict], is_text: bool=True):
    """
    Print the top-k relevant passages with their scores, page numbers, and sources.

    Parameters:
        query (str): The user's query.
        top_results (list[dict]): List of the top relevant passages.
        top_k (int): Number of top results to print.
    """
    print(f"Query: {query}\n")
    print("Top Results:\n")

    for i, result in enumerate(top_results): #[:top_k]):
        print(f"Result {i+1}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page Number: {result['page_number']}")
        if is_text:
            print("Passage:")
            print_wrapped(result["sentence_chunk"])
        else: 
            print("Code Snippet:")
            print(result["code_snippet"])
            formatted_code = format_code_snippet(result["code_snippet"])
            # print_wrapped(formatted_code)
            result["code_snippet"] = formatted_code

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

def write_top_result_to_file(query: str, top_result: dict, filename: str="results.txt", is_text: bool=True):
    """
    Write the top result for a query to a text file.
    Parameters:
        query (str): The user's query.
        top_result (dict): The top relevant passage.
        filename (str): Name of the file to write the results to.
    """
    with open(filename, "a") as file:
        file.write("Cosine Similarity Scores on queries \n")
        file.write("Overlap = 2\n\n")
        file.write(f"Query: {query}\n")
        file.write(f"Top Result:\n")
        file.write(f"Score: {top_result['score']:.4f}\n")
        file.write(f"Source: {top_result['source']}\n")
        file.write(f"Page Number: {top_result['page_number']}\n")
        if not is_text: file.write(f"Code Snippet: {top_result['code_snippet']}\n")
        else: file.write(f"Passage: {top_result['sentence_chunk']}\n")
        file.write("\n-------------------------\n\n")


# List of CSV files containing embeddings from different textbooks
embedding_csvs = [
    "operating_systems_three_easy_pieces_embeddings.csv",
    "algorithm-design-manual_embeddings.csv",
#    "neural_network_text_chunks_and_embeddings.csv",
#    "cog_sci_foundations_text_chunks_and_embeddings.csv",
]

text_embedding_csvs = ["embeddings/text/" + str(csv_name) for csv_name in embedding_csvs]
code_embedding_csvs = ["embeddings/code/" + str(csv_name) for csv_name in embedding_csvs]

### Text or Code based Queries
all_embeddings = load_embeddings(code_embedding_csvs, is_text=False)

with open("code_queries.txt", "r") as file:
    queries = file.readlines()

for query in queries:
    if not query.startswith("#"):
        # Retrieve top passages from all textbooks
        top_results = retrieve_top_code_embeddings(query, all_embeddings, top_k=3)

        # Print the results
        print_top_results(query, top_results, is_text=False)

        if top_results:
            write_top_result_to_file(query, top_results[0], filename="results/code_search_results.txt", is_text=False)

### Testing Code
# code_embedding_csvs = ["embeddings/text/test_embeddings.csv"]
# all_embeddings = load_embeddings(code_embedding_csvs, is_text=False)

# query = "priority queue"
# top_results = get_top_code_embeddings(query, all_embeddings, top_k=3)
# print_top_results(query, top_results, is_text=False)
