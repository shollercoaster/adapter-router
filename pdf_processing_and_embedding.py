import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English

# Initialize NLP model and sentence transformer globally
nlp = English()
nlp.add_pipe("sentencizer")  # Add sentence segmentation capability
embedding_model = SentenceTransformer("all-mpnet-base-v2", trust_remote_code=True, device="cuda")

def text_formatter(text: str) -> str:
    """Cleans and formats text: removes extra newlines and trims whitespace."""
    return text.replace("\n", " ").strip()

def open_and_read_pdf(pdf_path: str, start_page: int, end_page: int) -> list[dict]:
    """
    Opens a PDF file and extracts text from the specified page range.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        start_page (int): Start page number (0-indexed).
        end_page (int): End page number (0-indexed).
    
    Returns:
        list[dict]: A list of dictionaries containing page metadata and text.
    """
    doc = fitz.open(pdf_path)  # Open the PDF document
    pages_and_texts = []

    # Iterate over the pages within the specified range
    for page_num, page in tqdm(enumerate(doc[start_page:end_page])):
        text = page.get_text()  # Extract text from the page
        text = text_formatter(text)  # Clean and format the text
        pages_and_texts.append({
            "page_number": page_num + 1,  # Adjust page number to 1-based indexing
            "page_char_count": len(text),  # Record character count for the page
            "page_word_count": len(text.split(" ")), # Record word count
            "page_sentence_count_raw": len(text.split(". ")), # Raw period-splitted sentence count
            "page_token_count": len(text) / 4,  # 1 token = ~4 chars
            "text": text  # Store the extracted text
        })

    return pages_and_texts

def sentence_chunking(pages_and_texts: list[dict]) -> None:
    """
    Splits text from each page into individual sentences using spaCy's NLP model.
    
    Parameters:
        pages_and_texts (list[dict]): List of dictionaries containing page text.
    """
    for item in pages_and_texts:
        doc = nlp(item["text"])  # Apply NLP pipeline to extract sentences
        item["sentences"] = [str(sent) for sent in doc.sents]  # Convert sentences to strings

def split_list(input_list: list, chunk_size: int) -> list[list[str]]:
    """
    Splits a list into smaller sublists of a given size.
    
    Parameters:
        input_list (list): List to be split.
        chunk_size (int): Number of items in each chunk.
    
    Returns:
        list[list]: List of sublists, each with up to chunk_size elements.
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def merge_and_filter_chunks(pages_and_texts: list[dict], num_sentence_chunk_size: int, min_token_length: int) -> list[dict]:
    """
    Merges sentences into chunks and filters chunks based on token count.
    
    Parameters:
        pages_and_texts (list[dict]): List of dictionaries containing page sentences.
        num_sentence_chunk_size (int): Number of sentences per chunk.
        min_token_length (int): Minimum token count for chunks to be considered valid.
    
    Returns:
        list[dict]: A list of filtered sentence chunks with metadata.
    """
    pages_and_chunks = []

    for item in pages_and_texts:
        sentence_chunks = split_list(item["sentences"], num_sentence_chunk_size)  # Split sentences into chunks
        
        for chunk in sentence_chunks:
            # Join sentences into a single string and clean up spacing
            chunk_text = "".join(chunk).replace("  ", " ").strip()
            chunk_text = re.sub(r'\.([A-Z])', r'. \1', chunk_text)  # Ensure correct spacing after periods
            
            chunk_info = {
                "page_number": item["page_number"],
                "sentence_chunk": chunk_text,
                "chunk_char_count": len(chunk_text),  # Record character count of the chunk
                "chunk_word_count": len([word for word in chunk_text.split(" ")]),
                "chunk_token_count": len(chunk_text) / 4  # Approximate 1 token as 4 characters
            }

            if chunk_info["chunk_token_count"] > min_token_length:  # Filter out chunks that are too small
                pages_and_chunks.append(chunk_info)

    return pages_and_chunks

def embed_chunks(pages_and_chunks: list[dict]) -> None:
    """
    Generates embeddings for each text chunk using the pre-loaded SentenceTransformer model.
    
    Parameters:
        pages_and_chunks (list[dict]): List of text chunks for embedding.
    """
    for item in tqdm(pages_and_chunks):
        # Generate and store the embedding for each chunk of text
        item["embeddings"] = embedding_model.encode(item["sentence_chunk"])

def process_pdf_for_embeddings(pdf_path: str, start_page: int, end_page: int, num_sentence_chunk_size: int, min_token_length: int, output_file: str) -> None:
    """
    Automates the process of extracting text from a PDF, chunking sentences, generating embeddings, and saving results to a CSV.
    
    Parameters:
        pdf_path (str): Path to the PDF file.
        start_page (int): Start page number (0-indexed).
        end_page (int): End page number (0-indexed).
        num_sentence_chunk_size (int): Number of sentences per chunk for embedding.
        min_token_length (int): Minimum token count for valid chunks.
        output_file (str): Path to the CSV file for saving results.
    """
    pages_and_texts = open_and_read_pdf(pdf_path, start_page, end_page)  # Extract text from PDF
    sentence_chunking(pages_and_texts)  # Split text into sentences
    pages_and_chunks = merge_and_filter_chunks(pages_and_texts, num_sentence_chunk_size, min_token_length)  # Create and filter chunks
    embed_chunks(pages_and_chunks)  # Generate embeddings for each chunk

    # Save the final embeddings to a CSV file
    df = pd.DataFrame(pages_and_chunks)
    df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")

# Statistical Analysis 
def calculate_page_statistics(pages_and_texts: list[dict]) -> pd.DataFrame:
    """
    Converts the list of page dictionaries to a DataFrame and calculates
    average and minimum token and sentence counts across all pages.
    
    Parameters:
        pages_and_texts (list[dict]): List of dictionaries containing page metadata and text.
    
    Returns:
        pd.DataFrame: A DataFrame with the token and sentence statistics.
    """
    # Convert list of dicts to a DataFrame
    df = pd.DataFrame(pages_and_texts)
    
    # Calculate average and minimum token count
    avg_token_count = df["page_token_count"].mean()
    min_token_count = df["page_token_count"].min()

    # Calculate average and minimum sentence count
    avg_sentence_count = df["page_sentence_count_raw"].mean()
    min_sentence_count = df["page_sentence_count_raw"].min()

    # Print statistics
    print(f"Average Token Count per Page: {avg_token_count:.2f}")
    print(f"Minimum Token Count per Page: {min_token_count}")
    print(f"Average Sentence Count per Page: {avg_sentence_count:.2f}")
    print(f"Minimum Sentence Count per Page: {min_sentence_count}")

    return df

# Example usage
pages_and_texts = open_and_read_pdf("data/algorithm-design-manual.pdf", start_page=14, end_page=665)
df = calculate_page_statistics(pages_and_texts)
